"""Feature engineering (Section 3). Every feature must be known at the decision point.

Rules enforced here (and tested in tests/test_features.py):
- A past loan counts towards history only if it was disbursed on an earlier
  *date* than the current loan (same-day loans are excluded: their order is unknown).
- A past loan's outcome counts only once it was observable: its due_date plus
  LABEL_LAG_DAYS is before the current disbursement date, and it has a label.
- Macro indicators for year Y are joined to loans disbursed in Y + 1, because
  annual figures are published after the year ends.

Loans from Test are included as history (NkwaPay knows a customer's past loans)
but contribute no outcomes, because their labels are unknown.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src import config

LABEL_LAG_DAYS = 0      # days after due_date before default/repaid is known; raise if NkwaPay has a grace period
PRIOR_STRENGTH = 5      # pseudo-loans for the smoothed past default rate
MACRO_INDICATORS = {
    "Inflation, consumer prices (annual %)": "macro_inflation",
    "Official exchange rate (LCU per US$, period average)": "macro_fx",
    "Deposit interest rate (%)": "macro_deposit_rate",
    "Unemployment rate": "macro_unemployment",
}

HISTORY = ["n_prev_loans", "n_prev_resolved", "n_prev_defaults", "prev_default_rate", "has_resolved_history",
           "n_active_loans", "active_amount_log", "days_since_last_loan", "days_since_last_default",
           "prev_amount_mean_log", "prev_amount_max_log", "amount_vs_prev_mean", "prev_duration_mean",
           "n_prev_lenders"]
CALENDAR = ["disb_month", "disb_weekday", "disb_day", "disb_month_end"]
MACRO = ["macro_inflation", "macro_fx_change", "macro_deposit_rate", "macro_unemployment"]
LOAN = ["log_amount", "duration", "amount_per_day_log", "Lender_portion_Funded", "n_lenders", "is_new"]
CATEGORICAL = ["loan_type", "lender_id"]
NUMERIC = LOAN + HISTORY + CALENDAR + MACRO


def history_features(loans: pd.DataFrame, lag_days: int = LABEL_LAG_DAYS, prior=None) -> pd.DataFrame:
    """Borrower history from strictly earlier loans, indexed like `loans`.

    `loans` should hold every loan NkwaPay knows about (Train and Test). A
    `target` that is NaN means the outcome is unknown to us. `prior` fixes the
    smoothing prior (the scoring service passes the portfolio rate stored at
    training time, since it only sees one customer's loans).
    """
    cur = loans[["customer_id", "disbursement_date", "Total_Amount"]].reset_index(names="_row")
    past = loans[["customer_id", "disbursement_date", "due_date", "Total_Amount", "duration", "lender_id"]].assign(
        target=loans["target"] if "target" in loans else np.nan)
    pair = cur.merge(past, on="customer_id", suffixes=("", "_p"))
    pair = pair[pair.disbursement_date_p < pair.disbursement_date]

    known = (pair.due_date + pd.to_timedelta(lag_days, unit="D") < pair.disbursement_date) & pair.target.notna()
    active = pair.due_date >= pair.disbursement_date
    pair = pair.assign(resolved=known, default=known & (pair.target == 1), active=active,
                       active_amt=np.where(active, pair.Total_Amount_p, 0.0),
                       default_date=pair.disbursement_date_p.where(known & (pair.target == 1)))
    g = pair.groupby("_row")
    h = pd.DataFrame({
        "n_prev_loans": g.size(),
        "n_prev_resolved": g.resolved.sum(),
        "n_prev_defaults": g.default.sum(),
        "n_active_loans": g.active.sum(),
        "active_amount": g.active_amt.sum(),
        "last_loan": g.disbursement_date_p.max(),
        "last_default": g.default_date.max(),
        "prev_amount_mean": g.Total_Amount_p.mean(),
        "prev_amount_max": g.Total_Amount_p.max(),
        "prev_duration_mean": g.duration.mean(),
        "n_prev_lenders": g.lender_id.nunique(),
    }).reindex(cur._row)

    out = pd.DataFrame(index=loans.index)
    for c in ["n_prev_loans", "n_prev_resolved", "n_prev_defaults", "n_active_loans", "n_prev_lenders"]:
        out[c] = h[c].fillna(0).to_numpy().astype(int)
    if prior is None:
        prior = known_default_rate(loans, lag_days)
    out["prev_default_rate"] = (out.n_prev_defaults + PRIOR_STRENGTH * prior) / (out.n_prev_resolved + PRIOR_STRENGTH)
    out["has_resolved_history"] = (out.n_prev_resolved > 0).astype(int)
    out["active_amount_log"] = np.log1p(h.active_amount.fillna(0).to_numpy())
    d = loans.disbursement_date.to_numpy()
    out["days_since_last_loan"] = (d - h.last_loan.to_numpy()) / np.timedelta64(1, "D")
    out["days_since_last_default"] = (d - h.last_default.to_numpy()) / np.timedelta64(1, "D")
    out["prev_amount_mean_log"] = np.log1p(h.prev_amount_mean.to_numpy())
    out["prev_amount_max_log"] = np.log1p(h.prev_amount_max.to_numpy())
    out["amount_vs_prev_mean"] = loans.Total_Amount.to_numpy() / h.prev_amount_mean.to_numpy()
    out["prev_duration_mean"] = h.prev_duration_mean.to_numpy()
    # No history: leave NaN for "days since" and amount comparisons; the model pipeline imputes and flags them.
    return out


def known_default_rate(loans: pd.DataFrame, lag_days: int = LABEL_LAG_DAYS) -> np.ndarray:
    """Portfolio default rate as NkwaPay knew it on each loan's disbursement date
    (all labelled loans whose outcome was observable by then). Used as the
    smoothing prior so the prior itself carries no future labels."""
    if "target" not in loans:
        return np.zeros(len(loans))
    lab = loans[loans.target.notna()]
    seen = (lab.due_date + pd.to_timedelta(lag_days, unit="D")).to_numpy()
    order = np.argsort(seen)
    seen, y = seen[order], lab.target.to_numpy()[order]
    cum_n, cum_d = np.arange(1, len(y) + 1), np.cumsum(y)
    k = np.searchsorted(seen, loans.disbursement_date.to_numpy(), side="left")  # outcomes strictly before
    n = np.where(k > 0, cum_n[np.maximum(k - 1, 0)], 0)
    d = np.where(k > 0, cum_d[np.maximum(k - 1, 0)], 0)
    return np.where(n > 0, d / np.maximum(n, 1), 0.0)


def calendar_features(loans: pd.DataFrame) -> pd.DataFrame:
    d = loans.disbursement_date
    return pd.DataFrame({"disb_month": d.dt.month, "disb_weekday": d.dt.weekday, "disb_day": d.dt.day,
                         "disb_month_end": (d.dt.days_in_month - d.dt.day < 3).astype(int)}, index=loans.index)


def load_macro() -> pd.DataFrame:
    """Long table: country, year, indicator columns. FX change is year-on-year % depreciation."""
    return _load_macro().copy()


@lru_cache(maxsize=1)
def _load_macro() -> pd.DataFrame:
    e = pd.read_csv(config.ECON_CSV)
    e = e[e.Indicator.isin(MACRO_INDICATORS)].assign(Indicator=lambda x: x.Indicator.map(MACRO_INDICATORS))
    long = e.melt(id_vars=["Country", "Indicator"], var_name="year", value_name="v")
    long["year"] = long.year.str[2:].astype(int)
    wide = long.pivot_table(index=["Country", "year"], columns="Indicator", values="v").reset_index()
    wide = wide.sort_values(["Country", "year"])
    wide["macro_fx_change"] = wide.groupby("Country").macro_fx.pct_change() * 100
    return wide.drop(columns="macro_fx").rename(columns={"Country": "country_id"})


def macro_features(loans: pd.DataFrame) -> pd.DataFrame:
    """Previous calendar year's indicators for the loan's country."""
    key = pd.DataFrame({"country_id": loans.country_id, "year": loans.disbursement_date.dt.year - 1}, index=loans.index)
    merged = key.reset_index().merge(load_macro(), on=["country_id", "year"], how="left").set_index("index")
    return merged.reindex(loans.index)[MACRO]


def loan_features(loans: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "log_amount": np.log1p(loans.Total_Amount),
        "duration": loans.duration,
        "amount_per_day_log": np.log1p(loans.Total_Amount / loans.duration.clip(lower=1)),
        "Lender_portion_Funded": loans.Lender_portion_Funded,
        "n_lenders": loans.n_lenders,
        "is_new": (loans.New_versus_Repeat == "New Loan").astype(int),
    }, index=loans.index)


def build_features(loans: pd.DataFrame, lag_days: int = LABEL_LAG_DAYS, prior=None) -> pd.DataFrame:
    """All row-wise features for `loans` (no fitting, so safe to compute once before CV)."""
    return pd.concat([loans[CATEGORICAL].astype(str), loan_features(loans), history_features(loans, lag_days, prior),
                      calendar_features(loans), macro_features(loans)], axis=1)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Fitted inside each training fold: levels seen fewer than `min_count` times
    become 'Other', levels never seen in training become 'Unknown'."""

    def __init__(self, min_count: int = 30):
        self.min_count = min_count

    def fit(self, X, y=None):
        X = pd.DataFrame(X).astype(str)
        self.keep_ = {c: set(X[c].value_counts()[lambda s: s >= self.min_count].index) for c in X}
        self.seen_ = {c: set(X[c]) for c in X}
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).astype(str).copy()
        for c in X:
            known = X[c].isin(self.seen_[c])
            X[c] = np.where(X[c].isin(self.keep_[c]), X[c], np.where(known, "Other", "Unknown"))
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
