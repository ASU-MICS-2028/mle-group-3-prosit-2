"""No feature may use information from after the decision point (Section 3).

Replay test: for a decision date D, rebuild the world as NkwaPay saw it on D
(only loans disbursed on or before D; outcomes only where observable before D)
and recompute features for the loans disbursed on D. They must equal the
features computed once on the full dataset.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import load_loans
from src.features import LABEL_LAG_DAYS, NUMERIC, RareCategoryGrouper, build_features, CATEGORICAL, history_features
from src.schema import LEAKY


@pytest.fixture(scope="module")
def world():
    loans = pd.concat([load_loans("train"), load_loans("test")], ignore_index=True)
    return loans, build_features(loans)


def as_of(loans, day):
    seen = loans[loans.disbursement_date <= day].copy()
    unobservable = seen.due_date + pd.to_timedelta(LABEL_LAG_DAYS, unit="D") >= day
    seen.loc[unobservable, "target"] = np.nan
    return seen


@pytest.mark.parametrize("day", ["2022-01-15", "2022-08-10", "2022-11-30", "2023-06-01", "2024-09-20"])
def test_features_unchanged_when_future_removed(world, day):
    loans, full = world
    day = loans.disbursement_date[loans.disbursement_date >= pd.Timestamp(day)].min()
    past = as_of(loans, day)
    today = past.index[past.disbursement_date == day]
    assert len(today) > 0
    replay = build_features(past).loc[today]
    pd.testing.assert_frame_equal(replay, full.loc[today], check_dtype=False)


def test_future_labels_do_not_move_features(world):
    """Flip every label after a cutoff: features before the cutoff must not change."""
    loans, full = world
    cut = pd.Timestamp("2022-09-01")
    flipped = loans.copy()
    future = flipped.due_date >= cut
    flipped.loc[future, "target"] = 1 - flipped.loc[future, "target"]
    before = loans.index[loans.disbursement_date <= cut]
    pd.testing.assert_frame_equal(build_features(flipped).loc[before], full.loc[before])


def test_no_leaky_inputs(world):
    _, full = world
    assert not set(LEAKY) & set(full.columns)
    assert set(full.columns) == set(NUMERIC + CATEGORICAL)


def test_rare_grouper():
    train = pd.DataFrame({"t": ["a"] * 40 + ["b"] * 5})
    g = RareCategoryGrouper(min_count=30).fit(train)
    assert g.transform(pd.DataFrame({"t": ["a", "b", "z"]})).t.tolist() == ["a", "Other", "Unknown"]


def test_history_matches_backward_merge_asof():
    """The brief asks for history via a backward merge_asof without exact matches. history_features uses a
    self-merge with a strict '<' instead; on made-up loans (many same-day ties, unknown labels) both agree."""
    rng = np.random.default_rng(0)
    n = 400
    disb = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 60, n), unit="D")
    dur = rng.choice([1, 7, 14, 30], n)
    loans = pd.DataFrame({
        "customer_id": rng.integers(0, 30, n), "disbursement_date": disb,
        "due_date": disb + pd.to_timedelta(dur, unit="D"), "duration": dur,
        "Total_Amount": rng.uniform(100, 5000, n), "lender_id": rng.choice(["a", "b"], n),
        "target": np.where(rng.random(n) < 0.2, np.nan, (rng.random(n) < 0.3).astype(float)),
    })
    h = history_features(loans, lag_days=0, prior=0.02)

    def asof(right, key, cols):
        left = loans[["customer_id", "disbursement_date"]].reset_index().sort_values("disbursement_date")
        right = right.sort_values(key).rename(columns={key: "disbursement_date"})
        m = pd.merge_asof(left, right, on="disbursement_date", by="customer_id",
                          direction="backward", allow_exact_matches=False)
        return m.set_index("index").reindex(loans.index)[cols]

    # loans per customer per day, cumulated, then the last day strictly before each loan
    days = loans.groupby(["customer_id", "disbursement_date"]).size().rename("n").reset_index()
    days = days.assign(n_cum=days.groupby("customer_id").n.cumsum(), last_loan=days.disbursement_date)
    a = asof(days, "disbursement_date", ["n_cum", "last_loan"])
    assert (h.n_prev_loans == a.n_cum.fillna(0).astype(int)).all()
    gap = (loans.disbursement_date - a.last_loan).dt.days
    pd.testing.assert_series_equal(h.days_since_last_loan, gap.astype(float), check_names=False)

    # outcomes count from their due date, strictly before the current loan
    dues = loans[loans.target.notna()].groupby(["customer_id", "due_date"]).target.agg(["size", "sum"]).reset_index()
    dues = dues.assign(resolved=dues.groupby("customer_id")["size"].cumsum(), defaults=dues.groupby("customer_id")["sum"].cumsum())
    b = asof(dues, "due_date", ["resolved", "defaults"]).fillna(0).astype(int)
    assert (h.n_prev_resolved == b.resolved).all() and (h.n_prev_defaults == b.defaults).all()
    assert h.n_prev_loans.gt(0).any() and h.n_prev_defaults.gt(0).any()   # the check is not vacuous
