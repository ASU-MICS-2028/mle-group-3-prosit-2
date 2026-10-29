"""Plain-language reason codes for the logistic regression (Section 9, reused in Part E).

For a linear model the log-odds decompose exactly:
    logit(x) = logit(reference) + sum_j beta_j * (z_j(x) - z_j(reference)),
where z is the preprocessed feature vector and the reference is the average
training applicant. Contributions are summed into reason families, because
single coefficients of correlated inputs (e.g. log_amount and
amount_per_day_log) are not interpretable on their own. Positive totals push
towards decline.

Calendar features are never offered as reasons: they carry almost no signal
(Section 9 permutation importance) and are not something a borrower can act on.
"""
import numpy as np
import pandas as pd

FAMILIES = {
    # Term and amount are one family: log_amount (+) and amount_per_day_log (-) are collinear and together
    # mostly encode log(duration), so splitting them would attribute the term effect to "size".
    "LOAN_TERMS": ["duration", "log_amount", "amount_per_day_log", "amount_vs_prev_mean"],
    "HISTORY": ["n_prev_loans", "n_prev_resolved", "has_resolved_history", "is_new", "prev_duration_mean", "n_prev_lenders"],
    "REPAYMENT": ["n_prev_defaults", "days_since_last_default", "prev_default_rate"],
    "PAST_SIZE": ["prev_amount_mean_log", "prev_amount_max_log"],
    "RECENT": ["n_active_loans", "active_amount_log", "days_since_last_loan"],
    "PRODUCT": ["loan_type"],
    "CHANNEL": ["lender_id", "Lender_portion_Funded", "n_lenders"],
    "CALENDAR": ["disb_month", "disb_weekday", "disb_day", "disb_month_end"],
}
NOT_A_REASON = {"CALENDAR"}

TEXT = {
    "LOAN_TERMS": "The amount and repayment period requested ({amount:,.0f} over {duration:.0f} days) are typical of loans that were "
                  "more often not repaid; longer repayment periods carry the most risk in our records.",
    "HISTORY": "Your borrowing history with us is short ({n_prev_loans:.0f} earlier loan(s)).",
    "REPAYMENT": "Our records show an earlier loan that was not repaid.",
    "PAST_SIZE": "The sizes of your previous loans are in a range where repayment problems have been more common.",
    "RECENT": "Your recent borrowing pattern (time since your last loan and loans currently running) differs from that of borrowers who usually repay.",
    "PRODUCT": "Loans of this product type have had higher repayment problems.",
    "CHANNEL": "Loans funded through this lending channel have had higher repayment problems.",
}


# Unseen product or lender: the model has no record, so say that instead of claiming a bad record.
TEXT_UNKNOWN = {
    "PRODUCT": "This product type is new to us, so we have no repayment record for it yet.",
    "CHANNEL": "This lending channel is new to us, so we have no repayment record for it yet.",
}
UNKNOWN_COLUMN = {"PRODUCT": "loan_type", "CHANNEL": "lender_id"}


def _family_of(name: str) -> str:
    base = name.removeprefix("missingindicator_")
    for fam, cols in FAMILIES.items():
        if base in cols or any(base.startswith(c + "_") for c in cols if c in ("loan_type", "lender_id")):
            return fam
    raise KeyError(name)


class ReasonCoder:
    """Fit on the training inputs of a fitted logreg pipeline (steps 'prep', 'model')."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X):
        prep, model = self.pipeline.named_steps["prep"], self.pipeline.named_steps["model"]
        self.names_ = prep.get_feature_names_out()
        self.coef_ = model.coef_.ravel()
        self.intercept_ = float(model.intercept_[0])
        self.ref_ = np.asarray(prep.transform(X)).mean(axis=0)
        self.family_ = np.array([_family_of(n) for n in self.names_])
        return self

    def contributions(self, X) -> pd.DataFrame:
        """Log-odds contribution of each reason family relative to the average applicant."""
        z = np.asarray(self.pipeline.named_steps["prep"].transform(X))
        c = (z - self.ref_) * self.coef_
        fam = pd.DataFrame(c, columns=self.family_, index=getattr(X, "index", None))
        return fam.T.groupby(level=0).sum().T

    def reference_logit(self) -> float:
        return self.intercept_ + float(self.ref_ @ self.coef_)

    def top_reasons(self, X: pd.DataFrame, k: int = 3) -> list:
        """Top-k risk-increasing reason families per applicant, with plain-language text."""
        contrib = self.contributions(X).drop(columns=list(NOT_A_REASON), errors="ignore")
        grouper = self.pipeline.named_steps["prep"].named_transformers_["cat"].steps[0][1]
        grouped = grouper.transform(X[list(grouper.feature_names_in_)])
        out = []
        for i, ((idx, row), (_, x)) in enumerate(zip(contrib.iterrows(), X.iterrows())):
            values = {**x.to_dict(), "amount": float(np.expm1(x["log_amount"]))}
            ranked = row[row > 0].sort_values(ascending=False).head(k)
            reasons = []
            for fam, v in ranked.items():
                unseen = fam in UNKNOWN_COLUMN and grouped[UNKNOWN_COLUMN[fam]].iloc[i] == "Unknown"
                code = f"{fam}_UNSEEN" if unseen else fam
                text = TEXT_UNKNOWN[fam] if unseen else TEXT[fam].format(**values)
                reasons.append({"code": code, "log_odds": round(float(v), 3), "text": text})
            out.append(reasons)
        return out
