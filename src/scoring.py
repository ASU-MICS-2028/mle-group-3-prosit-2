"""NkwaPay scoring service (Part E).

    scorer = Scorer.load()
    result = scorer.score(application, history)

`application` is one loan request (see APPLICATION_FIELDS); `history` is the
applicant's earlier loans with NkwaPay (see HISTORY_FIELDS), outcome included
where known. Features are built with the same code as training
(src/features.py), so there is no separate re-implementation to drift.

Decision policy (Sections 5, 6, 10 and 11):
- decline      P(default) > threshold and the applicant has a repayment record
- refer        thin-file applicant above threshold: starter-loan track or human
               review instead of an automatic decline (Section 11.3), or any
               market the model was not trained for (Section 10)
- approve      otherwise; offered amount = min(requested, predicted limit),
               capped further for first loans (Section 6.2)
"""
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from src import features as F
from src import models as M
from src.config import MODELS
from src.costs import LoanEconomics
from src.data import load_loans
from src.reasons import ReasonCoder

VERSION = "nkwapay-v1"
TRAINED_MARKETS = {"Kenya"}

APPLICATION_FIELDS = {
    "customer_id": "NkwaPay customer identifier",
    "country_id": "Market, e.g. 'Kenya'",
    "loan_type": "Product code, e.g. 'Type_1'",
    "lender_id": "Primary funding lender",
    "lenders": "All funding lenders with shares, e.g. [{'lender_id': 267278, 'share': 0.3}]",
    "Total_Amount": "Requested principal, local currency",
    "duration": "Requested term in days",
    "application_date": "Date of the decision (ISO), used as disbursement date",
    "New_versus_Repeat": "'New Loan' or 'Repeat Loan'",
}
HISTORY_FIELDS = {
    "disbursement_date": "Date the earlier loan was disbursed",
    "due_date": "Its due date",
    "Total_Amount": "Its principal",
    "duration": "Its term in days",
    "lender_id": "Its primary lender",
    "target": "1 defaulted, 0 repaid, missing if not yet known",
}


@dataclass
class Scorer:
    classifier: object
    limit_model: object
    reasons: ReasonCoder
    prior: float
    threshold: float
    starter_cap: float
    trained_through: str
    version: str = VERSION
    meta: dict = field(default_factory=dict)

    # ---- training -------------------------------------------------------
    @classmethod
    def train(cls, econ: LoanEconomics = LoanEconomics()) -> "Scorer":
        tr = load_loans("train")
        te = load_loans("test")
        world = pd.concat([tr, te], ignore_index=True)
        X = F.build_features(world)
        Xtr = X.iloc[: len(tr)]
        cols = M.inputs(M.DEPLOY_FEATURES)
        clf = M.logreg(numeric=M.DEPLOY_FEATURES, class_weight=None).fit(Xtr[cols], tr.target)
        repaid = (tr.target == 0).to_numpy()
        reg_cols = M.inputs(M.REG_FEATURES)
        lim = M.ridge().fit(Xtr.loc[repaid, reg_cols], np.log1p(tr.Total_Amount[repaid]))
        prior = float(tr.target.mean())
        first = tr[repaid & (Xtr.n_prev_loans == 0).to_numpy()]
        return cls(classifier=clf, limit_model=lim, reasons=ReasonCoder(clf).fit(Xtr[cols]), prior=prior,
                   threshold=econ.threshold, starter_cap=float(first.Total_Amount.quantile(0.5)),
                   trained_through=str(tr.disbursement_date.max().date()),
                   meta={"n_train": len(tr), "features": M.DEPLOY_FEATURES, "limit_features": M.REG_FEATURES})

    def save(self, path=None):
        path = path or MODELS / f"{self.version}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path=None) -> "Scorer":
        return joblib.load(path or MODELS / f"{VERSION}.joblib")

    # ---- scoring --------------------------------------------------------
    def features(self, application: dict, history=None) -> pd.DataFrame:
        """One feature row for the application, built exactly as in training."""
        app_date = pd.Timestamp(application["application_date"])
        lenders = application.get("lenders") or [{"lender_id": application["lender_id"], "share": np.nan}]
        current = {
            "customer_id": application["customer_id"], "country_id": application["country_id"],
            "loan_type": application["loan_type"], "lender_id": application["lender_id"],
            "Total_Amount": float(application["Total_Amount"]), "duration": int(application["duration"]),
            "disbursement_date": app_date, "due_date": app_date + pd.Timedelta(days=int(application["duration"])),
            "New_versus_Repeat": application["New_versus_Repeat"], "n_lenders": len(lenders),
            "Lender_portion_Funded": float(np.nansum([l["share"] for l in lenders])), "target": np.nan,
        }
        past = pd.DataFrame(history if history is not None else [], columns=list(HISTORY_FIELDS))
        past = past.assign(customer_id=application["customer_id"], country_id=application["country_id"],
                           disbursement_date=pd.to_datetime(past.disbursement_date), due_date=pd.to_datetime(past.due_date))
        loans = pd.concat([past, pd.DataFrame([current])], ignore_index=True)
        for c, default in [("loan_type", "Unknown"), ("New_versus_Repeat", "Repeat Loan"), ("n_lenders", 1),
                           ("Lender_portion_Funded", 0.0)]:
            loans[c] = loans[c].fillna(default) if c in loans else default
        loans = loans.astype({"Total_Amount": float, "duration": int, "target": float, "n_lenders": int,
                              "Lender_portion_Funded": float})
        return F.build_features(loans, prior=self.prior).iloc[[-1]]

    def score(self, application: dict, history=None) -> dict:
        t0 = time.perf_counter()
        X = self.features(application, history)
        p = float(self.classifier.predict_proba(X[M.inputs(M.DEPLOY_FEATURES)])[:, 1][0])
        top = self.reasons.top_reasons(X[M.inputs(M.DEPLOY_FEATURES)])[0]
        limit = float(np.expm1(self.limit_model.predict(X[M.inputs(M.REG_FEATURES)])[0]))
        thin_file = int(X.n_prev_resolved.iloc[0]) == 0
        requested = float(application["Total_Amount"])

        if application["country_id"] not in TRAINED_MARKETS:
            decision, note = "refer", f"Model not validated for {application['country_id']}; human review required (Part C)."
        elif p > self.threshold and thin_file:
            decision, note = "refer", "No repayment record: starter-loan track or human review, not automatic decline (Section 11.3)."
        elif p > self.threshold:
            decision, note = "decline", "Estimated default risk above the cost threshold."
        else:
            decision, note = "approve", ""
        offer = None
        if decision == "approve":
            offer = min(requested, limit)
            if thin_file:
                offer = min(offer, self.starter_cap)
        return {
            "model_version": self.version, "p_default": round(p, 4), "threshold": round(self.threshold, 4),
            "decision": decision, "note": note,
            "reasons": top if decision != "approve" else [],
            "requested_amount": requested, "suggested_limit": round(limit, 2),
            "offered_amount": None if offer is None else round(offer, 2),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
