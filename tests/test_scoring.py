"""The service must reproduce training-time features and return a well-formed decision."""
import numpy as np
import pandas as pd
import pytest

from src import features as F
from src import models as M
from src.data import load_loans
from src.scoring import HISTORY_FIELDS, Scorer


@pytest.fixture(scope="module")
def setup():
    scorer = Scorer.train()
    tr = load_loans("train")
    return scorer, tr


def as_application(row):
    return {"customer_id": row.customer_id, "country_id": row.country_id, "loan_type": row.loan_type,
            "lender_id": row.lender_id, "lenders": [{"lender_id": row.lender_id, "share": row.Lender_portion_Funded}] * 1
            if row.n_lenders == 1 else [{"lender_id": row.lender_id, "share": row.Lender_portion_Funded / 2}] * 2,
            "Total_Amount": row.Total_Amount, "duration": row.duration,
            "application_date": str(row.disbursement_date.date()), "New_versus_Repeat": row.New_versus_Repeat}


def test_service_features_match_training(setup):
    """For Train loans whose whole history is in Train, the service rebuilds the same features."""
    scorer, tr = setup
    full = F.build_features(tr, prior=scorer.prior)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(tr), 40, replace=False):
        row = tr.iloc[i]
        hist = tr[tr.customer_id == row.customer_id][list(HISTORY_FIELDS)]
        hist = hist[hist.disbursement_date < row.disbursement_date]
        got = scorer.features(as_application(row), hist.to_dict("records"))
        pd.testing.assert_frame_equal(got[M.DEPLOY_FEATURES].reset_index(drop=True),
                                      full.iloc[[i]][M.DEPLOY_FEATURES].reset_index(drop=True), check_dtype=False)


def test_score_contract(setup):
    scorer, tr = setup
    out = scorer.score(as_application(tr.iloc[0]), [])
    assert set(out) >= {"p_default", "decision", "reasons", "offered_amount", "latency_ms"}
    assert out["decision"] in {"approve", "decline", "refer"} and 0 <= out["p_default"] <= 1
    assert len(out["reasons"]) <= 3


def test_ghana_is_referred(setup):
    scorer, tr = setup
    app = {**as_application(tr.iloc[0]), "country_id": "Ghana"}
    assert scorer.score(app, [])["decision"] == "refer"
