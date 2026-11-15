"""The saved model in models/ must score the example applications with no training data present."""
import json
from pathlib import Path

import pytest

from src.config import MODELS, ROOT
from src.costs import LoanEconomics
from src.scoring import VERSION, Scorer

EXAMPLES = ROOT / "examples"
EXPECTED = {
    "application_repeat.json": "approve",
    "application_decline.json": "decline",
    "application_past_default.json": "decline",
    "application_first_loan.json": "refer",
    "application_ghana.json": "refer",
}


@pytest.fixture(scope="module")
def scorer():
    path = MODELS / f"{VERSION}.joblib"
    assert path.exists(), "models/nkwapay-v1.joblib is missing; rerun notebooks/12_scoring_service.ipynb"
    return Scorer.load(path)


def test_threshold_matches_cost_matrix(scorer):
    assert scorer.threshold == pytest.approx(LoanEconomics().threshold)


@pytest.mark.parametrize("name,decision", EXPECTED.items())
def test_examples(scorer, name, decision):
    request = json.loads(Path(EXAMPLES / name).read_text())
    out = scorer.score(request["application"], request["history"])
    assert out["decision"] == decision
    assert 0 <= out["p_default"] <= 1
    if decision == "approve":
        assert out["reasons"] == [] and out["offered_amount"] <= request["application"]["Total_Amount"]
    else:
        assert 1 <= len(out["reasons"]) <= 3 and out["offered_amount"] is None


def test_earlier_default_is_never_auto_approved(scorer):
    """The model alone scores this applicant just under the threshold; the policy rule must decline."""
    request = json.loads(Path(EXAMPLES / "application_past_default.json").read_text())
    out = scorer.score(request["application"], request["history"])
    assert out["p_default"] <= scorer.threshold
    assert out["decision"] == "decline" and out["reasons"][0]["code"] == "REPAYMENT"
