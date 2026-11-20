"""Policy settings and the reviewer step, on the saved model (no training data needed)."""
import json
from datetime import datetime, timezone

import pytest

from src.config import MODELS, ROOT
from src.policy import Policy, record_review
from src.scoring import VERSION, Scorer


def load(name):
    req = json.loads((ROOT / "examples" / name).read_text())
    return req["application"], req.get("history") or []


@pytest.fixture(scope="module")
def scorer():
    return Scorer.load(MODELS / f"{VERSION}.joblib")


def test_default_policy_uses_saved_first_loan_cap(scorer):
    assert scorer.default_policy().cap("Kenya") == scorer.starter_cap
    assert scorer.default_policy().cap("Ghana") is None


def test_approvals_and_declines_have_no_review(scorer):
    for name in ["application_repeat.json", "application_decline.json", "application_past_default.json"]:
        out = scorer.score(*load(name))
        assert out["decision"] in {"approve", "decline"} and out["review"] is None


def test_referral_suggests_starter_within_cap(scorer):
    app, hist = load("application_first_loan.json")
    out = scorer.score(app, hist)
    assert out["decision"] == "refer"
    expected = min(0.10 * min(out["requested_amount"], out["suggested_limit"]), scorer.starter_cap)
    assert out["review"]["suggested_amount"] == pytest.approx(expected, abs=0.01)
    assert out["review"]["max_amount"] == out["review"]["suggested_amount"]
    assert out["review"]["currency"] == "KES"


def test_cap_and_share_change_without_retraining(scorer):
    app, hist = load("application_first_loan.json")
    base = scorer.score(app, hist)["review"]["suggested_amount"]
    lower = scorer.score(app, hist, Policy(first_loan_cap={"Kenya": 2000}))["review"]["suggested_amount"]
    smaller = scorer.score(app, hist, Policy(first_loan_cap={"Kenya": 1e9}, starter_share=0.05))
    assert lower == 2000 and lower < base
    assert smaller["review"]["suggested_amount"] == pytest.approx(
        0.05 * min(smaller["requested_amount"], smaller["suggested_limit"]), abs=0.01)


def test_market_maximum_caps_approved_offer(scorer):
    app, hist = load("application_repeat.json")
    out = scorer.score(app, hist, Policy(first_loan_cap={"Kenya": scorer.starter_cap}, max_loan={"Kenya": 1000}))
    assert out["decision"] == "approve" and out["offered_amount"] == 1000


def test_policy_does_not_change_risk_or_decision(scorer):
    app, hist = load("application_first_loan.json")
    a = scorer.score(app, hist)
    b = scorer.score(app, hist, Policy(first_loan_cap={"Kenya": 100}, starter_share=0.5, max_loan={"Kenya": 50}))
    assert (a["p_default"], a["decision"], a["reasons"]) == (b["p_default"], b["decision"], b["reasons"])


def test_ghana_needs_a_cedi_cap_before_any_offer(scorer):
    app, hist = load("application_ghana.json")
    out = scorer.score(app, hist)
    assert out["decision"] == "refer" and out["review"]["max_amount"] is None and "Ghana" in out["review"]["note"]
    with pytest.raises(ValueError):
        record_review(out, app, 100, "reviewer-1", "pilot loan for a known customer")
    capped = scorer.score(app, hist, Policy(first_loan_cap={"Kenya": scorer.starter_cap, "Ghana": 300}))
    assert capped["review"]["max_amount"] == min(300, capped["requested_amount"])
    assert capped["review"]["currency"] == "GHS"


def test_reviewer_can_lower_or_decline_but_not_raise(scorer, tmp_path):
    app, hist = load("application_first_loan.json")
    out = scorer.score(app, hist)
    ceiling = out["review"]["max_amount"]
    log = tmp_path / "reviews.jsonl"
    now = datetime(2026, 11, 20, tzinfo=timezone.utc)

    lowered = record_review(out, app, ceiling / 2, "reviewer-1", "income checked, smaller first loan", log, now)
    assert lowered["lowered"] and lowered["outcome"] == "starter_loan"
    declined = record_review(out, app, 0, "reviewer-1", "could not confirm the mobile-money account", log, now)
    assert declined["outcome"] == "declined_by_reviewer"
    with pytest.raises(ValueError, match="raised"):
        record_review(out, app, ceiling + 1, "reviewer-1", "customer asked for more money", log, now)
    with pytest.raises(ValueError, match="reason"):
        record_review(out, app, ceiling, "reviewer-1", "ok", log, now)
    with pytest.raises(ValueError, match="named"):
        record_review(out, app, ceiling, " ", "income checked, full starter loan", log, now)

    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(lines) == 2 and all(l["channel"] == "digital" for l in lines)


def test_only_referrals_can_be_reviewed(scorer):
    app, hist = load("application_repeat.json")
    with pytest.raises(ValueError, match="Only referred"):
        record_review(scorer.score(app, hist), app, 100, "reviewer-1", "trying to change an approval")


def test_invalid_policy_is_rejected(scorer):
    app, hist = load("application_repeat.json")
    with pytest.raises(ValueError):
        scorer.score(app, hist, Policy(first_loan_cap={"Kenya": 5000}, starter_share=0))
    with pytest.raises(ValueError):
        scorer.score(app, hist, Policy(first_loan_cap={"Kenya": -1}))
