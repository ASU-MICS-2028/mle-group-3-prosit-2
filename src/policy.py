"""Lending policy settings and the reviewer step for referred applications.

The saved model decides the risk; the policy decides the money. Keeping the
amounts here, outside the model file, lets NkwaPay change them without
retraining:

- first_loan_cap   largest first loan per market, in that market's currency.
                   Kenya defaults to the model's starter_cap (median repaid
                   first loan, 5,000 KES). Ghana has no default: a cedi amount
                   must be set before reviewers can offer a pilot loan.
- starter_share    a referred first-time applicant is offered this share of
                   what an approved applicant would get (10%, Section 13.5).
- max_loan         a per-customer maximum per market, for the limit the Bank
                   of Ghana sets under Directive s.3.5(5). None = not set.

Approvals stay automatic. A referral carries a suggested starter amount that is
also the ceiling: the reviewer may lower it (or decline), never raise it, must
give a reason, and every decision is recorded (record_review). The Directive
(s.2.3(1)) bars manual approval at a physical office, so the review is a
digital step: a choice within the suggested range, not a free decision.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

CURRENCY = {"Kenya": "KES", "Ghana": "GHS"}
MIN_REASON_CHARS = 10


@dataclass
class Policy:
    first_loan_cap: dict = field(default_factory=dict)
    starter_share: float = 0.10
    max_loan: dict = field(default_factory=dict)

    @classmethod
    def default(cls, kenya_first_loan_cap: float) -> "Policy":
        return cls(first_loan_cap={"Kenya": float(kenya_first_loan_cap)})

    def cap(self, market: str):
        return self.first_loan_cap.get(market)

    def market_max(self, market: str):
        return self.max_loan.get(market)

    def validate(self):
        if not 0 < self.starter_share <= 1:
            raise ValueError("starter_share must be between 0 and 1")
        for name, d in (("first_loan_cap", self.first_loan_cap), ("max_loan", self.max_loan)):
            for market, v in d.items():
                if v is not None and v <= 0:
                    raise ValueError(f"{name} for {market} must be positive")
        return self


def _limit(amount, *caps):
    for c in caps:
        if c is not None and amount is not None:
            amount = min(amount, c)
    return amount


def approved_offer(requested, limit, thin_file, market, policy: Policy):
    """Automatic offer for an approval: request, model limit, first-loan cap, market maximum."""
    offer = min(requested, limit)
    if thin_file:
        offer = _limit(offer, policy.cap(market))
    return _limit(offer, policy.market_max(market))


def starter_suggestion(requested, limit, market, trained_market: bool, policy: Policy):
    """Suggested starter amount for a referral, which is also the reviewer's ceiling.

    In a trained market: starter_share of what an approval would get, within the
    first-loan cap. In an untrained market the model's limit is not valid (it was
    learned in another currency and market), so the suggestion is the request
    within that market's first-loan cap, and None until a cap is set.
    """
    if trained_market:
        amount = policy.starter_share * min(requested, limit)
    else:
        if policy.cap(market) is None:
            return None, (f"No first-loan cap is set for {market}; set one in the policy "
                          "before a reviewer can offer a pilot loan.")
        amount = requested
    amount = _limit(amount, policy.cap(market), policy.market_max(market))
    return round(float(amount), 2), ""


def review_block(suggested, note, market):
    return {
        "suggested_amount": suggested,
        "max_amount": suggested,
        "currency": CURRENCY.get(market, "local"),
        "rule": "The reviewer may offer up to max_amount or decline, never more, and must give a reason. "
                "Each decision is recorded.",
        "note": note,
    }


def record_review(result: dict, application: dict, offered_amount: float, reviewer: str, reason: str,
                  log_path=None, now=None) -> dict:
    """Validate and record a reviewer's decision on a referred application.

    offered_amount 0 means the reviewer declined. Raises ValueError if the
    application was not referred, no ceiling is set, the amount is above the
    ceiling or negative, or the reviewer or reason is missing. Appends the
    record as one JSON line to log_path when given.
    """
    if result.get("decision") != "refer" or not result.get("review"):
        raise ValueError("Only referred applications go to a reviewer; approvals and declines are automatic.")
    ceiling = result["review"]["max_amount"]
    if ceiling is None:
        raise ValueError(result["review"]["note"] or "No ceiling is set for this application.")
    offered_amount = float(offered_amount)
    if offered_amount < 0:
        raise ValueError("The amount cannot be negative.")
    if offered_amount > ceiling + 1e-9:
        raise ValueError(f"The amount cannot be raised above the suggested {ceiling:,.2f}.")
    if not str(reviewer).strip():
        raise ValueError("The reviewer must be named.")
    if len(str(reason).strip()) < MIN_REASON_CHARS:
        raise ValueError(f"Give a reason of at least {MIN_REASON_CHARS} characters.")

    record = {
        "time": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "customer_id": application.get("customer_id"),
        "market": application.get("country_id"),
        "model_version": result.get("model_version"),
        "p_default": result.get("p_default"),
        "referral_note": result.get("note"),
        "suggested_amount": ceiling,
        "offered_amount": round(offered_amount, 2),
        "currency": result["review"]["currency"],
        "outcome": "starter_loan" if offered_amount > 0 else "declined_by_reviewer",
        "lowered": offered_amount < ceiling - 1e-9,
        "reviewer": str(reviewer).strip(),
        "reason": str(reason).strip(),
        "channel": "digital",
    }
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
    return record
