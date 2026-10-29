"""Cedi cost matrix and the cost-minimising approval threshold (Part A 1.5).

All figures are our own assumptions for a representative NkwaPay loan, not
observed data. The BoG Digital Credit Directive (Notice 30, 2025) requires
interest to be disclosed and not compounded but sets no cap, so pricing is a
business choice. Change the defaults here and every notebook picks them up.
"""
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class LoanEconomics:
    principal: float = 500.0      # GHS, representative digital loan
    fee_rate: float = 0.08        # flat interest + fees over the term (not compounded)
    funding_rate: float = 0.02    # cost of capital over the term
    ops_cost: float = 5.0         # GHS per disbursed loan: MoMo transfer, SMS, KYC
    collection_cost: float = 15.0 # GHS extra spent chasing a defaulted loan
    recovery_rate: float = 0.30   # share of principal eventually recovered after default

    @property
    def gain_good(self) -> float:
        """Net profit from approving a borrower who repays (true negative)."""
        return self.principal * (self.fee_rate - self.funding_rate) - self.ops_cost

    @property
    def loss_bad(self) -> float:
        """Net loss from approving a borrower who defaults (false negative)."""
        unrecovered = (1 - self.recovery_rate) * self.principal
        return unrecovered + self.principal * self.funding_rate + self.ops_cost + self.collection_cost

    def matrix(self) -> dict:
        """Value to NkwaPay in GHS. Positive class = default; 'flag' = decline.

        Declining is the zero baseline. The profit forgone by declining a good
        borrower is already counted because approving them would have earned
        gain_good.
        """
        return {
            ("repays", "approve"): self.gain_good,
            ("repays", "decline"): 0.0,
            ("defaults", "approve"): -self.loss_bad,
            ("defaults", "decline"): 0.0,
        }

    @property
    def threshold(self) -> float:
        """Decline when P(default) > G / (G + L).

        Approving has expected value (1 - p) G - p L, which is positive
        exactly when p < G / (G + L).
        """
        g, l = self.gain_good, self.loss_bad
        return g / (g + l) if g > 0 else 0.0


def expected_value(y_true, p_default, threshold, econ: LoanEconomics = LoanEconomics()) -> float:
    """Total GHS value of approving every application with p_default <= threshold."""
    y = np.asarray(y_true)
    approve = np.asarray(p_default) <= threshold
    return float(econ.gain_good * np.sum(approve & (y == 0)) - econ.loss_bad * np.sum(approve & (y == 1)))


def threshold_vs(param: str, values, econ: LoanEconomics = LoanEconomics()):
    """Threshold as one assumption varies, e.g. threshold_vs('recovery_rate', np.linspace(0, .9, 10))."""
    return np.array([replace(econ, **{param: v}).threshold for v in values])
