"""Out-of-time recalibration (Section 13.9) must not use outcomes that were unknown when a fold began.
Made-up loans, so it runs without the data."""
import numpy as np
import pandas as pd
import pytest

from src import validation as V


def made_up_loans(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    disb = pd.Timestamp("2022-08-01") + pd.to_timedelta(rng.integers(0, 670, n), unit="D")
    due = disb + pd.to_timedelta(rng.choice([7, 30], n), unit="D")
    fold = np.searchsorted([pd.Timestamp(c) for c in V.TEMPORAL_CUTS], disb, side="right") - 1
    s = rng.uniform(0.01, 0.6, n)
    true_p = 1 / (1 + np.exp(-(0.5 * np.log(s / (1 - s)) - 1)))  # over-spread scores: true slope 0.5
    return s, rng.binomial(1, true_p), fold, due.to_numpy()


@pytest.mark.parametrize("window", ["previous", "all"])
@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_recalibration_ignores_outcomes_known_after_the_cutoff(method, window):
    s, y, fold, due = made_up_loans()
    for k in range(1, 5):
        late = due >= np.datetime64(pd.Timestamp(V.TEMPORAL_CUTS[k]))
        scrambled = np.where(late, 1 - y, y)
        a, _ = V.recalibrate(s, y, fold, due, method=method, window=window)
        b, _ = V.recalibrate(s, scrambled, fold, due, method=method, window=window)
        assert np.allclose(a[fold == k], b[fold == k]), f"fold {k} used an outcome known only after its cut-off"


def test_platt_recovers_the_slope():
    s, y, fold, due = made_up_loans()
    _, fits = V.recalibrate(s, y, fold, due, window="all")
    assert abs(fits.slope.iloc[-1] - 0.5) < 0.1
