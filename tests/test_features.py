"""No feature may use information from after the decision point (workplan 3, last task).

Replay test: for a decision date D, rebuild the world as NkwaPay saw it on D
(only loans disbursed on or before D; outcomes only where observable before D)
and recompute features for the loans disbursed on D. They must equal the
features computed once on the full dataset.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import load_loans
from src.features import LABEL_LAG_DAYS, NUMERIC, RareCategoryGrouper, build_features, CATEGORICAL
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
