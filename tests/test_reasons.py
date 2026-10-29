"""Reason-code contributions must reproduce the model's own log-odds exactly."""
import numpy as np
import pandas as pd
import pytest

from src import models as M
from src.config import DATA_PROCESSED
from src.reasons import NOT_A_REASON, ReasonCoder


@pytest.fixture(scope="module")
def fitted():
    path = DATA_PROCESSED / "features.pkl"
    if not path.exists():
        pytest.skip("run notebooks/03_features.ipynb first")
    t = pd.read_pickle(path).query("split == 'train'").reset_index(drop=True)
    X = t[M.inputs(M.DEPLOY_FEATURES)]
    pipe = M.logreg(numeric=M.DEPLOY_FEATURES, class_weight=None).fit(X, t.target)
    return pipe, ReasonCoder(pipe).fit(X), X


def test_contributions_sum_to_logit(fitted):
    pipe, rc, X = fitted
    sample = X.sample(500, random_state=0)
    p = pipe.predict_proba(sample)[:, 1]
    logit = rc.reference_logit() + rc.contributions(sample).sum(axis=1).to_numpy()
    assert np.allclose(1 / (1 + np.exp(-logit)), p, atol=1e-10)


def test_reasons_are_positive_and_never_calendar(fitted):
    _, rc, X = fitted
    for reasons in rc.top_reasons(X.sample(300, random_state=1)):
        assert len(reasons) <= 3
        assert all(r["log_odds"] > 0 and r["code"] not in NOT_A_REASON for r in reasons)


def test_unseen_product_is_described_as_new(fitted):
    _, rc, X = fitted
    x = X.sample(200, random_state=2).assign(loan_type="Type_3", duration=60)
    codes = {r["code"] for reasons in rc.top_reasons(x) for r in reasons}
    assert "PRODUCT" not in codes
