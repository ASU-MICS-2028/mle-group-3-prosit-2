"""Validation schemes (Section 4). Each splitter yields (train_idx, test_idx) as
positional indices into the Train-loan frame it was given."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.config import SEED
from src.features import LABEL_LAG_DAYS

# Expanding-window cut-offs: each test window runs to the next cut-off.
TEMPORAL_CUTS = ["2022-09-01", "2022-10-01", "2022-11-01", "2022-12-01", "2024-01-01"]


def stratified(df, k=5):
    return StratifiedKFold(k, shuffle=True, random_state=SEED).split(df, df.target)


def grouped(df, k=5):
    return StratifiedGroupKFold(k, shuffle=True, random_state=SEED).split(df, df.target, groups=df.customer_id)


def temporal(df, cuts=TEMPORAL_CUTS):
    """Train on loans whose outcome was observable before the cut-off; test on
    loans disbursed from the cut-off up to the next one."""
    d, due = df.disbursement_date.to_numpy(), (df.due_date + pd.to_timedelta(LABEL_LAG_DAYS, unit="D")).to_numpy()
    edges = [pd.Timestamp(c).to_datetime64() for c in cuts] + [np.datetime64("2100-01-01")]
    for lo, hi in zip(edges[:-1], edges[1:]):
        yield np.flatnonzero(due < lo), np.flatnonzero((d >= lo) & (d < hi))


SCHEMES = {"stratified": stratified, "grouped": grouped, "temporal": temporal}


def cross_validate(model, df, cols, splits):
    """Out-of-fold scores plus per-fold metrics. Rows never tested keep NaN."""
    oof = np.full(len(df), np.nan)
    fold_id = np.full(len(df), -1)
    rows = []
    X, y = df[cols], df.target.to_numpy().astype(int)
    for i, (a, b) in enumerate(splits):
        m = clone(model).fit(X.iloc[a], y[a])
        p = m.predict_proba(X.iloc[b])[:, 1]
        oof[b], fold_id[b] = p, i
        rows.append(dict(fold=i, n_train=len(a), n_test=len(b), defaults=int(y[b].sum()), base_rate=y[b].mean(),
                         pr_auc=average_precision_score(y[b], p), roc_auc=roc_auc_score(y[b], p)))
    folds = pd.DataFrame(rows)
    folds["pr_lift"] = folds.pr_auc / folds.base_rate
    return oof, fold_id, folds


def summarise(folds: pd.DataFrame) -> dict:
    return {"pr_auc": folds.pr_auc.mean(), "pr_auc_sd": folds.pr_auc.std(), "roc_auc": folds.roc_auc.mean(),
            "roc_auc_sd": folds.roc_auc.std(), "pr_lift": folds.pr_lift.mean(), "base_rate": folds.base_rate.mean()}
