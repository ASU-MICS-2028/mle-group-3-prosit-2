"""Customer-level (cluster) bootstrap for metric confidence intervals (Section 7).

Loans from the same customer are correlated, so resampling individual loans
would understate uncertainty. We resample customers with replacement and keep
all of each sampled customer's loans. Paired: every model is scored on the
same resampled set, so differences between models get their own interval.
"""
import numpy as np
import pandas as pd

from src.config import SEED


def cluster_indices(groups, n_boot=1000, seed=SEED):
    """Yield arrays of row positions, one per bootstrap replicate."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(np.asarray(groups))
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(codes))
    members = [order[s:e] for s, e in zip(starts, ends)]
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[i] for i in pick])


def per_fold(fn):
    """Average a ranking metric over validation folds instead of pooling scores.

    Pooling out-of-fold scores from different folds mixes each fold's offset
    into the ranking (a constant baseline then scores ROC 0.44, not 0.5)."""
    def wrapped(y, p, fold):
        vals = [fn(y[fold == f], p[fold == f]) for f in np.unique(fold) if 0 < y[fold == f].sum() < (fold == f).sum()]
        return float(np.mean(vals))
    return wrapped


def pooled(fn):
    """Metrics that add up over loans (counts, cedi value) can be pooled."""
    return lambda y, p, fold: fn(y, p)


def bootstrap(metric_fns: dict, y, scores: dict, groups, fold, n_boot=1000, seed=SEED) -> pd.DataFrame:
    """Replicates of every metric x model on shared customer resamples.

    metric_fns: {"pr_auc": per_fold(average_precision_score), ...} taking (y, p, fold);
    scores: {"logreg": p, ...}. Returns a long frame: replicate, model, metric, value.
    """
    y, fold = np.asarray(y), np.asarray(fold)
    rows = []
    for b, idx in enumerate(cluster_indices(groups, n_boot, seed)):
        yb, fb = y[idx], fold[idx]
        for m, p in scores.items():
            pb = np.asarray(p)[idx]
            for name, fn in metric_fns.items():
                rows.append((b, m, name, fn(yb, pb, fb)))
    return pd.DataFrame(rows, columns=["replicate", "model", "metric", "value"])


def intervals(reps: pd.DataFrame, point: pd.DataFrame, level=0.95) -> pd.DataFrame:
    """Percentile intervals; `point` holds the full-sample estimate (model x metric)."""
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    q = reps.groupby(["model", "metric"]).value.quantile([lo, hi]).unstack()
    q.columns = ["lower", "upper"]
    est = point.stack().rename("estimate").rename_axis(["model", "metric"]).to_frame()
    return est.join(q)


def paired_differences(reps: pd.DataFrame, a: str, b: str, level=0.95) -> pd.DataFrame:
    """Interval for metric(a) - metric(b) and the share of replicates where a wins."""
    w = reps.pivot_table(index=["replicate", "metric"], columns="model", values="value")
    d = (w[a] - w[b]).rename("diff").reset_index()
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return d.groupby("metric")["diff"].agg(mean="mean", lower=lambda s: s.quantile(lo),
                                           upper=lambda s: s.quantile(hi), share_a_better=lambda s: (s > 0).mean())
