import numpy as np

from src.bootstrap import cluster_indices


def test_cluster_bootstrap_keeps_customers_whole():
    groups = np.repeat(["a", "b", "c"], [1, 2, 3])
    for idx in cluster_indices(groups, n_boot=50, seed=0):
        picked = groups[idx]
        for g, n in zip("abc", [1, 2, 3]):
            assert (picked == g).sum() % n == 0
        assert len(np.unique(idx, return_counts=True)[1]) <= 6
