"""
ensemble_consensus.py

Drop-in, pure-Python replacement for `Cluster_Ensembles.cluster_ensembles`
that avoids the METIS/gpmetis/shmetis subprocess calls (which don't work
out of the box on Windows). Uses co-association matrix consensus
clustering (Fred & Jain, 2005) instead.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def _build_coassociation_matrix(cluster_runs):
    cluster_runs = np.asarray(cluster_runs)
    n_runs, n_samples = cluster_runs.shape
    coassoc = np.zeros((n_samples, n_samples), dtype=float)
    for run in cluster_runs:
        run = np.asarray(run)
        same_cluster = (run[:, None] == run[None, :])
        coassoc += same_cluster
    coassoc /= n_runs
    return coassoc


def cluster_ensembles(cluster_runs, verbose=False, N_clusters_max=None):
    if N_clusters_max is None:
        raise ValueError("N_clusters_max must be provided.")

    coassoc = _build_coassociation_matrix(cluster_runs)
    distance = 1.0 - coassoc
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0

    try:
        model = AgglomerativeClustering(
            n_clusters=N_clusters_max, metric="precomputed", linkage="average"
        )
        labels = model.fit_predict(distance)
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=N_clusters_max, affinity="precomputed", linkage="average"
        )
        labels = model.fit_predict(distance)

    return labels