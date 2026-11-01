# Code for spatial cross-validation
# -*- coding: utf-8 -*-
"""
Reference of the code:
Y Wang, 2023, "SPATIAL+ CROSS-VALIDATION (SP-CV) EXPERIMENTS DATASETS AND CODES (after review)", https://doi.org/10.17026/DANS-ZWJ-ZAUQ, DANS Data Station Physical and Technical Sciences, V2, UNF:6:pxaVakGPWU1VPv0c0/3k7w== [fileUNF]
spatialCV.py

Spatial cross-validation splitting methods, adapted from the original
BLK_CV.py and SP_CV.py scripts into a single reusable module.

Dataset convention:
    points[:, 0:38]  = covariates
    points[:, 38:45] = soil targets
    points[:, 45]    = X coordinate
    points[:, 46]    = Y coordinate

There is no explicit ID column.
The row index is used as the point ID.

Contains two families of spatial CV:

    1. Block CV (BLK_CV)
       Grid-based spatial blocking.

    2. SP-CV family (SP_CV)
       Agglomerative Hierarchical Clustering (AHC) blocks,
       optionally combined with coordinate / environmental /
       target clustering + cluster ensembling.

        - SP1_CV : AHC blocks only
        - SP_CV  : AHC blocks + coordinate/env/target k-means + ensemble
        - SP2_CV : same as SP_CV but each point starts as its own
                   cluster (sp_threshold = 0)

Public high-level functions
----------------------------

block_cv(points, k, sp_threshold, out_dir=None, save_csv=True,
         basename="blockCV")

sp1_cv(points, sp_threshold, k, out_dir=None, save_csv=True,
       basename="sp1CV")

sp_cv(points, env, tar, cate_col_start, cate_col_end, sp_threshold, k,
      out_dir=None, save_csv=True, basename="spCV")

sp2_cv(points, env, tar, cate_col_start, cate_col_end, k,
       out_dir=None, save_csv=True, basename="sp2CV")

Each returns fold assignments:

    [point_ID, fold_index, cluster_id]
"""

import os
import csv
import numpy as np
from random import shuffle

try:
    from cv_plot_utils import plot_folds
except ImportError:  # pragma: no cover - fallback for direct script execution
    from CVs.cv_plot_utils import plot_folds


# ---------------------------------------------------------------------------
# Dataset column definitions
# ---------------------------------------------------------------------------

X_COL = -2
Y_COL = -1


def _resolve_xy_columns(points):
    """Support both full dataset arrays and simple [x, y] arrays."""

    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points must be a 2D array with at least two columns")

    if points.shape[1] > Y_COL:
        return X_COL, Y_COL

    return 0, 1


# ---------------------------------------------------------------------------
# 1. Block CV
# ---------------------------------------------------------------------------

def BLK_CV(points, k, sp_threshold):
    """
    Spatial block k-fold split: lay a regular grid of `sp_threshold`-sized
    blocks over the points, randomly assign whole blocks to folds.

    Parameters
    ----------
    points : np.ndarray
        Dataset with 47 columns.

    k : int
        Number of folds.

    sp_threshold : float
        Spatial block size.

    Returns
    -------
    produced_folds : list
        List of [point_ID, fold_index, block_id].
        point_ID is the row index of the point.
    """

    # ---------------------------------------------------------
    # Extract coordinates from the actual dataset columns
    # ---------------------------------------------------------

    x_col, y_col = _resolve_xy_columns(points)

    points_x = points[:, x_col]
    points_y = points[:, y_col]

    points_x_range = points_x.max() - points_x.min()
    points_y_range = points_y.max() - points_y.min()

    x_block_num = int(points_x_range / sp_threshold) + 1
    y_block_num = int(points_y_range / sp_threshold) + 1

    block_coord_left = (
        points_x.min()
        - (x_block_num * sp_threshold - points_x_range) / 2.0
    )

    block_coord_top = (
        points_y.max()
        + (y_block_num * sp_threshold - points_y_range) / 2.0
    )

    blocks_points = []

    # ---------------------------------------------------------
    # Create spatial blocks
    # ---------------------------------------------------------

    for i in range(y_block_num):

        for j in range(x_block_num):

            ij_block_id = i * x_block_num + j

            ij_block_x_range = [
                block_coord_left + j * sp_threshold,
                block_coord_left + (j + 1) * sp_threshold
            ]

            ij_block_y_range = [
                block_coord_top - i * sp_threshold,
                block_coord_top - (i + 1) * sp_threshold
            ]

            # First element = block ID
            ij_block_points = [ij_block_id]

            # -------------------------------------------------
            # Store point index + point itself.
            #
            # This replaces the old assumption that point[0]
            # contained the point ID.
            # -------------------------------------------------

            for point_idx, point in enumerate(points):

                if (
                    (point[x_col] < ij_block_x_range[1])
                    and (point[x_col] >= ij_block_x_range[0])
                    and (point[y_col] > ij_block_y_range[1])
                    and (point[y_col] <= ij_block_y_range[0])
                ):
                    ij_block_points.append(
                        (point_idx, point)
                    )

            if len(ij_block_points) > 1:
                blocks_points.append(ij_block_points)

    # ---------------------------------------------------------
    # Randomly shuffle spatial blocks
    # ---------------------------------------------------------

    np.random.shuffle(blocks_points)

    # ---------------------------------------------------------
    # Assign whole blocks to folds
    # ---------------------------------------------------------

    fold_count = int(len(blocks_points) / k)
    print(f"Total blocks: {len(blocks_points)}, "
          f"Blocks per fold: {fold_count}, "
          f"Number of folds: {k}")
    folds_list = []

    for i in range(0, fold_count * k, fold_count):
        folds_list.append(
            blocks_points[i:i + fold_count]
        )

    if len(blocks_points) > fold_count * k:

        for tail in range(
            fold_count * k,
            len(blocks_points)
        ):
            add_fold = tail - fold_count * k
            tail_block = blocks_points[tail]
            folds_list[add_fold].append(tail_block)

    # ---------------------------------------------------------
    # Produce final fold assignments
    # ---------------------------------------------------------

    produced_folds = []

    block_id = 0

    for i in range(len(folds_list)):  # fold

        for j in range(len(folds_list[i])):  # block

            block_id += 1

            block = folds_list[i][j]

            # Skip block ID at index 0
            for pt_idx in range(1, len(block)):

                point_id = block[pt_idx][0]

                point_fold = [
                    point_id,
                    i,
                    block_id
                ]

                produced_folds.append(point_fold)

    return produced_folds


def block_cv(
    points,
    k,
    sp_threshold,
    out_dir=None,
    save_csv=True,
    basename="blockCV"
):
    """High-level wrapper: run BLK_CV, save plot (+ optional CSV),
    return folds."""

    if out_dir is None:
        out_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

    os.makedirs(out_dir, exist_ok=True)

    folds = BLK_CV(
        points,
        k,
        sp_threshold
    )

    plot_path = os.path.join(
        out_dir,
        f"{basename}_k{k}_thr{sp_threshold}.png"
    )

    plot_folds(
        points,
        folds,
        plot_path,
        title=f"Block CV — {k} folds (threshold={sp_threshold})",
        k=k
    )

    if save_csv:

        csv_path = os.path.join(
            out_dir,
            f"{basename}_k{k}_thr{sp_threshold}.csv"
        )

        with open(csv_path, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(
                ["ID", "fold", "block_id"]
            )

            writer.writerows(folds)

    return folds


# ---------------------------------------------------------------------------
# 2. SP-CV family
# ---------------------------------------------------------------------------

def array_norm(array, cate_col_start, cate_col_end):
    """
    Min-max normalize all non-categorical columns.

    Constant columns are left unchanged instead of producing NaNs.
    """

    # IMPORTANT: don't modify the original env/points array
    array = np.array(array, dtype=np.float64, copy=True)

    # Columns before categorical range
    for col in range(cate_col_start):

        col_max = np.max(array[:, col])
        col_min = np.min(array[:, col])

        # Constant column -> leave unchanged
        if col_max == col_min:
            continue

        array[:, col] = (
            (array[:, col] - col_min)
            / (col_max - col_min)
        )

    # Columns after categorical range
    for col in range(
        cate_col_end + 1,
        array.shape[1]
    ):

        col_max = np.max(array[:, col])
        col_min = np.min(array[:, col])

        # Constant column -> leave unchanged
        if col_max == col_min:
            continue

        array[:, col] = (
            (array[:, col] - col_min)
            / (col_max - col_min)
        )

    return array


class st_point(object):

    def __init__(
        self,
        ID=None,
        coords=None,
        time=0
    ):
        self.ID = ID
        self.coords = coords
        self.time = time


class NNH_cluster(object):

    def __init__(
        self,
        st_points=None,
        count=None
    ):
        self.st_points = st_points
        self.count = count


def sp_maxHierarchicalClustering(
    points,
    sp_threshold=1
):
    """
    Agglomerative Hierarchical Clustering (complete linkage)
    on X/Y coordinates, cut at distance `sp_threshold`.
    """

    from sklearn.cluster import AgglomerativeClustering

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # Original code used points[:, 1:], which is incorrect for
    # the new dataset because that would include all covariates
    # and targets.
    #
    # AHC must operate ONLY on spatial coordinates.
    # ---------------------------------------------------------

    x_col, y_col = _resolve_xy_columns(points)
    spatial_coords = points[:, [x_col, y_col]]

    maxNNH_clusters_model = AgglomerativeClustering(
        distance_threshold=sp_threshold,
        n_clusters=None,
        linkage="complete"
    ).fit(spatial_coords)

    maxNNH_counts = (
        maxNNH_clusters_model.n_clusters_
    )

    maxNNH_labels = (
        maxNNH_clusters_model.labels_
    )

    maxNNH_clusters = []

    for i in range(maxNNH_counts):

        maxNNH_i_cluster = NNH_cluster(
            st_points=[],
            count=0
        )

        for j in range(len(points)):

            if int(maxNNH_labels[j]) == i:

                # Row index is now the point ID
                point = st_point(
                    ID=j,
                    coords=[
                        points[j, x_col],
                        points[j, y_col]
                    ]
                )

                maxNNH_i_cluster.st_points.append(
                    point
                )

                maxNNH_i_cluster.count += 1

        maxNNH_clusters.append(
            maxNNH_i_cluster
        )

    return maxNNH_clusters


def clusters_var_avg(
    points,
    env,
    tar,
    cate_col_start,
    cate_col_end,
    clusters
):
    """
    Average coordinates / normalized env vars / target value
    within each cluster.
    """

    pure_env = env

    pure_normenv = array_norm(
        pure_env,
        cate_col_start,
        cate_col_end
    )

    clusters_coords_avg = []
    clusters_normenv_avg = []
    clusters_tar_avg = []

    x_col, y_col = _resolve_xy_columns(points)

    for cluster in clusters:

        # -----------------------------------------------------
        # Average X/Y coordinates
        # -----------------------------------------------------

        coord_sum = [0, 0]

        for point in cluster.st_points:

            st_point_ID = int(point.ID)

            coord = points[
                st_point_ID,
                [x_col, y_col]
            ]

            coord_sum = [
                coord_sum[0] + coord[0],
                coord_sum[1] + coord[1]
            ]

        coord_avg = [
            coord_sum[0] / float(cluster.count),
            coord_sum[1] / float(cluster.count)
        ]

        clusters_coords_avg.append(
            coord_avg
        )

        # -----------------------------------------------------
        # Average environmental variables
        #
        # env handling remains unchanged.
        # -----------------------------------------------------

        env_avg = []

        for i_env in range(
            pure_normenv.shape[1]
        ):

            i_env_sum = 0

            for point in cluster.st_points:

                st_point_ID = int(point.ID)

                i_env_sum += float(
                    pure_normenv[
                        st_point_ID,
                        i_env
                    ]
                )

            env_avg.append(
                i_env_sum / float(cluster.count)
            )

        clusters_normenv_avg.append(
            env_avg
        )

        # -----------------------------------------------------
        # Average target
        #
        # tar handling remains unchanged.
        # -----------------------------------------------------

        tar_sum = 0

        for point in cluster.st_points:

            st_point_ID = int(point.ID)

            tar_sum += float(
                tar[st_point_ID, -1]
            )

        clusters_tar_avg.append(
            [
                tar_sum / float(cluster.count)
            ]
        )

    return (
        np.array(clusters_coords_avg),
        np.array(clusters_normenv_avg),
        np.array(clusters_tar_avg)
    )


def SP1_CV(
    points,
    sp_threshold,
    k
):
    """
    SP-1-CV: AHC blocks only, randomly assigned to k folds.
    """

    NNH_clusters = sp_maxHierarchicalClustering(
        points,
        sp_threshold
    )

    print(f"Total clusters: {len(NNH_clusters)}")
    NNH_list = list(
        range(len(NNH_clusters))
    )

    shuffle(NNH_list)

    fold_count = int(
        len(NNH_list) / k
    )

    folds_list = []

    for i in range(
        0,
        fold_count * k,
        fold_count
    ):
        folds_list.append(
            NNH_list[
                i:i + fold_count
            ]
        )

    if len(NNH_list) > fold_count * k:

        for tail in range(
            fold_count * k,
            len(NNH_list)
        ):

            add_fold = (
                tail - fold_count * k
            )

            folds_list[
                add_fold
            ].append(
                int(NNH_list[tail])
            )

    produce_folds = []

    for fold_label in range(
        len(folds_list)
    ):

        for cluster_id in folds_list[
            fold_label
        ]:

            for pt in range(
                NNH_clusters[
                    cluster_id
                ].count
            ):

                point_id = (
                    NNH_clusters[
                        cluster_id
                    ].st_points[pt].ID
                )

                # point_id is already the row index
                origin_point_id = int(
                    point_id
                )

                produce_folds.append(
                    [
                        origin_point_id,
                        fold_label,
                        cluster_id
                    ]
                )

    return produce_folds


def _ensemble_cluster_folds(
    points,
    env,
    tar,
    cate_col_start,
    cate_col_end,
    k,
    final_clusters
):
    """
    Shared logic for SP_CV / SP2_CV:

    cluster on coords, env, target separately,
    then ensemble the three clusterings into k final folds.
    """

    from sklearn.cluster import KMeans
    
    try:
        import ensemble_consensus as CE
    except ImportError:  # pragma: no cover - fallback for direct script execution
        from CVs import ensemble_consensus as CE

    from kmodes.kprototypes import KPrototypes

    (
        clusters_coords_avg,
        clusters_normenv_avg,
        clusters_tar_avg
    ) = clusters_var_avg(
        points,
        env,
        tar,
        cate_col_start,
        cate_col_end,
        final_clusters
    )

    # ---------------------------------------------------------
    # Coordinate clustering
    # ---------------------------------------------------------

    clusters_points_klabels = (
        KMeans(n_clusters=k)
        .fit(clusters_coords_avg)
        .labels_
    )

    # ---------------------------------------------------------
    # Environmental clustering
    # ---------------------------------------------------------

    if cate_col_start < cate_col_end:

        clusters_normenv_klabels = (
            KPrototypes(
                n_clusters=k,
                init="Cao",
                verbose=0
            )
            .fit_predict(
                clusters_normenv_avg,
                categorical=list(
                    range(
                        cate_col_start,
                        cate_col_end + 1
                    )
                )
            )
        )

    else:

        clusters_normenv_klabels = (
            KMeans(n_clusters=k)
            .fit(clusters_normenv_avg)
            .labels_
        )

    # ---------------------------------------------------------
    # Target clustering
    # ---------------------------------------------------------

    clusters_target_klabels = (
        KMeans(n_clusters=k)
        .fit(clusters_tar_avg)
        .labels_
    )

    # ---------------------------------------------------------
    # Cluster ensemble
    # ---------------------------------------------------------

    cluster_runs = np.array([
        list(clusters_points_klabels),
        list(clusters_normenv_klabels),
        list(clusters_target_klabels)
    ])

    ensemble_clustering_labels = (
        CE.cluster_ensembles(
            cluster_runs,
            verbose=False,
            N_clusters_max=k
        )
    )

    # ---------------------------------------------------------
    # Produce final fold assignments
    # ---------------------------------------------------------

    produced_folds = []

    for cluster_id, cluster in enumerate(
        final_clusters
    ):

        fold_label = (
            ensemble_clustering_labels[
                cluster_id
            ]
        )

        for pt in range(
            cluster.count
        ):

            point_id = (
                cluster.st_points[
                    pt
                ].ID
            )

            # Row index = point ID
            origin_point_id = int(
                point_id
            )

            produced_folds.append(
                [
                    origin_point_id,
                    fold_label,
                    cluster_id
                ]
            )

    return produced_folds


def SP_CV(
    points,
    sp_threshold,
    env,
    tar,
    cate_col_start,
    cate_col_end,
    k
):
    """
    SP-CV: AHC blocks, then ensemble-cluster on
    coords + env + target.
    """

    final_clusters = (
        sp_maxHierarchicalClustering(
            points,
            sp_threshold
        )
    )

    return _ensemble_cluster_folds(
        points,
        env,
        tar,
        cate_col_start,
        cate_col_end,
        k,
        final_clusters
    )


def SP2_CV(
    points,
    env,
    tar,
    cate_col_start,
    cate_col_end,
    k
):
    """
    SP-2-CV: every point starts as its own cluster
    (threshold=0), then ensemble-cluster on
    coords + env + target.
    """

    final_clusters = (
        sp_maxHierarchicalClustering(
            points,
            0
        )
    )

    return _ensemble_cluster_folds(
        points,
        env,
        tar,
        cate_col_start,
        cate_col_end,
        k,
        final_clusters
    )


# ---------------------------------------------------------------------------
# High-level wrappers
# ---------------------------------------------------------------------------

def _save_folds(
    points,
    folds,
    out_dir,
    basename,
    title,
    k
):

    plot_path = os.path.join(
        out_dir,
        f"{basename}.png"
    )

    plot_folds(
        points,
        folds,
        plot_path,
        title=title,
        k=k
    )

    csv_path = os.path.join(
        out_dir,
        f"{basename}.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["ID", "fold", "cluster_id"]
        )

        writer.writerows(folds)

    return plot_path, csv_path


def sp1_cv(
    points,
    sp_threshold,
    k,
    out_dir=None,
    save_csv=True,
    basename="sp1CV"
):

    if out_dir is None:
        out_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    folds = SP1_CV(
        points,
        sp_threshold,
        k
    )

    title = (
        f"SP1-CV — {k} folds "
        f"(threshold={sp_threshold})"
    )

    fname = (
        f"{basename}_k{k}_thr{sp_threshold}"
    )

    plot_path = os.path.join(
        out_dir,
        f"{fname}.png"
    )

    plot_folds(
        points,
        folds,
        plot_path,
        title=title,
        k=k
    )

    if save_csv:

        csv_path = os.path.join(
            out_dir,
            f"{fname}.csv"
        )

        with open(
            csv_path,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                ["ID", "fold", "cluster_id"]
            )

            writer.writerows(folds)

    return folds


def sp_cv(
    points,
    env,
    tar,
    cate_col_start,
    cate_col_end,
    sp_threshold,
    k,
    out_dir=None,
    save_csv=True,
    basename="spCV"
):

    if out_dir is None:
        out_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    folds = SP_CV(
        points,
        sp_threshold,
        env,
        tar,
        cate_col_start,
        cate_col_end,
        k
    )

    title = (
        f"SP-CV — {k} folds "
        f"(threshold={sp_threshold})"
    )

    fname = (
        f"{basename}_k{k}_thr{sp_threshold}"
    )

    plot_path = os.path.join(
        out_dir,
        f"{fname}.png"
    )

    plot_folds(
        points,
        folds,
        plot_path,
        title=title,
        k=k
    )

    if save_csv:

        csv_path = os.path.join(
            out_dir,
            f"{fname}.csv"
        )

        with open(
            csv_path,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                ["ID", "fold", "cluster_id"]
            )

            writer.writerows(folds)

    return folds


def sp2_cv(
    points,
    env,
    tar,
    cate_col_start,
    cate_col_end,
    k,
    out_dir=None,
    save_csv=True,
    basename="sp2CV"
):

    if out_dir is None:
        out_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    folds = SP2_CV(
        points,
        env,
        tar,
        cate_col_start,
        cate_col_end,
        k
    )

    title = (
        f"SP2-CV — {k} folds"
    )

    fname = (
        f"{basename}_k{k}"
    )

    plot_path = os.path.join(
        out_dir,
        f"{fname}.png"
    )

    plot_folds(
        points,
        folds,
        plot_path,
        title=title,
        k=k
    )

    if save_csv:

        csv_path = os.path.join(
            out_dir,
            f"{fname}.csv"
        )

        with open(
            csv_path,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                ["ID", "fold", "cluster_id"]
            )

            writer.writerows(folds)

    return folds