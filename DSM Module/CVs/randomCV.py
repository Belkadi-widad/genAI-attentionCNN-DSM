# -*- coding: utf-8 -*-
"""
randomCV.py

Random (non-spatial) k-fold cross-validation for the soil dataset.

Dataset structure:
    columns 0:38   -> covariates
    columns 38:45  -> soil properties
    column 45      -> X
    column 46      -> Y

There is no explicit ID column, so the row index is used as the point ID.
"""

import os
import csv
import numpy as np

try:
    from cv_plot_utils import plot_folds
except ImportError:  # pragma: no cover - fallback for direct script execution
    from CVs.cv_plot_utils import plot_folds


def rdm_cv_split(points, k, random_state=None):
    """
    Randomly assign every point to one of k folds.

    Parameters
    ----------
    points : np.ndarray
        Dataset with shape (n_points, 47).

        Columns:
            0:38  -> covariates
            38:45 -> soil properties
            45    -> X
            46    -> Y

    k : int
        Number of folds.

    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    produce_folds : list
        List of [point_ID, fold_index].
    """

    n_points = len(points)

    # Use row indices as point IDs
    points_ids = np.arange(n_points)

    # Reproducible random generator
    rng = np.random.default_rng(random_state)

    # Shuffle point IDs
    rng.shuffle(points_ids)

    # Create approximately equal-sized folds
    folds = np.array_split(points_ids, k)

    produce_folds = []

    for fold_idx, fold_ids in enumerate(folds):
        for point_id in fold_ids:
            produce_folds.append([int(point_id), fold_idx])

    return produce_folds


def random_cv(
    points,
    k=10,
    n_repeats=1,
    out_dir=None,
    save_csv=True,
    basename="randomCV",
    random_state=None
):
    """
    Run random k-fold CV.

    Parameters
    ----------
    points : np.ndarray
        Dataset with shape (n_points, 47).

    k : int
        Number of folds.

    n_repeats : int
        Number of independent random CV repetitions.

    out_dir : str, optional
        Directory where plots and CSV files are saved.

    save_csv : bool
        Whether to save fold assignments.

    basename : str
        Prefix for output files.

    random_state : int, optional
        Base random seed.

    Returns
    -------
    all_folds : list
        Fold assignments for every repetition.
    """

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(out_dir, exist_ok=True)

    all_folds = []

    for rep in range(n_repeats):

        # Different seed for each repetition
        if random_state is not None:
            rep_seed = random_state + rep
        else:
            rep_seed = None

        folds = rdm_cv_split(
            points,
            k=k,
            random_state=rep_seed
        )

        all_folds.append(folds)

        # Plot using X/Y coordinates
        plot_path = os.path.join(
            out_dir,
            f"{basename}_rep{rep}_k{k}.png"
        )

        plot_folds(
            points,
            folds,
            plot_path,
            title=f"Random CV — {k} folds (repeat {rep})",
            k=k
        )

        # Save fold assignments
        if save_csv:

            csv_path = os.path.join(
                out_dir,
                f"{basename}_rep{rep}_k{k}.csv"
            )

            with open(csv_path, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow(["ID", "fold"])

                writer.writerows(folds)

    return all_folds