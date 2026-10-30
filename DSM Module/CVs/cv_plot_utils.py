
"""
cv_plots.py

Shared helper for getting/ creating folds, and plotting the spatial layout of cross-validation folds.

Dataset structure:
    columns 0:38   -> covariates
    columns 38:45  -> soil properties
    column 45      -> X coordinate
    column 46      -> Y coordinate

There is no explicit ID column in the dataset.
The row index is therefore used as the point ID.

fold_assignments convention:
    [point_ID, fold_index, ...]
"""

# --------------------------------------------------------- Plotting ---------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os


# Dataset column indices
X_COL = -2
Y_COL = -1


def _resolve_xy_columns(points):
    """Support both full dataset arrays and simple [x, y] arrays."""

    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points must be a 2D array with at least two columns")

    if points.shape[1] > Y_COL:
        return X_COL, Y_COL

    return 0, 1


def plot_folds(
    points,
    fold_assignments,
    out_path,
    title="CV folds",
    k=None
):
    """
    Scatter-plot points colored by their assigned fold and save to disk.

    Parameters
    ----------
    points : np.ndarray
        Dataset with shape (n_points, 47).

        Columns:
            0:38  -> covariates
            38:45 -> soil properties
            45    -> X coordinate
            46    -> Y coordinate

    fold_assignments : list or np.ndarray
        Rows of [point_ID, fold_index, ...].

        point_ID corresponds to the row index of `points`.

    out_path : str
        Full path where the figure will be saved.

    title : str
        Plot title.

    k : int, optional
        Number of folds. If None, inferred from the assignments.

    Returns
    -------
    str
        Path to the saved figure.
    """

    # Convert assignments to numpy array
    fold_assignments = np.asarray(
        fold_assignments,
        dtype=float
    )

    # ---------------------------------------------------------
    # Create mapping:
    #
    # row index / point ID -> fold
    # ---------------------------------------------------------

    id_to_fold = dict(
        zip(
            fold_assignments[:, 0].astype(int),
            fold_assignments[:, 1].astype(int)
        )
    )

    # ---------------------------------------------------------
    # Extract X/Y coordinates
    # ---------------------------------------------------------

    point_ids = np.arange(len(points))

    x_col, y_col = _resolve_xy_columns(points)
    xs = points[:, x_col]
    ys = points[:, y_col]

    # ---------------------------------------------------------
    # Get fold assigned to every point
    # ---------------------------------------------------------

    folds = np.array([
        id_to_fold.get(int(point_id), -1)
        for point_id in point_ids
    ])

    # ---------------------------------------------------------
    # Infer number of folds if not provided
    # ---------------------------------------------------------

    if k is None:

        valid_folds = folds[folds >= 0]

        if len(valid_folds):
            k = int(valid_folds.max()) + 1
        else:
            k = 0

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    output_dir = os.path.dirname(out_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 6))

    scatter = ax.scatter(
        xs,
        ys,
        c=folds,
        cmap="tab10",
        vmin=0,
        vmax=max(k - 1, 1),
        s=18,
        edgecolors="none"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)

    ax.set_aspect(
        "equal",
        adjustable="datalim"
    )

    # ---------------------------------------------------------
    # Colorbar
    # ---------------------------------------------------------

    if k > 0:

        cbar = fig.colorbar(
            scatter,
            ax=ax,
            ticks=range(k)
        )

        cbar.set_label("Fold")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return out_path