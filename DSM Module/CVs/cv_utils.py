# -*- coding: utf-8 -*-
"""
cv_utils.py

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

# --------------------------------------------------------- Getting / Creating folds by CV  ---------------------------------------------------------
import csv
import os

try:
    from .randomCV import random_cv as _random_cv
    from .spatialCV import block_cv as _block_cv
    from .spatialCV import sp1_cv as _sp1_cv
    from .spatialCV import sp2_cv as _sp2_cv
    from .spatialCV import sp_cv as _sp_cv

except ImportError:  # pragma: no cover - fallback for direct script execution
    from randomCV import random_cv as _random_cv
    from spatialCV import block_cv as _block_cv
    from spatialCV import sp1_cv as _sp1_cv
    from spatialCV import sp2_cv as _sp2_cv
    from spatialCV import sp_cv as _sp_cv



def _default_cv_plots_dir(region, *parts):
    """Resolve output paths relative to this module directory."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if region is None: 
        region = 'all'
    return os.path.join(base_dir, "cv_folds", region, *parts)

def _read_folds_csv(csv_path):
    """Load fold assignments from an existing CSV file."""

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fold CSV does not exist: {csv_path}")

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) <= 1:
        return []

    folds = []

    for row in rows[1:]:
        if not row or all(cell.strip() == "" for cell in row):
            continue

        values = []

        for value in row:
            try:
                values.append(int(float(value)))
            except ValueError:
                values.append(value)

        folds.append(values)

    return folds


def _write_folds_csv(csv_path, folds, header):
    """Save fold assignments to a CSV file."""

    out_dir = os.path.dirname(csv_path)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(folds)


def _get_or_create_folds(csv_path, build_fn, header, *args, **kwargs):
    """If the CSV already exists, load it; otherwise create, save, and return it."""

    if os.path.exists(csv_path):
        return _read_folds_csv(csv_path)

    folds = build_fn(*args, **kwargs)

    _write_folds_csv(csv_path, folds, header)

    return folds


def random_cv(
    points,
    k=10,
    n_repeats=1,
    out_dir=None,
    save_csv=True,
    basename="randomCV",
    random_state=None
):
    """Load existing random-CV CSVs or generate them and cache the result."""

    
    print("Generating random-CV folds.")

    if out_dir is None:
        print("out_dir is None, using default directory for random CV plots.")
        out_dir = _default_cv_plots_dir("random")

    os.makedirs(out_dir, exist_ok=True)

    all_folds = []

    for rep in range(n_repeats):
        rep_seed = None if random_state is None else random_state + rep
        csv_path = os.path.join(
            out_dir,
            f"{basename}_rep{rep}_k{k}.csv"
        )
        print(f"Checking for existing CSV: {csv_path}")
        if os.path.exists(csv_path):

            all_folds.append(_read_folds_csv(csv_path))
            continue

        rep_folds = _random_cv(
            points,
            k=k,
            n_repeats=1,
            out_dir=out_dir,
            save_csv=save_csv,
            basename=basename,
            random_state=rep_seed
        )[0]

        all_folds.append(rep_folds)

    return all_folds


def block_cv(
    points,
    k,
    sp_threshold,
    out_dir=None,
    save_csv=True,
    basename="blockCV"
):
    """Load existing block-CV CSV or generate it and cache the result."""

    print("Generating block-CV folds.")

    if out_dir is None:
        out_dir = _default_cv_plots_dir("block")

    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(
        out_dir,
        f"{basename}_k{k}_thr{sp_threshold}.csv"
    )

    if os.path.exists(csv_path):
        return _read_folds_csv(csv_path)

    return _block_cv(
        points,
        k=k,
        sp_threshold=sp_threshold,
        out_dir=out_dir,
        save_csv=save_csv,
        basename=basename
    )


def sp1_cv(
    points,
    sp_threshold,
    k,
    out_dir=None,
    save_csv=True,
    basename="sp1CV"
):
    """Load existing SP1-CV CSV or generate it and cache the result."""

    

    if out_dir is None:
        out_dir = _default_cv_plots_dir("sp1")

    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(
        out_dir,
        f"{basename}_k{k}_thr{sp_threshold}.csv"
    )

    if os.path.exists(csv_path):
        return _read_folds_csv(csv_path)

    return _sp1_cv(
        points,
        sp_threshold=sp_threshold,
        k=k,
        out_dir=out_dir,
        save_csv=save_csv,
        basename=basename
    )


def sp_cv(
    points,
    env,
    tar,
    sp_threshold,
    k,
    out_dir=None,
    save_csv=True,
    basename="sp2CV"
):
    """Load existing SP2-CV CSV or generate it and cache the result."""

    

    if out_dir is None:
        out_dir = _default_cv_plots_dir("sp2")

    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(
        out_dir,
        f"{basename}_k{k}_thr{sp_threshold}.csv"
    )

    if os.path.exists(csv_path):
        return _read_folds_csv(csv_path)

    cate_col_start = 0
    cate_col_end = -1   # -> no categoricals, falls back to plain KMeans on env

    return _sp_cv(
        points,
        env,
        tar,
        cate_col_start=cate_col_start,
        cate_col_end=cate_col_end,
        sp_threshold=sp_threshold,
        k=k,
        out_dir=out_dir,
        save_csv=save_csv,
        basename=basename
    )


def sp2_cv(
    points,
    env,
    tar,
    k,
    out_dir=None,
    save_csv=True,
    basename="sp2CV"
):
    """Load existing SP2-CV CSV or generate it and cache the result."""

    

    if out_dir is None:
        out_dir = _default_cv_plots_dir("sp2")

    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(
        out_dir,
        f"{basename}_k{k}.csv"
    )

    cate_col_start = 0
    cate_col_end = -1   # -> no categoricals, falls back to plain KMeans on env

    if os.path.exists(csv_path):
        return _read_folds_csv(csv_path)

    return _sp2_cv(
        points,
        env,
        tar,
        cate_col_start = cate_col_start,
        cate_col_end = cate_col_end,
        k=k,
        out_dir=out_dir,
        save_csv=save_csv,
        basename=basename
    )

def _as_threshold_list(thresholds):
    """Convert a threshold value or iterable of thresholds to a list."""

    if thresholds is None:
        return []

    if isinstance(thresholds, (list, tuple, set)):
        return list(thresholds)

    try:
        if len(thresholds) and not isinstance(thresholds, (str, bytes)):
            return list(thresholds)
    except TypeError:
        pass

    return [thresholds]


def build_folds_by_method(
    region, 
    points,
    env, 
    tar, 
    n_folds,
    block_sp_threshold=None,
    sp1_threshold=None,
    sp_threshold=None,
    random_repeats=1,
    random_state=None,
    out_dir=None,
    random_basename="randomCV",
    block_basename="blockCV",
    sp1_basename="sp1CV",
    sp2_basename="sp2CV",
    sp_basename="spCV",

):
    """Return a dictionary of fold assignments keyed by CV method.

    Parameters
    ----------
    block_sp_threshold, sp1_threshold : float or list-like
        A single threshold value or a list of threshold values to try.
        When a list is provided, the returned value for that method is a list
        of fold assignments corresponding to each threshold.
    """

    if out_dir is None:
        out_dir = _default_cv_plots_dir(region)

    folds_by_method = {}

    random_out_dir = os.path.join(out_dir, "random")
    print(random_out_dir)
    block_out_dir = os.path.join(out_dir, "block")
    sp1_out_dir = os.path.join(out_dir, "sp1")
    sp2_out_dir = os.path.join(out_dir, "sp2")
    sp_out_dir = os.path.join(out_dir, "sp")
    print(sp1_out_dir)
    print(sp2_out_dir)
    print(sp_out_dir)

    print(f"Building folds by method in {out_dir}...")
    if random_repeats:
        print(f"  - Random CV: {random_repeats} repeats, k={n_folds}")
        random_folds = random_cv(
            points,
            k=n_folds,
            n_repeats=random_repeats,
            out_dir=random_out_dir,
            save_csv=True,
            basename=random_basename,
            random_state=random_state
        )
        folds_by_method["random"] = random_folds[0]

    if block_sp_threshold is not None:
        print(f"  - Block CV: k={n_folds}, thresholds={block_sp_threshold}")
        block_thresholds = _as_threshold_list(block_sp_threshold)

        for threshold in block_thresholds:
            print(f"    - Threshold: {threshold}")
            folds_by_method[f"block_k{n_folds}_{threshold}"] = block_cv(
                points,
                k=n_folds,
                sp_threshold=threshold,
                out_dir=block_out_dir,
                save_csv=True,
                basename=block_basename
            )

        if len(block_thresholds) == 1:
            folds_by_method["block"] = folds_by_method[f"block_k{n_folds}_{block_thresholds[0]}"]

    if sp1_threshold is not None:
        print(f"  - SP1 CV: k={n_folds}, thresholds={sp1_threshold}")
        sp1_thresholds = _as_threshold_list(sp1_threshold)

        for threshold in sp1_thresholds:
            print(f"    - Threshold: {threshold}")
            folds_by_method[f"sp1_k{n_folds}_{threshold}"] = sp1_cv(
                points,
                sp_threshold=threshold,
                k=n_folds,
                out_dir=sp1_out_dir,
                save_csv=True,
                basename=sp1_basename
            )

        if len(sp1_thresholds) == 1:
            folds_by_method["sp1"] = folds_by_method[f"sp1_k{n_folds}_{sp1_thresholds[0]}"]

    # SP2 without threshold 
    print(f"  - SP2 CV: k={n_folds}")
    folds_by_method[f"sp2_k{n_folds}"] =sp2_cv(points,
                env,
                tar,
                k = n_folds,
                out_dir=sp2_out_dir,
                save_csv=True,
                basename=sp2_basename
            )

    # SP2 with threshold 
    if sp_threshold is not None:
        print(f"  - SP CV: k={n_folds}, thresholds={sp_threshold}")
        sp2_thresholds = _as_threshold_list(sp_threshold)

        for threshold in sp2_thresholds:
            print(f"    - Threshold: {threshold}")
            folds_by_method[f"sp_k{n_folds}_{threshold}"] = sp_cv(
                points,
                env, 
                tar, 
                sp_threshold=threshold,
                k=n_folds,
                out_dir=sp_out_dir,
                save_csv=True,
                basename=sp_basename
            )

        if len(sp2_thresholds) == 1:
            folds_by_method["sp"] = folds_by_method[f"sp_k{n_folds}_{sp2_thresholds[0]}"]
            folds_by_method[f"sp_k{n_folds}_{threshold}"] = sp_cv(
                    points,
                    env, 
                    tar, 
                    sp_threshold=threshold,
                    k=n_folds,
                    out_dir=sp_out_dir,
                    save_csv=True,
                    basename=sp_basename
                )
    

    return folds_by_method

