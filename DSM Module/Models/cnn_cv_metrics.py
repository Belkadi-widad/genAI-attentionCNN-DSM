"""Cross-validation metrics for CNN models tuned with Optuna."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _as_2d(values: Any) -> np.ndarray:
    """Convert a single-output or multi-output value to (samples, targets)."""

    if isinstance(values, (list, tuple)):
        arrays = [np.asarray(value).reshape(-1, 1) for value in values]
        return np.concatenate(arrays, axis=1)

    array = np.asarray(values)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array.reshape(array.shape[0], -1)


def _metric_row(
    fold: str | int,
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Calculate regression metrics for one fold/target pair."""

    return {
        "fold": fold,
        "target": target,
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_best_cnn_cv(
    model_builder: Callable[[Any, tuple[int, ...]], Any],
    best_params: dict[str, Any],
    inputs: np.ndarray,
    y: np.ndarray,
    input_shape: tuple[int, ...],
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int = 50,
    patience: int = 10,
    seed: int | None = None,
    target_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Evaluate fixed Optuna parameters with the exact supplied CV splits.

    ``model_builder`` must be the same function used during tuning, such as
    ``create_cnn_optuna``. The builder receives an Optuna ``FixedTrial`` so
    every architecture hyperparameter is replayed from ``best_params``.

    Returns
    -------
    fold_metrics:
        One row per fold and target.
    overall_metrics:
        Pooled out-of-fold metrics, calculated after concatenating all folds.
    y_true_oof, y_pred_oof:
        Arrays aligned to the original sample order.
    """

    import optuna
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam, AdamW

    x_values = np.asarray(inputs)
    y_values = _as_2d(y)
    n_samples = len(x_values)
    if len(y_values) != n_samples:
        raise ValueError("inputs and y must contain the same number of samples")
    if target_names is not None and len(target_names) != y_values.shape[1]:
        raise ValueError("target_names must match the number of target columns")

    def target_name(target_index: int) -> str:
        if target_names is not None:
            return target_names[target_index]
        return (
            f"target_{target_index + 1}"
            if y_values.shape[1] > 1
            else "target"
        )

    optimizer_name = best_params.get("optimizer", "adam")
    learning_rate = float(best_params["learning_rate"])
    batch_size = int(best_params["batch_size"])
    scaler_name = best_params.get("scaler", "standard")
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError(f"Unsupported optimizer in best_params: {optimizer_name}")
    if scaler_name not in {"standard", "minmax"}:
        raise ValueError(f"Unsupported scaler in best_params: {scaler_name}")

    y_true_oof = np.full_like(y_values, np.nan, dtype=float)
    y_pred_oof = np.full_like(y_values, np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []

    for fold_number, (train_idx, val_idx) in enumerate(cv_splits):
        if seed is not None:
            tf.keras.utils.set_random_seed(seed + fold_number)

        scaler = StandardScaler() if scaler_name == "standard" else MinMaxScaler()
        x_train = x_values[train_idx]
        x_val = x_values[val_idx]
        x_train = scaler.fit_transform(
            x_train.reshape(-1, x_train.shape[-1])
        ).reshape(x_train.shape)
        x_val = scaler.transform(
            x_val.reshape(-1, x_val.shape[-1])
        ).reshape(x_val.shape)

        fixed_trial = optuna.trial.FixedTrial(best_params)
        model = model_builder(fixed_trial, input_shape)
        optimizer = (
            Adam(learning_rate=learning_rate)
            if optimizer_name == "adam"
            else AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        )
        n_outputs = len(model.outputs)
        model.compile(
            optimizer=optimizer,
            loss=["mse"] * n_outputs if n_outputs > 1 else "mse",
        )

        y_train = y_values[train_idx]
        y_val = y_values[val_idx]
        y_train_fit = (
            [y_train[:, i] for i in range(n_outputs)]
            if n_outputs > 1
            else y_train
        )
        y_val_fit = (
            [y_val[:, i] for i in range(n_outputs)]
            if n_outputs > 1
            else y_val
        )
        model.fit(
            x_train,
            y_train_fit,
            validation_data=(x_val, y_val_fit),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                )
            ],
            verbose=0,
        )

        predictions = _as_2d(model.predict(x_val, verbose=0))
        if predictions.shape[1] != y_values.shape[1]:
            raise ValueError(
                "The model output count does not match the number of target columns"
            )
        y_true_oof[val_idx] = y_val
        y_pred_oof[val_idx] = predictions

        for target_index in range(y_values.shape[1]):
            fold_rows.append(
                _metric_row(
                    fold_number + 1,
                    target_name(target_index),
                    y_val[:, target_index],
                    predictions[:, target_index],
                )
            )

        del model
        K.clear_session()

    if np.isnan(y_pred_oof).any():
        raise ValueError("CV splits did not produce predictions for every sample")

    overall_rows = []
    for target_index in range(y_values.shape[1]):
        overall_rows.append(
            _metric_row(
                "overall",
                target_name(target_index),
                y_true_oof[:, target_index],
                y_pred_oof[:, target_index],
            )
        )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(overall_rows),
        y_true_oof,
        y_pred_oof,
    )


def evaluate_multitask_cnn_cv(
    model_builder: Callable[[Any, tuple[int, ...]], Any],
    best_params: dict[str, Any],
    inputs: np.ndarray,
    y: np.ndarray,
    input_shape: tuple[int, ...],
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int = 50,
    patience: int = 10,
    seed: int | None = None,
    task_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Evaluate a fixed multi-task CNN by task, fold, and globally.

    This uses the same fold assignments and model-training behavior as
    :func:`evaluate_best_cnn_cv`. The global table pools every out-of-fold
    prediction across every task, whereas ``overall_by_task`` calculates each
    task independently across all of its folds.

    Returns
    -------
    metrics_by_fold_and_task:
        One row for each fold/task combination.
    overall_by_task:
        One row per task, pooling that task's out-of-fold predictions.
    overall_metrics:
        One row pooling all task/sample pairs. This is useful as a compact
        multi-task summary, but should not replace ``overall_by_task`` when
        tasks have different units or scales.
    y_true_oof, y_pred_oof:
        Out-of-fold arrays with shape ``(n_samples, n_tasks)``.
    """

    y_values = _as_2d(y)
    if y_values.shape[1] < 2:
        raise ValueError("evaluate_multitask_cnn_cv requires at least two tasks")

    metrics_by_fold, overall_by_task, y_true_oof, y_pred_oof = (
        evaluate_best_cnn_cv(
            model_builder=model_builder,
            best_params=best_params,
            inputs=inputs,
            y=y_values,
            input_shape=input_shape,
            cv_splits=cv_splits,
            epochs=epochs,
            patience=patience,
            seed=seed,
            target_names=task_names,
        )
    )

    overall_metrics = pd.DataFrame(
        [
            _metric_row(
                "overall",
                "all_tasks",
                y_true_oof.reshape(-1),
                y_pred_oof.reshape(-1),
            )
        ]
    )

    return (
        metrics_by_fold,
        overall_by_task,
        overall_metrics,
        y_true_oof,
        y_pred_oof,
    )
