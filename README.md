# genAI-attentionCNN-DSM

Deep learning and machine-learning workflows for **digital soil mapping (DSM)** in Algeria. The project combines soil-profile data, environmental covariates, spatial cross-validation, attention-based convolutional neural networks, conventional neural networks, and tree-based regressors to estimate soil properties and produce spatial prediction maps.

The repository is primarily implemented as a collection of Jupyter notebooks supported by reusable Python modules. It is intended for experimentation and research workflows rather than as a packaged command-line application.

## What this project does

The workflow covers the main stages of a digital soil-mapping study:

1. **Prepare soil observations** from raw tabular data and scanned documents.
2. **Collect and harmonize covariates**, including raster and climate data.
3. **Construct spatial windows** around observation points for CNN inputs.
4. **Create random and spatially separated cross-validation folds**.
5. **Train and tune several model families**:
   - attention CNNs with CBAM-style channel and spatial attention;
   - multi-task CNNs with six regression heads;
   - a standard CNN baseline;
   - multilayer perceptrons;
   - random forest, decision tree, and LightGBM regressors.
6. **Evaluate predictions** with MSE, RMSE, and R² at fold and pooled out-of-fold levels.
7. **Generate maps and interactive HTML visualizations** for soil properties, uncertainty, and soil texture.

## Repository structure

```text
.
├── Data module/
│   ├── pre_processing.ipynb              # Clean and export tabular data
│   ├── stats.ipynb                       # Exploratory statistics
│   ├── covariates/                       # Gather and resample environmental rasters
│   ├── soil extraction/                  # Extract and validate soil observations from theses and scanned documents
│   └── window_construction/             # Build raster/CNN windows
│
├── DSM Module/
│   ├── Models/
│   │   ├── mtcnn.py                      # Multi-task attention CNN
│   │   ├── utcnn.py                      # Single-output attention CNN
│   │   ├── mlp.py                        # MLP training helper
│   │   ├── trees.py                      # Tree-model training helpers
│   │   └── cnn_cv_metrics.py             # CNN cross-validation metrics
│   │
│   ├── CVs/
│   │   ├── cv_utils.py                   # Fold creation/loading dispatcher
│   │   ├── randomCV.py                   # Random k-fold CV
│   │   ├── spatialCV.py                  # Block and spatial CV methods
│   │   ├── ensemble_consensus.py         # Dependency-free cluster consensus
│   │   └── cv_plot_utils.py               # Fold-layout plots
│   │
│   ├── Training/
│   │   ├── optuna_mtcnn_CV.ipynb         # Multi-task CNN tuning
│   │   ├── optuna_utcnn_CV.ipynb         # Single-output CNN tuning
│   │   ├── mlp_experiments.ipynb
│   │   ├── trees_experiments.ipynb
│   │   └── visualizations.ipynb
│   │
│   ├── Gen_Maps/
│   │   ├── generate_maps.ipynb           # Prediction-map generation
│   │   ├── predict_baslines.ipynb        # Baseline predictions
│   │   ├── predict_cnn.ipynb             # Full-grid CNN prediction
│   │   ├── uncertainty_estimation.ipynb  # Monte Carlo dropout uncertainty
│   │   ├── uncertainty_estimation_viz.ipynb # Uncertainty rasters and diagnostics
│   │   └── maps/                         # GeoTIFF and interactive HTML outputs
│   │
│   ├── Data/                             # DSM data and external geospatial inputs
│   └── CVs/                              # Saved/generated fold assignments
│
└── README.md
```

The repository contains generated notebooks, raster/map products, CSV files, NumPy arrays, and HTML visualizations. Some of these files are large; outputs should be regenerated when the upstream data or model configuration changes.

## Data contract

Several DSM modules use a NumPy array with **47 columns**:

| Columns | Meaning |
|---|---|
| `0:38` | Environmental covariates |
| `38:45` | Soil-property target columns |
| `45` | X coordinate |
| `46` | Y coordinate |

The row index is treated as the observation ID; there is no separate ID column in this array representation. The cross-validation helpers rely on this convention when they return assignments such as `[point_id, fold_index]` or `[point_id, fold_index, cluster_id]`.

The notebooks refer to soil variables including pH, electrical conductivity (`CE`), organic matter (`OM`), cation exchange capacity (`CEC`), and particle-size fractions such as clay, silt, and sand. Confirm the exact column order and units in the dataset being used before training. In particular, `mtcnn.py` currently creates six task outputs, so the target array supplied to that model must contain six compatible targets.

## Installation

No `requirements.txt` or `pyproject.toml` is currently provided. Create an isolated environment and install the core dependencies used by the modules and notebooks:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install \
  numpy pandas matplotlib scikit-learn tensorflow optuna \
  lightgbm jupyterlab geopandas rasterio xarray netcdf4 \
  shapely pyproj folium seaborn
```

The exact geospatial packages required depend on which data-preparation and mapping notebooks are run. On Windows, install a TensorFlow version compatible with the selected Python version and hardware. GPU acceleration is optional; the CNN modules fall back to CPU execution.

## Getting started

### 1. Start Jupyter

From the repository root:

```bash
jupyter lab
```

The notebooks contain relative paths, so launch Jupyter with the repository root as the working directory. If a notebook is opened from a different working directory, update its paths before running it.

### 2. Prepare the tabular data

The preprocessing notebook expects:

```text
Data module/datasets/raw_dataset.csv
```

Run [`Data module/pre_processing.ipynb`](Data%20module/pre_processing.ipynb) to inspect missing values and outliers and to export:

```text
Data module/datasets/preprocessed_dataset.csv
```

The data-extraction notebooks under `Data module/soil extraction/` document the upstream collection, OCR, cleaning, spline, and validation steps for soil observations.

### 3. Build covariates and CNN windows

Use the notebooks under [`Data module/covariates/`](Data%20module/covariates/) to gather and resample raster covariates. Then run the notebooks under [`Data module/window_construction/`](Data%20module/window_construction/) to stack raster layers and extract local windows around soil observations.

CNN inputs are expected to be shaped like:

```text
(number_of_samples, window_height, window_width, number_of_covariates)
```

The exact window size and covariate count are defined by the data-preparation notebook and must match the `input_shape` passed to the model builder.

### 4. Create cross-validation folds

The shared dispatcher in `DSM Module/CVs/cv_utils.py` supports:

- **Random CV**: random point-level k-fold splits, optionally repeated.
- **Block CV**: assigns spatial grid blocks to folds.
- **SP1 CV**: spatial clustering based on AHC-derived blocks.
- **SP2 CV**: spatial/environmental clustering with zero spatial threshold.
- **SP CV**: combines spatial, environmental, and target clustering using consensus clustering.

`build_folds_by_method(...)` always creates an SP2 assignment and can create random, block, SP1, and thresholded SP assignments when the corresponding arguments are supplied. A scalar threshold or a list of thresholds is accepted. With multiple thresholds, the result dictionary contains keys such as `block_k5_10`, `sp1_k5_10`, and `sp_k5_10`; with one threshold, convenient aliases such as `block`, `sp1`, and `sp` are also added.

Fold assignments and spatial fold plots are cached under a region-specific `cv_folds/` directory. Existing CSV files are loaded instead of regenerated, which helps keep repeated experiments comparable. Random fold generation can be made reproducible with `random_state`; each repeated run derives its seed by adding the repetition index.

Example:

```python
import sys
from pathlib import Path

DSM_MODULE = Path(r"D:\path\to\genAI-attentionCNN-DSM\DSM Module")
sys.path.insert(0, str(DSM_MODULE))

from CVs.cv_utils import build_folds_by_method

folds = build_folds_by_method(
    region="algeria",
    points=points,          # shape: (n_samples, 47)
    env=environment,        # environmental features used by SP-CV
    tar=targets,            # target features used by SP-CV
    n_folds=5,
    block_sp_threshold=10,
    sp1_threshold=10,
    sp_threshold=10,
    random_repeats=1,
    random_state=42,
)
```

For spatial CV, choose the threshold in the coordinate units of the dataset and inspect the generated fold plot. A threshold that is too small or too large can produce unbalanced or scientifically uninformative folds.

The spatial-CV implementation is based on the following reference dataset and code:

> Y. Wang (2023), “SPATIAL+ CROSS-VALIDATION (SP-CV) EXPERIMENTS DATASETS AND CODES (after review),” DANS Data Station Physical and Technical Sciences, V2. DOI: [10.17026/DANS-ZWJ-ZAUQ](https://doi.org/10.17026/DANS-ZWJ-ZAUQ).

### 5. Train models

The training notebooks use Optuna to search model architectures and training hyperparameters. Start with:

- [`DSM Module/Training/optuna_mtcnn_CV.ipynb`](DSM%20Module/Training/optuna_mtcnn_CV.ipynb) for the multi-task attention CNN;
- [`DSM Module/Training/optuna_utcnn_CV.ipynb`](DSM%20Module/Training/optuna_utcnn_CV.ipynb) for the single-output attention CNN;
- [`DSM Module/Training/mlp_experiments.ipynb`](DSM%20Module/Training/mlp_experiments.ipynb) for the MLP baseline;
- [`DSM Module/Training/trees_experiments.ipynb`](DSM%20Module/Training/trees_experiments.ipynb) for tree-based baselines.

The model modules can also be imported directly:

```python
from Models.mtcnn import create_cnn_optuna as create_multitask_cnn
from Models.utcnn import create_cnn_optuna as create_single_output_cnn
from Models.mlp import MLP_training
from Models.trees import (
    random_forest_training,
    decision_tree_training,
    lgboost_training,
)
```

`create_cnn_optuna(trial, input_shape)` receives an Optuna trial (or `optuna.trial.FixedTrial`) and returns a compiled-ready Keras model. The returned model is not compiled by the builder; the training notebook supplies the optimizer, loss, and callbacks.

### 6. Evaluate with out-of-fold predictions

[`DSM Module/Models/cnn_cv_metrics.py`](DSM%20Module/Models/cnn_cv_metrics.py) provides:

- `evaluate_best_cnn_cv(...)` for a single-output or compatible multi-output model;
- `evaluate_multitask_cnn_cv(...)` for multi-task evaluation by fold and task.

The helpers:

1. replay fixed Optuna parameters with `FixedTrial`;
2. fit a scaler on each training fold only;
3. train with early stopping;
4. collect predictions for the validation fold;
5. calculate MSE, RMSE, and R²;
6. return aligned out-of-fold true and predicted arrays.

This makes it possible to compare models without mixing validation information into the training scaler.

### 7. Generate maps

After selecting and training a model, use the notebooks in [`DSM Module/Gen_Maps/`](DSM%20Module/Gen_Maps/) to predict over the covariate grid and export raster or interactive products.

For a single-output CNN, [`DSM Module/Gen_Maps/predict_cnn.ipynb`](DSM%20Module/Gen_Maps/predict_cnn.ipynb) loads a saved Keras model, reads stacked grid windows from `inputs_stacked/inputs.npy`, scales the covariates, predicts in batches, associates predictions with `points.csv`, and writes a property-specific CSV such as `predictions/S%.csv`. Update the model path, input path, point file, output property name, and scaler behavior for a different trained model or dataset.

Existing map products include:

- GeoTIFF soil-property maps;
- interactive HTML maps;
- random-forest and CNN baseline maps;
- multi-depth predictions;
- CNN, MLP, and tree-model uncertainty outputs;
- mean and standard-deviation uncertainty rasters;
- uncertainty diagnostics such as prediction interval coverage probability (PICP), prediction interval width (PIW), and mean prediction interval width (MPIW);
- soil-texture maps.

The map notebooks also use project geospatial assets such as Algerian boundaries, Wilaya boundaries, FAO pH data, and Köppen-Geiger climate data.

## Uncertainty estimation

The uncertainty workflow is implemented in:

- [`uncertainty_estimation.ipynb`](DSM%20Module/Gen_Maps/uncertainty_estimation.ipynb), which applies Monte Carlo dropout by running a trained model repeatedly with `training=True`;
- [`uncertainty_estimation_viz.ipynb`](DSM%20Module/Gen_Maps/uncertainty_estimation_viz.ipynb), which converts prediction tables into georeferenced GeoTIFFs, removes extreme raster outliers where configured, and evaluates uncertainty against observations.

For each prediction location, repeated stochastic predictions are summarized by:

- `mean_unc`: the estimated predictive mean;
- `std_unc`: the estimated predictive standard deviation.

The notebooks support single-output U‑CNN results, multi-task M‑CNN results, and MLP uncertainty tables. They use `rasterio` for GeoTIFF output, `geopandas` for point geometry, and nearest-neighbor matching through `scipy.spatial.cKDTree` when predictions and observations do not share identical coordinates.

For a nominal confidence level such as 95%, the diagnostic notebook calculates:

```text
lower = mean_unc - z * std_unc
upper = mean_unc + z * std_unc
PICP   = fraction of observations inside [lower, upper]
PIW    = 2 * z * std_unc
MPIW   = mean(PIW)
```

These intervals are useful diagnostics, not a guarantee of calibrated uncertainty. Check the coordinate reference system, raster transform, point ordering, and target units before interpreting the resulting maps.

## Model details

### Multi-task attention CNN (`mtcnn.py`)

The multi-task model uses:

- 1–2 convolutional blocks;
- standard or separable convolutions;
- optional batch normalization, pooling, dropout, and dilation;
- CBAM-style channel and spatial attention;
- shared dense layers;
- six task-specific regression heads with linear outputs.

Optuna explores the number and configuration of convolutional blocks, attention, activations, shared layers, task-specific layers, and dropout.

### Single-output attention CNN (`utcnn.py`)

The single-output model uses the same general attention-CNN idea but ends in one linear regression output. It is intended for predicting one soil property at a time and serves as a useful comparison with the multi-task architecture.

### MLP (`mlp.py`)

`MLP_training` builds a configurable dense network from:

- `dropout_rate`;
- `nb_neurones_by_layer`;
- `hidden_activation`;
- `lr`;
- `output_activation`;
- `batch_size`;
- `epochs`.

It uses mean squared error, mean absolute error, and optional early stopping when validation data is supplied.

### Tree baselines (`trees.py`)

The tree helpers wrap:

- `sklearn.ensemble.RandomForestRegressor`;
- `sklearn.tree.DecisionTreeRegressor`;
- `lightgbm.LGBMRegressor`.

The LightGBM helper uses an evaluation set and early stopping. Pass model parameters explicitly through `param_dict`.

## Outputs and reproducibility

Typical generated artifacts include:

```text
DSM Module/CVs/cv_folds/<region>/
DSM Module/Gen_Maps/maps/
DSM Module/Gen_Maps/outputs_uncertainty_utcnn/
DSM Module/Gen_Maps/outputs_uncertainty_mtcnn/
DSM Module/Gen_Maps/outputs_uncertainty_mlp/
DSM Module/Training/
```

For reproducible experiments:

- set `random_state` for random folds;
- preserve the generated fold CSV files;
- record the selected Optuna parameters;
- keep the same preprocessing, target ordering, and scaler;
- use the same `input_shape` and window-construction settings;
- report spatial CV results separately from random CV results;
- use the same scaler-fitting convention for training and full-grid prediction;
- preserve the model file, input-grid array, point table, and raster reference used for each map;
- save out-of-fold predictions alongside summary metrics.

The project intentionally distinguishes random point-level validation from spatial validation. Random folds can place nearby observations in both training and validation sets and may therefore provide an optimistic estimate of map performance.

## Important implementation notes

- There is currently no package configuration or dependency lock file. Environment setup is therefore manual.
- Notebook paths are relative and should be run from the repository root unless they are adjusted.
- The core data convention is positional; changing column order without updating the CV and training code will silently produce invalid results.
- Coordinates are assumed to be the final two columns of the 47-column array.
- `mtcnn.py` hardcodes six task heads. The target matrix must match that design.
- `ensemble_consensus.py` replaces external METIS-based cluster ensemble calls with a co-association matrix and agglomerative clustering implementation that works without those subprocess dependencies.
- The uncertainty notebooks currently use relative paths such as `models_utcnn/`, `inputs_stacked/`, `points.csv`, and `cnn_basic_maps\A\A_map.tif`; run them from the directory expected by the notebook or update those paths first.
- The prediction notebook fits a `StandardScaler` on the full prediction grid. For strict deployment without data leakage, reuse the scaler fitted during training and persist it with the model.
- Generated HTML, raster, NumPy, and CSV files may be large and are not necessarily regenerated automatically.

## Citation and license

No license or citation metadata is currently included in the repository. Add the appropriate license, dataset acknowledgements, and publication citation before distributing or publishing derived results.
