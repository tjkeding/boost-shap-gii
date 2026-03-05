# INPUT_SPECIFICATION.md

## Technical Reference for boost_shap_gii

### 1. Pipeline Stages

#### Stage 0: Pre-flight (`check_env.py`)
- **Python Verification**: Imports `catboost`, `optuna`, `shap`, `pyarrow`, `sklearn`, `scipy`, `pandas`, `yaml`.
- **R Verification**: Checks `ggplot2`, `dplyr`, `arrow`, `tidyr`, `foreach`, `doParallel`, `gridExtra`, `stringr`, `yaml`.
- **Guard Logic**: Aborts `run_boost_shap_gii.sh` if any dependency is missing.

#### Stage 1: Data Ingestion
- **Formats**: `.csv` or `.parquet`.
- **Reactive Strategy**:
    1. Attempt `pd.read_csv(sep=",")` or `pd.read_parquet()`.
    2. Catch `ParserError`, `ValueError`, or general `Exception`.
    3. Fallback to `pd.read_csv(sep=None, engine="python")` for auto-detection.
- **Preprocessing**: 
    - Whitespace-only strings → `pd.NA`.
    - Drop rows with missing outcomes.

#### Stage 2: Feature Selection (`FeatureSelector`)
- **Groups**: `continuous`, `ordinal`, `nominal`.
- **Matching**: `exact`, `substring`, `prefix`, `suffix`.
- **Conflicts**: Raises `ValueError` if a column matches multiple types.

#### Stage 3: Training & SHAP
- **Nested CV**: Outer K-fold for OOF; Inner K-fold for Optuna tuning.
- **SHAP Interaction**: Returns 4D tensor `(N, C, F+1, F+1)` for multiclass/multi-regression.
- **Noise calibration**: Boruta-style shadow features per fold.

### 2. Mathematical GII Formula

$$GII = \sqrt{M \times V}$$

- **M (Magnitude)**: mean of absolute SHAP across bootstrap resamples.
- **V (Variability)**: Standard deviation of the systematic signal (spline or group-means) across feature values.
- **Stability Gate**: median(boot) / CI_width > 2.0.

### 3. Directory Structure & Artifacts

```
output_dir/
├── models/                  # .cbm fold models
├── feature_metadata.json    # Types and ordinal levels
├── shap_stats_global.csv    # Final GII results
├── predictions_oof.csv      # Cross-validated predictions
└── shap_analysis/           # SHAP plots and raw microdata
```

### 4. Categorical Sentinel Values
- **Nominal**: NaN -> "__NA__" (literal string level).
- **Ordinal**: NaN -> pd.NA (mapped to -1 then masked).
- **SHAP Matrix**: Category NaN -> max_code + 1.
