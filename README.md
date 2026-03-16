# boost-shap-gii

**Global Importance Index via Boosted SHAP**

A YAML-driven predictive modeling pipeline combining CatBoost and SHAP interaction values to produce Global Importance Indices (GII).

---

## Quickstart

### 1. Environment Setup
```bash
conda env create -f environment.yaml
conda activate boost-shap-gii
```
*Note: Install R packages separately as listed in environment.yaml.*

### 2. Configuration
Copy `example_config_minimal.yaml` and specify your:
- `paths.input_data` (CSV/Parquet)
- `paths.output_dir`
- `features` (column patterns for continuous/ordinal/nominal)
- `modeling.outcome`

### 3. Running the Pipeline
The orchestrator automatically verifies your environment and data paths before starting.

**Training & SHAP Analysis:**
```bash
bash run_boost-shap-gii.sh train config.yaml OUTCOME_RANGE NEGATE_SHAP Y_AXIS_LABEL
```

**Inference on New Data:**
```bash
bash run_boost-shap-gii.sh infer train_config.yaml new_data.csv sub_dir OUTCOME_RANGE NEGATE_SHAP Y_AXIS_LABEL
```

---

## Robust Data Loading
The pipeline features **Reactive Data Loading**. It assumes standard CSV (comma-separated) or Parquet by default, but automatically falls back to **delimiter auto-detection** if standard parsing fails. This ensures compatibility with TSV, semicolon, and other common research formats without manual configuration.

---

## Core Components
- **train.py**: Data preparation and nested CV model training.
- **predict.py**: OOF evaluation and SHAP analysis orchestration.
- **infer.py**: Independent dataset ensemble inference.
- **shap_utils.py**: GII calculation and noise calibration.
- **plot.R**: High-quality visualization of significant effects.
- **check_env.py**: Automated dependency verification.

---

## Missing Value Handling

### Nominal Features
Missing values in nominal (categorical) features are filled with the literal string `"__NA__"` before encoding. CatBoost treats `"__NA__"` as a valid, distinct category level, meaning the model can learn whether missingness itself is a predictor. This is an **implicit informativeness assumption**: if nominal missingness is non-informative in your data, this behavior is conservative but not harmful.

### Ordinal Features
Missing values in ordinal features are preserved as `pd.NA` (integer sentinel -1 in CatBoost's coded representation) and masked to `NaN` after encoding. CatBoost handles missing ordinal values natively.

### Continuous Features
Missing values in continuous features are handled natively by CatBoost during training. In the SHAP bootstrap, the NaN-aware routing in `_bootstrap_worker_chunk` uses the NaN indicator as a discrete grouping axis for spline fitting, so missingness patterns are captured without imputation artifacts.

---

## GII Interpretation

### GII = sqrt(M * V)
The Global Importance Index is the geometric mean of two components:
- **M (Magnitude)**: mean absolute SHAP value across bootstrap resamples — represents the average prediction contribution.
- **V (Variability)**: standard deviation of the systematic (spline- or group-mean-fitted) trend in SHAP values as a function of feature values — represents whether the effect has a structured dose-response relationship.

A feature can have large M (strong average effect) but near-zero V if its SHAP contribution is constant across all feature values (e.g., a binary feature that uniformly shifts predictions). Such features have low GII but are reported separately through significant M values in `shap_stats_global.csv`.

### GII Values Are Not Summable to Marginal SHAP
GII values decompose prediction contributions at the effect level (singletons and pairwise interactions). They cannot be summed to reconstruct marginal SHAP importance for a given feature. The GII decomposition identity differs from the standard SHAP additivity property.

### Singleton vs. Interaction Scale Convention
Singleton effects (Φ[i,i]) are extracted from the SHAP interaction matrix diagonal at full scale. Interaction effects (Φ[i,j] + Φ[j,i]) use both off-diagonal cells to recover the true Shapley interaction index, because the interaction matrix divides the total interaction contribution by 2 per cell (one cell per direction). The summed convention ensures that singleton and interaction GII values are on the same prediction-contribution scale, making cross-type comparisons valid.

### Boruta Noise Baseline is Model-Adaptive
The shadow noise distribution used for significance testing is derived from a shadow model trained jointly on real and permuted features. This means the noise baseline is adaptive: it represents how important noise features are *in the presence of the real signal*. When the real features are strongly predictive, shadow features receive lower SHAP attribution, reducing the noise threshold. This is the correct null for the Boruta framework and is not a conservatism concern.

---

See `INPUT_SPECIFICATION.md` for exhaustive technical details, data schemas, and mathematical formulas.
