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

See `INPUT_SPECIFICATION.md` for exhaustive technical details, data schemas, and mathematical formulas.
