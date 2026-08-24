# boost-shap-gii

**Global Importance Index via Boosted SHAP**

A YAML-driven predictive modeling pipeline combining CatBoost and SHAP interaction values to produce Global Importance Indices (GII).

---

## Installation

### From GitHub
```bash
pip install git+https://github.com/tjkeding/boost-shap-gii
```

### Development Install
```bash
git clone https://github.com/tjkeding/boost-shap-gii.git
cd boost-shap-gii
pip install -e .
```

### Conda Environment (Alternative)
A conda environment specification is provided for users who prefer isolated environment management:
```bash
conda env create -f environment.yaml
conda activate boost-shap-gii
pip install -e .
```

**R (optional):** R is required only for the `plot` subcommand. Install R packages separately as listed in `environment.yaml`.

---

## Quickstart

### 1. Configuration
Copy `example_config_minimal.yaml` and specify your:
- `paths.input_data` (CSV/Parquet)
- `paths.output_dir`
- `features` (column patterns for continuous/ordinal/nominal)
- `modeling.outcome`

### 2. Running the Pipeline

#### CLI Interface

The `boost-shap-gii` command provides independently callable subcommands for each pipeline stage.

**Verify dependencies:**
```bash
boost-shap-gii check-env
```

**Training and hyperparameter tuning:**
```bash
boost-shap-gii train --config config.yaml
```

**Out-of-fold evaluation and SHAP analysis:**
```bash
boost-shap-gii predict --config config.yaml
```

**Inference on an independent dataset:**
```bash
boost-shap-gii infer --config resolved_config.yaml --data new_data.csv --output-subdir subdir_name
```

**Visualization of significant effects (requires R):**
```bash
boost-shap-gii plot --config config.yaml
```

**Per-individual SHAP reports (optional, configuration-driven):** Set `shap.indiv_ci_nboot > 0` in the config to enable per-individual SHAP bootstrap CIs during `predict` and `infer`. Reports are written to `indiv_reports/` within the output directory.

#### Alternative: Shell Script

The shell script orchestrator is available as an alternative interface. It automatically chains training, prediction, and plotting into a single command.

**Training & SHAP Analysis:**
```bash
bash run_boost-shap-gii.sh train config.yaml
```

**Inference on New Data:**
```bash
bash run_boost-shap-gii.sh infer train_config.yaml new_data.csv sub_dir
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

## Aggregate SHAP (Group-Level GII)

When features form a natural group (e.g., items in a psychometric subscale, related biomarkers), the pipeline can compute group-level M, V, and GII by summing member SHAP values within user-defined groups.

### Configuration

Add an `aggregate_shap` block to the config (top-level, alongside `shap` and `plot`):

```yaml
aggregate_shap:
  subscale_A_total:
    - "subscale_A_item1"
    - "subscale_A_item2"
    - "subscale_A_item3"
  subscale_B_total:
    - "subscale_B_item1"
    - "subscale_B_item2"
```

### Constraints
- Each feature may belong to **at most one group** (disjoint membership).
- **Nominal features are not permitted** in aggregate groups (SHAP values for categorical features are not directly comparable across levels).
- Group names must not collide with existing feature column names.
- Single-member groups emit a warning (no aggregation benefit).

### What It Produces
For each group, the pipeline emits:
- **Singleton aggregate**: group-level main effect (sum of member singleton SHAP values).
- **Within-group interaction**: sum of all pairwise member-by-member interactions.
- **Between-group interactions**: sum of cross-group pairwise interactions (when multiple groups are defined).
- **Group-by-ungrouped interactions**: sum of interactions between group members and features not in any group.

All aggregate effects receive shadow-calibrated significance testing via block-permuted Boruta exceedance (Au et al. 2022), where grouped shadow features share a single permutation index per group to preserve within-group correlation structure.

Results appear in `shap_stats_global.csv` with `is_aggregate = True`. See `INPUT_SPECIFICATION.md` Section 4 for full algorithmic details.

---

## Per-individual SHAP reports

When `shap.indiv_ci_nboot > 0`, each `predict` and `infer` run generates per-individual SHAP attribution plots with bootstrap confidence intervals. Reports are written to `indiv_reports/` within the respective output directory.

**Required configuration keys:**

- `shap.indiv_ci_nboot` (integer): bootstrap iterations; set to `0` to disable. Minimum recommended `2500`; `5000` for peer-review runs.
- `shap.indiv_scaling_mode` (string): one of `raw`, `sd` (regression only), or `custom_value`.
- `shap.indiv_scaling_value` (number): user-supplied divisor; required when `indiv_scaling_mode: custom_value`.
- `shap.compute_global_on_inference` (bool, default `false`): when `true`, `infer` also emits population-level GII on the inference set.

See `INPUT_SPECIFICATION.md` Section 10 for the full algorithmic description, CI interpretation, and output schema.

---

## Cross-Validation Strategy

The pipeline supports three CV strategies via `modeling.cv_strategy`:

| Value | Splitter | Description |
|---|---|---|
| `"uniform"` (default) | `KFold` | Standard K-fold, task-type-independent. |
| `"stratified"` | `StratifiedKFold` / quantile-binned `StratifiedKFold` | Preserves class proportions (classification) or outcome quantile distribution (regression) across folds. |
| `"group"` | `GroupKFold` | Ensures all observations sharing a group label remain in the same fold. Requires `modeling.group_column`. |

### Group CV

When `cv_strategy: "group"`, set `modeling.group_column` to the column name containing group labels (e.g., `"subject_id"`). The group column is automatically excluded from the feature candidate set. Groups are assigned to folds via greedy scheduling (Graham, 1966), which minimizes fold-size imbalance under unequal group sizes; a warning is emitted when the max/min fold-size ratio exceeds 2.0. The number of unique groups must be at least `cv_folds` (and at least `inner_cv_folds` when tuning is configured).

**SHAP significance testing with group CV**: when the group strategy is active, population-level bootstrap significance testing uses cluster-aware resampling (resample entire groups with replacement, then expand to member rows). This preserves within-group correlation in bootstrap resamples, producing correctly calibrated confidence intervals (Cameron, Gelbach, & Miller, 2008). When the number of unique groups is below 20, cluster bootstrap falls back to i.i.d. resampling with a `RuntimeWarning` (Ukoumunne, Gulliford, Chinn, Sterne, & Burney, 2003). During per-individual bootstrap-of-CV inference, group structure cannot be preserved in bootstrap inner splits, so they fall back to plain `KFold`.

**FDR correction method**: the default multiple-comparison correction is Benjamini-Hochberg FDR (`shap.bootstrapping.fdr_method: "bh"`). Set to `"by"` for Benjamini-Yekutieli FDR when test statistics are positively dependent.

### Inner CV Repeats

Set `modeling.tuning.n_inner_repeats` (default `1`) to average inner CV scores across multiple repetitions per Optuna trial, reducing tuning variance at the cost of additional computation. A warning is emitted when `n_inner_repeats > 10` or when total inner fits exceed 5000.

---

## Outcome Transformations

The pipeline supports an optional, user-provided Python transform applied to the outcome before model training and reversed before evaluation. This is useful for outcomes that require a nonlinear reparameterization (e.g., a rate transform, a residualization against a baseline covariate) that the pipeline's own preprocessing does not cover.

### Configuration

Add a `transformations` block to the config (top-level, alongside `shap` and `plot`):

```yaml
transformations:
  file: "path/to/my_transform.py"
  params: {}
  required_cols: []
  back_transform_shap: false
```

The named script must define two functions:

- `input_transform(df_raw, train_idx, val_idx, outcome_col, params) -> (y_train, y_val, metadata)`, called once per CV fold to produce the transformed training and validation targets. `metadata` is any JSON-serializable object needed later to reverse the transform.
- `output_transform(predictions, metadata, params, *, df_raw=None, row_indices=None) -> predictions`, called to map predictions on the transformed scale back to the original outcome scale.

Before training begins, the pipeline runs an automatic smoke test on a small subset of the data, checking execution, output shapes, finiteness, JSON-serializability of `metadata`, and the round-trip correctness of `output_transform`.

### SHAP Back-Transformation

When `back_transform_shap: true`, the pipeline additionally verifies that the transform is affine (via a linear-regression check during the smoke test) and, if so, rescales SHAP values by the transform's slope so that M, V, and GII are reported in original-outcome units. The pipeline halts with an error if `back_transform_shap: true` is requested for a transform found to be non-affine, since SHAP additivity does not survive a nonlinear back-transformation.

### Interactions with Other Features

- When a `transformations` block is active, `multi_regression`'s automatic outcome z-scoring is skipped; the transform takes full ownership of the outcome space.
- `infer.py` applies `output_transform` using per-fold metadata persisted by `train.py` (`fold_transform_metadata.json`); it does not require access to the original training data file.

See `INPUT_SPECIFICATION.md` for the full transform API contract, the smoke test specification, and the `transform_config.json` / `fold_transform_metadata.json` artifact schemas.

---

See `INPUT_SPECIFICATION.md` for exhaustive technical details, data schemas, and mathematical formulas.
