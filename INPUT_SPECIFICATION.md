# INPUT_SPECIFICATION.md

## Technical Reference for boost-shap-gii

---

### 0. Package Structure and Invocation

#### Source Layout
The pipeline is organized as an installable Python package under the `src/` layout:
```
src/boost_shap_gii/
    __init__.py          # Package metadata (version)
    cli.py               # CLI entry point with subcommand dispatch
    train.py             # Data ingestion, feature selection, model training
    predict.py           # OOF evaluation and SHAP analysis
    infer.py             # Independent dataset inference
    shap_utils.py        # GII computation and noise calibration
    utils.py             # Shared utility functions
    check_env.py         # Dependency verification
    scripts/
        plot.R           # ggplot2-based visualization
        run_boost-shap-gii.sh  # Shell script orchestrator
```

#### Installation
```bash
pip install git+https://github.com/tjkeding/boost-shap-gii   # from GitHub
pip install -e .                                               # editable (development)
```

#### CLI Entry Points
After installation, the `boost-shap-gii` command is available on `PATH`:
```
boost-shap-gii check-env
boost-shap-gii train    --config CONFIG
boost-shap-gii predict  --config CONFIG
boost-shap-gii infer    --config CONFIG --data DATA --output-subdir SUBDIR
boost-shap-gii plot     --config CONFIG --outcome-range RANGE --negate-shap BOOL --y-axis-label LABEL [--run-dir DIR]
```

#### Module Invocation (Alternative)
Each pipeline stage can also be invoked as a Python module:
```bash
python -m boost_shap_gii.check_env
python -m boost_shap_gii.train    --config CONFIG
python -m boost_shap_gii.predict  --config CONFIG
python -m boost_shap_gii.infer    --config CONFIG --data DATA --output-subdir SUBDIR
```
The shell script `run_boost-shap-gii.sh` uses module invocation internally and remains available
as a pipeline orchestrator that chains training, prediction, and plotting.

---

### 1. Pipeline Stages

#### Stage 0: Pre-flight (`check_env.py`)
- **Invocation**: `boost-shap-gii check-env` or `python -m boost_shap_gii.check_env`.
- **Python Verification**: Imports `catboost`, `optuna`, `shap`, `pyarrow`, `sklearn`, `scipy`,
  `pandas`, `yaml`, `joblib`, `statsmodels`.
- **R Verification**: Checks `ggplot2`, `dplyr`, `arrow`, `tidyr`, `foreach`, `doParallel`,
  `gridExtra`, `stringr`, `yaml`. R is optional; missing R packages produce a warning but
  do not abort the pipeline (only the `plot` subcommand requires R).
- **Guard Logic**: Aborts if any Python dependency is missing.

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
- **Output**: Sorted `final_columns` list; column ordering is deterministic and consistent
  across train/predict/infer.

#### Stage 3: Type Enforcement & Preprocessing

**Nominal features**:
- `NaN` → `"__NA__"` (literal string) → encoded as `category`.
- CatBoost treats `"__NA__"` as a distinct, valid category level. This embeds an implicit
  assumption that nominal missingness is potentially informative. The model may learn splits
  that distinguish missing from non-missing observations. This behavior is fixed and is not
  configurable.

**Ordinal features** (two-tier validation):
- Tier 1 (hard error): if > 50% of *unique values* in the data are absent from `levels`,
  raises `ValueError`. Indicates misconfigured level definitions or systematic naming mismatch.
- Tier 2 (loud warning): if > 10% of *observations* (non-missing) have values absent from
  `levels`, prints a warning with the exact fraction. Indicates data quality issues not captured
  by unique-value fraction alone.
- After validation: levels mapped to integer codes via `pd.CategoricalDtype`; `NaN` → `pd.NA`.

**Continuous features**:
- `pd.to_numeric(errors='coerce').astype("float32")`.

#### Stage 4: Training (`train.py`)

**Nested Cross-Validation**:
- Outer CV: `KFold` (regression/multi_regression) or `StratifiedKFold` (classification),
  seeded with `config.execution.random_seed`.
- Inner CV (Optuna tuning): same type, seeded with `random_seed + fold_idx + 1` to ensure
  inner and outer folds use distinct split patterns.

**Phase 1 — Clean Model**:
- Optuna TPE hyperparameter tuning on inner CV folds.
- Final clean model trained with tuned `iterations` on the full outer training fold.
- No outer early stopping: iteration count set by inner CV mean `best_iteration_`.

**Phase 2 — Shadow Model (Noise Calibration)**:
- Shadow features: each column independently permuted (column-wise, not row-wise).
  Permutation seed: `random_seed + fold_idx`.
- Shadow model trained on concatenated real + shadow features (2p total).
- Early stopping on outer validation fold (`X_val_full`); ceiling = `tuned_iters * 2`.
  This allows the shadow model to find its own optimal iteration count for the larger
  feature space without leakage: shadow outputs are used only for SHAP noise calibration,
  not for predictive evaluation.
- `early_stopping_rounds` patience is taken from `config.modeling.tuning.early_stopping_rounds`.

#### Stage 5: Prediction / Evaluation (`predict.py`)
- Replicates outer CV splitter from training (same seed and type).
- Validates that the number of saved model files matches `splitter.get_n_splits()`.
  Raises `AssertionError` with a clear message if counts diverge (protects against
  incomplete training runs).
- Bootstrapped 95% CIs for OOF metrics; permutation test for model vs. chance.
- Triggers `run_shap_pipeline()` in OOF mode.

#### Stage 6: Inference (`infer.py`)
- Loads all K fold models; applies each to full dataset; averages predictions (soft voting
  for classification, mean for regression).
- Soft voting assumes calibrated probability outputs from CatBoost. For the supported loss
  functions (Logloss, MultiClass, RMSE, MultiRMSE), this assumption holds.
- Triggers `run_shap_pipeline()` in inference mode; cluster bootstrap used to propagate
  fold-level variation.

#### Stage 7: SHAP Analysis (`shap_utils.py`)
- Boruta-style exceedance test with stratified max shadow distributions.
- See Section 4 for full statistical specification.

---

### 2. Configuration Parameters

All parameters are read from a YAML config file. See `example_config_advanced.yaml` for the
complete template. Parameters omitted from the config are auto-filled by `fill_config_defaults()`
using data-driven heuristics. User-provided values are **never overwritten**.

#### `paths`
| Key | Type | Required | Description |
|---|---|---|---|
| `input_data` | str | Yes | Path to `.csv` or `.parquet` data file. |
| `output_dir` | str | Yes | Root directory for all pipeline outputs. |

#### `execution`
| Key | Type | Default | Description |
|---|---|---|---|
| `n_jobs` | int | `os.cpu_count()` | CPU threads for CatBoost, Optuna, and joblib Parallel. |
| `random_seed` | int | `42` | Master seed for all stochastic components. |

#### `features`
Feature groups are defined as lists under `continuous_groups`, `ordinal_groups`,
`nominal_groups`. Each group entry has:
| Key | Type | Required | Description |
|---|---|---|---|
| `pattern` | str | Yes | Column name pattern. |
| `match_mode` | str | No | `"exact"`, `"prefix"`, `"suffix"`, or `"substring"` (default). |
| `exclude` | list[str] | No | Substring exclusions applied after match. |
| `levels` | list | Ordinal only | Ordered list of valid ordinal values (low → high). |

#### `modeling`
| Key | Type | Default | Description |
|---|---|---|---|
| `outcome` | str or list[str] | Required | Outcome column name(s). List triggers `multi_regression`. |
| `task_type` | str | Auto-inferred | One of: `regression`, `binary_classification`, `multiclass_classification`, `multi_regression`. Inferred from `scoring` if omitted. |
| `loss_function` | str | Task-dependent | CatBoost loss. Regression: `RMSE`. Binary: `Logloss`. Multiclass: `MultiClass`. Multi-regression: `MultiRMSE`. |
| `cv_folds` | int | Data-driven | Outer CV folds. Default: 3, 5, or 10 (min 30 obs per val fold). |

#### `modeling.tuning`
| Key | Type | Default | Description |
|---|---|---|---|
| `n_iter` | int | `300` | Optuna trials. Bergstra et al. (2011) recommend ≥ 30/parameter for 10-param space. |
| `scoring` | str | Task-dependent | Tuning metric. Regression: `neg_rmse`. Binary: `roc_auc`. Multiclass: `balanced_accuracy`. Multi-regression: `neg_rmse`. |
| `inner_cv_folds` | int | Data-driven | Inner CV folds (min 20 obs per inner val fold). |
| `early_stopping_rounds` | int | `250` | CatBoost patience for inner CV and shadow model. |

#### `modeling.tuning.search_space`
Each parameter entry is either a list (categorical) or a dict with `low`, `high`, optional `log: true`.
| Parameter | Default Range | Notes |
|---|---|---|
| `iterations` | [100, 5000] | Boosting rounds. |
| `learning_rate` | [0.001, 0.3] log | Step size. |
| `depth` | [2, max(3, min(10, log2(n/5)))] | Tree depth. Floor of 3 ensures at least a [2,3] range even for small n. |
| `l2_leaf_reg` | [0.01, 100.0] log | L2 regularization. |
| `min_data_in_leaf` | [1, min(200, n/50)] | Minimum leaf observations. |
| `random_strength` | [0.001, 10.0] log | Randomization for split scoring. |
| `bagging_temperature` | [0.1, 1.0] | Bayesian bootstrap temperature. |
| `border_count` | [32, 255] | Quantization bins for continuous features. |
| `colsample_bylevel` | [0.05, 1.0] | Column subsampling per level. |
| `one_hot_max_size` | [2, 25] | Max cardinality for one-hot encoding. Fixed range, independent of feature count. |

#### `shap`
| Key | Type | Default | Description |
|---|---|---|---|
| `output_microdata_n` | int | `10` | Extra non-significant features to save microdata for (plotting). |

#### `shap.bootstrapping`
| Key | Type | Default | Description |
|---|---|---|---|
| `n_boot` | int | Data-driven | Bootstrap iterations. 2000 (n<100), 5000 (n<500), 10000 (n≥500). |
| `alpha` | float | `0.05` | Significance level for CIs, exceedance tests, and FDR. |
| `fdr_correct` | bool | `True` | Apply Benjamini-Hochberg FDR correction to exceedance p-values. |
| `stab_thresh` | float | `2.0` | Minimum stability (median / CI_width) for significance. |
| `output_boots_n` | int | `10` | Extra non-significant features to save bootstrap distributions for. |

#### `shap.splines`
| Key | Type | Default | Description |
|---|---|---|---|
| `n_knots` | int | `4` | Interior knots for 1D/2D splines. |
| `degree` | int | `3` | Polynomial degree (3 = cubic). Downgraded automatically when too few knots. |
| `discrete_threshold` | int | `15` | Features with ≤ this many unique values per resample are treated as discrete (group means instead of spline). |

---

### 3. Mathematical GII Formula

$$\text{GII} = \sqrt{M \times V}$$

- **M (Magnitude)**: `mean(|SHAP|)` across bootstrap resamples.
- **V (Variability)**: standard deviation of the systematic signal (spline or group means)
  fitted to `SHAP ~ feature_value`.
- **Stability Gate**: `median(boot) / CI_width > stab_thresh` (default 2.0). Prevents
  significance for effects whose bootstrap distribution is too wide relative to the point estimate.

**Significance Criteria (all must hold)**:
1. `q_exceed_GII < alpha` (FDR-corrected exceedance p-value).
2. `stab_pctl_GII > stab_thresh` (stability threshold).

Both M and V are independently tested; `sig_M` and `sig_V` are also reported.

**Exceedance P-Values**:
- Computed with the Davison & Hinkley (1997) / Phipson & Smyth (2010) +1 correction:
  `p = (sum(boot <= noise) + 1) / (n_boot + 1)`.
- Minimum achievable p = `1 / (n_boot + 1)`. Consistent with `compute_permutation_test()`.

---

### 4. SHAP Decomposition Details

#### Singleton vs. Interaction Extraction
- **Singletons**: `Φ[i,i]` (diagonal of the SHAP interaction matrix). Stored at full scale.
- **Interactions**: `Φ[i,j] + Φ[j,i]` (sum of both off-diagonal cells). CatBoost's
  `ShapInteractionValues` divides the total interaction contribution by 2 per cell (once per
  direction), so the full symmetric pair is required to recover the true Shapley interaction
  index. The summed convention ensures interactions and singletons are on the same
  prediction-contribution scale. Cross-type GII comparisons (singleton vs. interaction) are
  therefore valid.
- **Non-additivity**: GII values cannot be summed to reconstruct marginal SHAP importance.
  The decomposition identity `SHAP_total(i) = Φ(i,i) + Σ_{j≠i} [Φ(i,j) + Φ(j,i)]` equals
  `Φ(i,i) + 2·Σ_{j≠i} Φ(i,j)`, which is 2× the off-diagonal marginal. Users should not
  attempt to sum singleton + interaction GII to reconstruct marginal SHAP values.

#### Boruta Noise Calibration
- Shadow features are independently column-permuted copies of the real features.
- The shadow model trains jointly on real + shadow features, so shadow SHAP values are
  conditioned on the real signal. This is standard Boruta behavior (Kursa & Rudnicki, 2010):
  the null represents "how important is noise when real signal is present."
- The noise baseline is model-adaptive: when the real features are strongly predictive,
  shadow features receive lower SHAP attribution, reducing the noise threshold.
  This is the correct statistical null — not a conservatism concern.
- Noise distributions are stratified by measurement type (singleton_continuous,
  singleton_ordinal, singleton_nominal, interaction_continuous_continuous, etc.). Per-stratum
  maximum shadow GII is used as the noise threshold, preventing inflation from cross-type
  scale differences.

#### V-Component Method Selection
Per bootstrap resample, the SHAP-vs-feature trend is estimated by:
1. **1D spline** (`LSQUnivariateSpline`): when feature has > `discrete_threshold` unique values
   in the resample. Double-guarded: density check (falls back to group means if too few points)
   + energy gate (spline total variation must not exceed data total variation).
2. **1D group means**: when feature is nominal, or has ≤ `discrete_threshold` unique values.
3. **2D bivariate spline** (`LSQBivariateSpline`): for continuous × continuous interactions,
   when both axes have ≥ 2 interior knots.
4. **Stacked spline**: for continuous × low-resolution interactions. The low-resolution axis
   is used as a discrete grouping variable; 1D splines are fit along the well-resolved axis
   within each group. Inherits energy gate per group.
5. **2D group means**: for quasi-discrete × quasi-discrete interactions (both axes lack
   adequate knot resolution).

**Features near `discrete_threshold`** may switch methods across bootstrap iterations.
This is intentional: the switching reflects genuine uncertainty. The stability gate
(`stab_thresh`) filters effects with bimodal or wide bootstrap distributions caused by
method-switching.

#### Bootstrap CI Validity Conditions
- Bootstrap iterations where all resampled `y_true` values share a single class are dropped
  (the metric is undefined for single-class samples).
- `n_boot_effective` = number of valid iterations. A warning is emitted when the drop rate
  exceeds 5%: `"[WARNING] compute_bootstrap_ci: X.X% of bootstrap iterations dropped..."`.
- `n_boot_effective = n_boot` indicates maximum CI reliability. Severely imbalanced datasets
  (e.g., 95/5 split, n < 50) may have reduced `n_boot_effective`.
- CIs are not corrected for dropped iterations. The user should treat CIs with caution when
  drop rate is elevated. Unlike permutation tests (where retry is valid), bootstrapped CI
  failure rate is diagnostic of sample size vs. class imbalance and should not be suppressed.

#### Permutation Test
- Null distributions built by shuffling `y_true` while holding `y_pred` fixed (one-sided,
  higher = better).
- A while-loop guarantees exactly `n_perm` successful iterations (capped at `2 * n_perm`
  total attempts). Permutation failures are rare numerical artifacts, not diagnostic events,
  so retry is statistically valid (unlike bootstrap drops, which carry diagnostic meaning).
- P-value: `(sum(null >= observed) + 1) / (n_perm_effective + 1)` with +1 correction.

---

### 5. Directory Structure & Artifacts

#### Source Package Layout
```
boost-shap-gii/
├── pyproject.toml                # Package metadata, dependencies, CLI entry point
├── environment.yaml              # Conda environment specification
├── example_config_advanced.yaml  # Full config template with all parameters
├── example_config_minimal.yaml   # Minimal config template (defaults auto-filled)
├── README.md                     # User-facing documentation
├── INPUT_SPECIFICATION.md        # Technical reference (this file)
└── src/boost_shap_gii/
    ├── __init__.py               # Package version
    ├── cli.py                    # CLI entry point (boost-shap-gii command)
    ├── train.py                  # Data ingestion, feature selection, model training
    ├── predict.py                # OOF evaluation and SHAP analysis
    ├── infer.py                  # Independent dataset inference
    ├── shap_utils.py             # GII computation and noise calibration
    ├── utils.py                  # Shared utility functions
    ├── check_env.py              # Dependency verification
    └── scripts/
        ├── plot.R                # ggplot2-based visualization
        └── run_boost-shap-gii.sh # Shell script orchestrator (alternative interface)
```

#### Pipeline Output Layout
```
output_dir/
├── resolved_config.yaml          # Fully-expanded config with all defaults applied
├── train_matrix.parquet          # Clean feature matrix at training time
├── feature_names.json            # Ordered list of trained feature names
├── feature_types.json            # {name: type} map (continuous/ordinal/nominal)
├── feature_metadata.json         # Ordinal level definitions
├── feature_names_shadow.json     # Real + shadow feature names (for SHAP)
├── missingness_report.csv        # Per-feature missing rates
├── full_oof_predictions.csv      # OOF predictions (from train.py)
├── predictions_oof.csv           # OOF predictions with IDs (from predict.py)
├── metrics_oof.csv               # Per-fold + mean metrics
├── performance_final.csv         # Bootstrapped OOF performance with 95% CIs
├── permutation_test_results.csv  # Permutation test p-values
├── permutation_null_distributions.parquet
├── task_info.json                # {"task_type": "..."}
├── model_fold_<k>.cbm            # Clean CatBoost models (K folds)
├── shadow_model_fold_<k>.cbm     # Shadow CatBoost models (K folds)
└── shap_analysis/                # (or shap_<label>/ for multiclass/multi-regression)
    ├── shap_stats_global.csv          # Final GII results table
    ├── real_shap_interaction_matrix.parquet
    ├── shadow_shap_interaction_matrix.parquet
    ├── bootstrap_distributions_M.parquet
    ├── bootstrap_distributions_V.parquet
    ├── bootstrap_distributions_GII.parquet
    ├── stratified_noise_distributions_M.parquet
    ├── stratified_noise_distributions_V.parquet
    ├── stratified_noise_distributions_GII.parquet
    ├── microdata_M.parquet
    ├── microdata_V.parquet
    ├── microdata_GII.parquet
    └── plots/
        ├── 0_model_performance.png
        └── <rank>_<effect>_GII.png

# Inference subdirectory (infer.py):
output_dir/<subdir>/
├── predictions_ensemble.csv
├── performance_final.csv
├── performance_per_model.csv
├── permutation_test_results.csv
├── permutation_null_distributions.parquet
├── inference_metadata.json
└── shap_analysis/  (or shap_<label>/)
    └── ... (same structure as above)
```

---

### 6. `shap_stats_global.csv` Column Reference

| Column | Type | Description |
|---|---|---|
| `effect` | str | Effect name. Singletons: feature name. Interactions: `"feat_A x feat_B"`. |
| `type` | str | `"Singleton"` or `"Interaction"`. |
| `noise_stratum` | str | Measurement-type stratum used for noise calibration (e.g., `singleton_continuous`). |
| `GII` | float | Observed GII = mean bootstrap `sqrt(M * V)`. |
| `GII_ci_low` | float | Lower (alpha/2) bootstrap percentile for GII. |
| `GII_ci_high` | float | Upper (1 - alpha/2) bootstrap percentile for GII. |
| `p_exceed_GII` | float | Exceedance p-value (with +1 correction). |
| `q_exceed_GII` | float | BH FDR-corrected q-value. |
| `stab_pctl_GII` | float | Stability = median(boot_GII) / CI_width. |
| `sig_GII` | bool | `True` if `q < alpha` AND `stab > stab_thresh`. |
| `M`, `M_ci_*`, `p_exceed_M`, `q_exceed_M`, `stab_pctl_M`, `sig_M` | float/bool | Same columns for the M component. |
| `V`, `V_ci_*`, `p_exceed_V`, `q_exceed_V`, `stab_pctl_V`, `sig_V` | float/bool | Same columns for the V component. |
| `calc_failed` | bool | `True` if any point estimate (M, V, or GII) is NaN. |
| `v_failure_rate` | float | Fraction of bootstrap iterations where V spline fitting raised an exception (NaN result). High rates (> 0.05) indicate unreliable V estimates. |

---

### 7. Ordinal Feature Encoding

Ordinal levels are defined in the config as an ordered list. The pipeline maps observed values
to integer codes (0, 1, 2, …) preserving the user-specified order. Values absent from `levels`
trigger the two-tier validation (see Stage 3). `NaN` in ordinal features becomes `pd.NA`
(stored as Int64 -1 then masked), which CatBoost treats as a missing value.

---

### 8. Edge Cases and Known Limitations

- **High M, near-zero V**: A feature with consistently non-zero SHAP values across all
  feature values (constant effect, no dose-response) will have V ≈ 0 and GII ≈ 0. This
  is by design — GII measures structured variability. Such features will not reach
  significance on GII but may be significant on M alone (`sig_M = True`).

- **Severely imbalanced classification**: Bootstrap CI reliability degrades when the
  minority class fraction is very low relative to n. Monitor `v_failure_rate` and the
  bootstrap drop rate warning.

- **Features near `discrete_threshold`**: V estimates for these features may be computed
  by a mixture of spline and group-means methods across bootstrap iterations. Wide CIs
  and low stability scores are expected; the stability gate provides the primary protection.

- **Inference mode without outcomes**: SHAP analysis proceeds normally. Performance metrics
  and permutation tests are skipped.

- **Multi-output models**: Separate SHAP analyses are run per output class or target, stored
  in `shap_<label>/` subdirectories. Each analysis uses the corresponding SHAP slice
  (column `slice_idx` of the 4D interaction tensor).
