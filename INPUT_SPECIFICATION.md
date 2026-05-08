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
    indiv_reports.py     # Per-individual SHAP CI computation and emission
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
boost-shap-gii plot     --config CONFIG [--run-dir DIR]
```

Note: the `plot` subcommand previously accepted positional arguments `--outcome-range`,
`--negate-shap`, and `--y-axis-label`. These arguments have been removed. All plot
parameters are now read from the config file under the `plot.*` keys (see Section 2
and Section 10).

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
- **R Verification**: Checks `ggplot2`, `dplyr`, `nanoparquet`, `tidyr`, `foreach`, `doParallel`,
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
- Fixed iteration count: `tuned_iters * 2` (no `eval_set`, no early stopping). The doubled
  ceiling compensates for the expanded feature space (2p vs. p), consistent with Kursa &
  Rudnicki (2010). Shadow outputs are used only for SHAP noise calibration, not for
  predictive evaluation. Early stopping on the outer validation fold was removed to prevent
  leakage: allowing shadow-model selection to be influenced by validation outcomes
  would bias the noise calibration baseline toward the clean signal.

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
- Population-level `run_shap_pipeline()` is now **conditional**: invoked only when
  `shap.compute_global_on_inference: true` in the config (default `false`). Users who
  require population-level GII on the inference dataset (e.g., as a distribution-shift
  diagnostic) must opt in explicitly. When `false`, the `shap_analysis/` subdirectory is
  not written in the inference output directory.
- Per-individual SHAP reports (`indiv_reports/`) are emitted separately when
  `shap.indiv_ci_nboot > 0`. See Section 10.
- **Invariant**: inference individuals are never assigned to any training CV fold. Their
  per-individual CIs are computed from all B bootstrap iterations (ensemble-averaged
  replicates), not from OOB subsets. The concept of an OOB count does not apply to
  inference individuals; their effective replicate count equals B.

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
| `indiv_ci_nboot` | int | Required | Coupled bootstrap iterations for per-individual SHAP CIs. Value `0` disables the feature entirely (no `indiv_reports/` emitted, no bootstrap cache built). When `> 0`, minimum recommended value is `2500` (inference Efron-tier, training near-Efron); peer-review-facing runs should use `5000`. See Section 10 for algorithmic details. |
| `indiv_scaling_mode` | str | Required | Scaling mode for per-individual SHAP values. One of: `raw` (unscaled, all task types), `sd` (divide by training-outcome SD; regression and multi_regression only), `custom_value` (divide by `shap.indiv_scaling_value`; all task types). |
| `indiv_scaling_value` | number | Required when `indiv_scaling_mode: custom_value` | Positive divisor used when `indiv_scaling_mode: custom_value`. Examples: outcome theoretical maximum, a minimum-meaningful-difference threshold, or any domain-specific anchor. Ignored when `indiv_scaling_mode` is `raw` or `sd`. |
| `compute_global_on_inference` | bool | `false` | When `true`, `infer.py` additionally emits a population-level `shap_analysis/` GII on the inference dataset (a distribution-shift diagnostic for large inference sets). Default `false` because small inference sets produce degenerate GII estimates. Prior to this change, `infer.py` always emitted global SHAP on inference; users upgrading from older versions who depend on this behavior must set this key to `true`. |

See Section 10 for the full per-individual CI algorithm, output schema, and interpretation guidance for the `indiv_*` keys.

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

#### `plot`

All `plot.*` keys are consumed exclusively by the `plot` subcommand (`plot.R` via `boost-shap-gii plot`). They are not referenced by `train`, `predict`, or `infer`. All six keys below are required when the `plot` subcommand is invoked; missing keys cause a loud failure before any plot is generated.

| Key | Type | Required | Description |
|---|---|---|---|
| `outcome_max` | number | Yes | Theoretical maximum value of the outcome, used to scale GII magnitude axes on population-level GII plots. Formerly passed as the `--outcome-range` positional CLI argument. |
| `negate_shap` | bool | Yes | When `true`, SHAP y-axis values are sign-flipped on both GII and per-individual plots. The diverging color scale and x-axis rank ordering remain anchored to the raw signed SHAP values (not flipped). |
| `gii_y_label` | str | Yes | Y-axis label rendered verbatim on population-level GII plots (left panel = M, right panel = V). No programmatic composition or auto-appended annotation. |
| `gii_y_sublabel` | str | Yes | Y-axis subtitle rendered verbatim below `gii_y_label` on GII plots. Pass an empty string `""` to suppress. |
| `indiv_y_label` | str | Yes | Y-axis label rendered verbatim on per-individual SHAP plots (`indiv_reports/plots/`). |
| `indiv_y_sublabel` | str | Yes | Y-axis subtitle rendered verbatim below `indiv_y_label` on per-individual plots. Pass an empty string `""` to suppress. |

No plot titles are emitted anywhere in the pipeline. Label strings are the sole axis-description mechanism.

---

### 3. Mathematical GII Formula

$$\text{GII} = \sqrt{M \times V}$$

- **M (Magnitude)**: `mean(|SHAP|)` across bootstrap resamples.
- **V (Variability)**: sample standard deviation (ddof=1; Fisher, 1925) of the systematic
  signal (spline or group means) fitted to `SHAP ~ feature_value` within each bootstrap
  resample. Bessel's correction (ddof=1) is applied at all V-computation sites in
  `shap_utils.py` for unbiased variance estimation.
- **Stability Gate**: `median(boot) / CI_width > stab_thresh` (default 2.0). Prevents
  significance for effects whose bootstrap distribution is too wide relative to the point estimate.

**Decision-theoretic interpretation**: GII is structured as a geometric mean
of two utility components:

  - M (Magnitude): mean of absolute SHAP across bootstrap resamples; captures
    the average prediction-contribution utility of an effect.
  - V (Variability): standard deviation of the systematic SHAP signal
    (spline-fitted or group-mean-fitted) as a function of feature values;
    captures the dose-response informativeness utility, anchored conceptually
    to Hill (1910) dose-response framing and visualized via individual
    conditional expectation curves (Goldstein et al., 2015).

A feature is globally important when it BOTH drives model output (M) AND
exhibits feature-value-driven prediction variation (V). The geometric-mean
form requires both utilities to be meaningfully positive: a feature with
strong magnitude but no dose-response (V ~ 0) yields GII ~ 0, and vice
versa. This is the intended decision-theoretic semantics — neither
attribution magnitude alone nor trend informativeness alone constitutes
global importance under the GII framework.

**Significance Criteria (all must hold)**:
1. `q_exceed_GII < alpha` (BH-FDR-corrected exceedance p-value).
2. `stab_pctl_GII > stab_thresh` (stability threshold).

Both M and V are independently tested; `sig_M` and `sig_V` are also reported.

**FDR Control**: Three independent Benjamini-Hochberg FDR (BH-FDR; Benjamini &
Hochberg, 1995) calls are applied — one per component family (M exceedance p-values,
V exceedance p-values, GII exceedance p-values). Separating the three families into
independent BH calls prevents cross-component FDR inflation and preserves the ability
to interpret M and V significance separately from GII significance.

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
- **Degenerate fallback**: when `n_boot_effective = 0` (all iterations dropped, e.g., extreme
  class imbalance with n < 5), `compute_bootstrap_ci` returns `(base_score, NaN, NaN)` and
  emits a `RuntimeWarning`. The point estimate is the metric computed on the full sample; CI
  bounds are undefined. Callers may detect this state by checking for `NaN` CI bounds.

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
    ├── indiv_reports.py          # Per-individual SHAP CI computation and emission
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
├── train_outcome_stats.json      # Training-outcome summary statistics (n, mean, sd, min,
│                                 #   max, q25, q50, q75 per outcome column); written
│                                 #   unconditionally for regression/multi_regression;
│                                 #   written with empty stats{} for classification.
│                                 #   Consumed by indiv_reports for sd-scaling.
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
├── shap_analysis/                # (or shap_<label>/ for multiclass/multi-regression)
│   ├── shap_stats_global.csv          # Final GII results table
│   ├── real_shap_interaction_matrix.parquet
│   ├── shadow_shap_interaction_matrix.parquet
│   ├── bootstrap_distributions_M.parquet
│   ├── bootstrap_distributions_V.parquet
│   ├── bootstrap_distributions_GII.parquet
│   ├── stratified_noise_distributions_M.parquet
│   ├── stratified_noise_distributions_V.parquet
│   ├── stratified_noise_distributions_GII.parquet
│   ├── microdata_M.parquet
│   ├── microdata_V.parquet
│   ├── microdata_GII.parquet
│   └── plots/
│       ├── 0_model_performance.png
│       └── <rank>_<effect>_GII.png
├── bootstrap_refits/             # Per-individual CI cache (only when indiv_ci_nboot > 0)
│   ├── bootstrap_metadata.json  # Design summary: K, B, random_seed, HP per fold
│   ├── shared_indices.npz       # Bootstrap sample index matrix, shape (B, N_train)
│   ├── iter_00000/
│   │   ├── fold_0.cbm           # Bootstrap-refitted CatBoost models (one per fold)
│   │   └── fold_<K-1>.cbm
│   └── iter_<B-1:05d>/
│       └── ...
└── indiv_reports/               # Training-individual reports (from predict.py; only when
    ├── main_effects.parquet     #   indiv_ci_nboot > 0)
    ├── interactions.parquet     # See Section 10 for schema
    ├── predictions.parquet
    ├── indiv_reports_metadata.json
    └── plots/
        ├── <id>_main_effects.png
        └── <id>_interactions.png

# Inference subdirectory (infer.py):
output_dir/<subdir>/
├── predictions_ensemble.csv
├── performance_final.csv
├── performance_per_model.csv
├── permutation_test_results.csv
├── permutation_null_distributions.parquet
├── inference_metadata.json
├── shap_analysis/  (or shap_<label>/)      # Only when compute_global_on_inference: true
│   └── ... (same structure as train_dir/shap_analysis/)
└── indiv_reports/                           # Inference-individual reports (only when
    ├── main_effects.parquet                 #   indiv_ci_nboot > 0; bootstrap_refits/
    ├── interactions.parquet                 #   cache is read from train_dir, not infer_dir)
    ├── predictions.parquet
    ├── indiv_reports_metadata.json
    └── plots/
        ├── <id>_main_effects.png
        └── <id>_interactions.png
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

- **multi_regression SHAP units**: For `task_type: multi_regression` with StandardScaler
  applied to targets (`loss_function: MultiRMSE`), SHAP values are on the z-scaled target
  space (units: standard deviations of the original target column). The `plot` subcommand
  emits multi_regression SHAP plots WITHOUT applying `outcome_max`-based percentage
  rescaling, because percent-of-max rescaling on z-scaled SHAP would produce a unit
  mismatch. Plot y-axis units for multi_regression are therefore "SHAP value (z-scaled)"
  rather than "% of outcome_max."

- **CatBoost multi-thread bitwise determinism**: CatBoost (Prokhorenkova et al. 2018)
  does not provide a multi-thread bitwise-determinism flag (unlike LightGBM's
  `deterministic=true` and XGBoost's partial-determinism support). The pipeline
  runs multi-threaded for tractability, so independent runs on the same data with
  identical seeds may produce numerically slightly different SHAP values due to
  floating-point order-of-operations drift. The expected magnitude of drift is
  assumed to fall well below the shadow-bootstrap noise floor; users requiring
  bit-exact reproducibility may force `n_jobs: 1` at the cost of substantially
  longer runtimes.

---

### 9. Outcome Distribution Considerations

#### 9.1 Diagnostic Criteria

For **regression** and **multi_regression** tasks, the pipeline computes three distributional
diagnostics on the outcome vector(s) after data ingestion but before cross-validation. If any
threshold is exceeded, an advisory `[WARNING]` is emitted recommending consideration of Huber
loss. The warning is informational only --- the pipeline continues with whatever `loss_function`
the user specified, and the user's config is never modified at runtime.

For **multi_regression** tasks, each target column is diagnosed independently.

**Classification tasks are excluded** from this diagnostic because Logloss and CrossEntropy
loss functions compute gradients on predicted probabilities (log-odds scale), which inherently
bounds gradient magnitude regardless of class distribution. Moderate class imbalance does not
produce the unbounded gradient inflation that affects RMSE on pathological continuous outcomes.

| Criterion | Threshold | Rationale | Source |
|---|---|---|---|
| Zero-inflation rate | >= 15% | Semicontinuous outcome modeling literature recommends special treatment when zero proportions exceed 10--20%. The 15% threshold is a conservative midpoint. | Olsen & Schafer (2001, *Stat Methods Med Res*); Tooze et al. (2002, *Biometrics*) |
| \|Skewness\| | >= 2.0 | Classified as "extreme" skewness for purposes of assessing non-normality impact on estimation procedures. | Kim (2013, *Br J Math Stat Psychol*) |
| Excess kurtosis | >= 5.0 | Conservative threshold adapted from Kim's (2013) "extreme" kurtosis criterion of 7.0, adjusted downward because gradient boosting residual accumulation amplifies tail effects relative to single-pass estimation. | Kim (2013, *Br J Math Stat Psychol*) |

**Why formal normality tests are not used**: Formal tests (Shapiro-Wilk, Anderson-Darling,
Kolmogorov-Smirnov) are asymptotically consistent --- they reject for *any* departure from
normality as sample size grows, including trivially small departures with no practical impact
on model training (Razali & Wah, 2011). Effect-size-based diagnostics (zero fraction, skewness
magnitude, kurtosis magnitude) measure the *severity* of distributional pathology, not its
statistical detectability, making them appropriate as automated advisory triggers.

#### 9.2 Huber Loss and the MAD-Based Delta

When the diagnostic suggests RMSE may be suboptimal, the warning recommends Huber loss with
a data-driven delta parameter. Huber loss replaces squared residuals with absolute residuals
beyond a threshold delta, producing a piecewise loss function:

```
L(r) = 0.5 * r^2          if |r| <= delta
L(r) = delta * |r| - 0.5 * delta^2   if |r| > delta
```

This caps the influence of large residuals on gradient updates, preventing outliers and
distributional artifacts (zero-inflation spikes, heavy tails) from dominating the tree-building
process.

**Delta derivation** (Huber, 1981; Maronna, Martin & Yohai, 2006):

```
delta = k * sigma_hat = 1.345 * 1.4826 * MAD(y)
```

where:
- `MAD(y) = median(|y - median(y)|)` --- the median absolute deviation, the most robust
  scale estimator (50% breakdown point).
- `1.4826 = 1 / Phi^{-1}(3/4)` --- the consistency factor that makes `1.4826 * MAD` a
  consistent estimator of the standard deviation under the normal model.
- `k = 1.345` --- the tuning constant that yields **95% asymptotic relative efficiency (ARE)**
  at the normal distribution. That is, the Huber M-estimator with this constant has asymptotic
  variance equal to `1.0526 * sigma^2`, compared to `sigma^2` for OLS --- a 5.26% inflation
  that is negligible relative to the variance introduced by CatBoost's internal regularization
  (L2 leaf regularization, bagging, learning rate shrinkage).

**MAD = 0 fallback**: When more than 50% of outcome values share the median (common with
zero-inflation exceeding 50%), MAD collapses to zero. The pipeline falls back to IQR-based
scale estimation: `sigma_hat = IQR / 1.3489`, where `1.3489 = 2 * Phi^{-1}(3/4)` is the
IQR consistency factor at the normal (Maronna et al., 2006). The IQR has a 25% breakdown
point (vs. 50% for MAD) but remains a robust scale estimator. If IQR is also zero (constant
outcome), the pipeline falls back to the standard deviation as a last resort.

**CatBoost syntax**: `loss_function: "Huber:delta=VALUE"` (e.g., `"Huber:delta=4.9850"`).

#### 9.3 Asymmetric Risk Framework

The diagnostic operates under an asymmetric risk model:

- **False trigger** (diagnostic flags a well-behaved outcome): The user switches to Huber loss
  unnecessarily. Cost: exactly 5.26% ARE loss at the normal model --- bounded, small, and
  precisely quantified. CatBoost's internal regularization already introduces substantially
  larger variance inflation, so the practical impact is negligible.
- **Missed detection** (diagnostic does not flag a pathological outcome): The user trains under
  RMSE with outlier-dominated gradients. Cost: potentially large and unbounded RMSE gradient
  bias, manifesting as distorted tree structure, compressed predictions, and unreliable SHAP
  decomposition.

The conservative thresholds are calibrated to minimize the sum of weighted error costs, erring
on the side of flagging rather than missing pathological distributions.

#### 9.4 Scope and Limitations

- The diagnostic is computed on the **full outcome vector before CV splitting**, ensuring
  consistent loss function selection across all folds. Per-fold diagnosis would introduce
  noisy, fold-specific decisions that violate the assumption of identical model specification
  across folds.
- The MAD-based delta formula is optimal for location estimation under contaminated normal
  models (Huber, 1981). Its relationship to optimal CatBoost training is indirect --- mediated
  through gradient signals rather than direct M-estimation. The formula provides a principled
  starting point; users may wish to adjust delta based on domain knowledge.
- The thresholds are not formally validated for the specific question "when does RMSE degrade
  for gradient boosting." They are borrowed from adjacent literature (semicontinuous models,
  robust statistics, descriptive statistics guidelines) and represent a conservative heuristic.

---

### 10. Per-individual SHAP Reports (`indiv_reports`)

#### 10.1 Purpose

Per-individual SHAP reports surface the model's feature attributions for each individual in
the training or inference set, together with bootstrap-derived confidence intervals (CIs)
quantifying the sampling variability of those attributions. The feature is enabled when
`shap.indiv_ci_nboot > 0` in the config.

These reports surface the model's feature attributions for each individual, together with
bootstrap-derived confidence intervals quantifying the sampling variability of those
attributions. They are hypothesis-generating inspection tools, not prescriptive decision
outputs. Causal interpretation is not warranted from SHAP values alone (Ghassemi,
Oakden-Rayner & Beam, 2021; Covert & Lee, 2021).

The false-precision concern raised by Covert & Lee (2021) regarding SHAP estimation applies
here: CIs reflect training-sample sampling variability of the deployed estimator only, not
model-class uncertainty, hyperparameter-selection variability across independent tuning runs,
distribution shift, or label noise.

#### 10.2 Algorithm: Option E with Coupled Bootstrap

**Setup**:
- K = number of outer CV folds (from training; determined by `modeling.cv_folds` and
  data-driven defaults).
- B = `shap.indiv_ci_nboot` (user-specified coupled iterations).
- Total refits = K x B (embarrassingly parallel).

**Per-iteration bootstrap**:
Per iteration b in {1, ..., B}: one shared bootstrap sample s_b of size N (the full training
set size) is drawn with replacement. When `cluster_ids` are present in the data, bootstrap
resampling is cluster-aware (resample clusters, then expand to member rows), matching the
population-level bootstrap in `shap_utils.py`. This single shared sample drives all K fold
refits for iteration b: fold k is refitted using the shared sample s_b with the hyperparameters
embedded in `model_fold_k.cbm` (retrieved via `CatBoost.get_all_params()`). No new
hyperparameter tuning is performed; the HP from the deployed fold models are reused directly.

**Point estimates (deployed-product SHAP, not a bootstrap statistic)**:
The point estimate for each individual is the SHAP value and prediction that the deployed
pipeline product produces for that individual (Breiman, 2001):
- Training individual i (assigned to outer fold k_i during K-fold CV): OOF single-model SHAP
  and prediction from the original `model_fold_{k_i}.cbm`. This is leakage-free (the fold
  model was trained on rows not including individual i's validation fold).
- Inference individual: ensemble-mean SHAP and prediction averaged across all K original
  `model_fold_k.cbm` files. This matches `infer.py`'s existing ensemble-prediction logic.

Fold assignments for training individuals are reconstructed deterministically at predict-time
from the saved `config_resolved.yaml` (which contains `random_seed` and `cv_folds`) without
persisting a new artifact, using the same `get_cv_splitter()` call that `predict.py` uses
internally.

**CI aggregation (estimand-matched to the point estimate)**:
- Training individual i: CI is computed from the subset of iterations where i is **not** in
  s_b (i.e., i is out-of-bag at iteration b). For each such OOB iteration, only the fold-k_i
  refit's SHAP is used (single-model estimand match to the OOF point estimate). The expected
  number of OOB iterations per training individual is approximately 0.368 x B (Breiman, 2001).
- Inference individual: CI is computed from all B iterations. Per iteration b, the K coupled
  fold refits are ensemble-averaged to form one replicate, yielding B ensemble-estimand
  replicates (matched to the ensemble-mean point estimate). Inference individuals are
  never in any bootstrap sample; their effective replicate count is always B.

**CI-scale asymmetry between training and inference modes**: Training-mode CIs
target the OOF single-model SHAP estimand, with the bootstrap distribution
computed at the single-fold level. Inference-mode CIs target the ensemble-mean
SHAP estimand, with the bootstrap distribution computed at the K-fold ensemble
level via a bootstrap-of-CV design (Efron 1983; Davison & Hinkley 1997, ch. 5):
each bootstrap iteration draws one sample s_b, partitions s_b via a fresh K-fold
split, refits the K fold models with the original fold-specific hyperparameters
(no retuning), and averages SHAP across the K refits to produce one ensemble
replicate. Inference-mode intervals are computed as basic/reverse-percentile
intervals [2 * hat - q_hi, 2 * hat - q_lo] anchored on the original ensemble
point estimate (Davison & Hinkley 1997, sec. 5.2.1), guaranteeing point-estimate
containment by structural midpoint symmetry. Because inference-mode intervals
target an ensemble-level estimand and training-mode intervals target a
single-fold estimand, the two are NOT directly comparable across modes:
inference-mode intervals are typically narrower than training-mode intervals
for the same feature and individual, by approximately a factor of
sqrt((1 + (K - 1) * rho) / K) where rho is the inter-fold SHAP correlation
(rho ~ 0.3 yields a 1.65x ratio at K = 10). This asymmetry is principled, not
pathological, and reflects the matching inferential targets of the deployed
estimators (single fold model for training individuals, K-fold ensemble for
inference individuals).

**OOB floor**: Training individuals whose OOB count is below 50 (approximately 1/50 of B at
B = 2500) have NaN CI bounds. The point estimate is still emitted; `oob_count` records the
actual count for diagnostic purposes. At B = 2500, the floor corresponds to an individual
appearing in nearly all bootstrap samples (a rare edge case under simple random resampling
without clustering).

**CI bounds**: Percentile CIs at 2.5 / 97.5 (Efron & Tibshirani, 1993) computed via
`np.nanpercentile` on the accumulated CI distributions.

**Recommended B values**:
- B = 2500 (minimum recommended): inference individuals achieve Efron-tier CI reliability
  (Efron & Tibshirani, 1993, recommend B >= 1000 effective samples); training individuals
  achieve approximately 920 effective OOB samples (near-Efron tier).
- B = 5000 (peer-review-facing runs): both training (approximately 1840 OOB samples) and
  inference (5000 samples) exceed the Efron & Tibshirani (1993) threshold.
- Values below B = 2500 are permitted but reduce training OOB CIs to Carpenter & Bithell
  (2000) minimum tier (B >= 200 effective samples).

**Total refits at B = 2500, K = 10**: 25,000 model fits. With `thread_count = 1` per
CatBoost fit and 45 available cores, this completes in approximately 4-5 hours on
moderate-sized datasets.

**Memory guard**: At function entry, `generate_indiv_reports()` computes a projected memory
footprint (N_target x N_features x B x 8 bytes for main-effect CI accumulation, plus an
analogous term for significant interaction pairs). If the projected footprint exceeds 50% of
available system memory (`psutil.virtual_memory().available`), a `MemoryError` is raised
immediately with guidance to reduce `indiv_ci_nboot` or run on a higher-memory node. No
streaming fallback is implemented (err-on-kill philosophy).

#### 10.3 Config Keys

| Key | Type | Required | Notes |
|---|---|---|---|
| `shap.indiv_ci_nboot` | int | Yes | Number of coupled bootstrap iterations. `0` disables the feature. Minimum recommended: 2500. Peer-review: 5000. |
| `shap.indiv_scaling_mode` | str | Yes | `raw`, `sd`, or `custom_value`. `sd` restricted to regression/multi_regression. |
| `shap.indiv_scaling_value` | number | When `scaling_mode: custom_value` | Positive divisor. Must be > 0. |
| `shap.compute_global_on_inference` | bool | No (default `false`) | Opt-in for population-level GII on inference dataset. |
| `plot.outcome_max` | number | When `plot` invoked | Outcome theoretical maximum for GII axis scaling. |
| `plot.negate_shap` | bool | When `plot` invoked | Sign-flip SHAP y-axis on all plots. |
| `plot.gii_y_label` | str | When `plot` invoked | Y-axis label for GII plots. |
| `plot.gii_y_sublabel` | str | When `plot` invoked | Y-axis subtitle for GII plots. |
| `plot.indiv_y_label` | str | When `plot` invoked | Y-axis label for per-individual plots. |
| `plot.indiv_y_sublabel` | str | When `plot` invoked | Y-axis subtitle for per-individual plots. |

#### 10.4 Output Artifacts

**`train_outcome_stats.json`** (written by `train.py`; located at `output_dir/train_outcome_stats.json`):

Written unconditionally after the fold loop completes. Schema:

```json
{
  "task_type": "regression",
  "outcome_columns": ["score"],
  "n": 432,
  "stats": {
    "score": {
      "mean": 42.1,
      "sd": 18.4,
      "min": 0.0,
      "max": 100.0,
      "q25": 28.3,
      "q50": 41.7,
      "q75": 57.2
    }
  }
}
```

For classification tasks, `stats` is an empty dict `{}`. For multi_regression, `stats`
contains one entry per outcome column. SD is unbiased (ddof = 1). Consumed by
`indiv_reports.generate_indiv_reports()` to resolve the scaling divisor when
`indiv_scaling_mode: sd`.

**`bootstrap_refits/`** (written by `predict.py`; located at `output_dir/bootstrap_refits/`):

Present only when `indiv_ci_nboot > 0`. Contains:
- `bootstrap_metadata.json`: design summary (K, B, total_refits, random_seed, cluster_aware,
  per-fold HP summary, ISO8601 timestamp).
- `shared_indices.npz`: numpy archive containing the (B, N_train) int32 index matrix (one
  row per iteration, encoding which training rows were included in that iteration's bootstrap
  sample).
- `iter_00000/fold_0.cbm` ... `iter_{B-1:05d}/fold_{K-1}.cbm`: K x B bootstrap-refitted
  CatBoost models. Storage cost: approximately 5-20 MB per model, approximately 50-500 GB
  total at K = 10, B = 2500.

**`indiv_reports/`** (written by `predict.py` for training individuals; written by `infer.py`
for inference individuals; located at the respective `output_dir/indiv_reports/` or
`output_dir/<subdir>/indiv_reports/`):

Four files:

- **`main_effects.parquet`**: long format, one row per (individual, feature), all features
  regardless of `sig_GII` status (sig_GII column distinguishes significant from non-significant
  features).
- **`interactions.parquet`**: long format, one row per (individual, feature_a, feature_b);
  hard-filtered to model-level `sig_GII = True` interaction pairs. If no significant
  interactions exist, the file is emitted as a header-only (zero-row) parquet.
- **`predictions.parquet`**: one row per individual (or per individual x class for multiclass);
  contains point estimate and CI for predicted outcome.
- **`indiv_reports_metadata.json`**: design metadata (scaling mode, divisor, B, K, estimand
  sources, OOB floor, mode, timestamp).

Per-individual plots are emitted to `indiv_reports/plots/` by `plot.R` when the `plot`
subcommand is invoked after `predict` or `infer`.

**Parquet schema tables**:

`main_effects.parquet` (non-multiclass tasks):

| Column | Type | Description |
|---|---|---|
| `id` | str | Individual identifier. |
| `feature` | str | Feature name. |
| `feature_value_raw` | str | Stringified raw feature value. |
| `feature_type` | str | `continuous`, `ordinal`, or `nominal`. |
| `shap_value_raw` | float | Point-estimate SHAP (deployed-product; NOT a bootstrap statistic). |
| `shap_value_scaled` | float | `shap_value_raw` divided by scaling divisor. |
| `shap_value_ci_lo` | float | 2.5th percentile of CI distribution (NaN if `oob_count < 50`). |
| `shap_value_ci_hi` | float | 97.5th percentile of CI distribution (NaN if `oob_count < 50`). |
| `oob_count` | int | Number of OOB iterations (training) or B (inference). |
| `sig_GII` | bool | Model-level GII significance flag (from `shap_stats_global.csv`). |

For multiclass_classification tasks, one additional column `class` (str) is inserted after
`feature`, and the schema above is extended to one row per (individual, feature, class).
`oob_count` is per-individual (same value across all class-rows for a given individual).
`sig_GII` is model-level (broadcast across class-rows; GII is not class-specific in the
current pipeline).

`interactions.parquet` (non-multiclass tasks):

| Column | Type | Description |
|---|---|---|
| `id` | str | Individual identifier. |
| `feature_a` | str | First feature in the interaction pair. |
| `feature_b` | str | Second feature in the interaction pair. |
| `feature_a_value_raw` | str | Stringified raw value of `feature_a`. |
| `feature_b_value_raw` | str | Stringified raw value of `feature_b`. |
| `feature_a_type` | str | Feature type of `feature_a`. |
| `feature_b_type` | str | Feature type of `feature_b`. |
| `shap_value_raw` | float | Point-estimate interaction SHAP (deployed-product). |
| `shap_value_scaled` | float | `shap_value_raw` divided by scaling divisor. |
| `shap_value_ci_lo` | float | 2.5th percentile (NaN if `oob_count < 50`). |
| `shap_value_ci_hi` | float | 97.5th percentile (NaN if `oob_count < 50`). |
| `oob_count` | int | Number of OOB iterations (training) or B (inference). |

For multiclass_classification, an additional `class` column is inserted after `feature_b_type`.

`predictions.parquet` (regression and binary_classification):

| Column | Type | Description |
|---|---|---|
| `id` | str | Individual identifier. |
| `y_pred_raw` | float | Point-estimate prediction (OOF single-model for training; ensemble-mean for inference). |
| `y_pred_scaled` | float | `y_pred_raw` divided by scaling divisor (regression); equals `y_pred_raw` for binary_classification. |
| `y_pred_ci_lo` | float | 2.5th percentile CI bound. |
| `y_pred_ci_hi` | float | 97.5th percentile CI bound. |
| `y_pred_oob_count` | int | OOB iteration count (training) or B (inference). |
| `y_true` | float | Observed outcome (NaN if outcome absent in inference data). |

For multi_regression, per-outcome wide columns: `y_true_{col}`, `y_pred_raw_{col}`,
`y_pred_scaled_{col}`, `y_pred_ci_lo_{col}`, `y_pred_ci_hi_{col}`, plus one shared
`y_pred_oob_count` column.

For multiclass_classification, one row per (individual, class): columns `id`, `class`,
`prob` (point estimate), `prob_ci_lo`, `prob_ci_hi`, `prob_oob_count` (per-individual,
repeated across class-rows), `y_true` (observed class label; repeated across class-rows
or NaN if absent).

`indiv_reports_metadata.json`:

```json
{
  "design": "coupled",
  "scaling_mode": "sd",
  "scaling_divisor": 18.4,
  "B": 2500,
  "K": 10,
  "total_refits": 25000,
  "point_estimate_source": "OOF_single_model",
  "ci_aggregation": "OOB_single_model",
  "oob_count_floor": 50,
  "outcome_columns": ["score"],
  "mode": "training",
  "timestamp": "2026-04-24T12:00:00Z"
}
```

`point_estimate_source` and `ci_aggregation` are coupled to `mode`: training yields
`OOF_single_model` + `OOB_single_model`; inference yields `ensemble_mean` +
`ensemble_replicates`.

#### 10.5 Interpretation Guidance

**What the CIs represent**: The per-individual CIs reflect the sampling variability of the
deployed model's SHAP attribution under resampling of the training data (Breiman, 2001).
They answer the question: "If the model had been trained on a different bootstrap sample
of these N individuals, how different would this individual's SHAP attribution be?" This is
a useful measure of estimation stability but does not quantify:
- Model-class uncertainty (whether a different algorithm class would yield different attributions).
- Hyperparameter-selection variability across independent tuning runs.
- Distribution shift (whether the training distribution matches the deployment distribution).
- Label noise or measurement error in the outcome.

**Estimand match**: The bootstrap design ensures that the CI's sampling distribution is
estimand-matched to the point estimate. Training individuals' CIs target OOF single-model
SHAP (the same estimand as the point estimate). Inference individuals' CIs target
ensemble-mean SHAP (the same estimand as the point estimate). This eliminates the systematic
point-outside-CI risk that would arise from applying single-model variance to an ensemble
estimator (variance ratio approximately 1/K x [1 + (K-1)rho] favors ensemble; at K = 10
with rho approximately 0.3, single-model CIs would be approximately 1.65x too wide).

**Whiskers crossing y = 0**: When a feature's CI includes zero (lower bound < 0 < upper
bound), the feature's SHAP contribution for that individual is not distinguishable from null
at the 95% percentile level. This is NOT a formal hypothesis test and does not imply a
p-value; it is a descriptive summary of resampling variability (Cumming & Finch, 2005).

**Scaling modes**:
- `raw`: SHAP values in the model's native prediction units. Comparable across features
  within one individual; not directly comparable across individuals with different feature
  distributions.
- `sd`: SHAP values divided by the training-outcome SD. Produces a Cohen's-d-like
  standardized effect size (attributions expressed as fractions of one SD). Enables
  approximate cross-individual and cross-feature comparison. Restricted to regression and
  multi_regression tasks (classification SHAP values are on a log-odds scale for which
  SD standardization is not meaningful).
- `custom_value`: SHAP values divided by a user-supplied positive divisor (e.g., the
  outcome's theoretical maximum, or any other meaningful reference value). Produces
  attributions expressed as fractions of the chosen user-supplied anchor. Available for
  all task types.

**Below-floor individuals (oob_count < 50)**: The point estimate is still emitted and
defensible (it is the deployed-product SHAP, not a bootstrap statistic). Only the CI bounds
are NaN. These individuals appeared in nearly all bootstrap samples, leaving fewer than 50
OOB iterations to estimate tail percentiles reliably. At B = 2500, K = 10, and simple
random resampling, below-floor training individuals are rare (expected OOB rate is
approximately 36.8% of B; below-floor cases arise only when an individual is systematically
included in nearly all samples, which does not occur under uniform resampling). Below-floor
cases are more likely under cluster-aware resampling with small clusters and large B.

**Plot rendering**: For below-floor individuals, `plot.R` emits point markers without
whiskers and adds the caption "CI unavailable (oob_count < 50); point estimate shown only."
below the x-axis. File naming and output directory are identical to compliant plots; no
separate subdirectory is used for below-floor cases.

**References**:
- Breiman L. (2001) Random Forests. *Machine Learning* 45: 5-32. (OOB aggregation and
  expected OOB rate.)
- Carpenter J., Bithell J. (2000) Bootstrap confidence intervals: when, which, what? A
  practical guide for medical statisticians. *Statistics in Medicine* 19: 1141-1164.
  (Minimum B guidance for practical CI reliability.)
- Covert I., Lee S.-I. (2021) Improving KernelSHAP: Practical Shapley value estimation
  using linear regression. *AISTATS*. (False-precision concern with SHAP estimates.)
- Cumming G., Finch S. (2005) Inference by eye: confidence intervals and how to read
  pictures of data. *American Psychologist* 60: 170-180. (Dot-plus-whisker visualization
  and CI interpretation.)
- Davison A.C., Hinkley D.V. (1997) *Bootstrap Methods and their Application*.
  Cambridge University Press, ch. 5. (Bootstrap-of-CV design; basic/reverse-percentile
  CI construction, sec. 5.2.1.)
- Efron B. (1983) Estimating the error rate of a prediction rule: improvement on
  cross-validation. *Journal of the American Statistical Association* 78: 316-331.
  (Bootstrap-of-CV inferential framework.)
- Efron B., Tibshirani R.J. (1993) *An Introduction to the Bootstrap*. Chapman & Hall.
  (Percentile CI method; recommended B >= 1000.)
- Ghassemi M., Oakden-Rayner L., Beam A.L. (2021) The false hope of current approaches to
  explainable artificial intelligence in health care. *Lancet Digital Health* 3: e745-e750.
  (Hypothesis-generating vs. causal interpretation of SHAP attributions.)
- Goldstein A., Kapelner A., Bleich J., Pitkin E. (2015) Peeking inside the black box:
  visualizing statistical learning with plots of individual conditional expectation.
  *Journal of Computational and Graphical Statistics* 24: 44-65. (ICE plots; anchor for
  V-component dose-response informativeness utility.)
- Hill A.V. (1910) The possible effects of the aggregation of the molecules of haemoglobin
  on its dissociation curves. *Journal of Physiology* 40: i-vii. (Dose-response framing;
  conceptual anchor for V-component trend informativeness.)
