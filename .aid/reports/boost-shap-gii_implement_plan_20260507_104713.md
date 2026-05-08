<implement_plan>
 <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-05-07T10:47:18-04:00" />

 <input_reports>
 <report path="boost-shap-gii_brainstorm_20260507_103025.md" mode="brainstorm" key_items="19" />
 </input_reports>

 <changes>

 <!-- ============================================================ -->
 <!-- operational:.gitignore allowlist for indiv_reports -->
 <!-- ============================================================ -->
 <change id="change-1" priority="P0" source_item="">
 <file path=".gitignore" action="modify" />
 <description>Add the indiv_reports.py allowlist entry so the new module is tracked by git. The current.gitignore allowlist covers all pre-existing source files but silently excludes indiv_reports.py because the pattern at line 2 (`*`) ignores everything by default. Audit confirms indiv_reports.py is the only missing source-file entry.</description>
 <spec>
Insert the line `!src/boost_shap_gii/indiv_reports.py` between the existing entry for `check_env.py` (line 24) and the entry for `scripts/` (line 25). The resulting block (lines 14-28) will read:

```
# Source package (recursive whitelist)
!src/
!src/boost_shap_gii/
!src/boost_shap_gii/__init__.py
!src/boost_shap_gii/cli.py
!src/boost_shap_gii/train.py
!src/boost_shap_gii/predict.py
!src/boost_shap_gii/infer.py
!src/boost_shap_gii/shap_utils.py
!src/boost_shap_gii/utils.py
!src/boost_shap_gii/check_env.py
!src/boost_shap_gii/indiv_reports.py
!src/boost_shap_gii/scripts/
!src/boost_shap_gii/scripts/plot.R
!src/boost_shap_gii/scripts/run_boost-shap-gii.sh
```

This placement satisfies "alphabetical with the existing source allowlist" because `indiv_reports` (i-n-d) sorts after `check_env` (c) and before `infer` (i-n-f).
 </spec>
 <dependencies>none</dependencies>
 <risk>low - single-line allowlist insertion; no behavior change in any pipeline module.</risk>
 <rollback>git checkout --.gitignore</rollback>
 </change>

 <!-- =================================================== -->
 <!-- — Topic 20: config-load discrete_threshold guard -->
 <!-- =================================================== -->
 <change id="change-2" priority="P2" source_item="">
 <file path="src/boost_shap_gii/utils.py" action="modify" />
 <file path="example_config_advanced.yaml" action="modify" />
 <description>Hard-error config validation: at config load (after fill_config_defaults), raise ValueError when user-supplied `shap.splines.discrete_threshold &lt; n_knots + degree + 2`. Anchors the constraint to Wood (2017) GAM ch. 4 (basis must be supportable by available unique x values). Adds a one-line comment in the example config citing the constraint.</description>
 <spec>
**utils.py changes:**

Identify the existing `fill_config_defaults` function in `utils.py` (which already sets `shap.splines.discrete_threshold` default to 15, `n_knots` to 4, `degree` to 3). After this function (or in the existing `validate_*` config function chain that runs at load-time), add the following validation block. If a `validate_indiv_reports_config` or similar load-time validator already exists, add this check there; otherwise create a new function `validate_spline_config(config: dict) -> None` and ensure it is called from the same load-site that invokes `fill_config_defaults`.

```python
def validate_spline_config(config: dict) -> None:
 """Validate shap.splines configuration.

 Spline stability requires at least n_knots + degree + 2 unique x values to
 support the basis (Wood 2017, Generalized Additive Models, ch. 4). Below
 this threshold, the basis is rank-deficient and fits become unstable.
 """
 splines = config.get("shap", {}).get("splines", {})
 n_knots = splines.get("n_knots")
 degree = splines.get("degree")
 discrete_threshold = splines.get("discrete_threshold")
 if n_knots is None or degree is None or discrete_threshold is None:
 return # fill_config_defaults will set these; nothing to validate yet
 lower_bound = n_knots + degree + 2
 if discrete_threshold &lt; lower_bound:
 raise ValueError(f"shap.splines.discrete_threshold ({discrete_threshold}) must be "
 f"&gt;= n_knots + degree + 2 ({n_knots} + {degree} + 2 = {lower_bound}). "
 f"Spline basis is rank-deficient below this lower bound (Wood 2017, "
 f"Generalized Additive Models, ch. 4)."
)
```

The validator must be invoked AFTER `fill_config_defaults` so that user-omitted values are populated before validation. Identify the existing call site (likely in cli.py, train.py, predict.py, or infer.py) where `fill_config_defaults` is invoked, and add a `validate_spline_config(config)` call immediately after each invocation.

**example_config_advanced.yaml changes:**

At the line under `shap.splines:` containing `discrete_threshold: 15`, append the comment:

```yaml
 discrete_threshold: 15 # Must satisfy: discrete_threshold &gt;= n_knots + degree + 2 (Wood 2017 GAM ch. 4)
```

Preserve any pre-existing inline comment by combining with the new text if necessary; if the line already has a different inline comment, replace it with the new comment.
 </spec>
 <dependencies>none</dependencies>
 <risk>low - validation happens at config load before any pipeline work; raises ValueError on mis-config (project doctrine: err on kill). Default values (15, 4, 3) satisfy the constraint (15 &gt;= 9), so existing valid configs are unaffected.</risk>
 <rollback>git checkout -- src/boost_shap_gii/utils.py example_config_advanced.yaml</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- (++N#1+N#2): unified tensor-shape fix -->
 <!-- ============================================================ -->
 <change id="change-3" priority="P0" source_item="">
 <file path="src/boost_shap_gii/indiv_reports.py" action="modify" />
 <description>Fix three coupled bugs in indiv_reports.py: (1) memory guard miscalculation using ×8 (float64) for buffers that are actually ×4 (float32); (2) interaction projection mis-shapes the actual N×B×F×F float32 tensor; (3) multiclass interaction tensors point_shap_int (3D) and int_iter_folds (4D) lack the class dimension that _compute_interaction_values returns for multiclass tasks. Implements the brainstorm-locked unified.</description>
 <spec>
**1. Add `_output_dim` helper near the top of indiv_reports.py (after existing module-level helpers, before `_shap_single`):**

```python
def _output_dim(model) -> int:
 """Return the number of output classes/coordinates for a fitted CatBoost estimator.

 For multiclass classification, returns model.n_classes_ (n &gt;= 2).
 For binary classification, regression, and multi_regression, returns 1
 (these task types use 1D output along the class dimension; the singleton
 class axis is collapsed in tensor handling for these tasks).
 """
 classes = getattr(model, "classes_", None)
 if classes is not None and len(classes) &gt; 2:
 return int(len(classes))
 return 1
```

**2. Rewrite `_shap_interaction_single` (current lines ~126-129) to mirror the multiclass-aware bias-trim handling in `shap_utils.py:_compute_interaction_values` (lines 366-385):**

Current implementation:
```python
def _shap_interaction_single(model, pool: Pool) -> np.ndarray:
 sv = model.get_feature_importance(pool, type="ShapInteractionValues")
 return np.asarray(sv, dtype=np.float64)
```

Replace with:

```python
def _shap_interaction_single(model, pool: Pool) -> np.ndarray:
 """Return SHAP interaction values with bias-trim and a normalized class axis.

 Always returns a 4D tensor of shape (N, C, F, F) where:
 - N is the number of rows in `pool`,
 - C is the number of output classes (1 for non-multiclass tasks),
 - F is the number of features (bias column trimmed off both feature axes).
 For non-multiclass tasks, the class axis is inserted as a singleton
 dimension so downstream consumers can use uniform 4D indexing.
 """
 sv = model.get_feature_importance(pool, type="ShapInteractionValues")
 sv = np.asarray(sv, dtype=np.float32)
 if sv.ndim == 4:
 # Multiclass: shape (N, C, F+1, F+1) -&gt; trim bias on both feature axes.
 return sv[:::-1:-1]
 if sv.ndim == 3:
 # Non-multiclass: shape (N, F+1, F+1) -&gt; trim bias, then add singleton C axis.
 trimmed = sv[::-1:-1]
 return trimmed[:, np.newaxis::]
 raise ValueError(f"_shap_interaction_single: unexpected SHAP interaction tensor "
 f"with ndim={sv.ndim} (expected 3 or 4)."
)
```

Use `np.float32` (not `float64`) to match the existing buffer dtypes; multiclass shapes returned by CatBoost are now bias-trimmed to (N, C, F, F) and non-multiclass shapes are augmented with a singleton class axis to (N, 1, F, F).

**3. Audit `_shap_single` (current lines ~116-123): confirm that the existing 3D multiclass slice `sv[:::-1]` produces the correct (N, C, F) shape for multiclass and (N, F) for non-multiclass; if it does NOT already insert a singleton class axis for non-multiclass, modify it to do so. Required final shape: `(N, C, F)` uniformly across task types, where C=1 for non-multiclass.**

If `_shap_single` currently returns:
 - `(N, C, F)` for multiclass: keep as-is.
 - `(N, F)` for non-multiclass: insert singleton class axis as `result[:, np.newaxis:]`.

The function MUST return a 3D `(N, C, F)` tensor where C=1 for non-multiclass tasks, so all downstream tensor allocations and aggregations can use the same indexing pattern.

**4. Reallocate point-estimate and CI tensors with the class dimension:**

In the function that computes per-individual point estimates and CIs (`generate_indiv_reports` and its inner loops, currently around lines 727, 737, 765-772, 777, 817):

- `point_shap_main`: change shape from `(N_target, F)` to `(N_target, C, F)` where C = `_output_dim(reference_model)`.
- `point_shap_int`: change shape from `(N_target, F, F)` to `(N_target, C, F, F)`.
- `shap_ci_buf` (main effects): change shape from `(N_target, B, F)` to `(N_target, B, C, F)` (verify if currently already class-aware; per pre-compaction analysis, lines 765-772 already have this correctly — confirm at edit time and only modify if missing).
- `int_ci_buf` (interactions): change shape from `(N_target, B, F, F)` to `(N_target, B, C, F, F)`.
- `iter_folds` accumulators (per-iteration K-fold collection): main shape `(K, N_target, F)` -&gt; `(K, N_target, C, F)`; interaction shape `(K, N_target, F, F)` -&gt; `(K, N_target, C, F, F)`.

Use np.float32 for all CI accumulator buffers (matches the existing convention).

**5. Update the memory guard at lines ~632-634 to reflect the actual buffer footprint:**

Current:
```python
projected_main = N_target * N_features * effective_B_max * 8
projected_inter = N_target * N_pairs_sig * effective_B_max * 8
projected_total = projected_main + projected_inter
```

Replace with:
```python
n_outputs = _output_dim(reference_model)
projected_main = N_target * N_features * effective_B_max * n_outputs * 4
projected_inter = N_target * N_pairs_sig * effective_B_max * n_outputs * 4 * N_features
projected_total = projected_main + projected_inter
```

Note: the interaction projection is N_target × B × C × F × F (not N_target × B × F), so the per-pair-sig estimate becomes `N_target × N_pairs_sig × B × C × 4 × F` — but since N_pairs_sig already encodes the F×F pair selection, the simpler form is `N_target × N_pairs_sig × B × C × 4`. Verify the actual buffer that is allocated in step 4 above and compute the projection to match. The single hard rule: every multiplicative factor present in the actual allocated buffer must appear in the projection, and the dtype scalar must be `4` (float32), not `8` (float64).

The `reference_model` used by `_output_dim` should be one of the trained fold CatBoost models loaded earlier in the function; if multiple models are loaded, any one suffices because all K fold models share `n_classes_`.

**6. Aggregation and emission paths:**

Audit every site in `generate_indiv_reports` that reads from or writes to the tensors modified in steps 3-4. Each must add an explicit class-axis loop or broadcast. Specifically:

- OOB-bag aggregation for training individuals: `np.nanpercentile(buf[i, oob_mask::], [2.5, 97.5], axis=0)` becomes `np.nanpercentile(buf[i, oob_mask, c:], [2.5, 97.5], axis=0)` inside a `for c in range(C):` loop.
- Ensemble-replicate aggregation for inference individuals: `mean across folds` operations remain on the K axis; the C axis is preserved through aggregation.
- Parquet writers for `main_effects.parquet`, `interactions.parquet`, and `predictions.parquet`: the multiclass schema (per INPUT_SPECIFICATION.md Section 10.4) inserts a `class` column for multiclass tasks. For non-multiclass tasks, the C=1 axis collapses to scalar columns and the writer emits the existing single-row-per-(individual,feature) schema. Implement this with a runtime branch on `n_outputs &gt; 1` rather than a hard-coded multiclass detector.
- The emission fallback path at the previously-noted lines ~1053-1071 (multiclass interaction emission with "same interaction value per class") must be REMOVED. The new tensor structure carries genuine per-class interaction values; emit them per (individual, feature_a, feature_b, class) with the actual buffered values instead of repeating a single value across class rows.

**7. plot.R microdata audit:**

`plot.R` reads `main_effects.parquet` and `interactions.parquet` from `indiv_reports/`. After the parquet schema gains a `class` column for multiclass tasks, plot.R will emit one plot per (individual, class) for multiclass per-individual SHAP plots. Confirm that the existing plot.R per-individual plotting branch already handles multiclass correctly via the `shap_<label>/` directory pattern; if it does not, this is an additional plot.R modification. (Note: the existing pipeline emits population-level multiclass GII via shap_<label>/ subdirectories — the per-individual plots should follow the same naming convention, e.g., `<id>_main_effects_<label>.png`.) Implement this emission convention if not present.
 </spec>
 <dependencies> (config validation should land before the indiv_reports tensor reshape so that any test exercising indiv_reports starts from a valid spline config).</dependencies>
 <risk>medium - touches multiple coupled tensor allocations and downstream emission paths. Multiclass class-axis change is a schema change to indiv_reports parquet outputs (per INPUT_SPECIFICATION.md Section 10.4 multiclass schema). Mitigated by: (a) the schema is already specified in INPUT_SPECIFICATION.md, so the change closes a pre-existing implementation gap rather than introducing a new schema; (b) non-multiclass tasks see no schema change because the C=1 axis collapses to scalar columns.</risk>
 <rollback>git checkout -- src/boost_shap_gii/indiv_reports.py src/boost_shap_gii/scripts/plot.R</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- — Topic 2: bootstrap-of-CV with basic/reverse-percentile -->
 <!-- ============================================================ -->
 <change id="change-4" priority="P0" source_item="">
 <file path="src/boost_shap_gii/infer.py" action="modify" />
 <description>Replace the existing inference-mode bootstrap loop (which shared one bootstrap sample across K coupled refits per iteration b) with a bootstrap-of-CV design that draws s_b once and partitions s_b via a fresh K-fold split, refits K models with original fold-specific HPs, averages SHAP across K to produce one ensemble-replicate per b, and computes a basic/reverse-percentile interval [2·hat − q_hi, 2·hat − q_lo] using the original ensemble point estimate as the centering anchor. The point estimate hat is the deployed ensemble-mean SHAP from the original K fold models; the bootstrap distribution targets the same ensemble estimand.</description>
 <spec>
**Locate the inference-mode bootstrap routine in infer.py (currently at lines ~254-321 per the prior session's audit).**

The current loop draws one bootstrap sample s_b per iteration b, refits K models on s_b, and computes K coupled fold-specific replicates. Replace this with a bootstrap-of-CV design:

```python
def _bootstrap_of_cv_inference(X_train: pd.DataFrame,
 y_train: np.ndarray,
 inference_pool: Pool,
 fold_models: list, # K original deployed models (cv-fitted on full training data)
 fold_hyperparameters: list, # K dicts of CatBoost params extracted via get_all_params
 B: int, # shap.indiv_ci_nboot
 K: int, # number of fold splits
 random_seed: int,
 cluster_ids: np.ndarray | None,
 splitter_factory: Callable, # callable returning a fresh splitter (e.g., KFold(K, shuffle=True, random_state=seed))
 point_shap_main: np.ndarray, # shape (N_target, C, F), the ensemble-mean point estimate
 point_shap_int: np.ndarray, # shape (N_target, C, F, F), the ensemble-mean interaction point estimate
) -&gt; tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
 """Bootstrap-of-CV with basic/reverse-percentile intervals.

 For each iteration b in {1..., B}:
 1. Draw bootstrap sample s_b of size N from (X_train, y_train).
 2. Generate a fresh K-fold split on s_b (independent of the original training split).
 3. For each fold k in {1..., K}: refit a CatBoost model on the fold-train portion
 of s_b using fold_hyperparameters[k] (no HP retuning).
 4. Compute SHAP main and interaction values from each refitted fold model on
 inference_pool, average across the K fold models, and store as one ensemble-replicate.
 After B iterations, compute the basic/reverse-percentile interval at each
 (individual, feature [, feature_b]) cell:
 ci_lo = 2 * hat - q_hi
 ci_hi = 2 * hat - q_lo
 where (q_lo, q_hi) are the (alpha/2, 1-alpha/2) percentiles of the bootstrap
 distribution (default alpha=0.05 -&gt; (2.5, 97.5)) and hat is the original
 point estimate.
 """
 rng = np.random.default_rng(random_seed)
 N = len(X_train)
 C = point_shap_main.shape[1]
 F = point_shap_main.shape[2]
 N_target = point_shap_main.shape[0]
 main_replicates = np.full((B, N_target, C, F), np.nan, dtype=np.float32)
 int_replicates = np.full((B, N_target, C, F, F), np.nan, dtype=np.float32)

 # Reuse cluster-aware bootstrap from shap_utils when cluster_ids present
 for b in range(B):
 s_b_idx = _draw_bootstrap_sample(N=N, rng=rng, cluster_ids=cluster_ids
)
 X_b = X_train.iloc[s_b_idx].reset_index(drop=True)
 y_b = y_train[s_b_idx]

 # Fresh K-fold split on s_b
 splitter = splitter_factory(seed=random_seed + b + 1)
 fold_iter = splitter.split(X_b, y_b)

 per_fold_main = np.full((K, N_target, C, F), np.nan, dtype=np.float32)
 per_fold_int = np.full((K, N_target, C, F, F), np.nan, dtype=np.float32)

 for k, (train_idx, _) in enumerate(fold_iter):
 X_fold_train = X_b.iloc[train_idx].reset_index(drop=True)
 y_fold_train = y_b[train_idx]
 fold_train_pool = _make_pool(X_fold_train, y_fold_train...)

 # Refit with fold-specific HP, no HP retuning
 params = dict(fold_hyperparameters[k])
 refit_model = CatBoost(params)
 refit_model.fit(fold_train_pool, verbose=False)

 per_fold_main[k] = _shap_single(refit_model, inference_pool)
 per_fold_int[k] = _shap_interaction_single(refit_model, inference_pool)

 # Average across the K fold refits to get one ensemble-replicate
 main_replicates[b] = np.nanmean(per_fold_main, axis=0)
 int_replicates[b] = np.nanmean(per_fold_int, axis=0)

 # Basic/reverse-percentile intervals
 alpha = 0.05 # consistent with shap.bootstrapping.alpha; should be passed in
 q_lo, q_hi = (alpha 2) * 100.0, (1 - alpha 2) * 100.0
 main_q_lo = np.nanpercentile(main_replicates, q_lo, axis=0)
 main_q_hi = np.nanpercentile(main_replicates, q_hi, axis=0)
 int_q_lo = np.nanpercentile(int_replicates, q_lo, axis=0)
 int_q_hi = np.nanpercentile(int_replicates, q_hi, axis=0)

 main_ci_lo = 2.0 * point_shap_main - main_q_hi
 main_ci_hi = 2.0 * point_shap_main - main_q_lo
 int_ci_lo = 2.0 * point_shap_int - int_q_hi
 int_ci_hi = 2.0 * point_shap_int - int_q_lo
 return main_ci_lo, main_ci_hi, int_ci_lo, int_ci_hi
```

**Key requirements (all locked):**

1. The point estimate is the ORIGINAL ensemble-mean SHAP from the K deployed `model_fold_k.cbm` files, not a bootstrap statistic. This matches the existing infer.py behavior at lines ~254-321 prior to bootstrap.
2. The bootstrap distribution is at the ensemble level: each iteration b yields ONE replicate (the K-fold mean on s_b), not K replicates.
3. HP transfer: `fold_hyperparameters[k] = original_fold_models[k].get_all_params` — no retuning per iteration b.
4. The fresh K-fold split per iteration b is independent of the original training split; use `KFold(n_splits=K, shuffle=True, random_state=random_seed + b + 1)` (or `StratifiedKFold` for classification) to guarantee independence across iterations.
5. The bootstrap sample drawer (`_draw_bootstrap_sample`) must be cluster-aware when `cluster_ids` is non-None (mirror the existing cluster-aware bootstrap in shap_utils.py used for population-level GII).
6. Memory: per-iteration `per_fold_main` and `per_fold_int` are released at end of iteration b; only the (B, N_target, C, F) main and (B, N_target, C, F, F) interaction replicate stacks accumulate. The C=1 axis collapses for non-multiclass tasks; the int_replicates stack at (B=2500, N_target=100, F=20, F=20) × 4 bytes = ~32 MB, well within budget.
7. For training-mode CIs (predict.py): NO change. Training-mode keeps the existing per-iteration shared-bootstrap design and OOB CI aggregation against the OOF single-model point estimate. The asymmetry is documented in (INPUT_SPECIFICATION.md note).
8. The bootstrap_refits/ cache layout described in INPUT_SPECIFICATION.md Section 10.4 (`shared_indices.npz`, `iter_<b>/fold_<k>.cbm`) is updated: the `shared_indices.npz` records the s_b draws for inference replicates; the K refit models per iteration are written to `iter_<b>/fold_<k>.cbm` exactly as before. The fresh K-fold partition of s_b is not persisted explicitly because it can be reconstructed from `random_seed + b + 1` and the bootstrap-sample length.

**Audit downstream consumers:**

- `bootstrap_metadata.json`: add a field `inference_ci_design: "bootstrap_of_cv_basic_percentile"` and `training_ci_design: "shared_sample_oob_single_model"` to make the design explicit.
- `indiv_reports_metadata.json` for inference-mode runs: update `ci_aggregation` to `"ensemble_replicates_basic_percentile"`. The `point_estimate_source` remains `"ensemble_mean"`.

**Determinism contract:**

- Original training fold-specific HPs are extracted ONCE before the bootstrap loop and frozen for the duration of the run.
- Bootstrap-sample draws use `np.random.default_rng(random_seed)` advanced sequentially, so iteration b's sample s_b is reproducible from the seed alone.
- Each iteration's K-fold split uses `random_seed + b + 1` as its splitter seed to decouple from s_b's draw.
 </spec>
 <dependencies> (the inference-mode CI tensors must already carry the (N, C, F) and (N, C, F, F) shapes from the unified tensor-shape fix; without, the buffers would be allocated with the wrong shape and the bootstrap-of-CV refactor would not compose).</dependencies>
 <risk>medium - largest behavioral change in this remediation cycle. The inference-mode CI changes from the OOB-equivalent to a basic/reverse-percentile interval computed against the original ensemble point estimate. Mitigated by: (a) the new design is the brainstorm-locked statistical correction; (b) basic/reverse-percentile intervals structurally guarantee point-estimate containment; (c) the change is confined to infer.py's bootstrap loop and downstream consumers can be audited for the new schema.</risk>
 <rollback>git checkout -- src/boost_shap_gii/infer.py</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- (docs portion): CI-scale asymmetry in INPUT_SPEC -->
 <!-- ============================================================ -->
 <change id="change-5" priority="P0" source_item="">
 <file path="INPUT_SPECIFICATION.md" action="modify" />
 <description>Add a brief explanatory paragraph to Section 10 (Per-individual SHAP Reports) documenting the training-mode vs. inference-mode CI-scale asymmetry. Training-mode CIs reflect single-fold variability (wider); inference-mode CIs reflect K-fold ensemble variability on a single bootstrap-resampled cohort (narrower). The two are not directly comparable across modes.</description>
 <spec>
In INPUT_SPECIFICATION.md Section 10 (`### 10. Per-individual SHAP Reports (`indiv_reports`)`), within subsection 10.2 ("Algorithm: Option E with Coupled Bootstrap"), append the following paragraph immediately after the existing "**CI aggregation (estimand-matched to the point estimate)**" block (or in a new subsection if structurally cleaner, titled "**CI-scale asymmetry between training and inference modes**"):

```
**CI-scale asymmetry between training and inference modes**: Training-mode CIs
target the OOF single-model SHAP estimand, with the bootstrap distribution
computed at the single-fold level. Inference-mode CIs target the ensemble-mean
SHAP estimand, with the bootstrap distribution computed at the K-fold ensemble
level via a bootstrap-of-CV design (Efron 1983; Davison &amp; Hinkley 1997, ch. 5):
each bootstrap iteration draws one sample s_b, partitions s_b via a fresh K-fold
split, refits the K fold models with the original fold-specific hyperparameters
(no retuning), and averages SHAP across the K refits to produce one ensemble
replicate. Inference-mode intervals are computed as basic/reverse-percentile
intervals [2 * hat - q_hi, 2 * hat - q_lo] anchored on the original ensemble
point estimate (Davison &amp; Hinkley 1997, sec. 5.2.1), guaranteeing point-estimate
containment by structural midpoint symmetry. Because inference-mode intervals
target an ensemble-level estimand and training-mode intervals target a
single-fold estimand, the two are NOT directly comparable across modes:
inference-mode intervals are typically narrower than training-mode intervals
for the same feature and individual, by approximately a factor of
sqrt((1 + (K - 1) * rho) K) where rho is the inter-fold SHAP correlation
(rho ~ 0.3 yields a 1.65x ratio at K = 10). This asymmetry is principled, not
pathological, and reflects the matching inferential targets of the deployed
estimators (single fold model for training individuals, K-fold ensemble for
inference individuals).
```

Reference list at the end of Section 10 must include:
- Efron B. (1983) Estimating the error rate of a prediction rule: improvement
 on cross-validation. *Journal of the American Statistical Association* 78: 316-331.
- Davison A.C., Hinkley D.V. (1997) *Bootstrap Methods and their Application*.
 Cambridge University Press, ch. 5.

Verify the existing reference list block at the end of Section 10 already includes Efron &amp; Tibshirani (1993) and Carpenter &amp; Bithell (2000); add the two new references in alphabetical order alongside them.
 </spec>
 <dependencies> (the documentation describes the bootstrap-of-CV design implemented in and must reflect the actual implementation).</dependencies>
 <risk>low - documentation-only change.</risk>
 <rollback>git checkout -- INPUT_SPECIFICATION.md</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- — Topic 3: pooled BH-FDR across effects -->
 <!-- ============================================================ -->
 <change id="change-6" priority="P0" source_item="">
 <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
 <description>Apply BH-FDR to the pooled set of all F effects (rather than stratifying by feature type) for each of the three families (sig_M, sig_V, sig_GII). Preserve three independent FDR calls — one per family — with NO cross-family correction.</description>
 <spec>
At shap_utils.py lines 905 and 908 (and the surrounding `_nan_safe_fdr` function and its call sites at lines 910-912), make the following changes:

1. Locate `_nan_safe_fdr` (around line 900) which currently invokes `multipletests(p_clean, alpha=alpha_val, method='fdr_bh')` on a stratum-restricted slice of p-values.

2. Change the call sites at lines 910-912 (the three independent calls for `q_exceed_M`, `q_exceed_V`, `q_exceed_GII`) so each receives the FULL pooled vector of p-values across all effects (singletons + interactions, all noise strata) rather than a stratum-filtered subset.

3. Audit the existing implementation: if `_nan_safe_fdr` itself receives a pre-filtered p-value vector (i.e., the stratification happens at the call site), then the only change required is at the call site (pass the unfiltered, pooled p-value vector). If the stratification happens INSIDE `_nan_safe_fdr`, refactor the function to accept the pooled vector directly and remove the internal stratification logic.

4. Preserve three independent BH calls: one for `q_exceed_M`, one for `q_exceed_V`, one for `q_exceed_GII`. NO cross-family joint correction.

5. Verify the `effect` ordering is preserved across the three calls so downstream merging into `shap_stats_global.csv` aligns rows correctly.

6. NaN handling: preserve the existing convention where NaN p-values are skipped (passed through as NaN q-values) without contributing to the BH denominator. The `_nan_safe_fdr` helper name implies this is already implemented; verify and preserve.

7. Update the relevant docstring at the function entry to state: "Applied to the pooled set of all effects (singletons + interactions across all noise strata). No cross-family correction is applied between sig_M, sig_V, and sig_GII; each family receives an independent BH call."
 </spec>
 <dependencies>none</dependencies>
 <risk>low - changes the FDR pooling semantics from per-stratum to pooled-across-strata, which is the brainstorm-locked statistical correction. Existing test cases that rely on stratum-specific q-values will need to be updated; addressed in /test phase, not in this implementation plan.</risk>
 <rollback>git checkout -- src/boost_shap_gii/shap_utils.py</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- — Topic 5: ddof=1 + len&lt;2 guard at six np.std sites -->
 <!-- ============================================================ -->
 <change id="change-7" priority="P1" source_item="">
 <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
 <description>Switch all six V-component np.std calls to unbiased sample SD (ddof=1) and add a len(signal)&lt;2 guard returning np.nan to avoid divide-by-zero on degenerate inputs.</description>
 <spec>
At shap_utils.py lines 223, 229, 251, 264, 303, 360 (within `calculate_v_group_means_1d`, `calculate_v_group_means_2d`, `calculate_v_spline_1d`, `calculate_v_spline_2d`, `calculate_v_stacked_spline`, and any sixth site), apply the following pattern at each site:

Before:
```python
v_value = np.std(signal)
```

After:
```python
if len(signal) &lt; 2:
 return np.nan
v_value = np.std(signal, ddof=1)
```

Six replacements total. The exact `signal` variable name at each site must be preserved (it may be `group_signal`, `spline_signal`, etc., depending on the site).

Audit each site to confirm the surrounding control flow handles NaN return values: the existing V-spline failure path already returns NaN, and downstream consumers (`np.nanmean`, `np.nanpercentile`, `_nan_safe_fdr`) tolerate NaN. No additional downstream changes required.
 </spec>
 <dependencies>none</dependencies>
 <risk>low - small, well-localized change. ddof=1 is the unbiased sample SD (Fisher 1925); deviation from R's default sd is the documented contributing source of the plot.R/Python spline mismatch.</risk>
 <rollback>git checkout -- src/boost_shap_gii/shap_utils.py</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- — Topic 9: outcome-distribution citation anchors -->
 <!-- ============================================================ -->
 <change id="change-8" priority="P1" source_item="">
 <file path="src/boost_shap_gii/train.py" action="modify" />
 <description>Add Groeneveld &amp; Meeden (1984) citation anchor for skewness threshold and Joanes &amp; Gill (1998) citation anchor for kurtosis threshold to the docstring of `_diagnose_outcome_distribution`. Single sentence per citation, no extended discussion. No threshold value changes.</description>
 <spec>
Locate the docstring of `_diagnose_outcome_distribution` in train.py (function spans lines ~200-332 per the prior session's audit). Within the docstring near lines 252-255 (where the existing skewness and kurtosis threshold language appears), append two terse sentences:

Skewness anchor (single sentence):
```
The |skewness| &gt;= 2.0 threshold is anchored to Groeneveld &amp; Meeden (1984)
moment-based skewness measures for assessing distributional non-normality.
```

Kurtosis anchor (single sentence):
```
The excess kurtosis &gt;= 5.0 threshold derives from Joanes &amp; Gill (1998)
sample-kurtosis bias-correction analysis, adjusted downward to a conservative
heuristic for tail-effect amplification under gradient boosting residual accumulation.
```

Place these sentences immediately following the existing Kim (2013) citation in the docstring (which already provides the primary anchor). The ordering is: existing Olsen &amp; Schafer (2001), Tooze et al. (2002), Kim (2013) anchors first, then the two new sentences.

NO threshold value changes (15% zero-inflation, |skewness| &gt;= 2.0, excess kurtosis &gt;= 5.0 all preserved). NO logic change. NO new code paths.

References to add to any inline reference list block (if one exists in the docstring or module-level docstring):
- Groeneveld R.A., Meeden G. (1984) Measuring skewness and kurtosis. *The Statistician* 33: 391-399.
- Joanes D.N., Gill C.A. (1998) Comparing measures of sample skewness and kurtosis. *The Statistician* 47: 183-189.
 </spec>
 <dependencies>none</dependencies>
 <risk>low - docstring-only change; no behavioral change in the diagnostic.</risk>
 <rollback>git checkout -- src/boost_shap_gii/train.py</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- — Topic 10: plot.R spline parity with Python LSQ -->
 <!-- ============================================================ -->
 <change id="change-9" priority="P1" source_item="">
 <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
 <description>Replace plot.R's lm(bs) spline visualization with an R implementation using splines::splineDesign that mirrors scipy's LSQUnivariateSpline. Port Python's adaptive-knot logic (np.percentile-based knot construction, boundary exclusion, degree downgrade to 1 if fewer than 4 unique knots) from shap_utils.py:146-164 to plot.R as a helper function. Use R's quantile with type=7 (default, matches numpy's default percentile method) for knot construction.</description>
 <spec>
**1. Add helper function `get_adaptive_knots_and_degree` near the top of plot.R (after existing utility functions, before `calc_v_spline_pred`):**

```R
# Python equivalent: shap_utils.py:_get_adaptive_knots_and_degree (lines 146-164)
# Returns a list with `interior_knots` (vector) and `degree` (integer).
get_adaptive_knots_and_degree &lt;- function(x_values, n_knots_target, degree_target) {
 # Sort and uniquify
 x_unique &lt;- sort(unique(x_values[!is.na(x_values)]))
 n_unique &lt;- length(x_unique)

 if (n_unique &lt; 2) {
 return(list(interior_knots = numeric(0), degree = 1L))
 }

 # Percentile-based interior knots; type=7 matches numpy's default
 probs &lt;- seq(0, 1, length.out = n_knots_target + 2)
 probs &lt;- probs[2:(length(probs) - 1)] # exclude 0 and 1 (boundary exclusion)
 candidate_knots &lt;- quantile(x_unique, probs = probs, type = 7, names = FALSE)

 # Drop duplicate knots (occurs when the data is highly discrete)
 interior_knots &lt;- unique(candidate_knots)

 # Boundary exclusion: drop knots at min/max of x_unique
 x_min &lt;- min(x_unique)
 x_max &lt;- max(x_unique)
 interior_knots &lt;- interior_knots[interior_knots &gt; x_min &amp; interior_knots &lt; x_max]

 # Degree downgrade: if fewer than 4 unique interior knots, downgrade to linear (degree=1)
 effective_degree &lt;- ifelse(length(interior_knots) &lt; 4, 1L, as.integer(degree_target))
 return(list(interior_knots = interior_knots, degree = effective_degree))
}
```

Knot/degree defaults must match the Python defaults: `n_knots_target = 4`, `degree_target = 3` (cubic). These should be read from the config at call time, with `config$shap$splines$n_knots` and `config$shap$splines$degree` as the source.

**2. Replace `calc_v_spline_pred` body (currently plot.R lines ~118-148) with a splineDesign-based LSQ fit:**

```R
calc_v_spline_pred &lt;- function(xs, ys, cfg) {
 n_knots_target &lt;- cfg$shap$splines$n_knots %||% 4L
 degree_target &lt;- cfg$shap$splines$degree %||% 3L

 knot_info &lt;- get_adaptive_knots_and_degree(xs, n_knots_target, degree_target)
 interior_knots &lt;- knot_info$interior_knots
 degree &lt;- knot_info$degree

 if (length(xs) &lt; degree + length(interior_knots) + 2L) {
 # Insufficient unique x for stable LSQ; return NA-filled predictions
 return(rep(NA_real_, length(xs)))
 }

 # Construct knot sequence with degree+1 multiplicity at boundaries
 x_min &lt;- min(xs, na.rm = TRUE)
 x_max &lt;- max(xs, na.rm = TRUE)
 knot_seq &lt;- c(rep(x_min, degree + 1L), interior_knots, rep(x_max, degree + 1L))

 # Build B-spline design matrix
 basis &lt;- tryCatch(splines::splineDesign(knots = knot_seq, x = xs, ord = degree + 1L, outer.ok = TRUE),
 error = function(e) NULL
)
 if (is.null(basis)) return(rep(NA_real_, length(xs)))

 # LSQ fit (mirrors scipy.interpolate.LSQUnivariateSpline)
 # Solve: coef = (B^T B)^-1 B^T y
 fit &lt;- tryCatch(qr.solve(basis, ys), error = function(e) NULL)
 if (is.null(fit)) return(rep(NA_real_, length(xs)))

 preds &lt;- as.vector(basis %*% fit)
 return(preds)
}
```

**3. Update all call sites to `calc_v_spline_pred` to pass `cfg` (the loaded config object):**

Search plot.R for `calc_v_spline_pred(` and update each invocation to pass the config. If the function is called from a context that does not yet have `cfg` in scope, propagate `cfg` via function arguments (do NOT introduce a global).

**4. Verify R-side defaults match Python defaults:**

- `cfg$shap$splines$n_knots`: default 4 (matches Python).
- `cfg$shap$splines$degree`: default 3 (matches Python).
- `quantile(..., type = 7)`: matches numpy.percentile default (linear interpolation).
- Boundary exclusion logic matches `shap_utils.py:_get_adaptive_knots_and_degree` (drops knots at min/max).

**5. Document the parity in plot.R header comment:**

Add a comment block near the top of plot.R noting: "calc_v_spline_pred uses splines::splineDesign with adaptive-knot LSQ fitting that mirrors scipy.interpolate.LSQUnivariateSpline as used in shap_utils.py:146-164. Visualization fits are now consistent with the V-statistic shown beneath them (from the 2026-04-24 CR)."

NO change to the actual V-component computation in shap_utils.py for the visualization parity. The V-statistic emission path is unchanged; only the visualization path is altered.
 </spec>
 <dependencies> (the Python ddof=1 fix in shap_utils.py also contributes to numerical parity; both must land for full).</dependencies>
 <risk>medium - replacing the spline visualization basis can change the visual smoothness of fitted curves on real data. Mitigated by: (a) the Python adaptive-knot logic is the canonical reference implementation; (b) numerical parity verified during /test on representative datasets.</risk>
 <rollback>git checkout -- src/boost_shap_gii/scripts/plot.R</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- — Topic 12: phase-2 shadow leakage fix -->
 <!-- ============================================================ -->
 <change id="change-10" priority="P1" source_item="">
 <file path="src/boost_shap_gii/train.py" action="modify" />
 <description>Remove eval_set=pool_val_full and early_stopping_rounds from the phase-2 shadow CatBoost fit. Set shadow iterations to a fixed ceiling of 2 * tuned_iters with no early-stopping criterion in phase-2 shadow training. Eliminates outer-validation-pool leakage from the shadow model's stopping criterion.</description>
 <spec>
At train.py around lines 895-950 (the phase-2 shadow training block), make the following changes:

1. **Verify line 938** currently reads `shadow_params["iterations"] = tuned_iters * 2` (already a ceiling). If yes, no change at this line.

2. **At lines 945-950** (the `model_shadow.fit(...)` call), remove the following arguments:
 - `eval_set=pool_val_full`
 - `early_stopping_rounds=...` (any reference to the early stopping rounds parameter)

3. **Resulting fit call:**

```python
model_shadow.fit(pool_train_full,
 verbose=False,
 # NO eval_set=...
 # NO early_stopping_rounds=...
)
```

(Other arguments to the existing `.fit` call are preserved verbatim.)

4. **Audit `shadow_params`** (the dict passed to the CatBoost constructor for the shadow model). Confirm:
 - `iterations = 2 * tuned_iters` (fixed ceiling).
 - No `early_stopping_rounds` parameter is set in the constructor either.

5. **Update the inline comment block** in the phase-2 shadow training section to read: "Phase-2 shadow training uses a fixed iteration ceiling of 2 * tuned_iters with NO early stopping. The 2x ceiling preserves the original rationale that shadow models need additional iterations to converge with 2p shadow features added to the feature space (Boruta-style stratified shadow features; Kursa &amp; Rudnicki 2010). Removing eval_set=pool_val_full closes the outer-validation-pool leakage path identified in the 2026-04-24 CR."

6. **No changes** to phase-1 (clean model) training. Phase-1 retains its inner-CV early stopping mechanism, which is correct because inner-CV early stopping uses inner-fold validation pools that are disjoint from the outer-fold validation set.
 </spec>
 <dependencies>none</dependencies>
 <risk>low - eliminates a leakage path while preserving the principled 2x iteration ceiling. Shadow models may train slightly longer than necessary on some datasets but never under-train. No predictive metric impact (shadow outputs are used only for SHAP noise calibration, not predictive evaluation).</risk>
 <rollback>git checkout -- src/boost_shap_gii/train.py</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- — Topic 13: plot.R OUTCOME_MAX scaling for multi_regression -->
 <!-- ============================================================ -->
 <change id="change-11" priority="P1" source_item="">
 <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
 <file path="INPUT_SPECIFICATION.md" action="modify" />
 <description>For multi_regression task type only, skip the OUTCOME_MAX-based percentage scaling at plot.R:385 (which is a unit mismatch given that multi_regression SHAP values are computed on z-scaled targets). Non-multi_regression paths retain their existing scaling. Add a brief INPUT_SPECIFICATION.md note that multi_regression SHAP values are in z-scaled units of the target.</description>
 <spec>
**plot.R changes:**

1. At plot.R:385, the current line reads:
```R
df_m$shap_value &lt;- (df_m$shap_value OUTCOME_MAX) * 100
```

Replace with a task-type-aware branch:

```R
# Skip OUTCOME_MAX scaling for multi_regression: SHAP values are on z-scaled
# targets (StandardScaler applied in train.py), so percent-of-max rescaling
# would create a unit mismatch. Closes from the 2026-04-24 CR.
task_type &lt;- cfg$modeling$task_type
if (!identical(task_type, "multi_regression")) {
 df_m$shap_value &lt;- (df_m$shap_value OUTCOME_MAX) * 100
}
```

The `cfg` config object must be in scope at this site; if it is not, propagate it via the surrounding function's arguments (do NOT introduce a global).

2. **Verify predict.py inverse-transform behavior at lines 271-278**: confirm that the inverse-transform applies to PREDICTIONS only (not SHAP values), which is the correct existing behavior. NO change to predict.py for this finding.

3. **No change to non-multi_regression code paths**: regression, binary_classification, and multiclass_classification all retain the existing scaling logic.

**INPUT_SPECIFICATION.md change:**

In Section 4 (SHAP Decomposition Details) or Section 8 (Edge Cases and Known Limitations), append a brief note:

```
**multi_regression SHAP units**: For `task_type: multi_regression` with
StandardScaler applied to targets (loss_function: MultiRMSE), SHAP values are
on the z-scaled target space (units: standard deviations of the original
target column). The plot subcommand emits multi_regression SHAP plots WITHOUT
applying outcome_max-based percentage rescaling, because percent-of-max
rescaling on z-scaled SHAP would produce a unit mismatch. Plot y-axis units
for multi_regression are therefore "SHAP value (z-scaled)" rather than
"% of outcome_max."
```

Place this note in the most semantically appropriate location (Section 4 or Section 8). If the existing structure has a "Multi-output models" subsection in Section 8, the note belongs there.
 </spec>
 <dependencies>none</dependencies>
 <risk>low - removes a unit-mismatch bug that only affects multi_regression visualization; non-multi_regression behavior preserved.</risk>
 <rollback>git checkout -- src/boost_shap_gii/scripts/plot.R INPUT_SPECIFICATION.md</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- — Topic 14: CatBoost determinism caveat note -->
 <!-- ============================================================ -->
 <change id="change-12" priority="P1" source_item="">
 <file path="INPUT_SPECIFICATION.md" action="modify" />
 <description>Add a brief CatBoost multi-thread bitwise-determinism caveat to INPUT_SPECIFICATION.md. One or two terse sentences. No config flag, no code change.</description>
 <spec>
Append the following caveat to INPUT_SPECIFICATION.md Section 8 (Edge Cases and Known Limitations) as a new bullet:

```
- **CatBoost multi-thread bitwise determinism**: CatBoost (Prokhorenkova et al. 2018)
 does not provide a multi-thread bitwise-determinism flag (unlike LightGBM's
 `deterministic=true` and XGBoost's partial-determinism support). The pipeline
 runs multi-threaded for tractability, so independent runs on the same data with
 identical seeds may produce numerically slightly different SHAP values due to
 floating-point order-of-operations drift. The expected magnitude of drift is
 assumed to fall well below the shadow-bootstrap noise floor; users requiring
 bit-exact reproducibility may force `n_jobs: 1` at the cost of substantially
 longer runtimes.
```

NO config flag is added. NO code change. NO quantitative drift evaluation in the public repo (deferred to the -unified quarantined simulation study per the brainstorm lock).
 </spec>
 <dependencies>none</dependencies>
 <risk>low - documentation-only change.</risk>
 <rollback>git checkout -- INPUT_SPECIFICATION.md</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- — Topic 15: degenerate compute_bootstrap_ci fallback -->
 <!-- ============================================================ -->
 <change id="change-13" priority="P2" source_item="">
 <file path="src/boost_shap_gii/utils.py" action="modify" />
 <description>Replace the degenerate fallback at utils.py:665-666 (which silently returns base_score as both CI bounds) with `(base_score, np.nan, np.nan)` plus an explicit warning identifying the affected effect. Aligns with the pipeline's existing NaN-on-failure convention.</description>
 <spec>
At utils.py around lines 600-670 (within the `compute_bootstrap_ci` function):

1. Locate lines 665-666:
```python
if not scores:
 return base_score, base_score, base_score
```

2. Replace with:
```python
if not scores:
 effect_label = effect_name if "effect_name" in locals else "&lt;unknown effect&gt;"
 warnings.warn(f"compute_bootstrap_ci: all bootstrap iterations dropped for "
 f"'{effect_label}' (n_boot_effective = 0). Returning point estimate "
 f"with NaN CI bounds. This indicates severe data sparsity or class "
 f"imbalance for this effect; CI is undefined.",
 RuntimeWarning)
 return base_score, float("nan"), float("nan")
```

The variable `effect_name` may already be in scope as a parameter of `compute_bootstrap_ci`; if not, the function signature should be checked and an `effect_name` parameter added if appropriate. If the existing function signature does not carry an effect identifier, fall back to a generic `"&lt;unknown effect&gt;"` placeholder in the warning message (the warning is still emitted).

3. Ensure `import warnings` is present at the top of utils.py (it likely already is).

4. Audit downstream consumers in shap_utils.py and predict.py to confirm NaN propagation is handled correctly:
 - `np.nanmean`, `np.nanpercentile`, and `_nan_safe_fdr` already accommodate NaN input.
 - JSON serialization of NaN: existing pipeline emits NaN-as-null in metric JSON files; verify the same convention applies to bootstrap CI metric output.
 - parquet writers: pyarrow writes NaN as null in float columns; no schema change required.
 </spec>
 <dependencies>none</dependencies>
 <risk>low - small, well-localized change. Aligns with pipeline-wide NaN-on-failure convention.</risk>
 <rollback>git checkout -- src/boost_shap_gii/utils.py</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- — Topic 17: plot.R V-driver weighted top-5 selection -->
 <!-- ============================================================ -->
 <change id="change-14" priority="P2" source_item="">
 <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
 <description>Replace the frequency-only top-5 nominal-level selection at plot.R:481-485 with a V-contribution-ranked selection: count_k * (mean_SHAP_k - grand_mean_SHAP)^2. Annotate N_k below each level. NOMINAL features only — ordinal features retain their inherent ordering.</description>
 <spec>
At plot.R around lines 481-485 (the existing nominal top-5 selection block):

1. Current code:
```R
if (m_type == "nominal" && nlevels(fac) &gt; 5) {
 top &lt;- names(sort(table(fac), decreasing = TRUE))[1:5]
 df_m &lt;- df_m %&gt;% filter(create_ordered_factor(main_feature_raw, feature_value) %in% top)
 fac &lt;- create_ordered_factor(df_m$main_feature_raw, df_m$feature_value)
}
```

Replace with:

```R
# V-contribution-ranked top-5 selection (NOMINAL only).
# V_nominal is the ANOVA between-group SS contribution per level:
# contribution_k = count_k * (mean_SHAP_k - grand_mean_SHAP)^2
# Ranking by this contribution exactly matches the per-level contribution to
# the V-statistic shown in the plot (from the 2026-04-24 CR).
if (m_type == "nominal" &amp;&amp; nlevels(fac) &gt; 5) {
 grand_mean_shap &lt;- mean(df_m$shap_value, na.rm = TRUE)
 level_contrib &lt;- df_m %&gt;%
 group_by(feature_value) %&gt;%
 summarise(n_k = n,
 mean_shap_k = mean(shap_value, na.rm = TRUE),
 contribution = n * (mean(shap_value, na.rm = TRUE) - grand_mean_shap) ^ 2.groups = "drop"
) %&gt;%
 arrange(desc(contribution))

 top &lt;- as.character(level_contrib$feature_value[1:5])
 df_m &lt;- df_m %&gt;% filter(as.character(feature_value) %in% top)
 fac &lt;- create_ordered_factor(df_m$main_feature_raw, df_m$feature_value)

 # Annotate N_k below each surviving level for transparency.
 level_labels &lt;- df_m %&gt;%
 group_by(feature_value) %&gt;%
 summarise(n_k = n.groups = "drop")
 level_label_lookup &lt;- setNames(paste0(level_labels$feature_value, "\n(N=", level_labels$n_k, ")"),
 as.character(level_labels$feature_value)
)
 # Apply level_label_lookup as the x-axis label transform on the corresponding
 # ggplot scale_x_discrete call further downstream.
}
```

2. **Restrict to NOMINAL only**: the `if (m_type == "nominal" &&...)` guard is preserved. Ordinal features are NOT subject to permutation truncation; they retain all levels in their inherent order.

3. **N_k annotation rendering**: the level label lookup must be applied at the ggplot `scale_x_discrete` call site downstream. Locate the existing `scale_x_discrete(...)` invocation in plot.R that controls the x-axis labels for nominal-feature plots, and add `labels = level_label_lookup` (or equivalent labeller function) so each surviving level renders as `"<level>\n(N=<count>)"`.

4. **Verify against `shap_value` column name**: the actual column name in `df_m` may be `shap_value`, `shap_value_raw`, or another; use whatever name the existing pipeline uses for the SHAP value at this site. The semantic is: per-level mean and grand mean of the SHAP attribution.

5. **NaN handling**: `na.rm = TRUE` in the group-mean calculations skips NaN; if a level has all-NaN SHAP values, its contribution is `NaN * (...)` which propagates to NaN; arrange(desc(contribution)) places NaN at the bottom (R default), correctly excluding all-NaN levels from the top-5.
 </spec>
 <dependencies>none</dependencies>
 <risk>low - localized to nominal-feature visualization. The V-contribution criterion exactly matches the V-statistic shown beneath the plot, eliminating the visualization/statistic mismatch.</risk>
 <rollback>git checkout -- src/boost_shap_gii/scripts/plot.R</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- — Topic 18: _label_nominal helper with distinct sentinels -->
 <!-- ============================================================ -->
 <change id="change-15" priority="P2" source_item="">
 <file path="src/boost_shap_gii/predict.py" action="modify" />
 <file path="src/boost_shap_gii/infer.py" action="modify" />
 <file path="src/boost_shap_gii/utils.py" action="modify" />
 <description>Add a `_label_nominal` helper that distinguishes NaN -> "__NA__" from unseen-level -> "__UNSEEN__". Apply at predict.py:137 and infer.py:156. Mirror the existing ordinal tier-1/tier-2 validation pattern: tier-1 raises ValueError if more than 50% of unique nominal values are unknown; tier-2 emits a warning if more than 10% of observations are unknown. train.py:651 unchanged (training never produces "__UNSEEN__").</description>
 <spec>
**utils.py changes:**

Add a helper function in utils.py (near other type-handling helpers, ideally adjacent to existing ordinal validation logic):

```python
def _label_nominal(value, levels: set) -&gt; str:
 """Map a nominal value to a sentinel-aware label.

 Returns "__NA__" if value is NaN (training-time signal: missingness may
 itself be informative). Returns "__UNSEEN__" if value is non-NaN but not
 in the training-time codebook `levels` (out-of-distribution; routes to
 CatBoost prior-mean fallback). Returns the value unchanged otherwise.
 """
 if pd.isna(value):
 return "__NA__"
 if value not in levels:
 return "__UNSEEN__"
 return str(value)


def _validate_nominal_unseen(series: pd.Series,
 levels: set,
 column_name: str,
 *,
 tier1_unique_threshold: float = 0.50,
 tier2_obs_threshold: float = 0.10) -&gt; None:
 """Two-tier validation for nominal feature values not in the training codebook.

 Mirrors the ordinal validation pattern at predict.py:148-169.

 Tier 1 (hard error): if &gt; 50% of unique observed values are absent from
 `levels`, raises ValueError. Indicates misconfigured codebook or systematic
 naming mismatch between training and inference data.

 Tier 2 (loud warning): if &gt; 10% of observations (non-NaN) have values
 absent from `levels`, prints a warning with the exact fraction.
 """
 non_na = series.dropna
 if len(non_na) == 0:
 return # all-NaN column; no unseen levels possible
 unique_observed = set(non_na.unique)
 unseen_unique = unique_observed - levels
 unseen_obs = non_na.isin(unseen_unique)

 unique_unseen_frac = len(unseen_unique) max(len(unique_observed), 1)
 obs_unseen_frac = float(unseen_obs.mean)

 if unique_unseen_frac &gt; tier1_unique_threshold:
 raise ValueError(f"Nominal feature '{column_name}': "
 f"{unique_unseen_frac:.1%} of unique observed values are absent "
 f"from the training-time codebook (threshold: "
 f"{tier1_unique_threshold:.0%}). This indicates a misconfigured "
 f"codebook or systematic naming mismatch between training and "
 f"inference data. Retrain with an expanded codebook or correct "
 f"the inference data."
)
 if obs_unseen_frac &gt; tier2_obs_threshold:
 warnings.warn(f"Nominal feature '{column_name}': "
 f"{obs_unseen_frac:.1%} of observations have values absent from "
 f"the training-time codebook (threshold: "
 f"{tier2_obs_threshold:.0%}). These will route to CatBoost "
 f"prior-mean fallback via the '__UNSEEN__' sentinel.",
 UserWarning)
```

**predict.py changes:**

At predict.py:137 (the current `X[c] = df_raw[c].astype(object).fillna("__NA__").astype(str).astype("category")` line), replace with:

```python
# Distinguish NaN (informative-missing) from unseen-level (OOD) at predict-time.
levels = set(feature_metadata[c]["levels"]) # training-time codebook for column c
_validate_nominal_unseen(df_raw[c], levels, column_name=c)
X[c] = df_raw[c].apply(lambda v: _label_nominal(v, levels)).astype(str).astype("category")
```

The `feature_metadata` dict is loaded from `output_dir/feature_metadata.json` per the existing predict.py loading flow; verify that nominal feature codebooks are actually persisted in feature_metadata.json (they should be, since the pipeline already emits ordinal level definitions there). If nominal codebooks are NOT currently persisted, ADD their persistence in train.py: at the existing feature_metadata.json emission site, include nominal feature observed-level lists keyed by column name.

**infer.py changes:**

At infer.py:156 (the analogous nominal fillna site), apply the same pattern as predict.py:137 above.

**train.py changes:**

train.py:651 (the existing nominal handling for training data) is UNCHANGED. Training data never produces "__UNSEEN__" because the training-time levels ARE the codebook. The existing fillna("__NA__") behavior is preserved.

**feature_metadata.json codebook persistence:**

If nominal codebooks are not yet persisted in feature_metadata.json, add an emission step in train.py at the feature_metadata.json writing site:

```python
# Persist nominal-feature observed-level lists for predict/infer-time validation
nominal_codebooks = {
 col: sorted(map(str, df_raw[col].dropna.unique.tolist))
 for col in nominal_columns
}
feature_metadata["nominal_codebooks"] = nominal_codebooks
```

The downstream load sites (predict.py, infer.py) read `feature_metadata["nominal_codebooks"][col]` to obtain the `levels` set for `_label_nominal` and `_validate_nominal_unseen`.

**Downstream impact audit:**

- "__UNSEEN__" propagates through SHAP buffers and microdata parquets; CatBoost prior-mean fallback handles it correctly (treated as a category like any other at inference).
- plot.R will display "__UNSEEN__" as a literal categorical level. V-driver selection naturally handles its inclusion or exclusion based on V-contribution; no plot.R-specific changes for "__UNSEEN__" are needed.
- Tests that exercise predict.py infer.py with nominal features absent from the training codebook may need updates. /test phase will surface any failures.
 </spec>
 <dependencies> (the V-driver selection in plot.R interacts with the new "__UNSEEN__" sentinel through the categorical-axis rendering; both should land before /test).</dependencies>
 <risk>medium - tier-1 ValueError is a breaking change for workflows where the inference dataset contains a high fraction of novel nominal levels. Mitigated by: (a) the fail-loud behavior is the project's "err on kill" doctrine; (b) the tier-1 threshold (50%) is conservative and only triggers on systematic misconfiguration; (c) tier-2 warning surfaces problematic but non-fatal cases without aborting.</risk>
 <rollback>git checkout -- src/boost_shap_gii/predict.py src/boost_shap_gii/infer.py src/boost_shap_gii/utils.py src/boost_shap_gii/train.py</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- — Topic 19: NEGATE_SHAP plot-only comment -->
 <!-- ============================================================ -->
 <change id="change-16" priority="P2" source_item="">
 <file path="example_config_advanced.yaml" action="modify" />
 <description>Add a single inline comment "# ONLY affects plot.R rendering" to the plot.negate_shap entry in example_config_advanced.yaml. No additional documentation, no source-code change.</description>
 <spec>
At example_config_advanced.yaml line 181 (the `negate_shap: false` line under the `plot:` block), append the inline comment:

Before:
```yaml
 negate_shap: false
```

After:
```yaml
 negate_shap: false # ONLY affects plot.R rendering
```

If the line already has a different inline comment, replace it with the new comment. Preserve indentation exactly.

NO change to the example_config_minimal.yaml file (the minimal config does not include the plot block).
NO change to source code (NEGATE_SHAP is already correctly plot-only-scoped per the audit at brainstorm time).
 </spec>
 <dependencies>none</dependencies>
 <risk>low - single-line config comment.</risk>
 <rollback>git checkout -- example_config_advanced.yaml</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- — Topic 21: energy-gate tolerance anchor -->
 <!-- ============================================================ -->
 <change id="change-17" priority="P2" source_item="">
 <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
 <description>Expand the existing comments at shap_utils.py:178-180 (1D `_check_spline_energy_stability_1d`) and:209-211 (2D `_check_spline_energy_stability_2d`) to frame 0.1% as an empirical balance margin between false-pass and false-fail rates of the energy gate. Anchor to Higham (2002) Accuracy and Stability of Numerical Algorithms ch. 1. NO code change beyond the docstring expansion.</description>
 <spec>
At shap_utils.py lines 178-180 (the 1D energy gate comment block) and lines 209-211 (the 2D energy gate comment block), replace the existing brief comment with:

```python
# Energy-gate tolerance: the 0.1% multiplier (1.001) is an empirical balance
# margin. Pure machine epsilon for float64 (~2.2e-16), compounded through N
# spline-evaluation operations, produces relative errors typically around
# 1e-10 to 1e-8 — well below 1e-3. The 0.1% threshold is loose enough to
# avoid false fails from splev rounding plus diff/sum cancellation, and tight
# enough to catch genuine spline overshoot indicative of basis instability.
# This is an empirical heuristic, not a formal numerical analysis result;
# see Higham (2002), *Accuracy and Stability of Numerical Algorithms*, ch. 1
# for the role of empirical tolerances in numerical software design.
```

Apply the SAME comment block at both sites (1D at lines 178-180 and 2D at lines 209-211), preserving the surrounding code unchanged.

NO change to the comparison itself (`if tv_signal &gt; 1.001 * tv_reference: return False` is preserved exactly). NO config flag added. NO behavioral change.
 </spec>
 <dependencies>none</dependencies>
 <risk>low - comment-only change.</risk>
 <rollback>git checkout -- src/boost_shap_gii/shap_utils.py</rollback>
 </change>

 <!-- ============================================================ -->
 <!-- (docs): Cobb-Douglas decision-theoretic framing -->
 <!-- ============================================================ -->
 <change id="change-18" priority="P0" source_item="">
 <file path="INPUT_SPECIFICATION.md" action="modify" />
 <file path="README.md" action="modify" />
 <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
 <description>Insert decision-theoretic framing of the GII composite (Cobb-Douglas geometric mean of magnitude utility M and trend-informativeness utility V; Hill 1910 dose-response anchor; Goldstein et al. 2015 ICE anchor) into INPUT_SPECIFICATION.md, README.md, and shap_utils.py GII function docstrings. NO reference to calibration study, simulation, "in prep," or "see supplemental" anywhere in any public-repo file.</description>
 <spec>
**INPUT_SPECIFICATION.md changes:**

In Section 3 (Mathematical GII Formula), replace or augment the existing GII formula description with the decision-theoretic framing:

```
**Decision-theoretic interpretation**: GII is structured as a Cobb-Douglas
geometric mean (Cobb &amp; Douglas, 1928) of two utility components:

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
```

References to add to the INPUT_SPECIFICATION.md reference list (or to Section 3's inline citations):
- Cobb C.W., Douglas P.H. (1928) A theory of production. *American Economic Review* 18: 139-165.
- Hill A.V. (1910) The possible effects of the aggregation of the molecules of haemoglobin on its dissociation curves. *Journal of Physiology* 40: i-vii.
- Goldstein A., Kapelner A., Bleich J., Pitkin E. (2015) Peeking inside the black box: visualizing statistical learning with plots of individual conditional expectation. *Journal of Computational and Graphical Statistics* 24: 44-65.

**README.md changes:**

In the "GII Interpretation" section, append (or replace the existing GII = sqrt(M*V) discussion with) the decision-theoretic framing in plain language:

```
### GII as a Cobb-Douglas Composite

The GII formula structures global feature importance as a Cobb-Douglas
geometric mean (Cobb &amp; Douglas, 1928) of two utility components:

- **M (Magnitude)**: how strongly a feature drives model predictions on average.
- **V (Variability)**: whether the feature's contribution varies with its value
 (dose-response informativeness, conceptually anchored to Hill 1910 and
 visualized via ICE plots, Goldstein et al. 2015).

A globally important feature must score positively on BOTH dimensions: a
feature with strong magnitude but constant contribution (M high, V ~ 0) yields
GII ~ 0, and vice versa. This decision-theoretic framing is intentional — it
avoids inflating the importance of features whose attributions are large in
average magnitude but uninformative about feature-value variation.
```

**shap_utils.py changes:**

In the docstring of the primary GII-computing function (the function that returns sqrt(M * V) — verify name during implementation, likely in shap_utils.py around line 800-1000), update or add the docstring to read:

```python
"""Compute the Global Importance Index (GII) as a Cobb-Douglas composite.

GII = sqrt(M * V) is structured as a Cobb-Douglas geometric mean (Cobb &amp;
Douglas, 1928) of two utility components:

 M (magnitude utility): mean(|SHAP|) across bootstrap resamples,
 capturing average prediction-contribution magnitude.
 V (trend-informativeness utility): standard deviation of the systematic
 SHAP signal (spline-fitted or group-mean-fitted) as a function of
 feature values; conceptually anchored to Hill (1910) dose-response
 framing and visualized via Goldstein et al. (2015) ICE plots.

The geometric-mean form requires both utilities to be meaningfully positive
for a feature to be globally important: high M with V ~ 0 yields GII ~ 0
(strong magnitude, no dose-response — captured separately by sig_M), and
vice versa. This decision-theoretic framing avoids inflating the importance
of features with large attribution magnitudes but no feature-value-driven
prediction variation.

References:
 Cobb &amp; Douglas (1928), Am. Econ. Rev., 18: 139-165.
 Hill (1910), J. Physiol., 40: i-vii.
 Goldstein et al. (2015), J. Comput. Graph. Stat., 24: 44-65.
"""
```

**HARD RULE — public-repo quarantine of calibration study:**

NO reference to:
- Any calibration study
- Any simulation study
- "In prep" citations
- "See supplemental" pointers
- Finding 25 quarantined sub-project

May appear in INPUT_SPECIFICATION.md, README.md, or any other tracked public-repo file. The AID_LOG.md (transparency-only file) MAY reference the quarantined unified simulation study as transparency disclosure, but this is a separate file and outside the scope of /implement build for this cycle.
 </spec>
 <dependencies>none</dependencies>
 <risk>low - documentation-only changes; no behavioral impact. The decision-theoretic framing is the brainstorm-locked public-repo presentation of the GII composite.</risk>
 <rollback>git checkout -- INPUT_SPECIFICATION.md README.md src/boost_shap_gii/shap_utils.py</rollback>
 </change>

 </changes>

 <execution_order></execution_order>

</implement_plan>
