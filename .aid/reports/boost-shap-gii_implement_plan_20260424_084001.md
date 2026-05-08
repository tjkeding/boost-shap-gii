<implement_plan>
 <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-04-24T12:40:01Z" />

 <input_reports>
 <report path="<aid_history>/brainstorm_history/boost-shap-gii_brainstorm_20260423_191951.md" mode="brainstorm" key_items="7" />
 </input_reports>

 <plan_discipline_resolutions>
 <resolution id="resolution-1" topic="Per-individual OOB count floor ('e.g., 25')">OOB floor fixed at 50. Rationale: at B=500 the floor corresponds to ~1/10 of total refits (below which percentile tails are unreliable); at B=1000 it corresponds to ~1/20 (comfortable floor). Individuals whose OOB count is below 50 emit NaN CI bounds with oob_count preserved in the output schema.</resolution>
 <resolution id="resolution-2" topic="Outcome-statistics caching artifact ('extend target_scaler.json or a new training artifact')">New artifact train_outcome_stats.json (NOT an extension of target_scaler.json). Rationale: target_scaler.json is conditionally written only for multi_regression (when outcome is scaled via StandardScaler); overloading it would couple outcome-statistics availability to task type. The new artifact is written unconditionally for any regression task and provides a single, stable pointer for downstream SD lookup.</resolution>
 <resolution id="resolution-3" topic="R-package preflight dependency list (brainstorm the relevant topic concrete list open)">R_DEPS unchanged in check_env.py. Existing list {ggplot2, dplyr, nanoparquet, tidyr, foreach, doParallel, gridExtra, stringr, yaml} already covers all imports used by the extended plot.R; the remaining imports (splines, grid, grDevices, parallel) are base R packages that do not require installation. Preflight elevation is a call-site change (invoke at every CLI entry point), not a dependency change.</resolution>
 <resolution id="resolution-4" topic="Original cosmetic config_outcome_b.yaml fdr_correct issue">Descoped by explicit user directive (2026-04-24). The file is a user-study-specific external config, not a pipeline artifact.</resolution>
 <resolution id="resolution-5" topic="HP source and bootstrap-refit structure for per-individual CIs">Option E with coupled bootstrap. User config shap.indiv_ci_nboot = B specifies the number of coupled iterations. Per iteration b in {1...,B}: draw ONE shared bootstrap sample s_b of size N from the full training set D (with replacement; cluster-aware when cluster_ids present). For each fold k in {0...,K-1}: load model_fold_{k}.cbm, extract params_k = model.get_all_params (HP already embedded in the saved model), and refit a fresh CatBoost on (X_train[s_b], y_train[s_b]) using params_k with cat_features=nom_feats. Cache the K refits and the shared index vector s_b per iteration. Total refits = K × B. Per-individual CI aggregation is estimand-matched to the deployed-product point estimate: inference individuals receive B ensemble-averaged replicates (mean across K coupled fold refits per iteration), matching the deployed ensemble estimator's sampling distribution; training individual i (assigned to fold k_i during original K-fold CV) receives the subset of iterations where i ∉ s_b, using ONLY the fold-k_i refit from each such OOB iteration (expected ≈ 0.368 × B effective samples, Breiman 2001). Rationale: the coupled design binds the K fold refits per iteration to a single bootstrap sample so that ensemble replicates capture between-fold covariance correctly; estimand match between point and CI eliminates systematic point-outside-CI risk and false-negative inflation that would arise from applying single-model variance to an ensemble estimator (variance ratio ≈ 1/K × [1+(K-1)ρ] favors ensemble; at K=10 with ρ≈0.3, single-model CI would be ≈1.65× too wide). No new HP artifact needed (HP embedded in model_fold_{k}.cbm).</resolution>
 <resolution id="resolution-6" topic="Scaling-mode sd applicability to classification tasks (plan-phase observation)">sd scaling mode is restricted to regression and multi_regression tasks. Config-validation layer fails loudly if shap.indiv_scaling_mode == 'sd' and modeling.task_type is a classification variant, with error message "shap.indiv_scaling_mode='sd' requires a regression task; got task_type='{task}'. Use 'raw' or 'custom_value' instead." raw mode is available for all task types; custom_value mode is available for all task types and requires shap.indiv_scaling_value to be a positive number. For multi_regression (outcome is a list of N columns), sd mode uses a per-outcome SD vector (one SD per outcome column) and SHAP values for each outcome are scaled by the corresponding SD.</resolution>
 <resolution id="resolution-7" topic="indiv_ci_nboot semantics for classification tasks ('required, no default')">shap.indiv_ci_nboot is an integer; value 0 disables the feature (no bootstrap refit, no indiv_reports/ emission, no cache written, no indiv_scaling_mode or plot.indiv_* keys consulted). Value greater than 0 enables the feature. The config key is required (user must specify, even if 0). This preserves the brainstorm's "required, no default" directive while providing a clean opt-out for users who do not want the feature (including classification-task users). When enabled, minimum recommended value is 2500 (inference CIs at Efron & Tibshirani 1993 preferred tier of B≥1000 effective samples; training OOB CIs at ≈0.368 × 2500 = 920 effective samples, near-Efron tier). Peer-review-facing runs should use 5000 (both sides solidly above Efron threshold: inference 5000, training 1840). Values below 2500 are permitted but result in training OOB CIs at only Carpenter & Bithell 2000 minimum tier (B≥200); this trade-off is documented in INPUT_SPECIFICATION Section 10. Total refits executed = K × B (embarrassingly parallel; on 45-core systems with CatBoost thread_count=1, B=2500 at K=10 completes in ≈4-5 hours on moderate datasets).</resolution>
 <resolution id="resolution-8" topic="Point-estimate statistic for shap_value_raw and y_pred_raw">Deployed-product SHAP. The point estimate (shap_value_raw and y_pred_raw) is NOT a bootstrap-distribution statistic (mean/median); it is the value the deployed pipeline product produces on the individual in question. Training individuals: OOF single-model SHAP and prediction from model_fold_{k_i}.cbm (the original fold model for the fold to which individual i was assigned; leakage-free). Inference individuals: ensemble-mean SHAP and prediction across all K original fold models (matching infer.py's existing ensemble-prediction logic at infer.py:225-312). The bootstrap refits (K × B models cached under bootstrap_refits/) are consumed ONLY for CI computation, never for the point estimate. Rationale: (a) user principle that pipeline statistics must be tied to the deployed product; (b) estimand match with the coupled-bootstrap CI — both point and CI target the same statistical functional (OOF single-model SHAP for training individuals, ensemble-mean SHAP for inference individuals) and its resampling distribution; (c) conventional reporting in use-case-specific-ML literature uses the model's actual output as the point estimate, not a resampled aggregate. Implementation consequence: indiv_reports.py needs access to the original K fold models (model_fold_{k}.cbm) in addition to the bootstrap_refits cache to compute point estimates; training-individual fold assignments are reconstructed deterministically from existing saved objects.</resolution>
 <resolution id="resolution-11" topic="Training-individual fold-assignment source (gap without a new artifact)">Training-individual fold assignments are reconstructed deterministically from already-saved objects, not persisted as a new artifact. Mechanism: indiv_reports calls `splitter = get_cv_splitter(config, y_for_split)` (utils.py:110-122, which returns KFold or StratifiedKFold with shuffle=True, random_state=config["execution"]["random_seed"]) and iterates `for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_for_split))` to populate `fold_of[val_idx] = fold_idx`. All inputs required for reconstruction are already persisted: random_seed and cv_folds live in config_resolved.json, feature order is fixed by trained_features, and the splitter type is deterministic in task_type. This matches the existing reconstruction pattern already used by predict.py:222-236 and inherits its determinism assumptions (no new silent dependencies introduced by indiv_reports). Scope: reconstruction is performed ONLY in training mode (predict.py-invoked generate_indiv_reports), where predict.py's normal flow already loads X_train and y_train. Inference mode (infer.py-invoked) does not reconstruct fold assignments because inference individuals are never in any bootstrap sample; all B iterations are OOB for them and CI aggregation uses ensemble-averaged replicates. Consequence: no train.py change beyond train_outcome_stats.json; infer.py never needs access to the original training dataset. Helper _load_fold_assignments_or_fail is renamed _reconstruct_fold_assignments and performs the iteration above; its failure mode is an AssertionError if any position in fold_of remains unset (indicating splitter did not produce a full partition).</resolution>
 <resolution id="resolution-12" topic="Memory-budget threshold for per-individual CI accumulation (step-4b gap)">Hard threshold: `budget_bytes = 0.5 * psutil.virtual_memory.available` at the start of generate_indiv_reports. Projected memory footprint = `N_target * N_features * effective_B_max * 8` bytes for main effects (float64), plus an equivalent term for interaction tensors when sig_GII interactions exist (N_target * N_pairs_sig * effective_B_max * 8). If projected > budget_bytes, raise MemoryError with guidance: "indiv_reports CI accumulation would require ≈{projected_GB} GB but budget is {budget_GB} GB (50% of available {avail_GB} GB). Reduce shap.indiv_ci_nboot or run on a higher-memory node." Rationale. psutil is already a pipeline dependency (present in existing train.py via the n_jobs resource-accounting code path and installed per environment.yaml).</resolution>
 <resolution id="resolution-13" topic="Streaming-percentile fallback (implementation-note gap)">No streaming fallback. On projected memory overflow, the module raises MemoryError (err-on-kill). Rationale: (a) the existing shap_utils.py bootstrap CI pipeline uses in-memory `np.nanpercentile` exclusively (shap_utils.py:872-928) with no streaming machinery — adding t-digest here would introduce a new library dependency for a single module, violating the project principle of minimizing dependencies; (b) err-on-kill philosophy (user CLAUDE.md: "minimizes assumptions, hardcoded values, and defaults (err on kill)") prefers a loud failure with actionable guidance over a silent approximation switch; (c) the 50% budget is conservative enough that typical workloads (N_target=1000, N_features=100, B=2500 → 2.0 GB for main-effect CI accumulation) fit comfortably on standard HPC nodes.</resolution>
 <resolution id="resolution-14" topic="shap_stats CSV schema (path-resolution gap)">Single file: {run_dir}/shap_analysis/shap_stats_global.csv emitted by shap_utils.py:962. Per code inspection there is NO separate shap_stats_interactions.csv; main effects and interactions coexist in one CSV distinguished by two columns: `type` (value "Singleton" for main effects, else "Interaction" or analogous tag written by shap_utils) and `effect` (feature name for main effects, "feat_a x feat_b" literal with " x " separator for interactions). Parser contract for _load_sig_GII_from_shap_stats(run_dir): (a) read shap_stats_global.csv via pandas; (b) split into main_df = rows where type == "Singleton" and int_df = rows where type != "Singleton"; (c) build sig_GII_main = {row.effect: bool(row.sig_GII) for row in main_df}; (d) build sig_GII_interaction = {tuple(row.effect.split(" x ")): bool(row.sig_GII) for row in int_df if " x " in row.effect}. Raises RuntimeError with remediation guidance if shap_stats_global.csv missing. Multi-output mode: shap_utils emits per-output files at {run_dir}/shap_{slice_label}/shap_stats_global.csv (shap_utils.py:1219) — _load_sig_GII_from_shap_stats auto-detects: if run_dir/shap_analysis/ exists use it; else enumerate shap_*/ and use the first matching slice (multi-output indiv_reports is v1 scope-limited to a single representative slice; full per-slice indiv_reports emission is deferred to a future cycle).</resolution>
 <resolution id="resolution-15" topic="check_env run_preflight API (factoring gap)">Per code inspection of check_env.py (79 lines): main is nullary (no argparse); it calls check_python + check_r + sys.exit(0/1). is SIMPLIFIED to: add a new `run_preflight` function in check_env.py that directly calls check_python and check_r (the existing module-level helpers at check_env.py:18-63), prints "[ENV] Environment preflight passed." on success, and calls sys.exit(2) on failure (distinct exit code from main's sys.exit(1) so preflight failures are distinguishable in CI logs). No factoring of main is needed. main remains the argparse-free entry point for the standalone `check-env` CLI subcommand.</resolution>
 <resolution id="resolution-9" topic="Multiclass classification row structure in indiv_reports long-format schema">All-classes long format (Option B). For task=multiclass_classification: (a) predictions.parquet becomes one row per (individual, class) with columns id, class, prob, prob_ci_lo, prob_ci_hi, prob_oob_count, y_true (y_true equals the observed class label, repeated across the C rows for a given individual, or NaN when outcome absent); the argmax class is derivable per-individual as the row with max prob. (b) main_effects.parquet becomes one row per (individual, feature, class) with the existing schema plus a new class column; each class's row carries that class's SHAP contribution, scaled value, CI bounds, and oob_count (oob_count is per individual, not per class, since OOB membership is an individual-level property). (c) interactions.parquet becomes one row per (individual, feature_a, feature_b, class) with the analogous extension; still hard-filtered to model-level sig_GII=True interaction pairs. For task in {regression, multi_regression, binary_classification}, the class column is omitted from main_effects.parquet and interactions.parquet (schema unchanged from the pre-); predictions.parquet for binary_classification remains one row per individual with y_pred_raw = predict_proba[:,1]. For multi_regression, predictions.parquet continues to use one row per individual with per-outcome columns (y_pred_raw_{col}, y_pred_ci_lo_{col}, y_pred_ci_hi_{col} for each col in outcome_cols), matching the existing wide-on-outcome convention at infer.py:470-477. Plot.R default behavior for multiclass: filter to the argmax-predicted class per individual and emit ONE plot per individual; a future config flag plot.indiv_emit_all_classes (out of scope v1) could emit C plots per individual. sig_GII is a model-level property (not class-specific in the current pipeline's sig_GII artifact); broadcast the same sig_GII value across all C class-rows per (individual, feature). Rationale: Option B preserves complete per-class SHAP information produced by CatBoost multiclass, honors the brainstorm's long-format schema invariant for all task types, cleanly supports research-level all-class inspection without breaking use-case-specific-default plot behavior, and integrates with (sd scaling regression-only; multiclass uses raw or custom_value, both scale-agnostic).</resolution>
 <resolution id="resolution-10" topic="Plot emission behavior for below-OOB-floor individuals">Point-only plot with in-plot caption (Option B). For any individual i where any feature has oob_count < 50: plot.R still emits {run_dir}/indiv_reports/plots/{id}_main_effects.png (and analogous _interactions.png), drawing dot markers for shap_value_scaled point estimates but omitting whiskers (geom_errorbar is either conditionally not added or its width is set to 0 so it renders invisibly). A caption/subtitle placed below the x-axis reads verbatim: "CI unavailable (oob_count < 50); point estimate shown only." The caption uses the same font and styling as other plot captions (no warning banner, no red highlighting, no color change). File naming and output directory are IDENTICAL to compliant plots (no separate below_floor/ subdirectory). Rationale: (a) point estimate is defensible (deployed-product SHAP, not bootstrap-derived) so rendering genuine information is correct; (b) caption makes the CI limitation explicit on the plot itself; (c) below-floor cases are rare at recommended B (B=2500, K=10 → expected 0.368 × 2500 = 920 OOB samples per training individual on average); (d) absent whiskers is itself a strong visual cue. Implementation: the below-floor detection is per-individual (if ANY feature's oob_count < 50, the entire plot for that individual uses the point-only convention; this is a conservative approach since the per-individual OOB count is actually constant across features under the coupled design — OOB membership is an individual-level property).</resolution>
 </plan_discipline_resolutions>

 <assumptions>
 <assumption>CatBoost's model.get_all_params returns all user-level hyperparameters (iterations, depth, learning_rate, loss_function, border_count, l2_leaf_reg, bagging_temperature, random_strength, thread_count) after load_model. Verified by CatBoost public API documentation; widely used pattern.</assumption>
 <assumption>The existing cluster-aware bootstrap mechanism in _run_bootstrap_pipeline (shap_utils.py:721) does not need to be reused for Option E; a new bootstrap-sampler function lives in indiv_reports.py because (a) the row-resample targets differ (row-matrix vs training-set individual indices), (b) the output artifacts differ (bootstrap CI on SHAP matrix vs B refit models + indices), (c) shap_utils.py is already 1231 lines and this logic is semantically distinct.</assumption>
 <assumption>Bootstrap OOB accounting under the coupled design: a training individual i is OOB at iteration b if i ∉ s_b (the shared bootstrap sample drawn for iteration b). Individual i's CI aggregation consumes ONLY iteration b's fold-k_i refit (not all K fold refits from iteration b), so that the aggregated distribution is single-model estimand-matched to the OOF single-model point estimate. Expected effective sample count per training individual ≈ 0.368 × B (Breiman 2001). Inference individuals are never in any bootstrap sample; for each iteration b they consume the ensemble-averaged SHAP across all K coupled fold refits (one ensemble replicate per iteration), producing B ensemble-estimand replicates matched to the ensemble-mean point estimate. The fold-assignment array for each training individual is reconstructed at predict-time via get_cv_splitter; no new artifact is persisted by train.py.</assumption>
 <assumption>plot.R auto-discovery of indiv_reports/ directory uses the same RUN_DIR root that already serves shap_analysis/ auto-discovery; no new path-resolution logic needed beyond listing file("indiv_reports/main_effects.parquet") etc. and checking existence.</assumption>
 <assumption>The 3 pre-existing test failures (test_hardening, test_package_structure::TestCLI.test_check_env_runs_successfully, test_plot_smoke) caused by missing nanoparquet R package are environment-only and out of scope for this implementation cycle; they do not block the build.</assumption>
 </assumptions>

 <changes>

 <change id="change-1" priority="P1" source_item="brainstorm action ">
 <file path="example_config_advanced.yaml" action="modify" />
 <description>Add six new config keys under shap.* and six new keys under plot.* for the indiv_reports feature and the plot.R config migration.</description>
 <spec><![CDATA[
Under shap: section, ADD:
 shap.indiv_ci_nboot: <int, required, no default>
 # Inline comment: "Number of coupled bootstrap iterations for per-individual SHAP CIs. 0 disables the feature. When enabled, minimum recommended 2500 (inference CIs Efron-tier, training OOB CIs near-Efron). Peer-review-facing runs: 5000 (both sides Efron-tier). Values below 2500 permitted but produce training OOB CIs only at Carpenter-Bithell 2000 minimum tier. Per iteration, K fold refits are performed on a shared bootstrap sample; total refits executed = K × B. Embarrassingly parallel."
 shap.indiv_scaling_mode: <str, required, no default, one of: raw | sd | custom_value>
 # Inline comment: "raw = unscaled SHAP values; sd = divide by SD of training outcome (regression only); custom_value = divide by shap.indiv_scaling_value."
 shap.indiv_scaling_value: <number, required when scaling_mode=custom_value, ignored otherwise>
 # Inline comment: "Divisor used when scaling_mode=custom_value (e.g., outcome theoretical maximum, minimum-meaningful-difference threshold, or domain-meaningful anchor). Must be > 0."
 shap.compute_global_on_inference: <bool, default false>
 # Inline comment: "When true, infer.py also emits shap_analysis/ population-level GII on the inference dataset (distribution-shift diagnostic at large N). Default false because small-N inference sets produce degenerate GII."

Under plot: section (CREATE the section if it does not yet exist), ADD:
 plot.outcome_max: <number, required when plot subcommand invoked>
 # Inline comment: "Theoretical maximum value of the outcome (used to scale GII magnitudes on plots). Renamed from the prior outcome_range CLI arg."
 plot.negate_shap: <bool, required when plot subcommand invoked>
 # Inline comment: "If true, SHAP y-axis values are sign-flipped on GII and indiv plots. Color gradient and x-axis ordering remain anchored to raw signed SHAP."
 plot.gii_y_label: <str, required when plot subcommand invoked>
 plot.gii_y_sublabel: <str, required when plot subcommand invoked>
 plot.indiv_y_label: <str, required when plot subcommand invoked>
 plot.indiv_y_sublabel: <str, required when plot subcommand invoked>
 # Inline comment for all four: "Label strings rendered verbatim on plots (no programmatic composition, no individual-ID substitution, no auto-appended CI annotation). No plot titles are emitted anywhere in the pipeline."

Preserve existing YAML ordering, key casing, and comment style. All new keys appear in the order listed above. Use advanced-config docstring conventions consistent with existing keys (explanatory comments on non-trivial keys).
 ]]></spec>
 <dependencies>none (schema-only)</dependencies>
 <risk>low — config schema addition, backward-compatible with explicit schema-evolution visible in diff.</risk>
 <rollback>git revert the hunk; fall back to prior config schema (no data loss, no artifact corruption).</rollback>
 </change>

 <change id="change-2" priority="P1" source_item="brainstorm action ">
 <file path="example_config_minimal.yaml" action="modify" />
 <description>Add the same keys as but only the required ones that cannot be auto-filled. Keys with defaults (shap.compute_global_on_inference) are omitted from the minimal config.</description>
 <spec><![CDATA[
ADD to example_config_minimal.yaml:
 shap.indiv_ci_nboot: <int>
 shap.indiv_scaling_mode: <str>
 shap.indiv_scaling_value: <number> # only present if mode=custom_value in example
 plot.outcome_max: <number>
 plot.negate_shap: <bool>
 plot.gii_y_label: <str>
 plot.gii_y_sublabel: <str>
 plot.indiv_y_label: <str>
 plot.indiv_y_sublabel: <str>

OMIT (defaulted by fill_config_defaults):
 shap.compute_global_on_inference (defaults to false)

Rationale for inclusion: each added key in the minimal config either has no default (per brainstorm decision) or represents a required input the user cannot omit. shap.compute_global_on_inference has a safe default (false) and can be left out of the minimal config; users who want true will set it in their own config.

Provide plausible minimal-config example values (e.g., indiv_ci_nboot: 2500, scaling_mode: sd, outcome_max: 100, negate_shap: false, label strings matching the advanced config's example).
 ]]></spec>
 <dependencies>none (schema-only, parallel to)</dependencies>
 <risk>low</risk>
 <rollback>git revert the hunk.</rollback>
 </change>

 <change id="change-3" priority="P1" source_item="brainstorm action ">
 <file path="src/boost_shap_gii/utils.py" action="modify" />
 <description>Extend fill_config_defaults to set default for shap.compute_global_on_inference and add validation for all new required keys. Add a helper validate_indiv_reports_config that performs deep consistency checks (scaling_mode in allowed set; scaling_value present when mode=custom_value; scaling_mode='sd' only for regression tasks; nboot >= 0).</description>
 <spec><![CDATA[
Modify fill_config_defaults(config: dict) -> dict at ~line 246:

1. In the shap.* block, add:
 config.setdefault("shap", {}).setdefault("compute_global_on_inference", False)
 No default for indiv_ci_nboot, indiv_scaling_mode, indiv_scaling_value — these are required.

2. In the plot.* block, do NOT set defaults for the six new plot keys. They are required at plot-subcommand invocation time; validation happens in plot.R (fail-loud on missing). Rationale: the plot step is optional (user may never invoke cmd_plot); forcing defaults would mask misconfiguration.

3. Add NEW function (module-level, public):
 def validate_indiv_reports_config(config: dict) -> None:
 """Validate shap.indiv_* and shap.compute_global_on_inference keys.

 Raises ValueError with precise messages on any violation:
 - shap.indiv_ci_nboot missing, non-integer, or negative
 - shap.indiv_scaling_mode missing or not in {raw, sd, custom_value}
 - shap.indiv_scaling_mode == 'sd' but task_type not in {regression, multi_regression}
 - shap.indiv_scaling_mode == 'custom_value' but shap.indiv_scaling_value missing or <= 0
 - shap.compute_global_on_inference present but not bool
 """

4. Add NEW function (module-level, public):
 def validate_plot_config(config: dict) -> None:
 """Validate plot.* required keys. Called only from cmd_plot (not from train/predict/infer).

 Raises ValueError with precise messages on missing or wrong-typed keys:
 - plot.outcome_max missing or non-positive number
 - plot.negate_shap missing or not bool
 - plot.gii_y_label plot.gii_y_sublabel plot.indiv_y_label plot.indiv_y_sublabel missing or empty string
 """

5. Call site: in predict.py and infer.py (in their main entry points), invoke validate_indiv_reports_config(config) immediately after fill_config_defaults. In cli.cmd_plot, invoke both validate_indiv_reports_config and validate_plot_config before launching plot.R.
 ]]></spec>
 <dependencies> (schema must exist before validator is meaningful; but can be authored first since it references the schema only by key name).</dependencies>
 <risk>low — validation is additive; existing configs without new keys will fail loudly at the new validator, which is the intended behavior per err-on-kill philosophy.</risk>
 <rollback>git revert; existing configs that lack the new keys will revert to working state.</rollback>
 </change>

 <change id="change-4" priority="P1" source_item="brainstorm action (d) and ">
 <file path="src/boost_shap_gii/train.py" action="modify" />
 <description>At end of train after fold loop completes, write train_outcome_stats.json unconditionally for any task type with numeric-valued outcome (regression or multi_regression). Task types without a continuous outcome (binary_classification, multiclass_classification) write a minimal placeholder with n and task_type only.</description>
 <spec><![CDATA[
Location: immediately after line 927 (save_json_atomic(task_info.json)) and before any SHAP-analysis block.

New block:

 # --- Training-outcome statistics artifact (consumed by indiv_reports at predict infer time) ---
 _write_train_outcome_stats(y, task, outcome_cols, run_dir)

Add NEW helper function (module-level) in train.py:

 def _write_train_outcome_stats(y: Union[pd.Series, pd.DataFrame],
 task: str,
 outcome_cols: list[str],
 run_dir: str) -> None:
 """Write train_outcome_stats.json containing training-outcome summary statistics.

 Schema:
 {
 "task_type": str,
 "outcome_columns": [str...], # list of outcome column names
 "n": int, # training-sample size (N_train)
 # For task in {regression, multi_regression}:
 "stats": {
 <col>: {
 "mean": float, "sd": float, "min": float, "max": float,
 "q25": float, "q50": float, "q75": float
 }...
 }
 # For classification tasks:
 "stats": {}
 }

 For multi_regression, stats contains one entry per outcome column.
 For classification, stats is an empty dict but the file is still written.
 SD is unbiased (ddof=1, per pandas default).
 """
 import json
 stats = {}
 if task in {"regression", "multi_regression"}:
 if task == "regression":
 series = y if isinstance(y, pd.Series) else y.iloc[:, 0]
 stats[outcome_cols[0]] = _summarize_series(series)
 else: # multi_regression
 for col in outcome_cols:
 stats[col] = _summarize_series(y[col])
 payload = {
 "task_type": task,
 "outcome_columns": list(outcome_cols),
 "n": int(len(y)),
 "stats": stats,
 }
 save_json_atomic(payload, os.path.join(run_dir, "train_outcome_stats.json"))

 def _summarize_series(s: pd.Series) -> dict:
 return {
 "mean": float(s.mean),
 "sd": float(s.std(ddof=1)),
 "min": float(s.min),
 "max": float(s.max),
 "q25": float(s.quantile(0.25)),
 "q50": float(s.quantile(0.50)),
 "q75": float(s.quantile(0.75)),
 }

Important: the raw outcome values y are used (pre-scaling for multi_regression). The target_scaler serialization at line 666-677 is unchanged.

Change 4 emits ONE new artifact at run_dir root: train_outcome_stats.json. Training-individual
fold assignments (needed by indiv_reports for point estimates) are NOT persisted;
they are reconstructed at predict-time by indiv_reports via get_cv_splitter.
 ]]></spec>
 <dependencies>none (independent of other changes, modifies train.py only).</dependencies>
 <risk>low — one new artifact, backward-compatible; no change to existing outputs.</risk>
 <rollback>git revert; consumers (indiv_reports module via predict.py/infer.py) would then fail validation when attempting to load the file, which is the desired err-on-kill behavior.</rollback>
 </change>

 <change id="change-5" priority="P1" source_item="brainstorm action ">
 <file path="src/boost_shap_gii/indiv_reports.py" action="create" />
 <description>New module encapsulating the indiv_reports feature: per-fold bootstrap-refit orchestration, OOB aggregation, CI percentile computation, outcome-scale translation, long-format parquet emission, metadata JSON emission. Public API is called by predict.py (training mode) and infer.py (inference mode) with symmetric signatures.</description>
 <spec><![CDATA[
Module docstring:
 """Per-individual SHAP reports with bootstrap CIs (individual-case inspection tool).

 Implements Option E with coupled bootstrap per implement_plan. The user config
 shap.indiv_ci_nboot = B specifies the number of coupled iterations. Per iteration b:
 draw ONE shared bootstrap sample s_b (size N, with replacement, cluster-aware when
 cluster_ids present); for each fold k in range(K) refit CatBoost on (X_train[s_b],
 y_train[s_b]) using params_k = get_all_params(model_fold_{k}.cbm). Total refits = K × B.

 Point estimates (, deployed-product SHAP; NOT bootstrap-distribution statistics):
 - Training individual i (assigned to fold k_i): OOF single-model SHAP and prediction
 from model_fold_{k_i}.cbm (leakage-free).
 - Inference individual: ensemble-mean SHAP and prediction across all K original
 model_fold_{k}.cbm files (matching infer.py:225-312 ensemble logic).

 Per-individual CIs (estimand-matched to the point estimate):
 - Training individual i: aggregate iterations where i ∉ s_b, using ONLY the fold-k_i
 refit from each such iteration (single-model estimand match to OOF point).
 Expected effective count ≈ 0.368 × B (Breiman 2001 OOB rate).
 Individuals with effective count < OOB_FLOOR_MIN emit NaN CI bounds with oob_count preserved.
 - Inference individual: per iteration, compute SHAP from each of the K coupled fold
 refits and average (ensemble-estimand match to ensemble-mean point). Aggregate all
 B ensemble replicates for percentile CI.

 Scaling: raw (unscaled) | sd (divide by training-outcome SD from train_outcome_stats.json)
 | custom_value (divide by user-supplied value).

 Output structure:
 {run_dir}/indiv_reports/
 main_effects.parquet # long format, ALL features, sig_GII column
 interactions.parquet # long format, HARD-FILTERED to sig_GII=True
 predictions.parquet # per-individual predicted outcome (raw + scaled) with CIs
 indiv_reports_metadata.json

 Called from:
 predict.py: generate_indiv_reports(..., mode='training') # after bootstrap cache built
 infer.py: generate_indiv_reports(..., mode='inference') # after bootstrap cache loaded
 """

Module-level constants:
 OOB_FLOOR_MIN: int = 50 # Individuals with fewer OOB refits than this emit NaN CI bounds.
 CI_LO_PCT: float = 2.5
 CI_HI_PCT: float = 97.5

PUBLIC API:

 def orchestrate_bootstrap_cache(run_dir: str,
 X_train: pd.DataFrame,
 y_train: Union[pd.Series, pd.DataFrame],
 task: str,
 outcome_cols: list[str],
 nom_feats: list[str],
 config: dict,
 n_jobs: int,
 random_seed: int) -> dict:
 """Build the coupled bootstrap-refit cache at run_dir/bootstrap_refits/.

 Called once from predict.py (after the standard prediction/evaluation phase,
 before generate_indiv_reports). No-ops and returns immediately if
 config['shap']['indiv_ci_nboot'] == 0.

 Orchestration (coupled bootstrap):
 1. K = count of model_fold_*.cbm files under run_dir.
 B = config['shap']['indiv_ci_nboot'] (coupled iterations, integer >= 1).
 Total refits = K × B.
 2. Preload all K original fold models ONCE and extract user-level HP dicts:
 params = [None] * K
 for k in range(K):
 mdl_k = CatBoost*.load_model(run_dir/model_fold_{k}.cbm)
 params[k] = _extract_user_level_params(mdl_k.get_all_params)
 _extract_user_level_params returns the allowlist of user-facing HP
 (iterations, depth, learning_rate, loss_function, border_count,
 l2_leaf_reg, bagging_temperature, random_strength, random_seed,
 and task-type-specific keys). Fit-only or runtime-only keys are stripped
 (e.g., 'cat_features', 'used_ram_limit', deprecated entries).
 3. Draw shared index vectors for all B iterations up front:
 shared_indices: np.ndarray of shape (B, N_train)
 shared_indices[b] = bootstrap sample of row indices from X_train
 (cluster-aware if cluster_ids column present — resample clusters, then
 expand to member rows, matching shap_utils.py _run_bootstrap_pipeline).
 Save the full block:
 np.savez_compressed(run_dir/bootstrap_refits/shared_indices.npz,
 indices=shared_indices
)
 4. Dispatch all K × B refits as independent tasks to a process pool.
 Each task (k, b):
 - Fits a fresh CatBoost*(**params[k], thread_count=1) on
 X_train.iloc[shared_indices[b]], y_train.iloc[shared_indices[b]]
 with cat_features=nom_feats.
 - Saves model to run_dir/bootstrap_refits/iter_{b:05d}/fold_{k}.cbm.
 Tasks are embarrassingly parallel with no dependencies.
 5. Write run_dir/bootstrap_refits/bootstrap_metadata.json:
 {
 "design": "coupled",
 "K": K, "B": B, "total_refits": K * B,
 "random_seed": random_seed,
 "cluster_aware": bool(cluster_ids present),
 "fold_hp_summary": {k: params[k] for k in range(K)},
 "timestamp": <ISO8601>
 }
 6. Return a summary dict:
 {"K": K, "B": B, "total_refits": K * B, "cache_dir": <path>}

 Parallelism: process-level pool via concurrent.futures.ProcessPoolExecutor.
 Max workers = config['execution']['n_jobs'] (existing train-time budget; on
 typical 45-core HPC environments this saturates across iterations × folds).
 Each worker sets CatBoost thread_count=1 inside params[k]; oversubscription is
 avoided because task-level parallelism already saturates cores.

 Disk layout:
 run_dir/bootstrap_refits/
 bootstrap_metadata.json
 shared_indices.npz # (B, N_train) int32
 iter_00000/fold_0.cbm, fold_1.cbm..., fold_{K-1}.cbm
 iter_00001/fold_0.cbm...
...
 iter_{B-1:05d}/fold_{K-1}.cbm

 Storage cost: K × B model files; at ~5-20 MB per CatBoost model this is
 typically 50-500 GB for K=10, B=2500 — large but manageable on HPC scratch.
 A future optimization (out of scope v1) could serialize models to a single
 contiguous archive, but per-file is simplest for resumption and partial reads.

 def generate_indiv_reports(run_dir: str, # output directory (train_dir OR infer_dir)
 train_dir: str, # ALWAYS train_dir (source of cache + outcome stats)
 X_target: pd.DataFrame, # feature matrix for individuals being reported
 ids_target: pd.Series, # individual IDs aligned with X_target rows
 X_train: pd.DataFrame, # training features (for OOB accounting)
 y_target: Optional[pd.Series | pd.DataFrame], # true outcomes for target (may be None at inference)
 task: str,
 outcome_cols: list[str],
 nom_feats: list[str],
 config: dict,
 mode: Literal["training", "inference"],
 sig_GII_main: dict[str, bool], # {feature -> sig_GII} from shap_stats_global.csv
 sig_GII_interaction: dict[tuple[str, str], bool]) -> None:
 """Compute per-individual SHAP point estimates + coupled-bootstrap CIs and emit
 long-format parquets.

 Preconditions:
 - train_dir/bootstrap_refits/ exists (built by orchestrate_bootstrap_cache).
 - train_dir/train_outcome_stats.json exists.
 - train_dir/shap_analysis/shap_stats_global.csv exists (source of sig_GII).
 - train_dir/model_fold_*.cbm exist (K original fold models; consumed for
 deployed-product point estimates).
 - In training mode: fold assignments for training individuals are reconstructed
 from config + X_train + y_train via _reconstruct_fold_assignments.
 No train_dir artifact is required for this; the caller (predict.py) already
 has X_train and y_train in scope.
 If any precondition fails, raise FileNotFoundError with explicit guidance.

 No-ops and returns immediately if config['shap']['indiv_ci_nboot'] == 0.

 Steps:
 1. Load bootstrap_metadata.json; get K, B, total_refits. Load shared_indices.npz
 into shared_indices (shape (B, N_train)).
 2. Load train_outcome_stats.json; resolve scaling_divisor per Topic 3:
 - mode='raw': divisor = 1.0
 - mode='sd': divisor = stats[outcome_cols[0]]['sd'] (regression)
 OR per-outcome vector for multi_regression
 - mode='custom_value': divisor = config['shap']['indiv_scaling_value']
 3. Compute point-estimate SHAP + y_pred using the ORIGINAL K fold models per Resolution 8:
 a. Load all K original models: orig_models[k] = load_model(train_dir/model_fold_{k}.cbm)
 b. For training mode (X_target == X_train):
 - Reconstruct fold assignments via _reconstruct_fold_assignments(config, X_train, y_train) which calls get_cv_splitter(config, y_for_split)
 and iterates splitter.split(X_train, y_for_split) to populate
 fold_of = np.full(N_train, -1, dtype=np.int32); fold_of[val_idx] = fold_idx
 for each (fold_idx, (train_idx, val_idx)) tuple. Asserts (fold_of >= 0).all.
 - For each individual i: point_shap[i] = SHAP(orig_models[fold_of[i]], row i),
 point_y_pred[i] = predict(orig_models[fold_of[i]], row i).
 - This is leakage-free OOF SHAP and prediction.
 c. For inference mode (X_target == X_infer):
 - No fold reconstruction performed; inference individuals do not have a
 training-fold assignment.
 - For each individual i: point_shap[i] = mean_{k=0..K-1}(SHAP(orig_models[k], row i)),
 point_y_pred[i] = mean_{k=0..K-1}(predict(orig_models[k], row i)).
 - Matches infer.py:225-312 ensemble-mean prediction logic.
 d. Compute interaction-tensor point estimates analogously
 (ShapInteractionValues API on orig_models).
 4. Compute CI distribution from bootstrap refits:
 a. For each iteration b in range(B):
 - Load the K coupled refits: boot_models[k] = load_model(train_dir/bootstrap_refits/iter_{b:05d}/fold_{k}.cbm)
 - Compute SHAP for each of the K refits on X_target:
 shap_iter_b_fold_k[i, f] for main effects;
 shap_int_iter_b_fold_k[i, f_a, f_b] for interactions;
 y_pred_iter_b_fold_k[i] for predictions.
 - Also compute ensemble-averaged replicate:
 shap_iter_b_ens[i, f] = mean_{k=0..K-1}(shap_iter_b_fold_k[i, f])
 (used for inference-individual CI aggregation).
 - Record: for each training individual i, was i ∈ shared_indices[b]?
 If NOT (i is OOB at iteration b), the iteration b fold-k_i refit's
 SHAP for i is added to individual i's training-CI distribution.
 - For each inference individual j: shap_iter_b_ens[j, f] is added to
 individual j's inference-CI distribution.
 - Streaming accumulation: after processing iteration b, release the
 K loaded boot_models; only the per-individual CI distributions remain
 in memory.
 b. Memory guard: at function entry, compute
 budget_bytes = int(0.5 * psutil.virtual_memory.available)
 projected_bytes_main = N_target * N_features_main * effective_B_max * 8
 projected_bytes_inter = N_target * N_pairs_sig_GII * effective_B_max * 8
 projected_total = projected_bytes_main + projected_bytes_inter
 where effective_B_max = B (inference) or B (upper bound for training since
 any individual individual could have effective_count up to B under coupled
 resampling, though the mean is 0.368B). If projected_total > budget_bytes:
 raise MemoryError(f"indiv_reports CI accumulation would require ~{projected_total/1e9:.2f} GB "
 f"but the 50% budget is {budget_bytes/1e9:.2f} GB (of "
 f"{psutil.virtual_memory.available/1e9:.2f} GB available). "
 "Reduce shap.indiv_ci_nboot or run on a higher-memory node."
)
 No streaming fallback: err-on-kill is chosen over a silent
 approximation switch so the user can decide explicitly.
 5. Per-individual aggregation:
 For each individual i and feature f:
 - Point: shap_value_raw[i, f] = point_shap[i, f] (from step 3;).
 - Select CI distribution:
 training mode AND individual_i in train: fold-k_i refits from OOB
 iterations only (step 4a).
 inference mode: ensemble-averaged replicates across all B iterations.
 - effective_count = len(CI distribution for individual i, feature f).
 Training mode: ≈ 0.368 × B per individual (varies by individual).
 Inference mode: = B (all iterations are OOB for inference individuals).
 - If effective_count < OOB_FLOOR_MIN:
 shap_value_scaled = shap_value_raw divisor,
 shap_value_ci_lo = NaN, shap_value_ci_hi = NaN,
 oob_count = effective_count (preserved).
 Note: point estimate is still emitted (not NaN); only the CI is NaN.
 - Else:
 shap_value_scaled = shap_value_raw divisor.
 ci_distribution_scaled = CI_distribution_raw divisor.
 shap_value_ci_lo = 2.5 percentile of ci_distribution_scaled.
 shap_value_ci_hi = 97.5 percentile of ci_distribution_scaled.
 oob_count = effective_count.
 Analogous aggregation for interactions and y_pred.
 6. Emit long-format parquets schema (see <output_schemas>).
 7. Emit indiv_reports_metadata.json (see <output_schemas>).
 8. sig_GII for main_effects: broadcast from sig_GII_main dict.
 Interactions: filter ALL rows (across all individuals) to sig_GII=True pairs;
 if NO sig_GII=True interactions exist at the model level, emit an empty
 interactions.parquet with the full column schema (header-only).
 """

 def _load_bootstrap_cache_or_fail(run_dir: str) -> dict:
 """Internal helper. Loads bootstrap_metadata.json from run_dir/bootstrap_refits/.
 Raises FileNotFoundError with remediation hint if cache missing."""

 def _load_train_outcome_stats_or_fail(train_dir: str) -> dict:
 """Internal helper. Loads train_outcome_stats.json from train_dir.
 Raises FileNotFoundError with remediation hint if missing."""

 def _resolve_scaling_divisor(mode: str, value: float, task: str, outcome_cols: list[str], stats: dict
) -> Union[float, dict[str, float]]:
 """Return scalar divisor (single-outcome regression/classification) or
 per-outcome dict {col -> divisor} (multi_regression)."""

 def _emit_main_effects_parquet(run_dir: str,
 rows: list[dict], # one dict per (individual, feature)
) -> None:...

 def _emit_interactions_parquet(run_dir: str,
 rows: list[dict], # one dict per (individual, feature_a, feature_b) with sig_GII=True
) -> None:...

 def _emit_predictions_parquet(run_dir: str,
 rows: list[dict], # one dict per individual
) -> None:...

 def _emit_metadata_json(run_dir: str,
 scaling_mode: str,
 scaling_divisor: Union[float, dict],
 effective_B: int,
 oob_floor: int,
 outcome_names: list[str],
 negate_shap_flag: Optional[bool],
 timestamp: str) -> None:...

OUTPUT SCHEMAS (per topic, extended by for multiclass):

main_effects.parquet:
 Non-multiclass tasks (regression, multi_regression, binary_classification):
 one row per (individual, feature), ALL features.
 id str
 feature str
 feature_value_raw str (stringified raw feature value)
 feature_type str ('nominal' | 'ordinal' | 'continuous')
 shap_value_raw float
 shap_value_scaled float (NaN if below OOB floor)
 shap_value_ci_lo float (NaN if below OOB floor)
 shap_value_ci_hi float (NaN if below OOB floor)
 oob_count int
 sig_GII bool

 Multiclass_classification:
 one row per (individual, feature, class), ALL features and ALL classes.
 id str
 feature str
 class str (class label; matches class_labels order)
 feature_value_raw str
 feature_type str
 shap_value_raw float (class-specific log-odds SHAP contribution)
 shap_value_scaled float
 shap_value_ci_lo float
 shap_value_ci_hi float
 oob_count int (per-individual OOB count; SAME value across C class-rows per (individual, feature))
 sig_GII bool (model-level sig_GII; broadcast across C class-rows)

interactions.parquet (FILTERED to model-level sig_GII=True pairs):
 Non-multiclass tasks: one row per (individual, feature_a, feature_b).
 id str
 feature_a str
 feature_b str
 feature_a_value_raw str
 feature_b_value_raw str
 feature_a_type str
 feature_b_type str
 shap_value_raw float
 shap_value_scaled float
 shap_value_ci_lo float
 shap_value_ci_hi float
 oob_count int

 Multiclass_classification: one row per (individual, feature_a, feature_b, class).
 Schema as above with additional:
 class str

predictions.parquet:
 Non-multiclass tasks: one row per individual.
 id str
 y_pred_raw float
 y_pred_scaled float (regression/multi_regression only; for binary_classification equals y_pred_raw)
 y_pred_ci_lo float
 y_pred_ci_hi float
 y_pred_oob_count int
 y_true float (NaN if outcome not present)

 multi_regression (extends single-row-per-individual with per-outcome wide columns,
 matching infer.py:470-477 convention):
 id str
 y_true_{col} float (for each col in outcome_cols)
 y_pred_raw_{col} float
 y_pred_scaled_{col} float
 y_pred_ci_lo_{col} float
 y_pred_ci_hi_{col} float
 y_pred_oob_count int (shared across outcomes; per-individual)

 Multiclass_classification: one row per (individual, class).
 id str
 class str
 prob float (point estimate: OOF single-model probability for training,
 ensemble-mean probability for inference)
 prob_ci_lo float
 prob_ci_hi float
 prob_oob_count int (per-individual; same value across C class-rows)
 y_true str (observed class label; repeated across C class-rows per individual;
 NaN if outcome absent)

indiv_reports_metadata.json:
 {
 "design": "coupled",
 "scaling_mode": str,
 "scaling_divisor": number | dict {col: number},
 "B": int, # coupled iterations (config value)
 "K": int, # number of outer CV folds
 "total_refits": int, # K * B
 "point_estimate_source": "OOF_single_model" | "ensemble_mean",
 "ci_aggregation": "OOB_single_model" | "ensemble_replicates",
 "oob_count_floor": 50,
 "outcome_columns": [str...],
 "mode": "training" | "inference",
 "timestamp": ISO8601
 }
 Note: point_estimate_source and ci_aggregation fields are and respectively;
 their values are coupled to mode ("training" → OOF_single_model + OOB_single_model;
 "inference" → ensemble_mean + ensemble_replicates).

IMPLEMENTATION NOTES:
 - Use numpy for in-memory SHAP accumulation (avoid per-row Python object overhead).
 - Target memory profile: at most one refit's worth of SHAP values in memory at once
 during accumulation; results are percentile-reduced per-individual before emission.
 - Keep-in-memory only. Per /Resolution 13: if projected memory exceeds 50% of
 psutil.virtual_memory.available at function entry, raise MemoryError immediately
 (no streaming-percentile fallback). The in-memory path uses np.nanpercentile,
 consistent with shap_utils.py:872-928.
 - feature_type resolution: read from existing pipeline artifacts (feature_types.json
 or the analog artifact written by train.py); do not re-infer.
 - y_pred computation: regression → CatBoost.predict; multi_regression → per-outcome
 wide columns (matching infer.py:470-477); binary_classification → predict_proba[:, 1];
 multiclass_classification → long format, one row per (individual, class), with prob =
 predict_proba[:, c]; point estimate uses deployed-product convention (OOF
 single-model for training individuals, ensemble-mean across K original fold models for
 inference individuals); CI.
 ]]></spec>
 <dependencies> (config validators), (train_outcome_stats.json).</dependencies>
 <risk>medium — largest single change (~400-600 LOC); algorithmic correctness (OOB aggregation, scaling, CI percentiles) must be validated; memory profile must be confirmed at realistic N and B.</risk>
 <rollback>git rm the new module; and below detect absence of orchestrate_bootstrap_cache generate_indiv_reports via import guard and skip.</rollback>
 <!-- All plan-level decisions resolved via resolutions 1-10. No build-time surfacing. -->
 </change>

 <change id="change-6" priority="P1" source_item="brainstorm action ">
 <file path="src/boost_shap_gii/predict.py" action="modify" />
 <description>After the existing prediction + evaluation + run_shap_pipeline phases, invoke indiv_reports.orchestrate_bootstrap_cache to build the B-refit cache (in train_dir), then invoke indiv_reports.generate_indiv_reports(mode="training") to emit training-individual reports to train_dir/indiv_reports/.</description>
 <spec><![CDATA[
Insertion point: at the end of the predict main function, AFTER run_shap_pipeline(ctx) and BEFORE the function returns.

Add the following block:

 # --- Per-individual SHAP reports (indiv_reports) ---
 from.utils import validate_indiv_reports_config
 validate_indiv_reports_config(config)

 nboot_indiv = int(config["shap"]["indiv_ci_nboot"])
 if nboot_indiv > 0:
 from.indiv_reports import orchestrate_bootstrap_cache, generate_indiv_reports

 # 1. Build cache at run_dir/bootstrap_refits/
 cache_summary = orchestrate_bootstrap_cache(run_dir=run_dir,
 X_train=X_train,
 y_train=y_train,
 task=task,
 outcome_cols=outcome_cols,
 nom_feats=nom_feats,
 config=config,
 n_jobs=n_jobs,
 random_seed=config["execution"]["random_seed"])
 print(f"[INFO] Bootstrap cache built: effective_B={cache_summary['effective_B']} "
 f"across K={cache_summary['K']} folds ({cache_summary['B_per_fold']} per fold).")

 # 2. Emit training-individual indiv_reports
 sig_GII_main, sig_GII_interaction = _load_sig_GII_from_shap_stats(run_dir)
 generate_indiv_reports(run_dir=run_dir,
 train_dir=run_dir, # in predict.py, train_dir == run_dir
 X_target=X_train,
 ids_target=ids_train, # or resolve from the training DataFrame; match infer.py convention
 X_train=X_train,
 y_target=y_train,
 task=task,
 outcome_cols=outcome_cols,
 nom_feats=nom_feats,
 config=config,
 mode="training",
 sig_GII_main=sig_GII_main,
 sig_GII_interaction=sig_GII_interaction)
 print(f"[INFO] Training indiv_reports/ emitted to {run_dir}.")
 else:
 print("[INFO] shap.indiv_ci_nboot=0; skipping per-individual SHAP reports.")

Add NEW helper _load_sig_GII_from_shap_stats(run_dir) in utils.py (shared between
predict.py and infer.py's location-3 note). Per, shap_utils emits ONE CSV
containing both main effects and interactions, distinguished by `type` column:

 def _load_sig_GII_from_shap_stats(run_dir: str) -> tuple[dict, dict]:
 """Load sig_GII flags from the single shap_stats_global.csv.

 Returns (sig_GII_main, sig_GII_interaction):
 sig_GII_main: {feature_name: bool}
 sig_GII_interaction: {(feature_a, feature_b): bool} # order as written

 File path resolution:
 - Single-output mode: read run_dir/shap_analysis/shap_stats_global.csv.
 - Multi-output mode (no shap_analysis/ present, but shap_{slice}/ dirs exist):
 v1 scope-limited to the first shap_*/ slice found by glob("shap_*/");
 emits INFO-level log ("indiv_reports using representative slice: {slice_label}").
 Per-slice indiv_reports emission is deferred to a future cycle.

 Filtering:
 import pandas as pd
 df = pd.read_csv(csv_path)
 main_df = df[df["type"] == "Singleton"]
 int_df = df[df["type"] != "Singleton"]
 sig_GII_main = {
 str(row["effect"]): bool(row["sig_GII"])
 for _, row in main_df.iterrows
 }
 sig_GII_interaction = {}
 for _, row in int_df.iterrows:
 eff = str(row["effect"])
 if " x " in eff:
 a, b = eff.split(" x ", 1)
 sig_GII_interaction[(a, b)] = bool(row["sig_GII"])
 return sig_GII_main, sig_GII_interaction
 - Exposed as module-level public: from.utils import _load_sig_GII_from_shap_stats.

Failure mode: if no shap_stats_global.csv can be located (neither run_dir/shap_analysis/
nor any run_dir/shap_*/ directory contains one), raise RuntimeError with guidance
"shap_stats_global.csv not found in run_dir; run full predict with shap computation
enabled before indiv_reports."

No other changes to predict.py (existing prediction/evaluation/SHAP flow preserved).
 ]]></spec>
 <dependencies> (module must exist; train_outcome_stats.json consumed by).</dependencies>
 <risk>medium — modifies predict.py's primary control flow; must not regress existing prediction outputs.</risk>
 <rollback>git revert the hunk; existing predict.py flow restored unchanged.</rollback>
 </change>

 <change id="change-7" priority="P1" source_item="brainstorm action ">
 <file path="src/boost_shap_gii/infer.py" action="modify" />
 <description>(a) Config-gate the existing run_shap_pipeline(ctx) call with shap.compute_global_on_inference; (b) after inference evaluation, invoke indiv_reports.generate_indiv_reports(mode="inference") loading cached refits from train_dir/bootstrap_refits/.</description>
 <spec><![CDATA[
Location 1 (config-gate on global SHAP):

 Current: infer.py contains a call to run_shap_pipeline(ctx) for population-level GII
 on the inference dataset.

 Change: wrap that call in:

 if config["shap"].get("compute_global_on_inference", False):
 run_shap_pipeline(ctx)
 print("[INFO] Global SHAP analysis emitted on inference dataset "
 "(shap.compute_global_on_inference=true).")
 else:
 print("[INFO] Global SHAP analysis on inference dataset skipped "
 "(shap.compute_global_on_inference=false; default).")

 This change preserves backward compatibility only when users explicitly set the new
 key to true. Users upgrading from prior versions will see the new default behavior
 (skip global SHAP on inference). This is documented in AID_LOG.md and
 INPUT_SPECIFICATION.md.

Location 2 (per-individual indiv_reports on inference):

 Insertion point: at end of infer main function, AFTER the global-SHAP config gate
 above and BEFORE the function returns.

 Add block:

 from.utils import validate_indiv_reports_config
 validate_indiv_reports_config(config)

 nboot_indiv = int(config["shap"]["indiv_ci_nboot"])
 if nboot_indiv > 0:
 from.indiv_reports import generate_indiv_reports, _load_bootstrap_cache_or_fail
 # Loud error if cache absent at train_dir:
 _load_bootstrap_cache_or_fail(train_dir)

 # Reuse the same sig_GII source: ALWAYS from train_dir (not infer_dir),
 # because sig_GII is a property of the trained model, not of inference data.
 sig_GII_main, sig_GII_interaction = _load_sig_GII_from_shap_stats(train_dir)

 generate_indiv_reports(run_dir=infer_dir, # per-individual outputs go to infer_dir
 train_dir=train_dir, # cache + outcome stats live in train_dir
 X_target=X_infer,
 ids_target=ids_infer,
 X_train=X_train, # loaded from train_dir/train_matrix.parquet
 y_target=y_infer, # may be None if inference data lacks outcomes
 task=task,
 outcome_cols=outcome_cols,
 nom_feats=nom_feats,
 config=config,
 mode="inference",
 sig_GII_main=sig_GII_main,
 sig_GII_interaction=sig_GII_interaction)
 print(f"[INFO] Inference indiv_reports/ emitted to {infer_dir}.")
 else:
 print("[INFO] shap.indiv_ci_nboot=0; skipping per-individual SHAP reports.")

 Add helper _load_sig_GII_from_shap_stats: place this helper in utils.py (NOT
 duplicated across predict.py and infer.py) so that both and import it from the
 same module. 's spec is amended to call utils._load_sig_GII_from_shap_stats
 rather than defining the helper in predict.py. This matches the codebase's existing
 pattern of cross-module helpers in utils.py.

 X_train must be loaded from train_dir/train_matrix.parquet; this is needed for OOB
 accounting of training individuals present ONLY when training-individual SHAP is to be
 re-emitted during inference (NOT the case here — inference mode aggregates all
 refits for inference individuals). However, X_train is still needed to determine which
 rows were "training" for the OOB-vs-all-refits selection logic in generate_indiv_reports;
 passing it ensures the function can distinguish training ids from inference ids even
 in inference mode if the two sets overlap (they generally should NOT, but the
 function must handle it correctly).

 If the overlap concern is negligible in practice (inference dataset is always
 disjoint from training), X_train argument is still required by the generate_indiv_reports
 signature for consistency — pass it regardless.

Location 3 (no other changes): the existing ensemble-prediction loop over
model_fold_*.cbm is preserved verbatim.
 ]]></spec>
 <dependencies> (module and validator must exist); (bootstrap cache must be built at predict.py time — infer.py is a pure consumer).</dependencies>
 <risk>medium — modifies infer.py default behavior for global SHAP (now skipped by default). Documented explicitly in.</risk>
 <rollback>git revert; prior infer.py behavior (always emit global SHAP, no indiv_reports) restored.</rollback>
 </change>

 <change id="change-8" priority="P1" source_item="brainstorm action ">
 <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
 <description>Migrate from positional CLI args (4-5 args) to config-driven (CONFIG_PATH required + optional RUN_DIR). Remove all plot titles. Add per-individual dot-plus-whisker plot generation that auto-discovers {RUN_DIR}/indiv_reports/ and emits plots/{id}_main_effects.png and plots/{id}_interactions.png.</description>
 <spec><![CDATA[
PART A: CLI argument migration
 Current (plot.R line 11-28, 62):
 args[1] = CONFIG_PATH
 args[2] = OUTCOME_RANGE # → read from config.plot.outcome_max
 args[3] = NEGATE_SHAP # → read from config.plot.negate_shap
 args[4] = Y_AXIS_LABEL # → read from config.plot.gii_y_label + gii_y_sublabel
 args[5] = RUN_DIR (optional) # preserved

 New:
 args[1] = CONFIG_PATH (required; fail-loud if missing)
 args[2] = RUN_DIR (optional; defaults to config.paths.output_dir)

 After loading config via yaml::read_yaml(CONFIG_PATH):
 Fail loudly on missing required plot.* keys:
 plot_cfg <- config$plot
 required_keys <- c("outcome_max", "negate_shap", "gii_y_label", "gii_y_sublabel",
 "indiv_y_label", "indiv_y_sublabel")
 missing <- setdiff(required_keys, names(plot_cfg))
 if (length(missing) > 0) {
 stop(sprintf("Missing required plot.* config keys: %s", paste(missing, collapse=", ")))
 }
 OUTCOME_MAX <- as.numeric(plot_cfg$outcome_max)
 NEGATE_SHAP <- as.logical(plot_cfg$negate_shap)
 GII_Y_LABEL <- plot_cfg$gii_y_label
 GII_Y_SUBLABEL <- plot_cfg$gii_y_sublabel
 INDIV_Y_LABEL <- plot_cfg$indiv_y_label
 INDIV_Y_SUBLABEL <- plot_cfg$indiv_y_sublabel

PART B: Remove all plot titles
 Identify every ggtitle and labs(title=...) call. Remove them (or set title = NULL
 explicitly and use element_blank in theme where needed). Y-axis labels retained
 from GII_Y_LABEL GII_Y_SUBLABEL INDIV_Y_LABEL INDIV_Y_SUBLABEL verbatim.

PART C: Existing GII plot emission
 Preserve the existing GII plot emission flow (left panel = M, right panel = V).
 Update its label consumption to use GII_Y_LABEL and GII_Y_SUBLABEL (instead of the
 old Y_AXIS_LABEL). Apply NEGATE_SHAP per existing behavior (sign-flip y-values only;
 colors and ordering unchanged, matching).

PART D: New per-individual plot emission
 Add a new section at the end of plot.R:

 # --- Per-individual SHAP plots (indiv_reports) ---
 indiv_dir <- file.path(RUN_DIR, "indiv_reports")
 if (dir.exists(indiv_dir)) {
 main_path <- file.path(indiv_dir, "main_effects.parquet")
 int_path <- file.path(indiv_dir, "interactions.parquet")
 if (file.exists(main_path)) {
 render_indiv_main_effects_plots(main_path, indiv_dir, INDIV_Y_LABEL,
 INDIV_Y_SUBLABEL, NEGATE_SHAP)
 }
 if (file.exists(int_path)) {
 render_indiv_interactions_plots(int_path, indiv_dir, INDIV_Y_LABEL,
 INDIV_Y_SUBLABEL, NEGATE_SHAP)
 }
 }

 New R functions:

 render_indiv_main_effects_plots(path, out_dir, y_label, y_sublabel, negate_flag):
 # Load long-format parquet (one row per individual × feature, all features).
 # Group by id (individual). For each individual:
 # - Filter to sig_GII=TRUE features (decision).
 # - Order features along x by this individual's RAW signed SHAP value:
 # most positive left → most negative right (descending signed rank).
 # - y-axis: shap_value_scaled with sign-flip applied IF negate_flag=TRUE
 # (y-values flip only; x-ordering and color anchored to raw signed SHAP).
 # - Whiskers: shap_value_ci_lo to shap_value_ci_hi (also sign-flipped if
 # negate_flag=TRUE; bounds may swap: after sign flip, new_lo = -old_hi,
 # new_hi = -old_lo).
 # - Color: diverging blue-positive red-negative, anchored to RAW signed SHAP
 # (NOT flipped by negate_flag). Use scale_color_gradient2 with midpoint=0,
 # low=red-dark, mid=white, high=blue-dark.
 # - Horizontal reference line at y=0.
 # - y-axis label: y_label, y-axis subtitle: y_sublabel (if non-empty).
 # - No title, no legend title. No caption on compliant plots.
 # - Output: {out_dir}/plots/{id}_main_effects.png at a consistent size (default
 # 10 × 5 inches at 300 dpi).
 # - Below-OOB-floor rendering per Resolution 10: if any feature for the individual has
 # oob_count < 50, omit whiskers (or draw with width=0) and add the caption
 # "CI unavailable (oob_count < 50); point estimate shown only." below the
 # x-axis (same font/styling as other plot text; no warning banner, no red
 # highlight). File naming and directory identical to compliant plots.

 render_indiv_interactions_plots(path, out_dir, y_label, y_sublabel, negate_flag):
 # Same as main_effects but for interaction pairs. x-axis label = feature_a × feature_b
 # (composite). Interactions parquet already filtered to sig_GII=TRUE at emission.
 # If no sig_GII interactions exist globally, interactions.parquet is empty (header
 # only); this function emits nothing and returns without error.

PART E: Parallelism
 Preserve existing foreach + doParallel usage for GII plots. For per-individual plots,
 parallelize over individuals using parallel::mclapply (Unix) with cores = config-derived
 n_jobs or detectCores - 1 (match the existing convention in plot.R for the GII loop).
 On Windows, fall back to sequential; this is consistent with the existing plot.R.

PART F: R-package imports
 No new imports needed (confirmed). Existing library calls unchanged.
 ]]></spec>
 <dependencies> (indiv_reports/ output must exist before plot.R can consume it; but plot.R runs in its own subcommand invocation, so build-time order is flexible).</dependencies>
 <risk>medium — significant rewrite of plot.R entry point; regression risk on existing GII plot rendering.</risk>
 <rollback>git revert; existing plot.R restored. run_boost-shap-gii.sh CLI contract compatibility: update the wrapper script's plot invocation in the same commit (includes this; see run_boost-shap-gii.sh hunk below).</rollback>
 <additional_file_changes>
 Also modify src/boost_shap_gii/scripts/run_boost-shap-gii.sh and src/boost_shap_gii/cli.py's cmd_plot to pass only CONFIG_PATH (+ optional RUN_DIR) to plot.R, removing the existing positional-arg pipeline. Remove any bash-side construction of OUTCOME_RANGE/NEGATE_SHAP/Y_AXIS_LABEL from the orchestrator.
 </additional_file_changes>
 </change>

 <change id="change-9" priority="P1" source_item="brainstorm action ">
 <file path="src/boost_shap_gii/cli.py" action="modify" />
 <description>Add mandatory check_env preflight at the start of every CLI subcommand (train, predict, infer, plot, check-env). check_env currently exists in check_env.py; promote to a gate that fails fast before any other work.</description>
 <spec><![CDATA[
Location: cli.py, at the top of each command handler (cmd_train, cmd_predict, cmd_infer, cmd_plot).
Exception: cmd_check_env is the check_env command itself; it invokes check_env but does
not need a preflight (would be circular).

For each of cmd_train, cmd_predict, cmd_infer, cmd_plot, insert at the top of the function:

 def cmd_<name>(args: argparse.Namespace) -> None:
 from.check_env import run_preflight
 run_preflight # raises SystemExit(2) on failure with clear remediation guidance
 #... existing body...

Add NEW function run_preflight in check_env.py. Per, check_env.py main is
nullary (no argparse) and delegates to module-level check_python and check_r
helpers. run_preflight directly invokes those same helpers:

 def run_preflight -> None:
 """Run all environment checks; exit with status 2 on failure.

 Intended to be called from within other CLI command handlers (cmd_train,
 cmd_predict, cmd_infer, cmd_plot) before any other work, so environment
 problems surface as a fast early exit with actionable guidance.

 On success: prints "[ENV] Environment preflight passed." and returns.
 On failure: check_python check_r have already printed the concrete list
 of missing packages with install commands; this function then calls
 sys.exit(2). Exit code 2 is distinct from main's sys.exit(1) so that CI and
 log scrapers can distinguish a preflight-gate failure from a standalone
 check-env invocation failure.
 """
 py_ok = check_python
 r_ok = check_r
 if not (py_ok and r_ok):
 sys.exit(2)
 print("[ENV] Environment preflight passed.")

R_DEPS is unchanged. main is untouched (remains the entry point for the
standalone `check-env` CLI subcommand).
 ]]></spec>
 <dependencies>none (independent of changes 1-8; can run in parallel with any other build group).</dependencies>
 <risk>low — additive; failure surfaces as an early exit with clear message, preventing downstream confusion.</risk>
 <rollback>git revert the cli.py hunks; run_preflight function in check_env.py is unused but harmless.</rollback>
 </change>

 <change id="change-10" priority="P2" source_item="brainstorm action ">
 <file path="INPUT_SPECIFICATION.md" action="modify" />
 <description>Add a new top-level section documenting the indiv_reports feature, config keys, output schema, and the hypothesis-generating inspection framing. Update section on shap.* config keys and plot.* config keys.</description>
 <spec><![CDATA[
Add new Section 10 (or appropriate successor; current Section 9 is outcome-distribution
diagnostics added in Session 6) titled:

 ## Section 10: Per-individual SHAP reports (indiv_reports)

With subsections:
 - 10.1 Purpose (individual-case inspection tool; explicit non-causal framing;
 reference to planned multi-arm experimental study as the causal-validation vehicle)
 - 10.2 Algorithm: Option E with coupled bootstrap
 - K outer CV folds; B coupled iterations; per-iteration ONE shared bootstrap sample s_b
 drives K fold-specific refits; total refits = K × B; per-fold HP from
 model_fold_{k}.cbm embedded parameters (retrieved via get_all_params)
 - Cluster-aware bootstrap (when cluster_ids present)
 - Point estimates (deployed-product SHAP): OOF single-model SHAP from model_fold_{k_i}.cbm
 for training individuals; ensemble-mean SHAP across K original fold models for inference
 individuals; NOT a bootstrap-distribution statistic
 - CI aggregation (estimand-matched to point): training individuals → OOB iterations only,
 using fold-k_i refit per iteration (single-model CI, Breiman 2001 OOB); inference
 individuals → ensemble-averaged replicate per iteration across K coupled fold refits
 (ensemble-estimand CI); percentile CIs at 2.5/97.5 (Efron & Tibshirani 1993)
 - OOB floor = 50 individuals; below-floor individuals emit NaN CI bounds; point estimate
 is still emitted regardless of OOB count
 - Recommended B: minimum 2500 (inference Efron-tier, training near-Efron); 5000 for
 peer-review-facing runs (both sides solidly Efron-tier)
 - 10.3 Config keys
 - shap.indiv_ci_nboot (required, int, 0 disables feature)
 - shap.indiv_scaling_mode (required, one of {raw, sd, custom_value})
 - shap.indiv_scaling_value (required when scaling_mode=custom_value)
 - shap.compute_global_on_inference (bool, default false; distribution-shift diagnostic)
 - 10.4 Outputs
 - train_outcome_stats.json (written by train.py; consumed by predict.py/infer.py)
 - bootstrap_refits/ cache at train_dir (iter_{b:05d}/fold_{k}.cbm for coupled design;
 shared_indices.npz; bootstrap_metadata.json)
 - indiv_reports/ at both train_dir (from predict.py) and infer_dir (from infer.py):
 main_effects.parquet, interactions.parquet, predictions.parquet,
 indiv_reports_metadata.json, plots/{id}_main_effects.png, plots/{id}_interactions.png
 - Full schema tables for each parquet
 - 10.5 Interpretation guidance
 - CIs reflect training-sample sampling variability of the deployed product
 (OOF single-model for training, ensemble-mean for inference); estimand-matched
 to the point estimate via the coupled bootstrap design
 - CIs do NOT capture model-class uncertainty, HP-tuning uncertainty across new tuning
 runs, distribution shift, or label noise
 - Whiskers crossing y=0 indicate feature contribution not distinguishable from null
 at the 95% percentile level (NOT a formal hypothesis test)
 - Scaling modes: raw (model-output units), sd (Cohen's-d-like), custom_value
 (user-supplied user-supplied anchor)
 - Below-floor individuals (oob_count < 50): point estimate emitted, CI bounds NaN;
 interpret with caution (insufficient OOB resampling for stable tail percentiles)

Update existing sections:
 - shap.* config section: add cross-reference to Section 10 for new keys
 - plot.* config section: document the migrated keys; note the CLI simplification
 (CONFIG_PATH + optional RUN_DIR only)

References to add (inline citations at appropriate points):
 - Breiman L. (2001) Random Forests. Machine Learning 45: 5-32. (OOB aggregation)
 - Efron B., Tibshirani R.J. (1993) An Introduction to the Bootstrap. Chapman & Hall. (percentile CIs)
 - Covert I., Lee S.-I. (2021) Improving KernelSHAP: Practical Shapley value estimation
 using linear regression. AISTATS. (false-precision concern)
 - Ghassemi M., Oakden-Rayner L., Beam A.L. (2021) The false hope of current approaches
 to explainable artificial intelligence in health care. Lancet Digital Health 3: e745-e750.
 (hypothesis-generating vs causal framing)
 - Cumming G., Finch S. (2005) Inference by eye: confidence intervals and how to read
 pictures of data. American Psychologist 60: 170-180. (dot-plus-whisker visualization)
 ]]></spec>
 <dependencies> (documentation reflects the implemented behavior).</dependencies>
 <risk>low — documentation only.</risk>
 <rollback>git revert; documentation reverts to pre-feature state.</rollback>
 </change>

 <change id="change-11" priority="P2" source_item="brainstorm action ">
 <file path="README.md" action="modify" />
 <description>Add a brief user-facing section on the indiv_reports feature under the hypothesis-generating inspection framing. Keep the README high-level; link to INPUT_SPECIFICATION.md Section 10 for details.</description>
 <spec><![CDATA[
Add a new section after the existing Usage section (or appropriate location), titled:

 ## Per-individual SHAP reports

With content:
 - Brief purpose paragraph (2-3 sentences) positioning the feature as a
 individual-case inspection tool, NOT a prescriptive decision tool. Reference the
 planned multi-arm experimental study as the causal-validation vehicle.
 - Bullet list of required config keys (shap.indiv_ci_nboot, shap.indiv_scaling_mode,
 shap.indiv_scaling_value when applicable) with one-line descriptions.
 - Example config snippet (copied from the advanced config with the new keys populated).
 - Output directory structure (one paragraph describing train_dir/indiv_reports/ and
 infer_dir/indiv_reports/).
 - Link to INPUT_SPECIFICATION.md Section 10 for algorithmic details, CI interpretation,
 and limitations.

Explicit language to include verbatim:
 "Per-individual SHAP reports surface candidate predictor features for user consideration,
 not use-case decisions. Causal interpretation is reserved for experimental validation
 (e.g., a multi-arm experimental study comparing control, baseline-intervention, and model-informed-intervention)."

Do NOT include causal language anywhere else in the section.
 ]]></spec>
 <dependencies> (feature must be implemented before README documents it).</dependencies>
 <risk>low — documentation only.</risk>
 <rollback>git revert.</rollback>
 </change>

 <change id="change-12" priority="P2" source_item="brainstorm action (AID disclosure)">
 <file path="AID_LOG.md" action="modify" />
 <description>Append a session disclosure for the indiv_reports implementation cycle. Update metrics (test count, LOC added). Update the Version Release Notes section to reflect the bundled Option B release (patch + indiv_reports + Session 5/6 documentation updates).</description>
 <spec><![CDATA[
Append to AID_LOG.md a new session entry (date: 2026-04-24) following the existing
template pattern used for prior sessions. Key disclosures:

 - Session scope: indiv_reports feature (new module), pandas-3.0 Categorical
 fillna patch (carried forward from prior session, bundled per Option B), docs
 updates (INPUT_SPECIFICATION Section 10, README, AID_LOG), config migration
 (plot.R CLI simplified to CONFIG_PATH + optional RUN_DIR), R-package preflight
 elevated to mandatory gate on every CLI entry.
 - LLM tools used: Claude (Opus 4.7 orchestrator; Sonnet 4.6 for build-agent
 dispatch). No LLM co-authorship.
 - Algorithmic decisions:
 - Option E with coupled bootstrap (shared sample per iteration across K fold refits)
 for estimand-matched per-individual CIs
 - Point estimate = deployed-product SHAP: OOF single-model for training individuals,
 ensemble-mean for inference individuals
 - Minimum recommended B = 2500; 5000 for peer-review runs
 - OOB floor = 50 individuals (below floor: point emitted, CI NaN)
 - Path-dependent SHAP retained
 - hypothesis-generating inspection framing
 - Three-mode scaling: raw | sd | custom_value
 - Dot-plus-whisker plot format with signed-rank x-ordering
 - Test metrics: target total tests post-build = 461 + <new indiv_reports tests>
 (exact value populated in end-session report after /test phase completes).
 - Breaking change in default behavior: infer.py no longer emits population-level
 shap_analysis/ by default. Users who want it must set
 shap.compute_global_on_inference: true in config.

Do NOT include personal identifying information, absolute local paths, or usernames.
Follow the existing AID_LOG entry template for formatting.
 ]]></spec>
 <dependencies>changes 1-11 complete (AID_LOG disclosure reflects the full cycle).</dependencies>
 <risk>low — documentation only.</risk>
 <rollback>git revert.</rollback>
 </change>

 </changes>

 <execution_order>
 <step order="1" change_ids="" rationale="Config-defaults + validators must exist before downstream code references them. No external-file dependency." />
 <step order="2" change_ids="" rationale="Schema additions to config YAMLs and the new train_outcome_stats.json artifact in train.py — parallelizable across three distinct files." />
 <step order="3" change_ids="" rationale="CLI preflight change is independent of the feature implementation; run in parallel with config changes." />
 <step order="4" change_ids="" rationale="New module indiv_reports.py; consumes validators and outcome-stats artifact; prerequisite for." />
 <step order="5" change_ids="" rationale="predict.py orchestrates bootstrap cache + training-individual indiv_reports; depends on." />
 <step order="6" change_ids="" rationale="infer.py consumer + plot.R renderer can run in parallel; both depend on (and in the case of infer.py, on having built the cache at train time — but this is a runtime-ordering concern, not a build-ordering concern)." />
 <step order="7" change_ids="" rationale="Documentation last; reflects the fully implemented behavior." />
 </execution_order>

 <agent_dispatch_plan>
 <group id="group-1" order="1" change_ids="" files="src/boost_shap_gii/utils.py" rationale="Single file; single agent." />
 <group id="group-2" order="2" change_ids="" files="example_config_advanced.yaml" rationale="Single file." />
 <group id="group-3" order="2" change_ids="" files="example_config_minimal.yaml" rationale="Single file; parallel with." />
 <group id="group-4" order="2" change_ids="" files="src/boost_shap_gii/train.py" rationale="Single file; parallel with." />
 <group id="group-5" order="2" change_ids="" files="src/boost_shap_gii/cli.py,src/boost_shap_gii/check_env.py" rationale="Two files; preflight change is cross-cutting but isolated to these two." />
 <group id="group-6" order="3" change_ids="" files="src/boost_shap_gii/indiv_reports.py" rationale="New file; largest single agent task." />
 <group id="group-7" order="4" change_ids="" files="src/boost_shap_gii/predict.py" rationale="Depends on completing." />
 <group id="group-8" order="5" change_ids="" files="src/boost_shap_gii/infer.py" rationale="Depends on." />
 <group id="group-9" order="5" change_ids="" files="src/boost_shap_gii/scripts/plot.R,src/boost_shap_gii/scripts/run_boost-shap-gii.sh,src/boost_shap_gii/cli.py" rationale="plot.R primary + run_boost-shap-gii.sh + cli.py cmd_plot updates; parallel with. NOTE: cli.py edits in (preflight) and (plot subcommand CLI simplification) affect the same file; serialize after to avoid edit conflicts, OR merge into a single agent for cli.py." />
 <group id="group-10" order="6" change_ids="" files="INPUT_SPECIFICATION.md,README.md,AID_LOG.md" rationale="Docs; one agent per file or single docs agent." />
 </agent_dispatch_plan>

 <summary>
 <total_changes>12</total_changes>
 <estimated_loc>~900-1200 added/modified across 11 distinct files (src/boost_shap_gii/indiv_reports.py new ~400-600 LOC; predict.py ~100; infer.py ~80; plot.R ~200; train.py ~50; utils.py ~60; cli.py ~30; check_env.py ~30; configs ~40 lines across two YAMLs; docs ~150-200 lines across three markdown files).</estimated_loc>
 <risk_profile>Largest risks: (algorithmic correctness of OOB aggregation and memory profile at realistic N × B), (default-behavior change in infer.py global SHAP), (plot.R rewrite regression risk on GII plots). All other changes are additive or cosmetic.</risk_profile>
 <breaking_changes>
 1. infer.py default behavior change: global SHAP on inference now requires shap.compute_global_on_inference: true. Documented in.
 2. plot.R CLI surface simplified: args 2-4 (OUTCOME_RANGE, NEGATE_SHAP, Y_AXIS_LABEL) removed; corresponding config keys required. Users invoking plot.R directly with the old 4-arg signature will see a clear fail-loud error. run_boost-shap-gii.sh wrapper updated in the same change.
 3. All configs must now include the six new shap.* keys (indiv_ci_nboot, indiv_scaling_mode, indiv_scaling_value, compute_global_on_inference) and six new plot.* keys (outcome_max, negate_shap, gii_y_label, gii_y_sublabel, indiv_y_label, indiv_y_sublabel). Existing configs without them will fail validate_indiv_reports_config validate_plot_config (err-on-kill per project philosophy).
 </breaking_changes>
 <flagged_at_build_time>
 (None. All plan-level decisions resolved via resolutions 1-10. Plan is fully executable.)
 </flagged_at_build_time>
 </summary>
</implement_plan>
