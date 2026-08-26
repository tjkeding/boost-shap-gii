<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-26T15:26:00-04:00" />
  <pre_design_run>
    <total>895</total>
    <passed>862</passed>
    <failed>1</failed>
    <errors>32</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="TestTrainPyWiring::test_first_fold_alpha_computation_gated_correctly" file="tests/test_transformations_wiring.py" line="96">
        <error_type>AssertionError</error_type>
        <message>Source string 'if transform_module is not None and fold_idx == 0:' not found in train.py source</message>
        <traceback>tests/test_transformations_wiring.py:96: in test_first_fold_alpha_computation_gated_correctly
    assert "if transform_module is not None and fold_idx == 0:" in src
E   AssertionError</traceback>
      </failure>
      <failure test="TestTrainArtifacts / TestPredictArtifacts / TestPredictShapOutputs / TestInferArtifacts / TestInferShapOutputs (32 tests, shared fixture)" file="tests/test_dry_run_transformations.py" line="170">
        <error_type>ValueError</error_type>
        <message>Cross-fold alpha inconsistency detected: [2.339..., 2.416...]. The SHAP scale factor must be constant across folds for the global shap_scale_factor to be well-defined.</message>
        <traceback>tests/test_dry_run_transformations.py:170: in dry_run_artifacts
    train_main()
src/boost_shap_gii/train.py:1212: in main
    raise ValueError(
E   ValueError: Cross-fold alpha inconsistency detected: [2.339..., 2.416...]</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
    <disposition test="TestTrainPyWiring::test_first_fold_alpha_computation_gated_correctly" file="tests/test_transformations_wiring.py" classification="obsolete-test">
      <intended_contract>Per the prior design, alpha (the affine output_transform slope) was computed once at fold 0 and reused as the global shap_scale_factor for every fold. The recent per-fold-per-bootstrap transform-fitting change intentionally removed this gate: every fold now computes its own alpha from its own held-out slice, matching CatBoost's own per-fold re-estimation structure, and each fold's alpha is injected into that fold's own fold_meta record.</intended_contract>
      <current_test_claim>Asserted the literal source string "if transform_module is not None and fold_idx == 0:" is present in train.py, plus the (now renamed) variable assignments "shap_scale_factor = float(alpha_vec[0])" and "shap_scale_factor = 1.0".</current_test_claim>
      <evidence>src/boost_shap_gii/train.py:1013-1050 (fold loop; alpha computed unconditionally inside "if transform_module is not None:" for every fold_idx, variable renamed to fold_alpha); src/boost_shap_gii/train.py:1209-1219 (post-loop cross-fold aggregation now derives the single shap_scale_factor from all folds' alphas, not fold 0's alone).</evidence>
      <action>re-express: split into test_per_fold_alpha_computed_for_every_fold (verifies the fold_idx==0 gate is absent and the renamed fold_alpha assignments are present) and test_per_fold_alpha_injected_into_fold_meta (a NEW, strictly stronger assertion verifying every fold's alpha is written into that fold's own fold_meta before it is appended to all_fold_transform_meta). Neither new assertion references the post-loop cross-fold consistency check's specific tolerance value, since that logic is the subject of the product-bug disposition below and is expected to change under a future fix.</action>
    </disposition>
    <disposition test="TestTrainArtifacts / TestPredictArtifacts / TestPredictShapOutputs / TestInferArtifacts / TestInferShapOutputs (32 tests, shared dry_run_artifacts fixture)" file="tests/test_dry_run_transformations.py" classification="product-bug">
      <intended_contract>train.py's post-loop cross-fold alpha validation (added under the per-fold-per-bootstrap transform-fitting change) exists to guard against a shap_scale_factor that is not well-defined as a single global scalar. Given that guard's stated purpose, it must correctly distinguish "the transform's scale parameter is genuinely unstable across folds" (a real defect worth halting on) from "two independent finite-sample estimates of a data-dependent scale parameter differ by ordinary sampling variability" (expected, not exceptional).</intended_contract>
      <current_test_claim>N/A -- this is a production-code hard halt encountered during fixture setup, not a test assertion. The fixture's dry-run config exercises a z-score affine transform whose sigma is fit per fold from that fold's own training partition (tests/test_dry_run_transformations.py:52-64), exactly the per-fold-fitting design the recent change was built to support.</current_test_claim>
      <evidence>
        (1) src/boost_shap_gii/train.py:1209-1219: `np.allclose(fold_alphas, fold_alphas[0], rtol=1e-6)` requires two independently-fit, finite-sample scale-parameter estimates to agree to 6 decimal places. With cv_folds=2 and n=80, two disjoint ~40-row partitions produced sigma estimates of 2.339 and 2.416 (~3.2% apart), which is ordinary sampling variability for a standard deviation estimated from n approximately 40, not evidence of instability. This will hard-halt for essentially any affine transform whose scale parameter is legitimately fit per fold.
        (2) Even absent the hard halt, `shap_scale_factor = fold_alphas[0]` (train.py:1217) assigns whichever fold happens to run first as the single global scale factor. Consumption trace: predict.py:490,559 -> shap_utils.py:961,972 (`SHAP_vals = SHAP_vals * shap_scale_factor`) -> shap_utils.py:1314, confirmed to operate on the pooled out-of-fold SHAP matrix (one row per observation, sourced from whichever fold held it out). Applying one arbitrarily-chosen fold's alpha to every row is incorrect whenever fold-level alphas differ, which is the expected case for any per-fold-fit scale parameter -- structurally the same class of defect the per-fold-per-bootstrap change was built to eliminate, relocated from the input-transform side to the output-scale side.
      </evidence>
      <action>route-to-implement (P0). No test edit: the defect is in production code (train.py) and, on the current design, also implicates predict.py/shap_utils.py's single-global-scalar consumption of a per-fold-varying quantity. See action_items below. This finding was surfaced to and confirmed with the user before test design proceeded; the user elected to continue test design now (disposition-and-route) rather than halt mid-cycle to resolve the design question inline.</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>10</tests_created>
    <tests_modified>1</tests_modified>
    <files_created>
      <file path="tests/test_transformations_api.py" test_count="4" coverage_target="resolve_transform_path() (utils.py): no-transformations-key returns None, absolute path passthrough, relative path resolved against config[data][data_dir], relative path resolved against os.getcwd() fallback. Zero prior coverage; this utility is new (introduced to let per-bootstrap-resample workers re-resolve and re-import the transform script by path in a separate process)." />
      <file path="tests/test_indiv_reports_unit.py" test_count="6" coverage_target="Two new gaps closed. (1) _fit_and_save_refit's per-bootstrap-resample transform conditioning: confirms input_transform is fit on the bootstrap resample's own (duplicated, out-of-order) indices rather than the full training sample when transform_module_path is supplied, and confirms the legacy y_train_path-array-indexing path still applies when it is None -- captured via monkeypatching catboost.Pool to intercept the label kwarg before any real CatBoost fit occurs. (2) orchestrate_bootstrap_cache's transform-arg threading: confirms df_raw is serialized to a temp parquet and transform_module_path/tx_params/outcome_col are threaded into every submitted worker task when transform args are supplied, and confirms no serialization occurs when they are absent -- captured via a stub ProcessPoolExecutor that intercepts the first submitted task's args without spawning a real subprocess. (3) shap_scale_factor end-to-end: confirms shap_value_raw in the emitted main_effects.parquet scales linearly with shap_scale_factor via two calls to generate_indiv_reports against the same persisted cached models (bit-for-bit deterministic across calls), and confirms the default (omitted) value behaves identically to an explicit 1.0. Zero prior behavioral coverage existed for any of these three code paths; only source-inspection wiring tests existed for the surrounding call sites." />
    </files_created>
    <design_rationale>Coverage gaps were identified by cross-referencing the implement build report's changes (task-type guard, load assertion, metric labeling, shap_scale_factor threading, per-fold-per-bootstrap transforms) against existing test files. test_transformations_wiring.py's own stated scope explicitly excludes algorithmic verification (source-inspection only), and test_transformations_api.py / test_indiv_reports_unit.py had zero hits for resolve_transform_path, transform_module_path, df_raw_parquet_path, or shap_scale_factor prior to this design pass. The three new indiv_reports.py test classes were designed as behavioral (not source-inspection) tests, since these code paths involve genuine runtime computation (per-resample transform fitting, cross-process argument threading, linear SHAP scaling) where a wiring-only check would not catch a logic error.</design_rationale>
  </design_phase>
  <post_design_run>
    <total>906</total>
    <passed>874</passed>
    <failed>0</failed>
    <errors>32</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="TestTrainArtifacts / TestPredictArtifacts / TestPredictShapOutputs / TestInferArtifacts / TestInferShapOutputs (32 tests, shared fixture)" file="tests/test_dry_run_transformations.py" line="170">
        <error_type>ValueError</error_type>
        <message>Cross-fold alpha inconsistency detected: [2.339..., 2.416...]. The SHAP scale factor must be constant across folds for the global shap_scale_factor to be well-defined.</message>
        <traceback>tests/test_dry_run_transformations.py:170: in dry_run_artifacts
    train_main()
src/boost_shap_gii/train.py:1212: in main
    raise ValueError(
E   ValueError: Cross-fold alpha inconsistency detected: [2.339..., 2.416...]</traceback>
        <likely_cause>Unchanged from the pre-design run: the product-bug disposed above (rtol=1e-6 cross-fold gate mis-specified for a legitimately per-fold-fit scale parameter). Not addressed in this cycle by design -- resolution requires editing train.py, out of test's write scope, and is routed to /implement below.</likely_cause>
      </failure>
    </failures>
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>1</bugs_routed_to_implement>
    <recommendation>implement_fixes</recommendation>
  </summary>
  <action_items>
    <item priority="P0" target_mode="implement" finding_ref="test-design-disposition-2" description="Cross-fold shap_scale_factor logic in train.py:1209-1219 is scientifically mis-specified and blocks the entire affine-transform dry-run pipeline (32 tests). Two compounding issues: (1) rtol=1e-6 cross-fold consistency gate cannot be satisfied by two independent finite-sample estimates of a data-dependent scale parameter (e.g. per-fold sigma), so it will hard-halt for essentially any legitimately per-fold-fit affine transform; (2) even if relaxed, shap_scale_factor = fold_alphas[0] applies one arbitrarily-chosen fold's alpha uniformly to the pooled out-of-fold SHAP matrix (predict.py -> shap_utils.py:972), which is incorrect for rows originating from any other fold whenever fold-level alphas differ. This requires a genuine methodological decision (recommend /brainstorm before /implement) among candidates including: computing the reporting-side scale parameter once from the full dataset (decoupled from the leakage-sensitive per-fold-per-bootstrap input_transform used for model fitting); applying per-row, per-originating-fold scale factors to the pooled SHAP matrix before any pooled statistic is computed; or a pooled/averaged estimator (e.g. mean of fold alphas) paired with a materially looser, principled consistency check in place of rtol=1e-6." />
  </action_items>
</test_report>
