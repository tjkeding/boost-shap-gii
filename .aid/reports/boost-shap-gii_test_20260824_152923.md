<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-24T15:29:23Z" />

  <pre_design_run>
    <total>822</total>
    <passed>822</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures/>
  </pre_design_run>

  <failing_test_dispositions>
    <disposition test="TestGroupCvTrainArtifacts::test_no_transform_config_written (and 10 other tests in the same file, all failing at fixture setup)" file="tests/test_dry_run_no_transform_group_cv.py" classification="product-bug">
      <intended_contract>predict.py's OOF SHAP pipeline must complete successfully when cv_strategy="group" is combined with the default (nonzero) shap.output_microdata_n setting, producing well-formed per-observation microdata parquets, exactly as it does for every other task type and CV strategy already covered end-to-end.</intended_contract>
      <current_test_claim>N/A — this disposition did not originate from a pre-design baseline failure (the pre-design run was 822/822 clean). It was discovered while authoring test_dry_run_no_transform_group_cv.py during this design phase: the fixture itself (train + predict + infer) raises during predict_main(), so every test depending on it errors at setup. The disposition framework is applied here by the same underlying principle as the pre-design ledger: a newly-authored test that correctly encodes the intended contract and fails due to a genuine implementation defect is a product-bug, not grounds for weakening the test.</current_test_claim>
      <evidence>
        src/boost_shap_gii/shap_utils.py, function _run_bootstrap_pipeline, microdata-saving block (~lines 1330-1345). The block's single `if original_cluster_ids is not None:` branch conflates two distinct semantics of cluster_ids:
        (1) infer.py (always active): cluster_ids = each SHAP row's original per-observation index, used to collapse K ensemble-model duplicates back to N rows via groupby(level=0).mean(); ids (length N) correctly matches the post-groupby result.
        (2) predict.py with cv_strategy="group" (shap_utils.py ~line 1489: cluster_ids = groups): cluster_ids holds the user's group_column values, e.g. 20 unique cluster labels repeated across 80 observations. The same groupby(level=0).mean() collapses 80 rows to 20 (one per cluster, not per observation), but ids_micro = ids is left at its original length-80 array. _process_and_save_microdata (shap_utils.py:924) then builds pd.DataFrame({"id": id_vals (len 80), "effect_name": eff, "shap_value": phi (len 20)}), raising ValueError: All arrays must be of the same length.
        Verified in isolation: with shap.output_microdata_n=0 and zero significant effects (n_save=0, early return before the crash site), the identical cv_strategy="group" pipeline completes cleanly through the grouped-cluster bootstrap and SHAP significance stages — confirming the defect is isolated to the microdata-saving step's id/length reconciliation, not the cluster-bootstrap significance mechanism itself.
        Severity: shap.output_microdata_n defaults to 10 (nonzero) in utils.py when unset, so this crashes by default for any real (multi-member) cv_strategy="group" usage the moment any effect is significant or output_microdata_n > 0. For the degenerate case of singleton clusters, the same code would not crash but would silently mislabel microdata rows (id array misaligned against cluster-grouped SHAP values), a silent correctness defect rather than a crash.
      </evidence>
      <action>route-to-implement: see action_items below. Test left unmodified (output_microdata_n=2, matching the other four new dry-run files' convention) so it continues to fail visibly until the underlying defect is fixed. No assertion was weakened, skipped, or replaced with a tautology.</action>
    </disposition>
  </failing_test_dispositions>

  <design_phase>
    <tests_created>41</tests_created>
    <tests_modified>0</tests_modified>
    <files_created>
      <file path="tests/test_dry_run_binary_classification.py" test_count="11" coverage_target="First end-to-end (train-&gt;predict-&gt;infer) exercise of task_type=binary_classification via main(); prior coverage was unit-level only. Verifies probability-bounded OOF/infer predictions, SHAP/GII sanity, and the no-transform code branch." />
      <file path="tests/test_dry_run_multiclass_classification.py" test_count="8" coverage_target="First end-to-end exercise of task_type=multiclass_classification (3 classes) via main(); prior coverage was a single unit-level probability-shape test. Verifies class_labels.json, per-class shap_&lt;label&gt; directory structure, and the probability simplex constraint on ensemble predictions." />
      <file path="tests/test_dry_run_multi_regression.py" test_count="9" coverage_target="First end-to-end exercise of task_type=multi_regression (2 outcomes) via main(); previously untested end-to-end. Verifies the auto z-score target_scaler.json, per-outcome shap_&lt;label&gt; directory structure, and multi-output prediction columns." />
      <file path="tests/test_dry_run_no_transform_group_cv.py" test_count="11" coverage_target="First end-to-end exercise of (a) the vanilla no-transform code path (every prior dry run had a transform active) and (b) cv_strategy=&quot;group&quot; / grouped-cluster bootstrap (Session 14 feature) via main(). Discovered a product-bug (see failing_test_dispositions above); all 11 tests currently error at fixture setup pending the fix." />
      <file path="tests/test_dry_run_plot_r.py" test_count="2" coverage_target="First actual Rscript execution of plot.R anywhere in the suite (prior coverage was source-inspection only). Runs plot.R as a real subprocess against genuine train+predict output and verifies clean exit plus the model-performance PNG artifact." />
    </files_created>
    <design_rationale>
      This design phase was triggered not by pre-design failures (baseline was 822/822 clean) but by a direct user request to close a coverage gap identified through review of the existing dry-run test's scope: only one pipeline configuration (regression, transform active, aggregate SHAP) had ever been exercised end-to-end via main(); every other task type, the no-transform path, group/cluster-bootstrap CV, and plot.R itself were either unit-tested in isolation or only source-inspected. Five new dry-run fixtures were added, each mirroring test_dry_run_transformations.py's low-cost profile (n approx 80-90, cv_folds=2, tuning n_iter=5, n_boot=10) to keep total suite runtime low per the user's explicit "brief, low-cost" requirement. Four of the five fixtures passed cleanly on first execution. The fifth (no-transform + group CV) surfaced a genuine, previously-latent product defect in the interaction between cv_strategy="group" and the default per-individual microdata output, exactly the kind of finding this design phase existed to surface before the user proceeds to /document and /publish. Per explicit user instruction (asked via clarifying question), the test was left in its originally-designed, failing state rather than adjusted to avoid the crash, and the defect was routed to /implement with a precise, verified root-cause citation.
    </design_rationale>
  </design_phase>

  <post_design_run>
    <total>863</total>
    <passed>852</passed>
    <failed>0</failed>
    <errors>11</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="TestGroupCvTrainArtifacts::test_no_transform_config_written" file="tests/test_dry_run_no_transform_group_cv.py" line="161">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:1354 in _run_bootstrap_pipeline -&gt; shap_utils.py:924 in _process_and_save_microdata -&gt; pd.DataFrame({"id": id_vals, "effect_name": eff, "shap_value": phi}) -&gt; pandas construction.py:643 _extract_index raises ValueError</traceback>
        <likely_cause>Fixture-level failure (train + predict + infer run once, module-scoped): predict_main() raises during the microdata-saving step described in failing_test_dispositions above. All 11 tests in this file depend on the same fixture and therefore error identically at setup, not 11 independent defects.</likely_cause>
      </failure>
      <failure test="TestGroupCvTrainArtifacts::test_no_fold_transform_metadata_written" file="tests/test_dry_run_no_transform_group_cv.py" line="161">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above (fixture setup failure).</traceback>
        <likely_cause>Same as above.</likely_cause>
      </failure>
      <failure test="TestGroupCvTrainArtifacts::test_models_saved" file="tests/test_dry_run_no_transform_group_cv.py" line="161">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above (fixture setup failure).</traceback>
        <likely_cause>Same as above.</likely_cause>
      </failure>
      <failure test="TestGroupCvTrainArtifacts::test_fold_assignments_respect_group_integrity" file="tests/test_dry_run_no_transform_group_cv.py" line="161">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above (fixture setup failure).</traceback>
        <likely_cause>Same as above.</likely_cause>
      </failure>
      <failure test="TestGroupCvTrainArtifacts::test_group_column_excluded_from_features" file="tests/test_dry_run_no_transform_group_cv.py" line="161">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above (fixture setup failure).</traceback>
        <likely_cause>Same as above.</likely_cause>
      </failure>
      <failure test="TestGroupCvPredictShapOutputs::test_shap_stats_global_exists" file="tests/test_dry_run_no_transform_group_cv.py" line="161">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above (fixture setup failure).</traceback>
        <likely_cause>Same as above.</likely_cause>
      </failure>
      <failure test="TestGroupCvPredictShapOutputs::test_gii_values_finite_and_nonnegative" file="tests/test_dry_run_no_transform_group_cv.py" line="161">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above (fixture setup failure).</traceback>
        <likely_cause>Same as above.</likely_cause>
      </failure>
      <failure test="TestGroupCvPredictShapOutputs::test_bootstrap_distributions_saved_under_cluster_resampling" file="tests/test_dry_run_no_transform_group_cv.py" line="161">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above (fixture setup failure).</traceback>
        <likely_cause>Same as above.</likely_cause>
      </failure>
      <failure test="TestGroupCvInferArtifacts::test_infer_predictions_finite" file="tests/test_dry_run_no_transform_group_cv.py" line="161">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above (fixture setup failure).</traceback>
        <likely_cause>Same as above.</likely_cause>
      </failure>
      <failure test="TestGroupCvInferArtifacts::test_infer_shap_stats_global_exists" file="tests/test_dry_run_no_transform_group_cv.py" line="161">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above (fixture setup failure).</traceback>
        <likely_cause>Same as above.</likely_cause>
      </failure>
      <failure test="TestGroupCvInferArtifacts::test_infer_gii_values_finite_and_nonnegative" file="tests/test_dry_run_no_transform_group_cv.py" line="161">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above (fixture setup failure).</traceback>
        <likely_cause>Same as above.</likely_cause>
      </failure>
    </failures>
  </post_design_run>

  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>1</bugs_routed_to_implement>
    <recommendation>implement_fixes</recommendation>
  </summary>

  <action_items>
    <item priority="P0" target_mode="implement" description="Fix src/boost_shap_gii/shap_utils.py::_run_bootstrap_pipeline's microdata-saving block (~lines 1330-1345). The `if original_cluster_ids is not None:` branch conflates infer.py's K-duplicate-collapse semantics (correct: ids stays length-N, matching the post-groupby-mean result) with predict.py's cv_strategy=&quot;group&quot; semantics (broken: groupby(level=0).mean() collapses to n_unique_clusters rows, but ids_micro=ids is never reduced to match, causing a length-mismatch ValueError in _process_and_save_microdata at shap_utils.py:924 whenever any cluster has &gt;1 member and any effect is significant or output_microdata_n &gt; 0; for singleton clusters the same code would silently misalign id-to-row mapping instead of crashing). The fix must distinguish these two call sites (e.g., via an explicit mode flag rather than inferring intent from cluster_ids being non-None) and, for the predict-mode group-CV case, either produce cluster-level microdata correctly labeled by cluster id (not observation id) or explicitly document/guard that per-observation microdata output is unsupported under cv_strategy=&quot;group&quot;, raising a clear, actionable error instead of the current opaque pandas ValueError. After the fix, tests/test_dry_run_no_transform_group_cv.py (11 tests, currently erroring) should be re-run via /test to confirm resolution without modification to its assertions." />
  </action_items>
</test_report>
