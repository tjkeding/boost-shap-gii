<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-24T16:36:00Z" />

  <pre_design_run>
    <total>863</total>
    <passed>853</passed>
    <failed>10</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="TestClusterBootstrapUnequalSizes::test_unequal_cluster_sizes_completes_with_fallback" file="tests/test_inference_shap.py" line="924">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:924 in _process_and_save_microdata -&gt; pd.DataFrame({"id": id_vals, ...}) raises ValueError</traceback>
      </failure>
      <failure test="TestClusterBootstrapSeedReproducibility::test_same_seed_same_results" file="tests/test_inference_shap.py" line="924">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above.</traceback>
      </failure>
      <failure test="TestClusterBootstrapTinyN::test_tiny_n_no_crash" file="tests/test_inference_shap.py" line="924">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above.</traceback>
      </failure>
      <failure test="TestInferenceShapParquetHasNRows::test_microdata_has_n_rows" file="tests/test_inference_shap.py" line="924">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above.</traceback>
      </failure>
      <failure test="TestInferenceMicrodataHasNRows::test_microdata_x_values_align" file="tests/test_inference_shap.py" line="924">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above.</traceback>
      </failure>
      <failure test="TestClusterBootstrapMicrodataNoFallback::test_large_n_microdata_has_n_rows_no_fallback" file="tests/test_inference_shap.py" line="924">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above.</traceback>
      </failure>
      <failure test="TestInferenceShadowShapSameStructure::test_shadow_uses_cluster_bootstrap_path" file="tests/test_inference_shap.py" line="924">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above.</traceback>
      </failure>
      <failure test="TestAllZeroShapValues::test_zero_shap_produces_zero_gii" file="tests/test_inference_shap.py" line="924">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above.</traceback>
      </failure>
      <failure test="TestZeroVarianceFeatureInInference::test_constant_feature_produces_zero_v" file="tests/test_inference_shap.py" line="924">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above.</traceback>
      </failure>
      <failure test="TestInferenceSingleObservation::test_n1_k3_no_crash" file="tests/test_inference_shap.py" line="924">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>Same root cause as above.</traceback>
      </failure>
    </failures>
  </pre_design_run>

  <failing_test_dispositions>
    <disposition test="TestClusterBootstrapUnequalSizes::test_unequal_cluster_sizes_completes_with_fallback (and 9 other tests in the same file, all sharing one of three call sites)" file="tests/test_inference_shap.py" classification="obsolete-test">
      <intended_contract>These tests directly invoke the internal function shap_utils.py::_run_bootstrap_pipeline (bypassing the production _run_shap_for_slice wiring) to unit-test cluster-bootstrap and K-fold-duplicate-collapse microdata behavior in isolation. Their synthetic data (K*N SHAP rows tiled/repeated from N unique observations) unambiguously models inference-mode semantics, as documented in the shared helper's own docstring ("Mimics inference mode: K*N SHAP rows, N-row X_raw...") and in inline comments at the two standalone call sites ("matching inference mode conventions").</intended_contract>
      <current_test_claim>Three call sites (tests/test_inference_shap.py line 118 inside the shared helper _run_pipeline_with_synthetics, used by 8 of the 10 failing tests; line 207 in TestClusterBootstrapUnequalSizes; line 740 in TestZeroVarianceFeatureInInference) call _run_bootstrap_pipeline with cluster_ids set but without the new inference_mode parameter introduced by the immediately preceding /implement cycle (boost-shap-gii_implement_build_20260824_160500.md), which fixed the predict-mode group-CV microdata crash reported in boost-shap-gii_test_20260824_152923.md. Because inference_mode now defaults to False, the K-duplicate-collapse groupby these three call sites rely on no longer executes, reproducing the identical length-mismatch crash (shap_utils.py:924) for a different reason: a stale, pre-fix calling convention rather than the original predict-mode/infer-mode conflation.</current_test_claim>
      <evidence>
        This is not a flaw in the implement fix's design. The fix replaced an implicit signal (cluster_ids is not None, which became ambiguous once predict-mode group-CV bootstrap also began populating cluster_ids for a different purpose) with an explicit inference_mode flag. An implicit, data-shape-derived alternative (e.g., inferring collapse-need by comparing row counts) would reintroduce exactly the silent-mislabeling risk the original bug report flagged for singleton clusters, so the explicit-flag design is the more defensible choice under the project's minimize-implicit-assumptions posture. The gap is that these three test-internal call sites, which bypass the sole production caller (_run_shap_for_slice, which does correctly thread inference_mode=inference_mode per the implement fix), were not updated as part of that fix, since Input Scoping and Scope Discipline restricted that implement cycle strictly to the production code path named in its action item. Verified: exactly 3 call sites exist in the file (grep-confirmed no others), and they account for exactly the 10 observed failures (8 via the shared helper + 1 + 1 via the two standalone calls).
      </evidence>
      <action>re-express: added inference_mode=True to all 3 call sites. No assertion was touched, weakened, or removed; every existing postcondition (N-row or cluster-count-row collapsed microdata, correct M/V/GII values, RuntimeWarning fallback behavior) is preserved exactly, since inference_mode=True is precisely the signal these tests' own synthetic data and docstrings already claimed to model.</action>
    </disposition>
  </failing_test_dispositions>

  <design_phase>
    <tests_created>0</tests_created>
    <tests_modified>3</tests_modified>
    <files_created>
    </files_created>
    <design_rationale>
      This invocation followed directly from an /implement cycle that fixed the P0 microdata length-mismatch bug identified in the prior /test report (boost-shap-gii_test_20260824_152923.md: predict-mode cv_strategy="group" crashing shap_utils.py's microdata-saving step). The pre-design run_suite (fresh baseline, not carried over from the prior report) surfaced a new regression: the fix's chosen mechanism (an explicit inference_mode parameter disambiguating K-duplicate-collapse semantics from predict-mode group-CV bootstrap semantics) was correctly wired into the sole production call site but left 3 test-internal direct-call sites in tests/test_inference_shap.py on their pre-fix calling convention, where the parameter silently defaulted to False and reproduced the crash. All 10 failures were traced to these exact 3 call sites (verified via exhaustive grep of every _run_bootstrap_pipeline invocation in the file) and dispositioned obsolete-test: the fix's design is sound (an explicit flag is more defensible than implicit shape-based inference, given the original bug's documented silent-mislabeling risk under implicit signals), and the test file's calling convention needed updating to match, with zero change to any assertion. This closes the loop opened by the original test report: tests/test_dry_run_no_transform_group_cv.py (the file that discovered the original bug) is confirmed passing in the post-design run below, and no other test in the 863-test suite was affected.
    </design_rationale>
  </design_phase>

  <post_design_run>
    <total>863</total>
    <passed>863</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures/>
  </post_design_run>

  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>proceed_to_document</recommendation>
  </summary>

  <action_items>
  </action_items>
</test_report>
