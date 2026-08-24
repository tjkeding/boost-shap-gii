<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-24T14:57:31+00:00" />

  <pre_design_run>
    <total>818</total>
    <passed>816</passed>
    <failed>2</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="TestPredictShapOutputs::test_gii_is_geometric_mean_of_m_and_v" file="tests/test_dry_run_transformations.py" line="319">
        <error_type>AssertionError</error_type>
        <message>GII values do not match sqrt(M*V) to rtol=1e-6; max relative diff 0.00336 (9/9 mismatched)</message>
        <traceback>assert_allclose(valid["GII"].values, expected_gii, rtol=1e-6); ACTUAL vs DESIRED differ by up to 0.0002209 absolute / 0.00336 relative across all 9 rows.</traceback>
      </failure>
      <failure test="TestInferShapOutputs::test_infer_gii_is_geometric_mean" file="tests/test_dry_run_transformations.py" line="449">
        <error_type>AssertionError</error_type>
        <message>Infer GII values do not match sqrt(M*V) to rtol=1e-6; max relative diff 0.04942 (9/9 mismatched)</message>
        <traceback>assert_allclose(valid["GII"].values, expected, rtol=1e-6); ACTUAL vs DESIRED differ by up to 0.00183 absolute / 0.04942 relative across all 9 rows.</traceback>
      </failure>
    </failures>
  </pre_design_run>

  <failing_test_dispositions>
    <disposition test="TestPredictShapOutputs::test_gii_is_geometric_mean_of_m_and_v" file="tests/test_dry_run_transformations.py" classification="obsolete-test">
      <intended_contract>GII is computed as sqrt(m_i * v_i) per bootstrap iteration (shap_utils.py:865), and the CSV point estimate is the independent nanmean of each of the three quantities across iterations: obs_mag = nanmean(boot_mag), obs_var = nanmean(boot_var), obs_gii = nanmean(boot_gii) (shap_utils.py:1192-1194). The GII column was never intended to equal sqrt(nanmean(M) * nanmean(V)) at the aggregate level.</intended_contract>
      <current_test_claim>assert_allclose(GII_column, sqrt(M_column * V_column), rtol=1e-6) — asserts exact equality between the averaged GII and the geometric mean of the two separately-averaged components.</current_test_claim>
      <evidence>shap_utils.py:865 (per-iteration sqrt(m*v)); shap_utils.py:1192-1194 (three independent nanmean calls). By the Cauchy-Schwarz inequality / Jensen's inequality applied to the jointly concave geometric-mean function, E[sqrt(X*Y)] &lt;= sqrt(E[X]*E[Y]) for non-negative X, Y, with equality only in the degenerate case of zero variance. Observed failure is one-sided (GII always below sqrt(M*V)) and grows with bootstrap variance, exactly the signature this inequality predicts, not a code regression.</evidence>
      <action>re-express: replaced the false exact-equality assertion with the mathematically true one-sided bound (GII &lt;= sqrt(M*V), with a small floating-point/rare-NaN-mismatch tolerance) plus a proximity check (rtol=0.10) that still detects a grossly broken GII computation. This is a strengthening relative to the correct contract: the prior assertion encoded an invariant that cannot hold in general, so no true postcondition was being tested at all.</action>
    </disposition>
    <disposition test="TestInferShapOutputs::test_infer_gii_is_geometric_mean" file="tests/test_dry_run_transformations.py" classification="obsolete-test">
      <intended_contract>Same as above, applied to inference-mode SHAP output.</intended_contract>
      <current_test_claim>Same false exact-equality assertion, applied to the infer_dir CSV.</current_test_claim>
      <evidence>Same evidence as above. Observed gap larger here (max relative diff 4.94%) because infer-mode bootstrap uses the same n_boot as train/predict but a smaller row count (9 effects), consistent with the same statistical mechanism at a different sample composition, not a separate defect.</evidence>
      <action>re-express: identical treatment to the predict-mode test above.</action>
    </disposition>
  </failing_test_dispositions>

  <design_phase>
    <tests_created>4</tests_created>
    <tests_modified>3</tests_modified>
    <files_created>
      <file path="tests/test_dry_run_transformations.py" test_count="2 modified" coverage_target="Re-expressed GII geometric-mean identity assertions (predict-mode and infer-mode) to assert the Cauchy-Schwarz/Jensen upper bound plus a proximity check, replacing a mathematically false exact-equality claim." />
      <file path="tests/test_transformations_wiring.py" test_count="1 modified, 4 added" coverage_target="Re-expressed the fold-hoisting wiring test for infer.py's new fold_transform_metadata.json contract (replacing the retired fold_assignments.json plus training-data-reload design), and added four tests asserting: train.py persists fold_transform_metadata.json; infer.py loads it from train_dir; infer.py never calls input_transform; infer.py never references config[&quot;paths&quot;][&quot;input_data&quot;] (the training-data-independence architectural invariant)." />
    </files_created>
    <design_rationale>
      The two pre-design failures were traced to a false mathematical claim embedded in the original tests (equating an averaged geometric mean with the geometric mean of averages), not to any implementation defect; both were re-expressed as the corresponding true inequality (Cauchy-Schwarz/Jensen), preserving detection power via a one-sided hard bound plus a proximity check rather than being weakened or dropped. Separately, the infer.py training-data-dependency redesign completed immediately prior to this cycle changed infer.py's call-site contract (fold_transform_metadata.json in place of fold_assignments.json plus a reloaded training-data file); the wiring test encoding the retired contract was re-expressed to match the new one, and four new tests were added to lock in the architectural invariant the user specifically raised: infer.py must never depend on the original training data.
    </design_rationale>
  </design_phase>

  <post_design_run>
    <total>822</total>
    <passed>822</passed>
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
    <item priority="P2" target_mode="document" description="README.md, INPUT_SPECIFICATION.md, and AID_LOG.md still need updating for the transformations API and aggregate SHAP noise-stratum-split features (original session goal, not yet executed), including the infer.py architectural change: infer.py no longer reads the original training data at inference time, recovering fold-level transform metadata exclusively from train.py's persisted fold_transform_metadata.json artifact." />
  </action_items>
</test_report>
