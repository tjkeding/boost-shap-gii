<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-26T19:34:43Z" />
  <pre_design_run>
    <total>906</total>
    <passed>899</passed>
    <failed>7</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="TestTrainArtifacts::test_transform_config_has_all_fields" file="tests/test_dry_run_transformations.py" line="216">
        <error_type>AssertionError</error_type>
        <message>Missing field: shap_scale_factor -- key absent from transform_config.json (dict has fold_shap_scale_factors but not shap_scale_factor)</message>
        <traceback>tests/test_dry_run_transformations.py:216: AssertionError: Missing field: shap_scale_factor</traceback>
      </failure>
      <failure test="TestTrainArtifacts::test_shap_scale_factor_is_sigma" file="tests/test_dry_run_transformations.py" line="241">
        <error_type>KeyError</error_type>
        <message>KeyError: 'shap_scale_factor' -- transform_config.json missing top-level shap_scale_factor key</message>
        <traceback>tests/test_dry_run_transformations.py:241: KeyError: 'shap_scale_factor'</traceback>
      </failure>
      <failure test="TestShapScaleFactorApplication::test_shap_scale_factor_scales_point_shap_linearly" file="tests/test_indiv_reports_unit.py" line="1363">
        <error_type>TypeError</error_type>
        <message>generate_indiv_reports() got an unexpected keyword argument 'shap_scale_factor'</message>
        <traceback>tests/test_indiv_reports_unit.py:1363: TypeError: generate_indiv_reports() got an unexpected keyword argument 'shap_scale_factor'</traceback>
      </failure>
      <failure test="TestShapScaleFactorApplication::test_default_shap_scale_factor_is_noop" file="tests/test_indiv_reports_unit.py" line="1383">
        <error_type>TypeError</error_type>
        <message>generate_indiv_reports() got an unexpected keyword argument 'shap_scale_factor'</message>
        <traceback>tests/test_indiv_reports_unit.py:1383: TypeError: generate_indiv_reports() got an unexpected keyword argument 'shap_scale_factor'</traceback>
      </failure>
      <failure test="TestTrainPyWiring::test_transform_config_json_written_with_all_seven_fields" file="tests/test_transformations_wiring.py" line="131">
        <error_type>AssertionError</error_type>
        <message>assert '"shap_scale_factor"' in train.py source block -- field absent from tx_artifact dict literal in train.py</message>
        <traceback>tests/test_transformations_wiring.py:131: AssertionError</traceback>
      </failure>
      <failure test="TestPredictPyWiring::test_shap_scale_factor_passed_conditionally" file="tests/test_transformations_wiring.py" line="183">
        <error_type>AssertionError</error_type>
        <message>assert 'if shap_scale_factor != 1.0:' in predict.py source -- conditional guard absent</message>
        <traceback>tests/test_transformations_wiring.py:183: AssertionError</traceback>
      </failure>
      <failure test="TestInferPyWiring::test_shap_scale_factor_passed_conditionally" file="tests/test_transformations_wiring.py" line="311">
        <error_type>AssertionError</error_type>
        <message>assert 'if shap_scale_factor != 1.0:' in infer.py source -- conditional guard absent</message>
        <traceback>tests/test_transformations_wiring.py:311: AssertionError</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
    <disposition test="TestTrainArtifacts::test_transform_config_has_all_fields" file="tests/test_dry_run_transformations.py" classification="obsolete-test">
      <intended_contract>transform_config.json must contain the complete set of metadata fields needed for downstream (predict.py, infer.py, indiv_reports.py) consumption to correctly back-transform SHAP values.</intended_contract>
      <current_test_claim>Asserted the field list includes the literal key "shap_scale_factor".</current_test_claim>
      <evidence>src/boost_shap_gii/train.py:1220-1228 (tx_artifact dict literal): the field is now "fold_shap_scale_factors" (a per-fold list), per the locked design decision (T1: A2 per-row fold-specific scaling) that replaced the single-scalar architecture this test was written against.</evidence>
      <action>re-express: swap the literal field name in the checked-field tuple from "shap_scale_factor" to "fold_shap_scale_factors".</action>
    </disposition>
    <disposition test="TestTrainArtifacts::test_shap_scale_factor_is_sigma" file="tests/test_dry_run_transformations.py" classification="obsolete-test">
      <intended_contract>For a z-score standardization transform, each fold's affine scale parameter (alpha) must equal the sample standard deviation computed from that fold's own training partition.</intended_contract>
      <current_test_claim>Asserted only that the single global scalar tx["shap_scale_factor"] equals sigma computed from fold 0's training partition.</current_test_claim>
      <evidence>train.py:1210 now emits fold_alphas (a list, one entry per fold), not a single scalar. The old test validated only fold 0's alpha, which is precisely the blind spot that allowed an arbitrarily-chosen fold's alpha to be applied uniformly to every row under the prior design.</evidence>
      <action>re-express and strengthen: renamed to test_fold_shap_scale_factors_are_per_fold_sigma; validates EVERY fold's alpha against that fold's own training-partition sigma, not just fold 0. This is a strictly stronger postcondition than the original assertion.</action>
    </disposition>
    <disposition test="TestShapScaleFactorApplication::test_shap_scale_factor_scales_point_shap_linearly" file="tests/test_indiv_reports_unit.py" classification="obsolete-test">
      <intended_contract>generate_indiv_reports must scale point-level SHAP estimates in the emitted main_effects.parquet proportionally to the supplied scale factor(s), so back-transformed SHAP values are expressed in original outcome units.</intended_contract>
      <current_test_claim>Called generate_indiv_reports(shap_scale_factor=1.0) and generate_indiv_reports(shap_scale_factor=3.0), a keyword argument removed by the fix.</current_test_claim>
      <evidence>indiv_reports.py generate_indiv_reports signature now takes fold_shap_scale_factors (Optional[List[float]]), and requires bootstrap_alphas.npy to exist in the cache when it is not None.</evidence>
      <action>re-express: renamed to test_fold_shap_scale_factors_scale_point_shap_linearly; fixture extended to write bootstrap_alphas.npy; calls use fold_shap_scale_factors=[1.0,1.0] vs. [3.0,3.0] (uniform across folds, preserving the original linear-scaling postcondition exactly).</action>
    </disposition>
    <disposition test="TestShapScaleFactorApplication::test_default_shap_scale_factor_is_noop" file="tests/test_indiv_reports_unit.py" classification="obsolete-test">
      <intended_contract>Omitting the scale-factor argument must be behaviorally identical to passing the neutral (1.0) value explicitly.</intended_contract>
      <current_test_claim>Compared shap_scale_factor=1.0 (explicit) against the kwarg omitted entirely, using the old parameter name.</current_test_claim>
      <evidence>Same signature change as above.</evidence>
      <action>re-express: renamed to test_default_fold_shap_scale_factors_is_noop; same comparison structure (explicit uniform list vs. omitted kwarg), preserving the original no-op postcondition exactly.</action>
    </disposition>
    <disposition test="TestTrainPyWiring::test_transform_config_json_written_with_all_seven_fields" file="tests/test_transformations_wiring.py" classification="obsolete-test">
      <intended_contract>Source-inspection check that train.py's tx_artifact dict literal contains all seven required metadata field names.</intended_contract>
      <current_test_claim>Asserted the literal string '"shap_scale_factor"' appears within the tx_artifact block.</current_test_claim>
      <evidence>train.py:1227 (tx_artifact literal): the field is now '"fold_shap_scale_factors"'.</evidence>
      <action>re-express: swap the literal field-name string checked in the source-inspection list.</action>
    </disposition>
    <disposition test="TestPredictPyWiring::test_shap_scale_factor_passed_conditionally" file="tests/test_transformations_wiring.py" classification="obsolete-test">
      <intended_contract>predict.py must conditionally pass the SHAP scale-factor artifact into shap_ctx only when it differs from the no-op default, so the downstream pipeline performs zero extra work when transforms are absent or non-scaling.</intended_contract>
      <current_test_claim>Asserted the literal strings "if shap_scale_factor != 1.0:" and 'shap_ctx["shap_scale_factor"] = shap_scale_factor' appear in predict.py source.</current_test_claim>
      <evidence>predict.py:497-498 (verified by direct file read): the guard is now "if fold_shap_scale_factors is not None:" gating 'shap_ctx["fold_shap_scale_factors"] = fold_shap_scale_factors'.</evidence>
      <action>re-express: renamed to test_fold_shap_scale_factors_passed_conditionally; literal strings updated to match the new guard and assignment.</action>
    </disposition>
    <disposition test="TestInferPyWiring::test_shap_scale_factor_passed_conditionally" file="tests/test_transformations_wiring.py" classification="obsolete-test">
      <intended_contract>Same as the predict.py counterpart above, applied to infer.py.</intended_contract>
      <current_test_claim>Same legacy literal-string assertions, targeting infer.py source.</current_test_claim>
      <evidence>infer.py:577-578 (verified by direct file read): same new guard/assignment pattern as predict.py.</evidence>
      <action>re-express: renamed to test_fold_shap_scale_factors_passed_conditionally; literal strings updated.</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>8</tests_created>
    <tests_modified>7</tests_modified>
    <files_created>
      <file path="tests/test_shap_utils.py" test_count="3" coverage_target="New class TestPerRowFoldShapScaleFactors: the per-row fold-specific alpha vector construction added to _run_shap_for_slice (predict-mode fold-index alignment via fold_assignments, inference-mode np.repeat fold-block alignment with an explicit check ruling out the np.tile misalignment that would silently corrupt results, and the None no-op path). Zero prior coverage existed for this new construction logic; the merge-logic-replication pattern follows this file's existing TestInferenceModeMergeLogic convention." />
      <file path="tests/test_indiv_reports_unit.py" test_count="5" coverage_target="New class TestFitAndSaveRefitBootstrapAlpha (3 tests): _fit_and_save_refit's exact per-bootstrap alpha probe against a KNOWN fixed affine slope (3.5, not data-dependent), confirming exactness rather than approximation, plus its two no-op return paths (back_transform_shap=False; transform absent). New test in TestShapScaleFactorApplication: test_fold_shap_scale_factors_uses_originating_folds_alpha, a regression test proving non-uniform per-fold alphas produce exactly the fold-weighted average the accumulation formula predicts -- a result the prior single-scalar design could not express, directly targeting the original defect class. New test in TestOrchestrateBootstrapCacheTransformConditioning: test_back_transform_shap_and_is_affine_threaded_to_workers, confirming the two new worker-task-tuple positions (indices 13, 14) carry True when the caller passes both flags. Zero prior coverage existed for any of these three code paths." />
    </files_created>
    <design_rationale>The 7 pre-design failures were all disposed obsolete-test: the intended contract (SHAP values correctly scaled to original-outcome units) is unchanged, only the mechanism changed (single global scalar to per-fold array plus exact per-bootstrap alpha), per the locked A2 design decision from the upstream brainstorm and the six-change implement plan. One re-expression (test_shap_scale_factor_is_sigma) was strengthened beyond a literal rename: the original only ever validated fold 0's alpha, which is the exact blind spot that let the original single-scalar defect ship undetected. Beyond the obsolete-test re-expressions, three genuinely new code paths introduced by the implement build had zero direct test coverage: the per-row alpha vector construction in shap_utils.py, the exact per-bootstrap alpha probe in indiv_reports.py's _fit_and_save_refit, and the two-new-parameter threading through orchestrate_bootstrap_cache's worker task tuple. Coverage was added for all three, prioritizing the wiring/alignment risk class (fold-to-row indexing, tuple-position threading) that has been the source of three prior incorrect specifications in this feature area.</design_rationale>
  </design_phase>
  <post_design_run>
    <total>914</total>
    <passed>914</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures />
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>proceed_to_document</recommendation>
  </summary>
  <action_items />
</test_report>
