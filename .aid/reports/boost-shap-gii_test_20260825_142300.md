<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-25T14:23:00Z" />
  <pre_design_run>
    <total>895</total>
    <passed>892</passed>
    <failed>3</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="TestC3TwoTierOrdinalCheck::test_train_py_has_tier2_block" file="tests/test_implementation_changes.py" line="160">
        <error_type>AssertionError</error_type>
        <message>assert "obs_frac" in content</message>
        <traceback>tests/test_implementation_changes.py:160: in test_train_py_has_tier2_block
    assert "obs_frac" in content
E   AssertionError</traceback>
      </failure>
      <failure test="TestC3TwoTierOrdinalCheck::test_predict_py_has_tier2_block" file="tests/test_implementation_changes.py" line="169">
        <error_type>AssertionError</error_type>
        <message>assert "obs_frac" in content</message>
        <traceback>tests/test_implementation_changes.py:169: in test_predict_py_has_tier2_block
    assert "obs_frac" in content
E   AssertionError</traceback>
      </failure>
      <failure test="TestC3TwoTierOrdinalCheck::test_infer_py_has_tier2_block" file="tests/test_implementation_changes.py" line="177">
        <error_type>AssertionError</error_type>
        <message>assert "obs_frac" in content</message>
        <traceback>tests/test_implementation_changes.py:177: in test_infer_py_has_tier2_block
    assert "obs_frac" in content
E   AssertionError</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
    <disposition test="TestC3TwoTierOrdinalCheck::test_train_py_has_tier2_block" file="tests/test_implementation_changes.py" classification="obsolete-test">
      <intended_contract>The ordinal-coercion pathway in train.py implements two-tier unknown-value validation, including an observation-level fraction check (obs_frac/n_unknown_obs) that warns when more than 10% of non-missing observations fall outside the configured YAML levels.</intended_contract>
      <current_test_claim>assert "obs_frac" in content; assert "n_unknown_obs" in content; assert "0.10" in content or "0.1" in content, where content is the raw source text of train.py.</current_test_claim>
      <evidence>The prior /implement build (change C7, boost-shap-gii_implement_plan_20260825_140200.md) extracted the two-tier validation logic, including the obs_frac/n_unknown_obs computation, out of train.py/predict.py/infer.py into a single shared function coerce_ordinal_column in utils.py (utils.py lines 164-169). train.py now calls coerce_ordinal_column(X[c], levels, c) (train.py line 899) rather than containing the tier-2 logic inline. This was a deliberate, user-approved refactor, not a regression.</evidence>
      <action>re-express: assert train.py invokes coerce_ordinal_column, and assert utils.py contains the tier-2 obs_frac/n_unknown_obs/threshold logic. This is a strictly stronger postcondition than the original substring check, since it verifies actual wiring to functioning logic rather than mere text presence.</action>
    </disposition>
    <disposition test="TestC3TwoTierOrdinalCheck::test_predict_py_has_tier2_block" file="tests/test_implementation_changes.py" classification="obsolete-test">
      <intended_contract>The ordinal-coercion pathway in predict.py implements two-tier unknown-value validation, including the observation-level fraction check.</intended_contract>
      <current_test_claim>assert "obs_frac" in content; assert "n_unknown_obs" in content, where content is the raw source text of predict.py.</current_test_claim>
      <evidence>Same C7 extraction as above; predict.py now calls coerce_ordinal_column(df_raw[c], levels, c) (predict.py line 171).</evidence>
      <action>re-express: assert predict.py invokes coerce_ordinal_column, and assert utils.py contains the tier-2 logic.</action>
    </disposition>
    <disposition test="TestC3TwoTierOrdinalCheck::test_infer_py_has_tier2_block" file="tests/test_implementation_changes.py" classification="obsolete-test">
      <intended_contract>The ordinal-coercion pathway in infer.py implements two-tier unknown-value validation, including the observation-level fraction check.</intended_contract>
      <current_test_claim>assert "obs_frac" in content; assert "n_unknown_obs" in content, where content is the raw source text of infer.py.</current_test_claim>
      <evidence>Same C7 extraction as above; infer.py now calls coerce_ordinal_column(df_raw[c], levels, c) (infer.py line 165).</evidence>
      <action>re-express: assert infer.py invokes coerce_ordinal_column, and assert utils.py contains the tier-2 logic.</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>0</tests_created>
    <tests_modified>3</tests_modified>
    <files_created>
    </files_created>
    <design_rationale>All 3 failures traced to a single cause: the prior implement:build session's C7 change extracted duplicated ordinal-coercion logic (including the tier-2 obs_frac validation) from train.py, predict.py, and infer.py into a shared utils.py function, coerce_ordinal_column. The pre-existing marker tests checked for the literal string "obs_frac" in each caller file's source text, which is no longer true post-extraction even though the underlying validation behavior is fully intact and unchanged. Re-expressed each assertion to check (a) the caller file wires in coerce_ordinal_column, and (b) utils.py contains the tier-2 logic. This strengthens the original postcondition: the new assertions verify actual behavioral wiring rather than mere text-presence, and correctly locate the tier-2 logic in its new single-source-of-truth location. No other coverage gaps were identified; no other test files required modification.</design_rationale>
  </design_phase>
  <post_design_run>
    <total>895</total>
    <passed>895</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
    </failures>
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>proceed_to_document</recommendation>
  </summary>
  <action_items>
  </action_items>
</test_report>
