<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-11T23:45:00Z" />
  <pre_design_run>
    <total>625</total>
    <passed>621</passed>
    <failed>4</failed>
    <errors>0</errors>
    <coverage_pct></coverage_pct>
    <failures>
      <failure test="test_catboost_user_param_allowlist_contents" file="tests/test_indiv_reports_unit.py" line="55">
        <error_type>AttributeError</error_type>
        <message>module 'boost_shap_gii.indiv_reports' has no attribute '_CATBOOST_USER_PARAM_ALLOWLIST'</message>
        <traceback>AttributeError: module 'boost_shap_gii.indiv_reports' has no attribute '_CATBOOST_USER_PARAM_ALLOWLIST'</traceback>
      </failure>
      <failure test="TestExtractUserLevelParams::test_filters_out_non_allowlisted_keys" file="tests/test_indiv_reports_unit.py" line="76">
        <error_type>AssertionError</error_type>
        <message>assert output equals allowlist-filtered dict; output instead retains blocklist-only-excluded keys</message>
        <traceback>AssertionError: assert {'_internal_flag': True, 'thread_count': 16, ...} == {'depth': 6, ...}</traceback>
      </failure>
      <failure test="TestExtractUserLevelParams::test_all_unknown_keys_returns_empty" file="tests/test_indiv_reports_unit.py" line="86">
        <error_type>AssertionError</error_type>
        <message>assert unknown keys filtered to empty dict; output instead passes them through</message>
        <traceback>AssertionError: assert {'another_internal_key': 'value', 'some_new_param': 42} == {}</traceback>
      </failure>
      <failure test="TestMemoryGuard::test_memory_overflow_raises" file="tests/test_indiv_reports_unit.py" line="null">
        <error_type>FileNotFoundError</error_type>
        <message>Expected 2 model_fold_*.cbm files in tmpdir/run, found 0.</message>
        <traceback>FileNotFoundError: Expected 2 model_fold_*.cbm files in .../test_memory_overflow_raises0/run, found 0.</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
    <disposition test="test_catboost_user_param_allowlist_contents" file="tests/test_indiv_reports_unit.py" classification="obsolete-test">
      <intended_contract>Verify the CatBoost refit-parameter filter constant, whatever its current strategy, is internally correct.</intended_contract>
      <current_test_claim>Asserted specific keys were members of _CATBOOST_USER_PARAM_ALLOWLIST, an 18-entry inclusion set.</current_test_claim>
      <evidence>Locked brainstorm decision T4/B2 (boost-shap-gii_brainstorm_20260811_214642.md): allowlist replaced with a 5-entry _CATBOOST_REFIT_BLOCKLIST per user directive "be very selective about what is blocked (we want maximum flexibility)". implement_plan change C8 executed this; git diff confirms _CATBOOST_USER_PARAM_ALLOWLIST no longer exists in indiv_reports.py.</evidence>
      <action>re-express: renamed to test_catboost_refit_blocklist_contents, asserting exact 5-entry blocklist set.</action>
    </disposition>
    <disposition test="TestExtractUserLevelParams::test_filters_out_non_allowlisted_keys" file="tests/test_indiv_reports_unit.py" classification="obsolete-test">
      <intended_contract>Verify _extract_user_level_params correctly filters a params dict per the active strategy.</intended_contract>
      <current_test_claim>Asserted only allowlisted keys survive filtering (inclusion-filter semantics).</current_test_claim>
      <evidence>Same locked decision as above; filter logic inverted from inclusion to exclusion in indiv_reports.py:87-89.</evidence>
      <action>re-express: renamed to test_filters_out_blocklisted_keys, asserting only the 5 blocklisted keys are excluded and all other keys (including previously-unlisted thread_count and _internal_flag) pass through.</action>
    </disposition>
    <disposition test="TestExtractUserLevelParams::test_all_unknown_keys_returns_empty" file="tests/test_indiv_reports_unit.py" classification="obsolete-test">
      <intended_contract>Verify novel/unrecognized CatBoost params are handled per the active strategy's flexibility guarantee.</intended_contract>
      <current_test_claim>Asserted unrecognized keys are dropped entirely (inclusion-filter semantics).</current_test_claim>
      <evidence>User directive explicitly requires "maximum flexibility for those users that know what they're doing"; blocklist strategy by design passes through anything not explicitly blocklisted.</evidence>
      <action>re-express: renamed to test_unknown_keys_pass_through, asserting novel keys are NOT dropped (strengthened postcondition: pins pass-through behavior explicitly rather than only asserting non-crash).</action>
    </disposition>
    <disposition test="TestMemoryGuard::test_memory_overflow_raises" file="tests/test_indiv_reports_unit.py" classification="aligned">
      <intended_contract>generate_indiv_reports must raise MemoryError when projected refit memory exceeds the configured budget (R12/R13 err-on-kill guard).</intended_contract>
      <current_test_claim>pytest.raises(MemoryError, match="would require") around a call with a forced 1KB memory budget.</current_test_claim>
      <evidence>Confirmed via git stash against pre-build code: identical failure reproduces, ruling out C1-C9 as the cause. Root cause: the _make_tiny_cache fixture never wrote model_fold_*.cbm files, and indiv_reports.py:834-838 raises FileNotFoundError on that precondition before the memory-guard logic under test ever executes. Pre-existing, unrelated fixture defect.</evidence>
      <action>User elected to fix in this pass (not deferred): _make_tiny_cache extended to train and save K minimal CatBoostRegressor models as model_fold_{i}.cbm, satisfying the precondition so the memory-guard assertion now actually executes. No assertion text was changed.</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>27</tests_created>
    <tests_modified>4</tests_modified>
    <files_created>
      <file path="tests/test_aggregate_shap.py" test_count="27" coverage_target="Aggregate SHAP feature (implement_plan C2, C5, C6, C7): _validate_aggregate_shap (6 invariants), _aggregate_effects (singleton/within-group/between-group/group-x-ungrouped aggregation with shadow equivalents, known-answer fixture), group-total X column NaN handling, _is_aggregate_effect classification, block-permutation (S1) statistical property and source-invariant checks." />
    </files_created>
    <design_rationale>
      Three tests in test_indiv_reports_unit.py were re-expressed to match the locked allowlist-to-blocklist architecture change (T4/B2); postconditions were preserved or strengthened, never weakened. One pre-existing, build-unrelated fixture defect (TestMemoryGuard) was fixed at the user's explicit direction after disposition as aligned/test-environment-issue.

      The aggregate SHAP feature (C2, C5, C6, C7) had zero test coverage prior to this design pass, per the brainstorm action item. A new file provides known-answer coverage for the pure, directly-callable functions (_validate_aggregate_shap, _aggregate_effects, _is_aggregate_effect). Constructing the known-answer fixture for _aggregate_effects surfaced a genuine implementation/spec mismatch: the group-total X column is built via DataFrame.sum(axis=1), whose pandas default (skipna=True) does not propagate NaN, contradicting the implement_plan C5 spec's explicit NaN-propagation requirement. The test encodes the intended contract (per Test Design Discipline's product-bug routing rule) rather than being weakened to match the current behavior; see the routed action item below.

      Block-permutation (C3, C4) is embedded inline in non-extractable scopes (train.py's CLI-level main(), and a nested closure in shap_utils.py's _run_shap_for_slice), making direct unit-execution of the production code infeasible without either a disproportionately expensive full-pipeline integration test or a src-level refactor outside /test's write scope. Per explicit user direction, coverage here consists of (a) an algorithmic property test validating the specified shared-permutation-index behavior's statistical correctness in isolation, and (b) a structural source-invariant check confirming the actual production files still implement that pattern. Extraction of the duplicated logic into a shared, independently-testable helper is routed as a follow-up action item.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>652</total>
    <passed>651</passed>
    <failed>1</failed>
    <errors>0</errors>
    <coverage_pct></coverage_pct>
    <failures>
      <failure test="TestGroupTotalNanPropagation::test_nan_in_one_constituent_yields_nan_group_total" file="tests/test_aggregate_shap.py" line="251">
        <error_type>AssertionError</error_type>
        <message>group-total X column must be NaN when any constituent is NaN (implement_plan C5 spec); DataFrame.sum(axis=1) defaults to skipna=True and silently drops the missing constituent instead.</message>
        <traceback>AssertionError: assert np.False_ == True; where np.False_ = np.isnan(np.float64(20.0)); tests/test_aggregate_shap.py:251</traceback>
        <likely_cause>_aggregate_effects (shap_utils.py, added by implement change C5) computes group-total X columns via X_stacked[existing].sum(axis=1) without min_count=len(existing). Pandas' default skipna=True treats a missing constituent as 0 rather than propagating NaN through the sum, understating missingness in the group total and in the downstream nan_mask computed immediately afterward. This is a genuine deviation from the C5 spec's explicit "NaN propagation" requirement, not a test artifact.</likely_cause>
      </failure>
    </failures>
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>1</bugs_routed_to_implement>
    <recommendation>implement_fixes</recommendation>
  </summary>
  <action_items>
    <item priority="P0" target_mode="implement" description="Fix NaN propagation in the group-total X column computed by _aggregate_effects (shap_utils.py, near line 575): replace X_stacked[existing].sum(axis=1) with X_stacked[existing].sum(axis=1, min_count=len(existing)) so that any NaN constituent yields a NaN group total, matching the implement_plan C5 spec. This also affects the nan_mask computed immediately downstream (shap_utils.py, post-C6 call site), which currently undercounts missingness for aggregate-group individuals with partial constituent missingness. Covered by tests/test_aggregate_shap.py::TestGroupTotalNanPropagation::test_nan_in_one_constituent_yields_nan_group_total, which will pass once fixed; do not weaken or remove this assertion." />
    <item priority="P2" target_mode="implement" description="Extract the duplicated block-permutation logic (implement_plan C3 in train.py's shadow-generation loop inside main(), and C4 in shap_utils.py's _process_boruta_fold closure inside _run_shap_for_slice) into a single shared, independently-importable helper (e.g., a _block_permute_shadow(df_train, df_val, agg_groups, rng) function in utils.py or shap_utils.py). This eliminates the current code duplication between the two near-identical inline implementations and would let a future /test pass exercise the actual production code path directly, rather than the algorithmic mirror and source-text invariant checks in tests/test_aggregate_shap.py::TestBlockPermutationProperty / TestBlockPermutationSourceInvariant, which validate the specified algorithm and its structural presence in source but do not execute the production code path itself." />
  </action_items>
</test_report>
