<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-12T11:42:01Z" />
  <pre_design_run>
    <total>652</total>
    <passed>650</passed>
    <failed>2</failed>
    <errors>0</errors>
    <coverage_pct></coverage_pct>
    <failures>
      <failure test="TestBlockPermutationSourceInvariant::test_train_py_shares_permutation_index_across_group_members" file="tests/test_aggregate_shap.py" line="359">
        <error_type>AssertionError</error_type>
        <message>assert 'for group_name, members in agg_groups.items():' in train.py source</message>
        <traceback>AssertionError: inline block-permutation pattern no longer present in train.py; logic refactored to shared _block_permute_shadow helper in utils.py (implement build C2, this session).</traceback>
      </failure>
      <failure test="TestBlockPermutationSourceInvariant::test_shap_utils_py_shares_permutation_index_across_group_members" file="tests/test_aggregate_shap.py" line="367">
        <error_type>AssertionError</error_type>
        <message>assert 'for group_name, members in agg_groups.items():' in shap_utils.py source</message>
        <traceback>AssertionError: inline block-permutation pattern no longer present in shap_utils.py; logic refactored to shared _block_permute_shadow helper in utils.py (implement build C2, this session).</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
    <disposition test="TestBlockPermutationSourceInvariant::test_train_py_shares_permutation_index_across_group_members" file="tests/test_aggregate_shap.py" classification="obsolete-test">
      <intended_contract>Confirm that train.py's shadow generation uses block-permutation (shared permutation index per aggregate group) rather than independent per-column permutation.</intended_contract>
      <current_test_claim>Asserts specific inline source patterns (for group_name, members in agg_groups.items():, perm_idx_train = rng.permutation(n_train), etc.) exist verbatim in train.py.</current_test_claim>
      <evidence>Implement build C2 (boost-shap-gii_implement_build_20260812_111658.md) extracted the inline block-permutation code from train.py into _block_permute_shadow in utils.py. train.py now calls _block_permute_shadow(X_train_shadow, agg_groups, rng) at line 995. The intended contract (block-permutation is used) is unchanged; the assertion mechanism (specific inline source strings) is obsolete.</evidence>
      <action>re-express: split into two tests: (1) test_utils_py_defines_block_permute_shadow_with_shared_index asserts _block_permute_shadow exists in utils.py with algorithmic markers (perm_idx = rng.permutation(n), df[c] = df[c].values[perm_idx]); (2) test_train_py_imports_and_calls_block_permute_shadow asserts train.py contains _block_permute_shadow call-site strings for both X_train_shadow and X_val_shadow. Postcondition strengthened: the new assertions verify both the helper definition and the call-site wiring, whereas the old assertions only verified inline patterns that could exist without being connected to the actual code path.</action>
    </disposition>
    <disposition test="TestBlockPermutationSourceInvariant::test_shap_utils_py_shares_permutation_index_across_group_members" file="tests/test_aggregate_shap.py" classification="obsolete-test">
      <intended_contract>Confirm that shap_utils.py's Boruta shadow generation uses block-permutation for aggregate groups.</intended_contract>
      <current_test_claim>Asserts specific inline source patterns (for group_name, members in agg_groups.items():, perm_idx = rng.permutation(n_val), etc.) exist verbatim in shap_utils.py.</current_test_claim>
      <evidence>Same implement build C2 refactor. shap_utils.py now calls _block_permute_shadow(X_val_shadow, agg_groups, rng) at line 1333.</evidence>
      <action>re-express: test_shap_utils_py_imports_and_calls_block_permute_shadow asserts shap_utils.py contains the _block_permute_shadow import and call-site string. Postcondition strengthened: verifies call-site wiring to the shared helper rather than asserting inline patterns.</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>0</tests_created>
    <tests_modified>5</tests_modified>
    <files_created></files_created>
    <design_rationale>
      The two failing source-invariant tests were re-expressed to match the refactored architecture (block-permutation logic extracted from inline to shared _block_permute_shadow helper in utils.py). The re-expressed TestBlockPermutationSourceInvariant class now has 3 tests (up from 2): one verifying the helper definition in utils.py with algorithmic markers, one verifying the train.py call-site wiring (both X_train_shadow and X_val_shadow calls), and one verifying the shap_utils.py call-site wiring. All postconditions are strengthened: the new assertions verify both the helper's algorithmic structure and the call-site wiring, whereas the old assertions only verified inline string patterns that could theoretically exist without being connected to the execution path.

      Additionally, TestBlockPermutationProperty's 3 tests were strengthened by replacing the static _block_permute mirror method (which re-implemented the algorithm independently) with direct calls to the production _block_permute_shadow function imported from boost_shap_gii.utils. The property tests now exercise the actual production code path rather than a test-local mirror of the algorithm.

      No new test files were created. No assertions were removed, weakened, or replaced with tautologies.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>653</total>
    <passed>653</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct></coverage_pct>
    <failures></failures>
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>proceed_to_document</recommendation>
  </summary>
  <action_items></action_items>
</test_report>
