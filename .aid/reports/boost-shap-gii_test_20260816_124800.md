<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-16T12:48:00-04:00" />
  <pre_design_run>
    <total>725</total>
    <passed>725</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct></coverage_pct>
    <failures>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>8</tests_created>
    <tests_modified>2</tests_modified>
    <files_created>
      <file path="tests/test_shap_utils.py" test_count="7" coverage_target="New _diagnose_spline_downgrades function (6 tests: degree-le-1 no-print, sufficient-knots no-print, downgraded feature name/count in output, nominal exclusion, shadow exclusion, multi-feature listing) plus 1 regression test locking in that _get_adaptive_knots_and_degree no longer prints on downgrade" />
      <file path="tests/test_preflight.py" test_count="1" coverage_target="Direct membership assertion that psutil is present in check_env.PYTHON_DEPS, mirroring the existing TestRDepsIncludesNewRequirements pattern for R_DEPS" />
      <file path="tests/test_package_structure.py" test_count="0" coverage_target="Extended existing test_dependencies_listed required-dependency list with psutil (pins presence in pyproject.toml)" />
      <file path="tests/test_shell_and_config.py" test_count="0" coverage_target="Extended existing test_key_dependencies_present required-dependency list with psutil (pins presence in environment.yaml)" />
    </files_created>
    <design_rationale>
      Two implemented changes required new coverage: (1) the new _diagnose_spline_downgrades function had zero prior test coverage since it did not exist before this cycle; (2) the psutil dependency addition across three manifest files (pyproject.toml, environment.yaml, check_env.py) had no test pinning its presence, following the exact pattern already established for R_DEPS in test_preflight.py and for other required dependencies in test_package_structure.py and test_shell_and_config.py. No integration-level test was added for the call site wiring _diagnose_spline_downgrades into run_shap_pipeline; the codebase's existing convention unit-tests comparable helper functions (_get_effect_stratum, _get_adaptive_knots_and_degree) directly rather than through full pipeline fixtures, and the call site itself is a single-line, low-risk addition.

      Design-phase self-correction: the initial version of test_lists_multiple_downgraded_features constructed a DataFrame from three arrays of unequal length (two length-7 arrays and one length-100 array), which raised ValueError at construction time on the first post-design run. This was a fixture bug in the newly written test itself (disposition: not a product-under-test failure), fixed in place by resizing the third array to length 7 (linspace(0, 10, 7), independently verified to retain 4 interior knots and therefore not trigger a downgrade) before the corrected post-design run reported below.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>733</total>
    <passed>733</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct></coverage_pct>
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
