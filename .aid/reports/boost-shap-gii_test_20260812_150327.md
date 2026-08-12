<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-12T15:03:27Z" />
  <pre_design_run>
    <total>653</total>
    <passed>653</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures></failures>
  </pre_design_run>
  <failing_test_dispositions>
    No failing tests from the pre-design run; the per-failure ledger is empty. Test design
    proceeded directly to coverage-gap analysis against the three implemented changes
    (docstring fix, pre-flight CatBoost refit probe, inference-mode X_stacked tiling fix)
    per the implement build report.
  </failing_test_dispositions>
  <design_phase>
    <tests_created>5</tests_created>
    <tests_modified>1</tests_modified>
    <files_created>
      <file path="tests/test_indiv_reports_unit.py" test_count="5" coverage_target="New TestProbeAndStripRefitParams class: covers the previously-uncovered _probe_and_strip_refit_params helper (clean pass with no warning, single-param discovery and cross-fold stripping, multi-param discovery within the 5-retry cap, unparseable-TypeError safety break, regressor-vs-classifier task-based class selection)." />
      <file path="tests/test_shap_utils.py" test_count="1" coverage_target="Re-expressed test_x_stacked_takes_first_in_inference (renamed test_x_stacked_concatenates_all_folds_in_inference) to match the corrected inference-mode X_stacked contract: concatenates all K folds instead of taking fold 0 only, preserving per-fold shadow feature values." />
    </files_created>
    <design_rationale>
      All three action items from the implement build report were assessed for coverage gaps.
      The docstring-only fix requires no new test (existing TestNanSafeFdr already exercises
      the unchanged numerical behavior). The new _probe_and_strip_refit_params helper had zero
      existing coverage and was fully unit-tested using fake CatBoost-like classes that mimic
      CPython's real "unexpected keyword argument" TypeError message format, avoiding real
      CatBoost fits. The X_stacked tiling fix directly invalidated an existing test's assertion:
      test_x_stacked_takes_first_in_inference hand-replicated the OLD (buggy) "take first fold"
      logic inline and asserted it as the correct contract. Since the implement build changed the
      actual intended contract (CR finding F3), this test was re-expressed rather than left in
      place, where it would have continued to pass while documenting behavior that no longer
      matches the implementation. The re-expression strengthens the postcondition: the original
      test used K identical copies of the same DataFrame, which cannot distinguish
      "take first fold" from "concatenate all folds" (both produce equal-content single-fold-worth
      of data when folds are identical). The new version uses per-fold-distinct shadow columns
      specifically so a regression back to the old behavior is caught, and asserts the full K*N
      row count plus per-block shadow-value preservation.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>658</total>
    <passed>658</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures></failures>
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>proceed_to_document</recommendation>
  </summary>
  <action_items></action_items>
</test_report>
