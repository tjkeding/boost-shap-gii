<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-14T16:05:00Z" />
  <pre_design_run>
    <total>724</total>
    <passed>724</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures></failures>
  </pre_design_run>
  <failing_test_dispositions>
    <!-- No failures in the pre-design run; no disposition ledger entries required.
    The implement build cycle (boost-shap-gii_implement_build_20260814_155500.md)
    resolved all 9 previously failing tests from the prior test cycle by
    introducing original_cluster_ids in _run_bootstrap_pipeline and
    re-expressing the obsolete test_unequal_cluster_sizes_raises. -->
  </failing_test_dispositions>
  <design_phase>
    <tests_created>1</tests_created>
    <tests_modified>0</tests_modified>
    <files_created>
      <file path="tests/test_inference_shap.py" test_count="1" coverage_target="Non-fallback (n_clusters &gt;= 20) branch of the cluster-bootstrap microdata deduplication path. The just-applied fix (original_cluster_ids) only has behavioral significance when the i.i.d. fallback fires; every pre-existing microdata-row-count test used N below the 20-cluster fallback threshold, so none exercised the complementary branch. TestClusterBootstrapMicrodataNoFallback::test_large_n_microdata_has_n_rows_no_fallback (N=25, K=4) asserts no fallback warning fires and microdata parquets still deduplicate K*N rows to exactly N rows per effect." />
    </files_created>
    <design_rationale>
      Pre-design run showed 724/724 passing (0 failures), confirming the prior implement cycle's fix and test re-expression resolved the previously routed product bug. With no failures to disposition, the design phase's task reduced to a first-class coverage-gap analysis of the change just applied. The original_cluster_ids fix in _run_bootstrap_pipeline is only observably different from the pre-fix behavior when the i.i.d. fallback guard reassigns cluster_ids to None (n_clusters &lt; 20, per Ukoumunne et al. 2003); at n_clusters &gt;= 20, original_cluster_ids and the post-fallback cluster_ids are always equal, so the two code paths are behaviorally indistinguishable in that regime. All pre-existing tests asserting microdata row counts (test_microdata_has_n_rows at N=15, test_microdata_x_values_align at N=10) use N below the fallback threshold; the only large-N test in the file (TestClusterBootstrapVsIIDCIWidth, N=50) is a standalone statistical simulation that never calls _run_bootstrap_pipeline and does not touch the microdata path. This left the non-fallback branch of the same conditional without direct regression coverage. One test was added, following the existing test_microdata_has_n_rows pattern (strong-signal synthetic SHAP values to guarantee effect significance, X_display_override for microdata), at N=25/K=4 to place n_clusters above the fallback threshold. The test asserts (a) no RuntimeWarning matching the fallback message is emitted, confirming the non-fallback branch was actually exercised, and (b) all three microdata parquets (M, V, GII) deduplicate to exactly N rows per effect. The assertion was independently executed against the live codebase in isolation (pytest tests/test_inference_shap.py::TestClusterBootstrapMicrodataNoFallback -v) before being included in the post-design suite run, per Design by Contract discipline.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>725</total>
    <passed>725</passed>
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
  <action_items>
    <!-- None. All prior action items closed; no new product bugs identified this cycle. -->
  </action_items>
</test_report>
