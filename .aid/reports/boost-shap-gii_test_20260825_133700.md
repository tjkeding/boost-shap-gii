<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-25T13:37:00Z" />
  <pre_design_run>
    <total>888</total>
    <passed>886</passed>
    <failed>2</failed>
    <errors>0</errors>
    <coverage_pct />
    <failures>
      <failure test="TestOrchestrateBootstrapCacheClusterIdsResolution::test_resolution_activates_when_group_column_present_in_x_train" file="tests/test_indiv_reports_unit.py" line="877">
        <error_type>AssertionError</error_type>
        <message>captured["cluster_ids"] expected to equal grp via internal X_train.columns-based resolution; resolved to None instead.</message>
        <traceback>tests/test_indiv_reports_unit.py:877: AssertionError</traceback>
      </failure>
      <failure test="TestClusterIdsSourceWiring::test_group_column_resolution_pattern_present_at_both_sites" file="tests/test_indiv_reports_unit.py" line="877">
        <error_type>AssertionError</error_type>
        <message>Expected exactly two sites (training-mode orchestrate_bootstrap_cache and inference-mode generate_indiv_reports) to resolve group_column from config; a count other than 2 indicates the two sites have drifted out of sync.</message>
        <traceback>tests/test_indiv_reports_unit.py:877: AssertionError</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
    <disposition test="TestOrchestrateBootstrapCacheClusterIdsResolution::test_resolution_activates_when_group_column_present_in_x_train" file="tests/test_indiv_reports_unit.py" classification="obsolete-test">
      <intended_contract>As written, the test isolated an internal resolution block inside orchestrate_bootstrap_cache that read config['modeling']['group_column'] and checked X_train.columns to derive cluster identity, confirming that logic was correct in isolation (the fault lying entirely in what production callers passed as X_train).</intended_contract>
      <current_test_claim>Calls orchestrate_bootstrap_cache without a cluster_ids keyword argument, with X_train containing the configured group column, and asserts the captured cluster_ids equals that column's values.</current_test_claim>
      <evidence>The current implement build (boost-shap-gii_implement_build_20260825_133000.md, change C1, user-directed modification) deliberately deleted the internal X_train.columns-based resolution block from orchestrate_bootstrap_cache and replaced it with an explicit keyword-only cluster_ids parameter supplied by the caller. grep -c 'config.get("modeling", {}).get("group_column")' src/boost_shap_gii/indiv_reports.py returns 0 (was 2 before this build). There is no longer any internal resolution logic to isolate.</evidence>
      <action>re-express: rewritten as a pure passthrough test proving cluster_ids is used verbatim when supplied by the caller, independent of X_train's columns entirely (X_train no longer contains the group column at all in the rewritten test, strengthening the prior version which coincidentally depended on X_train and cluster_ids agreeing).</action>
    </disposition>
    <disposition test="TestClusterIdsSourceWiring::test_group_column_resolution_pattern_present_at_both_sites" file="tests/test_indiv_reports_unit.py" classification="obsolete-test">
      <intended_contract>Guard the two group_column-resolution call sites inside indiv_reports.py (training-mode orchestrate_bootstrap_cache, inference-mode generate_indiv_reports) against drifting out of sync with each other.</intended_contract>
      <current_test_claim>src.count('config.get("modeling", {}).get("group_column")') == 2</current_test_claim>
      <evidence>Both internal resolution sites were intentionally removed in this session's corrected build: training-mode resolution moved to the caller (predict.py resolves from df_raw, which has training-data cluster identity), and inference-mode resolution was replaced entirely by loading a persisted cluster_ids.npy artifact (infer.py's df_raw is the inference dataset and cannot supply training-cluster identity; inference mode is architecturally restricted to persisted artifacts). See boost-shap-gii_implement_build_20260825_133000.md, changes C1 and C3.</evidence>
      <action>re-express: split into two guards matching the new architecture -- zero occurrences of the config-based resolution pattern remaining in indiv_reports.py (test_no_internal_group_column_config_resolution_remains), and exactly two references to the new cluster_ids.npy artifact, the write site and the read site (test_cluster_ids_artifact_referenced_at_exactly_two_sites). This preserves the same underlying two-site-parity invariant, applied to the corrected architecture.</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>7</tests_created>
    <tests_modified>4</tests_modified>
    <files_created>
      <file path="tests/test_indiv_reports_unit.py" test_count="7 created, 4 modified (added to 60+ pre-existing)" coverage_target="Coverage for this session's implement-mode correction to the F1 fix: (1) orchestrate_bootstrap_cache's cluster_ids parameter is a pure passthrough with no internal config/X_train-based resolution; (2) orchestrate_bootstrap_cache persists cluster_ids as bootstrap_refits/cluster_ids.npy when supplied, and omits the artifact when cluster_ids is None; (3) generate_indiv_reports' inference-mode branch loads cluster identity from the persisted cluster_ids.npy artifact (falling back to None when absent) rather than accepting it from the caller; (4) predict.py's caller-side resolution of cluster_ids from df_raw[group_column], threaded to both call sites; (5) a regression guard proving infer.py does not resolve cluster identity from its own df_raw (the inference dataset), which was the exact defect caught and corrected during this session's implementation review before any test cycle observed it." />
    </files_created>
    <design_rationale>
      The pre-design run surfaced exactly two failures, both direct and
      expected consequences of a mid-session architectural correction: the
      original F1 fix (from the prior implement build) threaded cluster_ids
      from the caller's df_raw at all three call sites uniformly, including
      infer.py. During user review of that build, it was identified that
      this was itself incorrect for infer.py specifically, since infer.py's
      df_raw is the inference dataset (not the training data) and its
      group_column, if present at all, may carry entirely different cluster
      semantics than the training data's. Inference mode is also
      architecturally restricted to persisted artifacts (models, JSON/parquet
      metadata, transformations) and must never depend on training data
      access. The build was corrected in-session: orchestrate_bootstrap_cache
      now persists cluster_ids as a new artifact (bootstrap_refits/cluster_ids.npy,
      mirroring the existing y_train.npy pattern), generate_indiv_reports'
      inference-mode branch loads cluster identity from that artifact instead
      of from a caller-supplied argument, and infer.py's caller-side
      resolution from its own df_raw was fully reverted.

      Both pre-design failures trace directly to this correction and are
      classified obsolete-test: the tests encoded the intended contract of
      the ORIGINAL (uncorrected) build, which itself has since been
      superseded by explicit user direction grounded in a real architectural
      constraint (inference mode's artifact-only access, and the inference
      dataset's potentially divergent group semantics), not by a
      subsequently-discovered code defect. Both were re-expressed to encode
      the corrected contract, with postconditions preserved or strengthened
      per Test Design Discipline: the passthrough test now proves stronger
      decoupling (no dependency on X_train's columns at all, rather than a
      coincidental agreement), and the source-parity guard was split into two
      guards matching the new two-artifact-site architecture rather than the
      removed two-config-resolution-site architecture.

      Beyond the two failures, design-phase analysis of the corrected
      build's diff surfaced a complete absence of test coverage for three
      behaviors introduced by the correction itself: the cluster_ids.npy
      artifact's write path (including the negative case of omitting the
      artifact when cluster_ids is None, so a stale artifact from a prior
      grouped run cannot leak into a later ungrouped run sharing the same
      cache directory), the artifact's read path in inference mode (including
      graceful fallback to None for pre-existing caches built before this fix),
      and the caller-side wiring in both predict.py (positive coverage) and
      infer.py (a regression guard against reintroducing the exact defect
      this session's review caught, following the same "no legacy key
      referenced" guard convention already established in this file for a
      prior fixed mistake). All three gaps are now covered.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>895</total>
    <passed>895</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct />
    <failures />
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>proceed_to_document</recommendation>
  </summary>
  <action_items>
  </action_items>
</test_report>
