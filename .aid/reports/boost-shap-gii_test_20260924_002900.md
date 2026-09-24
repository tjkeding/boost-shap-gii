<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-09-24T00:29:00-04:00" />
  <pre_design_run>
    <total>1004</total>
    <passed>993</passed>
    <failed>6</failed>
    <errors>5</errors>
    <coverage_pct></coverage_pct>
    <failures>
      <failure test="TestPlotR_VContributionRanking::test_underscore_wordwrap_replaces_n_count_lookup" file="tests/test_build_20260507.py" line="313">
        <error_type>AssertionError</error_type>
        <message>Expected the underscore-to-space word-wrap transform to appear exactly twice; found 0 occurrence(s).</message>
        <traceback>assert 0 == 2</traceback>
      </failure>
      <failure test="TestMPanelPerFeatureAxis::test_legend_labels_plain_and_stats_annotated_below" file="tests/test_build_20260921b.py" line="257">
        <error_type>AssertionError</error_type>
        <message>'signal_label &lt;- "Signal"' not in plot.R source.</message>
        <traceback>assert 'signal_label &lt;- "Signal"' in plot_r_source</traceback>
      </failure>
      <failure test="TestMPanelPerFeatureAxis::test_legend_repositioned_and_resized" file="tests/test_build_20260921b.py" line="298">
        <error_type>AssertionError</error_type>
        <message>'legend.text = element_text(size = 3.8)' not in plot.R source.</message>
        <traceback>assert 'legend.text = element_text(size = 3.8)' in plot_r_source</traceback>
      </failure>
      <failure test="TestCoordCartesianClipOff::test_clip_off_present_in_all_three_panels" file="tests/test_build_20260922.py" line="161">
        <error_type>AssertionError</error_type>
        <message>coord_cartesian(clip = "off") count is 2, expected 3.</message>
        <traceback>assert 2 == 3</traceback>
      </failure>
      <failure test="TestBelowAxisLabelPositioning::test_vjust_uniform_at_all_five_sites" file="tests/test_build_20260922b.py" line="183">
        <error_type>AssertionError</error_type>
        <message>vjust=0.5 stat-label block count is 3, expected 5.</message>
        <traceback>assert 3 == 5</traceback>
      </failure>
      <failure test="TestStatLabelWhiteBackground::test_white_background_block_partitioned_by_panel_type" file="tests/test_build_20260923c.py" line="145">
        <error_type>AssertionError</error_type>
        <message>M-panel white-background block count is 0, expected 2.</message>
        <traceback>assert 0 == 2</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <interim_note>
    A mid-design investigation into test_underscore_wordwrap_replaces_n_count_lookup surfaced a documented, prior-cycle-discovered risk: newline-separated category labels combined with the "__NA__" sentinel were previously found to render as four blank lines around "NA". A functional guard (ifelse(levels(fac) == "__NA__", "NA", gsub(...))) was routed through /implement and applied. Further investigation (triggered by a newly-surfaced test, TestNASentinelSimplified in test_build_20260923.py) established that the upstream NA-recoding sites (plot.R lines 682 and 759) already normalize all missing-value representations to the bare 2-character string "NA" before this word-wrap step runs, so the literal "__NA__" sentinel structurally cannot reach it. The guard was therefore unnecessary dead code that also reintroduced a deprecated string an existing test explicitly forbids; it was reverted via a second /implement cycle. The pre-design run above reflects the suite state after this revert (i.e., the stable, correct baseline), confirmed identical across three independent runs bracketing the guard-add/guard-revert cycle.
  </interim_note>
  <failing_test_dispositions>
    <disposition test="test_underscore_wordwrap_replaces_n_count_lookup" file="tests/test_build_20260507.py" classification="obsolete-test">
      <intended_contract>Discrete x-axis category labels receive an underscore-to-readable-text word-wrap transform at both the singleton and interaction V-panel sites.</intended_contract>
      <current_test_claim>Asserted the separator is a space (gsub("_", " ", levels(fac))), per an earlier same-day critique cycle (T7).</current_test_claim>
      <evidence>plot.R lines 838/1023 (pre-session-change) now read gsub("_", "\n", levels(fac)) at both sites, per explicit user direction this session to eliminate horizontal label-to-label overlap. TestNASentinelSimplified (test_build_20260923.py) confirms the "__NA__" sentinel that motivated the space-separated form cannot reach this step.</evidence>
      <action>Re-expressed to assert the newline-separated form (count == 2) and the absence of the deprecated space-separated form. Level_label_lookup absence assertion preserved unchanged.</action>
    </disposition>
    <disposition test="test_legend_labels_plain_and_stats_annotated_below" file="tests/test_build_20260921b.py" classification="obsolete-test">
      <intended_contract>M-panel legend uses plain "Signal"/"Noise" labels; mean+SD statistics render as separately-positioned annotate("label", ...) calls with a non-overlap guarantee from nudge_stat_labels().</intended_contract>
      <current_test_claim>Asserted plain string labels, nudge_stat_labels() presence, and both annotate("label", ...) call sites.</current_test_claim>
      <evidence>Implement build boost-shap-gii_implement_build_20260923_220500.md (change C1): eight sub-cycles of spatial-annotation tuning failed to eliminate collision; the entire architecture was replaced with legend-integrated stats via ggtext::element_markdown(). nudge_stat_labels, stat_pos, and both annotate("label", ...) sites are confirmed absent from plot.R.</evidence>
      <action>Renamed to test_legend_integrated_stats_replace_spatial_annotation; re-expressed to assert the sprintf/markdown label format, its presence in the scale_*_manual labels mapping, and the absence of every superseded mechanism (plain strings, nudge_stat_labels, stat_pos, both annotate forms).</action>
    </disposition>
    <disposition test="test_legend_repositioned_and_resized" file="tests/test_build_20260921b.py" classification="obsolete-test">
      <intended_contract>M-panel legend is bottom-positioned, horizontal, with element_text(size = 3.8).</intended_contract>
      <current_test_claim>Asserted legend.text = element_text(size = 3.8).</current_test_claim>
      <evidence>plot.R line 644 now reads legend.text = ggtext::element_markdown(size = 4.2), required because the legend labels carry embedded markdown (bold prefix + span-sized stat) that plain element_text cannot render.</evidence>
      <action>Re-expressed to assert ggtext::element_markdown(size = 4.2) and the absence of the deprecated element_text(size = 3.8) form. The bottom/horizontal positioning assertions (still valid, unchanged by this redesign) were preserved.</action>
    </disposition>
    <disposition test="test_clip_off_present_in_all_three_panels" file="tests/test_build_20260922.py" classification="obsolete-test">
      <intended_contract>Three panels (performance has_boot_perf branch, performance CI-band fallback branch, M-panel) each need coord_cartesian(clip = "off") to render below-axis annotations outside their plot area.</intended_contract>
      <current_test_claim>Asserted count == 3.</current_test_claim>
      <evidence>The M-panel's legend-integration redesign removed all below-axis spatial annotation from that panel (see disposition above); with no element extending outside the panel boundary, clip = "off" is no longer needed there. plot.R now contains exactly 2 occurrences (both performance-panel branches, unaffected by this redesign).</evidence>
      <action>Renamed to test_clip_off_present_in_exactly_two_panels; re-expressed to assert count == 2.</action>
    </disposition>
    <disposition test="test_vjust_uniform_at_all_five_sites" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>All five below-axis stat-label sites (2 performance has_boot_perf, 1 performance CI-band fallback, 2 M-panel) share identical vjust = 0.5 styling.</intended_contract>
      <current_test_claim>Asserted count == 5.</current_test_claim>
      <evidence>The M-panel's two former vjust = 0.5 sites (noise, signal) no longer exist as annotate("label", ...) calls; they were removed by the legend-integration redesign. Only the three performance-panel sites remain.</evidence>
      <action>Renamed to test_vjust_uniform_at_remaining_three_sites; re-expressed to assert count == 3, plus an explicit absence check for the M-panel's former annotate("label", x = stat_pos$x1/x2, ...) call sites.</action>
    </disposition>
    <disposition test="test_white_background_block_partitioned_by_panel_type" file="tests/test_build_20260923c.py" classification="obsolete-test">
      <intended_contract>Performance-panel sites use a white-background block with label.padding = unit(0.15, "lines") (count 3); M-panel sites use the same block with label.padding = unit(0.5, "lines") (count 2).</intended_contract>
      <current_test_claim>Asserted PERF_PANEL block count == 3 and M_PANEL block count == 2.</current_test_claim>
      <evidence>The M-panel's two annotate("label", ...) sites (which carried the enlarged-padding white-background block) were removed entirely by the legend-integration redesign. Only the three performance-panel sites retain any white-background annotation.</evidence>
      <action>Re-expressed: PERF_PANEL_WHITE_BACKGROUND_BLOCK count == 3 assertion preserved unchanged; M_PANEL_WHITE_BACKGROUND_BLOCK count == 2 re-expressed to count == 0, with rationale that a nonzero count would indicate reintroduction of the superseded spatial-annotation form.</action>
    </disposition>
    <disposition test="TestNudgeStatLabels (5 tests: test_no_overlap_returns_original_means, test_overlap_nudges_symmetrically_about_midpoint, test_nudge_preserves_relative_ordering_when_reversed, test_nudged_separation_meets_required_width, test_identical_means_still_separates)" file="tests/test_build_20260922.py" classification="obsolete-test">
      <intended_contract>The nudge_stat_labels() R helper symmetrically nudges two mean positions apart when their required text-width separation exceeds the actual gap, per a documented formula.</intended_contract>
      <current_test_claim>Five known-answer behavioral tests against the extracted nudge_stat_labels function, erroring with "substring not found" because the function no longer exists in plot.R.</current_test_claim>
      <evidence>nudge_stat_labels was deliberately deleted as dead code in the legend-integration redesign (implement plan boost-shap-gii_implement_plan_20260923_220000.md, change C1, edit site A). Confirmed zero occurrences of the string "nudge_stat_labels" anywhere in plot.R. No successor function performs equivalent collision-avoidance nudging; the entire spatial-annotation approach was abandoned, not replaced.</evidence>
      <action>Removed the entire TestNudgeStatLabels class, its supporting fixture (nudge_fn_source) and helper functions (_extract_r_function, _run_nudge), and the now-unused subprocess import. No successor tests are needed since there is no successor mechanism to test; the dead-code-absence guard is instead covered by TestMPanelLegendIntegratedStats.test_no_spatial_annotation_dead_code_remains in the new test_build_20260924.py.</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>6</tests_created>
    <tests_modified>4</tests_modified>
    <files_created>
      <file path="tests/test_build_20260924.py" test_count="6" coverage_target="M-panel legend-integrated stats (ggtext::element_markdown, sprintf/markdown label format, scale_*_manual wiring, spatial-annotation dead-code absence) and V-panel discrete-label newline word-wrap (both sites, plus the __NA__-sentinel-guard-unnecessary confirmation)" />
    </files_created>
    <design_rationale>
      All 6 pre-design failures and 5 errors were clean obsolete-test dispositions, every one traceable to a single, deliberate, user-directed architectural change this session (the M-panel legend-integration redesign) plus the V-panel newline-wrap fix. Per Meyer's postcondition-strengthening rule, each re-expression preserves every assertion that remains valid under the new contract (e.g., the bottom/horizontal legend positioning, the level_label_lookup absence check, the PERF_PANEL_WHITE_BACKGROUND_BLOCK count) and adds regression guards against the just-deprecated forms. The TestNudgeStatLabels removal is not a weakening: the function under test was deliberately deleted as dead code with no successor, so no assertion could be re-expressed to test a mechanism that no longer exists; its dead-code-absence guard is preserved via a new test. The new test_build_20260924.py closes the coverage gap the removed/re-expressed tests could not: it provides direct, positive-contract coverage of the two genuinely new behaviors this session introduced (legend-integrated stats, newline word-wrap) rather than only negative (absence-of-old-form) coverage. No product bugs were found; every failure was the correct, expected consequence of a user-directed and now-verified-correct implementation change.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>1005</total>
    <passed>1005</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct></coverage_pct>
    <failures></failures>
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>proceed_to_publish</recommendation>
  </summary>
  <action_items>
    <item priority="P2" target_mode="publish" description="Publish v1.7.0. Remember KI-001: the version bump should target both pyproject.toml and __init__.py, not __init__.py alone." />
  </action_items>
</test_report>
