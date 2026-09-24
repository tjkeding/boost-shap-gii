<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-09-23T19:55:00-04:00" />
  <pre_design_run>
    <total>1002</total>
    <passed>997</passed>
    <failed>5</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="test_shared_legend_stats_annotation_present" file="tests/test_build_20260921b.py" line="97">
        <error_type>AssertionError</error_type>
        <message>geom_text(data = df_obs, aes(x = boot_mean, y = -Inf, not found</message>
        <traceback>FAILED tests/test_build_20260921b.py::TestPerfPlotStatsAndLegend::test_shared_legend_stats_annotation_present</traceback>
      </failure>
      <failure test="test_fallback_branch_null_only_annotation_present" file="tests/test_build_20260921b.py" line="136">
        <error_type>AssertionError</error_type>
        <message>geom_text(data = df_obs, aes(x = null_mean, y = -Inf, not found</message>
        <traceback>FAILED tests/test_build_20260921b.py::TestPerfPlotStatsAndLegend::test_fallback_branch_null_only_annotation_present</traceback>
      </failure>
      <failure test="test_legend_labels_plain_and_stats_annotated_below" file="tests/test_build_20260921b.py" line="213">
        <error_type>AssertionError</error_type>
        <message>annotate("text", x = stat_pos$x1, y = -Inf, not found</message>
        <traceback>FAILED tests/test_build_20260921b.py::TestMPanelPerFeatureAxis::test_legend_labels_plain_and_stats_annotated_below</traceback>
      </failure>
      <failure test="test_vjust_0_5_present_at_all_five_sites" file="tests/test_build_20260922b.py" line="153">
        <error_type>AssertionError</error_type>
        <message>vjust=0.5 count == 5 failed (actual: 3); vjust=1.5 absence check failed (now present, 2 occurrences)</message>
        <traceback>FAILED tests/test_build_20260922b.py::TestBelowAxisLabelPositioning::test_vjust_0_5_present_at_all_five_sites</traceback>
      </failure>
      <failure test="test_m_panel_bottom_margin_reduced" file="tests/test_build_20260922b.py" line="248">
        <error_type>AssertionError</error_type>
        <message>plot.margin = unit(c(1, 0.5, 1, 1), "mm") not found (now 3mm)</message>
        <traceback>FAILED tests/test_build_20260922b.py::TestBelowAxisMarginAccommodation::test_m_panel_bottom_margin_reduced</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
    <disposition test="test_shared_legend_stats_annotation_present" file="tests/test_build_20260921b.py" classification="obsolete-test">
      <intended_contract>A mean+SD annotation for the trained-model and permutation-null distributions in the performance panel's has_boot_perf branch, centered below each distribution's own mean line.</intended_contract>
      <current_test_claim>Asserted the literal substring "geom_text(data = df_obs, aes(x = boot_mean, y = -Inf," is present.</current_test_claim>
      <evidence>plot.R:486 and plot.R:491 now read "geom_label(data = df_obs, aes(x = boot_mean/null_mean, y = -Inf," per implement build report boost-shap-gii_implement_build_20260923_190500.md change C1, driven by the user's explicit direct instruction this session requesting a white background so the label occludes the geom_vline mean line beneath it.</evidence>
      <action>Re-expressed to assert geom_label presence, added a regression guard forbidding the deprecated geom_text form at this site, and preserved all unaffected assertions (deprecated distribution-prefix format, deprecated two-line format).</action>
    </disposition>
    <disposition test="test_fallback_branch_null_only_annotation_present" file="tests/test_build_20260921b.py" classification="obsolete-test">
      <intended_contract>A mean+SD annotation for the permutation-null distribution in the performance panel's CI-band fallback branch, centered below its mean line.</intended_contract>
      <current_test_claim>Asserted the literal substring "geom_text(data = df_obs, aes(x = null_mean, y = -Inf," is present.</current_test_claim>
      <evidence>plot.R:530 (and plot.R:491, sharing the identical substring) now reads "geom_label(...)" per build report change C1.</evidence>
      <action>Re-expressed to assert geom_label presence, added a regression guard forbidding the deprecated geom_text form, preserved the deprecated-prefix guard.</action>
    </disposition>
    <disposition test="test_legend_labels_plain_and_stats_annotated_below" file="tests/test_build_20260921b.py" classification="obsolete-test">
      <intended_contract>M-panel legend labels are plain ("Noise"/"Signal"); mean+SD statistics are shown via separate annotate() calls at nudge_stat_labels()-computed positions.</intended_contract>
      <current_test_claim>Asserted the literal substrings 'annotate("text", x = stat_pos$x1, y = -Inf,' and 'annotate("text", x = stat_pos$x2, y = -Inf,' are present.</current_test_claim>
      <evidence>plot.R:655 and plot.R:659 now read 'annotate("label", x = stat_pos$x1/x2, y = -Inf,' per implement build report change C2, driven by the user's explicit instruction requesting the same white-background occlusion fix for the M-panel.</evidence>
      <action>Re-expressed both assertions to the "label" variant, added regression guards forbidding the deprecated annotate("text", ...) form at both sites, preserved all unaffected assertions (signal_label/noise_label definitions, nudge_stat_labels function/call, deprecated legend-embedded-stats format guard).</action>
    </disposition>
    <disposition test="test_vjust_0_5_present_at_all_five_sites" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>Prior to this session: a uniform vjust=0.5 styling invariant across all five stat-label annotation sites, established across five prior critique cycles documented in the class docstring.</intended_contract>
      <current_test_claim>Asserted exactly 5 occurrences of vjust=0.5 styling, and asserted vjust=1.5 styling is absent everywhere (a blanket exclusion left over from a fourth-critique-cycle deprecation of a prior, unrelated attempt at vjust=1.5 applied panel-wide).</current_test_claim>
      <evidence>plot.R:657 and plot.R:661 (M-panel noise/signal sites) now read vjust=1.5, while plot.R:488, 493, 532 (performance panel sites) remain at vjust=0.5, per implement build report change C2. This is a deliberate, user-directed re-partition: instruction (1) this session requested only a background fix for the performance panel (no spacing change), while instruction (2) explicitly requested "more space" for the M-panel specifically.</evidence>
      <action>Re-expressed and renamed to test_vjust_partitioned_by_panel_type: asserts exactly 3 occurrences of vjust=0.5 (performance panel only) and exactly 2 occurrences of vjust=1.5 (M-panel only, replacing the old blanket-absence guard with a stricter, position-specific exact-count assertion for the new contract). Preserved the still-valid vjust=1.0 and vjust=3.5 deprecated-value absence guards. Updated the class docstring with a sixth critique-cycle entry documenting the rationale, matching this file's established documentation convention.</action>
    </disposition>
    <disposition test="test_m_panel_bottom_margin_reduced" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>Prior to this session: the M-panel bottom margin is minimized (1 mm), since vjust=0.5 required no extra clearance.</intended_contract>
      <current_test_claim>Asserted 'plot.margin = unit(c(1, 0.5, 1, 1), "mm")' is present; asserted 4mm/12mm/6mm are absent.</current_test_claim>
      <evidence>plot.R:685 now reads 'plot.margin = unit(c(1, 0.5, 3, 1), "mm")' per implement build report change C2, driven directly by the vjust increase to 1.5 in the same change (more clearance requires more canvas room, or ggsave clips the label boxes at the bottom edge).</evidence>
      <action>Re-expressed to assert 3mm is present; added 1mm to the deprecated-value absence list (now itself superseded); preserved the still-valid 4mm/12mm/6mm absence guards. Updated the class docstring with a sixth critique-cycle entry.</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>2</tests_created>
    <tests_modified>5</tests_modified>
    <files_created>
      <file path="tests/test_build_20260923c.py" test_count="2" coverage_target="Cross-cutting completeness census for the white-background occlusion fix (build report boost-shap-gii_implement_build_20260923_190500.md, changes C1+C2): exact count of 5 for the shared white-background parameter block across all five stat-label sites, plus a consolidated absence guard for all deprecated transparent-background geom_text/annotate(&quot;text&quot;) forms." />
    </files_created>
    <design_rationale>
      All 5 pre-design failures were clean obsolete-test dispositions: each failing assertion source-string-locked the exact pre-fix rendering syntax (geom_text, annotate("text", ...), a uniform vjust=0.5, and a 1mm M-panel margin) from prior critique cycles, all of which the user's explicit, direct instruction this session (already implemented via the completed /implement cycle, build report boost-shap-gii_implement_build_20260923_190500.md) deliberately supersedes. No product bugs were found; every failure was the correct, expected consequence of a user-directed implementation change.

      Per Meyer's postcondition-strengthening rule, each re-expression preserves every assertion that remains valid under the new contract and adds a regression guard against the deprecated form at that exact site, so no re-expression is merely "make it pass" but is at least as strong as the prior version for everything the prior version verified, while accurately reflecting the new intended contract for what changed. The vjust re-partition (test_vjust_0_5_present_at_all_five_sites, re-expressed to test_vjust_partitioned_by_panel_type) required inverting a blanket "vjust=1.5 absent everywhere" guard into an exact-count "vjust=1.5 present exactly twice, at the M-panel sites" assertion; this is not a weakening because the two assertions describe genuinely different, mutually exclusive contracts, and the new contract (a deliberate two-tier partition established by direct user instruction) is a real design change, not an oversight the old assertion should have continued to catch.

      Beyond the 5 re-expressions, a coverage gap was identified: the white-background fix itself (fill = "white", label.size = NA, label.padding) was entirely uncovered by any existing assertion prior to this design pass, and the 5 per-site re-expressions above are split across two files, with 2 of the 5 sites (the has_boot_perf branch's null-mean label and the CI-band fallback branch's null-mean label) sharing identical null_mean-anchored source text that a per-site substring search cannot distinguish between. A new file, test_build_20260923c.py, adds a single test class with an exact-count assertion (count == 5) for the shared white-background parameter block across the whole file, independently guarding against a future edit that touches one site's white-background text without regard for the other four, in a way the existing per-site tests cannot jointly detect.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>1004</total>
    <passed>1004</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures />
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>proceed_to_run_local</recommendation>
  </summary>
  <action_items>
    <item priority="P1" target_mode="run-local" description="Run /run-local to visually re-verify the white-background fix (build report boost-shap-gii_implement_build_20260923_190500.md): confirm the performance panel's three stat labels (trained mean+SD, permutation-null mean+SD x2) now render as opaque white boxes occluding the geom_vline mean lines beneath them, and confirm the M-panel's two stat labels (noise mean+SD, signal mean+SD) now render as opaque white boxes with visible clearance below the density distributions, across all three <external_project> child-child run directories." />
  </action_items>
</test_report>
