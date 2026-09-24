<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-09-23T20:28:00-04:00" />
  <pre_design_run>
    <total>1004</total>
    <passed>1001</passed>
    <failed>3</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="test_vjust_partitioned_by_panel_type" file="tests/test_build_20260922b.py" line="167">
        <error_type>AssertionError</error_type>
        <message>Expected exactly 3 occurrences of vjust=0.5 stat-label styling (perf panel only); found 5. M-panel sites not yet moved to vjust=1.5.</message>
        <traceback>tests/test_build_20260922b.py:167: AssertionError: assert 5 == 3</traceback>
      </failure>
      <failure test="test_m_panel_bottom_margin_reduced" file="tests/test_build_20260922b.py" line="279">
        <error_type>AssertionError</error_type>
        <message>Expected M-panel bottom margin 3 mm; string not found in plot.R.</message>
        <traceback>tests/test_build_20260922b.py:279: AssertionError: 'plot.margin = unit(c(1, 0.5, 3, 1), "mm")' not in plot_r_source</traceback>
      </failure>
      <failure test="test_white_background_block_present_at_all_five_sites" file="tests/test_build_20260923c.py" line="95">
        <error_type>AssertionError</error_type>
        <message>Expected 5 occurrences of the identical white-background param block; found 3 (M-panel sites now use different padding).</message>
        <traceback>tests/test_build_20260923c.py:95: AssertionError: assert 3 == 5</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
    <disposition test="test_vjust_partitioned_by_panel_type" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>The M-panel's two stat labels (noise, signal) sit at vjust=1.5 (below-panel) while the performance panel's three sites remain at vjust=0.5, per the sixth critique cycle's attempt to create clearance from the density distributions.</intended_contract>
      <current_test_claim>Asserted exactly 3 occurrences of vjust=0.5 (performance panel only) and exactly 2 occurrences of vjust=1.5 (M-panel only).</current_test_claim>
      <evidence>plot.R:657 and plot.R:661 (M-panel sites) now read vjust=0.5 per implement build report boost-shap-gii_implement_build_20260923_203500.md, change C1. This is a direct consequence of the user's visual inspection of the /run-local re-render (this session), which found that vjust=1.5 pushed the M-panel stat labels into the x-axis tick-label zone (overlapping "0.0", "1.0", etc.) rather than creating clearance from the distributions, and that the white background was imperceptible at the prior padding.</evidence>
      <action>Re-expressed and renamed to test_vjust_uniform_at_all_five_sites: asserts exactly 5 occurrences of vjust=0.5 (uniform across all sites, restoring the pre-sixth-cycle invariant) and explicit absence of the deprecated vjust=1.5 form. Preserved the still-valid vjust=1.0 and vjust=3.5 deprecated-value absence guards. Updated the class docstring with a seventh critique-cycle entry documenting the user-visual-feedback-driven reversion.</action>
    </disposition>
    <disposition test="test_m_panel_bottom_margin_reduced" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>The M-panel bottom margin is 3 mm, sized to accommodate the vjust=1.5 below-panel label positioning from the sixth critique cycle.</intended_contract>
      <current_test_claim>Asserted 'plot.margin = unit(c(1, 0.5, 3, 1), "mm")' is present; asserted 1mm/4mm/12mm/6mm are absent.</current_test_claim>
      <evidence>plot.R:685 now reads 'plot.margin = unit(c(1, 0.5, 1, 1), "mm")' per implement build report change C1, a direct consequence of the vjust reversion in the same change: with labels no longer extending below the panel edge, the extra 3mm of canvas room is unneeded.</evidence>
      <action>Re-expressed and renamed to test_m_panel_bottom_margin_reverted_to_minimal: asserts 1mm is present; added 3mm to the deprecated-value absence list (now itself superseded); preserved the still-valid 4mm/12mm/6mm absence guards. Updated the class docstring with a seventh critique-cycle entry.</action>
    </disposition>
    <disposition test="test_white_background_block_present_at_all_five_sites" file="tests/test_build_20260923c.py" classification="obsolete-test">
      <intended_contract>All five stat-label sites share an identical white-background parameter block, including label.padding = unit(0.15, "lines"), so each opaque box occludes the geom_vline or density distribution beneath it.</intended_contract>
      <current_test_claim>Asserted the exact substring 'fill = "white", label.size = NA, label.padding = unit(0.15, "lines")' appears exactly 5 times.</current_test_claim>
      <evidence>plot.R:658 and plot.R:662 (M-panel sites) now read label.padding = unit(0.5, "lines") per implement build report change C1. This is a direct consequence of the user's visual inspection finding the shared 0.15-line padding produced an imperceptibly small white rectangle at the M-panel sites, where it needed to visibly occlude density distribution tails (a taller, more visually dense element than the thin geom_vline the performance panel's white boxes occlude, where 0.15 lines remains sufficient per user confirmation).</evidence>
      <action>Re-expressed and renamed to test_white_background_block_partitioned_by_panel_type: split the single shared-block constant into PERF_PANEL_WHITE_BACKGROUND_BLOCK (padding 0.15, asserted count==3) and M_PANEL_WHITE_BACKGROUND_BLOCK (padding 0.5, asserted count==2). This is not a weakening: the new assertion pins both panels' padding values exactly, which is at least as specific as the prior single-count assertion, and additionally distinguishes which sites use which value (a property the prior test could not express). The companion test_no_transparent_stat_label_geoms_remain in the same class was unaffected (aligned, no change) since it checks only for absence of the deprecated transparent-background geom_text/annotate("text") forms, independent of padding value. Updated the class docstring with a seventh critique-cycle entry.</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>0</tests_created>
    <tests_modified>3</tests_modified>
    <files_created />
    <design_rationale>
      All 3 pre-design failures were clean obsolete-test dispositions, all traceable to a single implement build (boost-shap-gii_implement_build_20260923_203500.md, change C1) that reverted the sixth critique cycle's M-panel fix after the user's direct visual inspection of the regenerated plots found it did not work: vjust=1.5 moved the stat-label overlap problem from the density distributions to the x-axis tick labels instead of resolving it, and the white background was too small to be visually effective at the M-panel's shared 0.15-line padding.

      Per Meyer's postcondition-strengthening rule, each re-expression preserves every assertion that remains valid under the new contract and adds a regression guard against the just-deprecated form (vjust=1.5, 3mm margin, 0.15-line padding at M-panel sites specifically). The white-background re-expression is a net strengthening, not merely a re-expression: splitting the single shared-block constant into a performance-panel constant and a distinct M-panel constant lets the test verify both panels' exact padding values independently, catching a class of regression (e.g., an M-panel site reverting to the smaller padding, or the larger padding leaking into a performance-panel site) that the prior single-constant, single-count assertion could not distinguish.

      No product bugs were found. Every failure was the correct, expected consequence of a user-directed implementation reversion following visual feedback that the sixth critique cycle's fix was ineffective.
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
    <item priority="P1" target_mode="run-local" description="Run /run-local to visually re-verify the M-panel stat-label fix (implement build boost-shap-gii_implement_build_20260923_203500.md): confirm the M-panel's two stat labels (noise mean+SD, signal mean+SD) now render as opaque white boxes centered on the panel bottom edge (matching the performance panel's confirmed-working vjust=0.5 approach) with a visibly larger white rectangle (label.padding = unit(0.5, lines)) that occludes the density distribution tails, and confirm they no longer overlap the x-axis tick labels, across all three <external_project> child-child run directories." />
  </action_items>
</test_report>
