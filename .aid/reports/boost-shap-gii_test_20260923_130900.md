<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-09-23T13:09:00-04:00" />
  <pre_design_run>
    <total>985</total>
    <passed>978</passed>
    <failed>4</failed>
    <errors>3</errors>
    <coverage_pct></coverage_pct>
    <failures>
      <failure test="TestMPanelPerFeatureAxis::test_legend_repositioned_and_resized" file="tests/test_build_20260921b.py" line="219">
        <error_type>AssertionError</error_type>
        <message>assert "dist_midpoint &lt;- (noise_m + signal_m) / 2" in plot_r_source</message>
        <traceback>tests/test_build_20260921b.py:219: AssertionError</traceback>
      </failure>
      <failure test="TestBelowAxisLabelPositioning::test_vjust_1_5_present_at_all_five_sites" file="tests/test_build_20260922b.py" line="174">
        <error_type>AssertionError</error_type>
        <message>Expected 5 occurrences of vjust=1.5 stat-label styling; assert 0 == 5</message>
        <traceback>tests/test_build_20260922b.py:174: AssertionError</traceback>
      </failure>
      <failure test="TestBelowAxisMarginAccommodation::test_perf_panel_bottom_margin_increased" file="tests/test_build_20260922b.py" line="208">
        <error_type>AssertionError</error_type>
        <message>Expected exactly 2 occurrences of "plot.margin = margin(5.5, 5.5, 12, 5.5)"; assert 0 == 2</message>
        <traceback>tests/test_build_20260922b.py:208: AssertionError</traceback>
      </failure>
      <failure test="TestBelowAxisMarginAccommodation::test_m_panel_bottom_margin_increased" file="tests/test_build_20260922b.py" line="220">
        <error_type>AssertionError</error_type>
        <message>assert 'plot.margin = unit(c(1, 0.5, 6, 1), "mm")' in plot_r_source</message>
        <traceback>tests/test_build_20260922b.py:220: AssertionError</traceback>
      </failure>
      <failure test="TestAdaptiveLegendPlacement::test_left_clustered_distributions_place_legend_right" file="tests/test_build_20260922b.py" line="92">
        <error_type>ValueError</error_type>
        <message>substring not found (start_marker in plot.R source)</message>
        <traceback>tests/test_build_20260922b.py:92 -&gt; tests/test_build_20260922b.py:68 (_extract_r_conditional_block): ValueError: substring not found</traceback>
      </failure>
      <failure test="TestAdaptiveLegendPlacement::test_right_clustered_distributions_place_legend_left" file="tests/test_build_20260922b.py" line="92">
        <error_type>ValueError</error_type>
        <message>substring not found (start_marker in plot.R source)</message>
        <traceback>tests/test_build_20260922b.py:92 -&gt; tests/test_build_20260922b.py:68 (_extract_r_conditional_block): ValueError: substring not found</traceback>
      </failure>
      <failure test="TestAdaptiveLegendPlacement::test_boundary_midpoint_places_legend_left" file="tests/test_build_20260922b.py" line="92">
        <error_type>ValueError</error_type>
        <message>substring not found (start_marker in plot.R source)</message>
        <traceback>tests/test_build_20260922b.py:92 -&gt; tests/test_build_20260922b.py:68 (_extract_r_conditional_block): ValueError: substring not found</traceback>
      </failure>
    </failures>
  </pre_design_run>

  <failing_test_dispositions>
    <disposition test="TestMPanelPerFeatureAxis::test_legend_repositioned_and_resized" file="tests/test_build_20260921b.py" classification="obsolete-test">
      <intended_contract>Per the locked brainstorm decision (boost-shap-gii_brainstorm_20260922_184900.md, Topic T3), the M-panel legend uses a fixed bottom horizontal legend (legend.position = "bottom", legend.direction = "horizontal"); the prior adaptive left/right inset heuristic (dist_midpoint, if/else branch, legend_x, legend_just_x, positional legend.position/legend.justification theme entries) is removed as dead code.</intended_contract>
      <current_test_claim>Asserted presence of "dist_midpoint &lt;- (noise_m + signal_m) / 2", the if/else branch, legend_x/legend_just_x assignments, and the positional legend.position = c(legend_x, 0.95)/legend.justification = c(legend_just_x, 1) theme entries.</current_test_claim>
      <evidence>boost-shap-gii_brainstorm_20260922_184900.md Topic T3 (locked): "Replace the adaptive inset legend with a fixed bottom-positioned horizontal legend... Remove the adaptive heuristic code as dead code." Confirmed implemented at src/boost_shap_gii/scripts/plot.R:644-645 (legend.position = "bottom", legend.direction = "horizontal"); confirmed absent via grep (zero occurrences of dist_midpoint/legend_x/legend_just_x anywhere in plot.R) in boost-shap-gii_implement_build_20260923_121000.md, change C3.</evidence>
      <action>re-express: assert fixed legend.position/legend.direction presence plus explicit absence of all 6 removed adaptive-heuristic constructs (strengthens the prior single-negative-check test).</action>
    </disposition>
    <disposition test="TestBelowAxisLabelPositioning::test_vjust_1_5_present_at_all_five_sites" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>Per the locked brainstorm (Topics T1/T2), all 5 below-axis stat-label annotation sites use vjust=3.5 (increased from 1.5), because vjust=1.5 still overlapped the x-axis tick marks/labels.</intended_contract>
      <current_test_claim>plot_r_source.count('vjust = 1.5, hjust = 0.5, size = 1.5, color = "black"') == 5</current_test_claim>
      <evidence>boost-shap-gii_brainstorm_20260922_184900.md Topics T1, T2 (locked): "Increase vjust from 1.5 to 3.5 at all performance-panel annotation sites" / "at both M-panel annotate() call sites." Confirmed implemented: 5 occurrences of vjust = 3.5 at plot.R:469,473,511,645,648; 0 occurrences of vjust = 1.5 remain.</evidence>
      <action>re-express: assert 5 occurrences of the vjust=3.5 styling string, plus explicit absence of the deprecated vjust=1.5 string (strengthens the prior test, which had no negative check for this exact deprecated value).</action>
    </disposition>
    <disposition test="TestBelowAxisMarginAccommodation::test_perf_panel_bottom_margin_increased" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>Per the locked brainstorm (Topic T1), the performance panel bottom margin is increased from 12 to 20 in both branches (has_boot_perf and CI-band fallback), to accommodate the vjust=3.5 displacement.</intended_contract>
      <current_test_claim>plot_r_source.count("plot.margin = margin(5.5, 5.5, 12, 5.5)") == 2; asserted absence of margin(5.5, 5.5, 9, 5.5).</current_test_claim>
      <evidence>boost-shap-gii_brainstorm_20260922_184900.md Topic T1 (locked): "Increase bottom margin from 12 to 20 in both branches." Confirmed implemented at plot.R:487,525 (margin(5.5, 5.5, 20, 5.5)); 0 occurrences of margin(...,12,...) remain.</evidence>
      <action>re-express: assert count == 2 for margin value 20, plus chained absence of both prior deprecated values (12 and 9), strengthening regression coverage across the full margin-value history.</action>
    </disposition>
    <disposition test="TestBelowAxisMarginAccommodation::test_m_panel_bottom_margin_increased" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>Per the locked brainstorm (Topic T2), the M-panel bottom margin is increased from 6mm to 12mm, to accommodate the vjust=3.5 displacement.</intended_contract>
      <current_test_claim>Asserted presence of unit(c(1, 0.5, 6, 1), "mm"); asserted absence of unit(c(1, 0.5, 4, 1), "mm").</current_test_claim>
      <evidence>boost-shap-gii_brainstorm_20260922_184900.md Topic T2 (locked): "Increase bottom margin from 6mm to 12mm." Confirmed implemented at plot.R:661 (unit(c(1, 0.5, 12, 1), "mm")).</evidence>
      <action>re-express: assert presence of the 12mm value, plus chained absence of both prior deprecated values (6mm and 4mm).</action>
    </disposition>
    <disposition test="TestAdaptiveLegendPlacement::test_left_clustered_distributions_place_legend_right" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>Per the locked brainstorm (Topic T3), the adaptive distribution-mass-center legend-placement heuristic no longer exists in plot.R; it has been replaced unconditionally by a fixed bottom horizontal legend.</intended_contract>
      <current_test_claim>Behavioral known-answer test extracting the dist_midpoint-anchored if/else block via balanced-brace scanning and executing it via Rscript for 3 input configurations; extraction itself fails (ValueError) because the anchor string no longer exists.</current_test_claim>
      <evidence>boost-shap-gii_brainstorm_20260922_184900.md Topic T3 (locked): the adaptive heuristic is explicitly removed as dead code. Confirmed absent via grep (zero occurrences of dist_midpoint anywhere in plot.R) in boost-shap-gii_implement_build_20260923_121000.md, change C3, which also flagged this exact test file for expected obsolescence.</evidence>
      <action>re-express: the code-under-test has been deleted, not merely changed, so a like-for-like re-express is not possible; the 3-test class is replaced with a new class (TestFixedBottomHorizontalLegend) verifying an unconditional invariant (fixed bottom horizontal legend holds regardless of distribution configuration) plus full dead-code absence -- a strictly stronger guarantee than the 3-case conditional the deleted heuristic provided, per Meyer 1992.</action>
    </disposition>
    <disposition test="TestAdaptiveLegendPlacement::test_right_clustered_distributions_place_legend_left" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>Same as above.</intended_contract>
      <current_test_claim>Same extraction failure, second input configuration.</current_test_claim>
      <evidence>Same as above.</evidence>
      <action>Same re-express action as above (single replacement class covers all 3 deleted tests).</action>
    </disposition>
    <disposition test="TestAdaptiveLegendPlacement::test_boundary_midpoint_places_legend_left" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>Same as above.</intended_contract>
      <current_test_claim>Same extraction failure, third (boundary) input configuration.</current_test_claim>
      <evidence>Same as above.</evidence>
      <action>Same re-express action as above.</action>
    </disposition>
  </failing_test_dispositions>

  <design_phase>
    <tests_created>0</tests_created>
    <tests_modified>2</tests_modified>
    <files_created>
      <file path="tests/test_build_20260921b.py" test_count="1 method re-expressed" coverage_target="M-panel fixed bottom horizontal legend contract (was: adaptive left/right inset placement)" />
      <file path="tests/test_build_20260922b.py" test_count="3 methods re-expressed, 1 class (3 tests) replaced" coverage_target="vjust=3.5 below-axis stat labels; performance/M-panel margin increases (20 / 12mm); dead-code absence + unconditional fixed-bottom-horizontal-legend presence (replacing deleted adaptive-heuristic behavioral tests)" />
    </files_created>
    <design_rationale>Critique cycle 3 (boost-shap-gii_brainstorm_20260922_184900.md, T1-T3) is a values-only refinement of already-tested plot.R constructs (vjust, margin, legend placement) with no new code paths, so no new test file was warranted. All 7 pre-design failures were confirmed obsolete-test (locked design decision superseded the asserted values/logic, and the implementation was verified exact against the tech spec). Re-expressions strictly preserve or strengthen postconditions: negative absence checks were added or chained across the full deprecated-value history everywhere a positive value assertion changed, and the 3 deleted-heuristic behavioral tests were replaced with a class asserting a strictly stronger unconditional invariant plus explicit dead-code absence, rather than being silently dropped. No new test file was needed; total collected test count is unchanged at 985 (net zero: 1 method renamed, 3 methods replaced 1:1).</design_rationale>
  </design_phase>

  <post_design_run>
    <total>985</total>
    <passed>985</passed>
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
  <action_items>
    <item priority="P1" target_mode="run-local" description="Re-verify critique cycle 3 rendering (below-axis label/margin spacing and the fixed bottom horizontal legend) against real <external_project> child-child data before considering the plotting feature set visually stable." />
    <item priority="P1" target_mode="document" description="Once run-local re-verification passes, update README, INPUT_SPECIFICATION, and AID_LOG for the full v1.7.0 plotting feature set across Session 19 and all three Session 20/23 critique cycles, then publish v1.7.0." />
  </action_items>
</test_report>
