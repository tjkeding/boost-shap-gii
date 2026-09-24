<test_report>
  <meta project="boost-shap-gii" mode="test" submodule="run_suite" timestamp="2026-09-23T17:05:00-04:00" />
  <pre_design_run>
    <total>995</total>
    <passed>993</passed>
    <failed>2</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="TestBelowAxisLabelPositioning::test_vjust_1_0_present_at_all_five_sites" file="tests/test_build_20260922b.py" line="147">
        <error_type>AssertionError</error_type>
        <message>Expected 5 occurrences of the below-axis vjust=1.0 stat-label styling; found 0 (plot.R now uses vjust=0.5 at all 5 sites per critique-cycle-5 change C2).</message>
        <traceback>FAILED tests/test_build_20260922b.py::TestBelowAxisLabelPositioning::test_vjust_1_0_present_at_all_five_sites</traceback>
      </failure>
      <failure test="TestBelowAxisMarginAccommodation::test_m_panel_bottom_margin_reduced" file="tests/test_build_20260922b.py" line="225">
        <error_type>AssertionError</error_type>
        <message>Expected M-panel plot.margin bottom value of 4mm; plot.R now uses 1mm per critique-cycle-5 change C3.</message>
        <traceback>FAILED tests/test_build_20260922b.py::TestBelowAxisMarginAccommodation::test_m_panel_bottom_margin_reduced</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
    <disposition test="TestBelowAxisLabelPositioning::test_vjust_1_0_present_at_all_five_sites" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>Prior contract (fourth critique cycle, 2026-09-23): all 5 below-axis stat-label annotation sites use vjust=1.0. Critique cycle 5 (implement plan boost-shap-gii_implement_plan_20260923_130000.md, change C2) explicitly superseded this: vjust reduced from 1.0 to 0.5 at all 5 sites per direct user feedback that 1.0 was close but needed additional gap above the x-axis tick marks.</intended_contract>
      <current_test_claim>Asserts exactly 5 occurrences of 'vjust = 1.0, hjust = 0.5, size = 1.5, color = "black"' in plot.R, and asserts vjust=3.5 and vjust=1.5 are absent (prior deprecated values).</current_test_claim>
      <evidence>boost-shap-gii_implement_plan_20260923_130000.md change C2 spec: "change vjust from 1.0 to 0.5 at all 5 sites". Direct grep of current plot.R confirms 5 occurrences of vjust=0.5 and 0 occurrences of vjust=1.0.</evidence>
      <action>route-to-design: re-express the assertion to check for vjust=0.5 (count==5) and add vjust=1.0 to the deprecated-value exclusion list, preserving the existing exclusions for 3.5 and 1.5. This strengthens the postcondition (now forbids three deprecated values instead of two) without weakening any existing check.</action>
    </disposition>
    <disposition test="TestBelowAxisMarginAccommodation::test_m_panel_bottom_margin_reduced" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>Prior contract (fourth critique cycle, 2026-09-23): M-panel plot.margin bottom is 4mm. Critique cycle 5 (implement plan boost-shap-gii_implement_plan_20260923_130000.md, change C3) explicitly superseded this: bottom margin reduced from 4mm to 1mm, driven by the vjust reduction in C2 needing less below-panel accommodation and the user's explicit request to maximize the M-panel distribution area.</intended_contract>
      <current_test_claim>Asserts 'plot.margin = unit(c(1, 0.5, 4, 1), "mm")' is present in plot.R, and asserts the 12mm and 6mm deprecated values are absent.</current_test_claim>
      <evidence>boost-shap-gii_implement_plan_20260923_130000.md change C3 spec: "plot.margin bottom from 4mm to 1mm". Direct grep of current plot.R confirms 'plot.margin = unit(c(1, 0.5, 1, 1), "mm")' is present.</evidence>
      <action>route-to-design: re-express the assertion to check for the 1mm value and add the 4mm value to the deprecated-value exclusion list, preserving the existing exclusions for 12mm and 6mm. This strengthens the postcondition without weakening any existing check.</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>0</tests_created>
    <tests_modified>0</tests_modified>
    <files_created />
    <design_rationale>Not executed. This invocation was scoped to the run_suite submodule only (per explicit /test run_suite invocation); no design-phase test edits were made. Both failures are dispositioned above as obsolete-test candidates for re-expression, pending a /test design invocation.</design_rationale>
  </design_phase>
  <post_design_run />
  <summary>
    <assertions_preserved_or_strengthened>n/a (design phase not executed this invocation)</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>Invoke /test design to re-express the two obsolete-test assertions identified above (both are direct consequences of the approved critique-cycle-5 changes C2 and C3; no product-bug disposition applies). Once the suite is at 995/995, proceed to /run-local for critique-cycle-5 visual re-verification.</recommendation>
  </summary>
  <action_items>
    <item priority="P1" target_mode="test" description="Run /test design to re-express test_vjust_1_0_present_at_all_five_sites and test_m_panel_bottom_margin_reduced in tests/test_build_20260922b.py to match the critique-cycle-5 values (vjust=0.5, M-panel bottom margin=1mm), per the dispositions recorded above." />
  </action_items>
</test_report>
