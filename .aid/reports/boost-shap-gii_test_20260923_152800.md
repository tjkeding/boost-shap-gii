<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-09-23T15:28:00-04:00" />
  <pre_design_run>
    <total>985</total>
    <passed>978</passed>
    <failed>7</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="TestPlotR_VContributionRanking::test_underscore_wordwrap_replaces_n_count_lookup" file="tests/test_build_20260507.py" line="305">
        <error_type>AssertionError</error_type>
        <message>Expected underscore word-wrap transform to appear exactly twice; found 0 occurrence(s).</message>
        <traceback>assert 0 == 2 (gsub separator changed from newline to space)</traceback>
      </failure>
      <failure test="TestBootstrapDensityRendering::test_conditional_shared_legend_density_present" file="tests/test_build_20260921.py" line="281">
        <error_type>AssertionError</error_type>
        <message>scale_fill_manual with name = "Distribution" not found in plot.R source.</message>
        <traceback>name = "Distribution" removed (name = NULL)</traceback>
      </failure>
      <failure test="TestPerfPlotStatsAndLegend::test_null_mean_vline_present" file="tests/test_build_20260921b.py" line="85">
        <error_type>AssertionError</error_type>
        <message>Dashed null mean vline styling not found in plot.R source.</message>
        <traceback>linetype = "dashed" removed from null mean geom_vline</traceback>
      </failure>
      <failure test="TestInteractionLegendAscendingOrder::test_singleton_legend_reverse_unaffected" file="tests/test_build_20260922.py" line="198">
        <error_type>AssertionError</error_type>
        <message>Singleton discrete V-panel legend must retain reverse=TRUE; not found.</message>
        <traceback>reverse = TRUE removed from singleton legend per T8</traceback>
      </failure>
      <failure test="TestBelowAxisLabelPositioning::test_vjust_3_5_present_at_all_five_sites" file="tests/test_build_20260922b.py" line="143">
        <error_type>AssertionError</error_type>
        <message>vjust=3.5 stat-label styling absent; expected 5 occurrences, found 0.</message>
        <traceback>vjust changed from 3.5 to 1.0 at all five sites</traceback>
      </failure>
      <failure test="TestBelowAxisMarginAccommodation::test_perf_panel_bottom_margin_increased" file="tests/test_build_20260922b.py" line="189">
        <error_type>AssertionError</error_type>
        <message>plot.margin bottom=20 not found; expected 2 occurrences, found 0.</message>
        <traceback>margin bottom changed from 20 to 8 at both branches</traceback>
      </failure>
      <failure test="TestBelowAxisMarginAccommodation::test_m_panel_bottom_margin_increased" file="tests/test_build_20260922b.py" line="205">
        <error_type>AssertionError</error_type>
        <message>plot.margin unit(c(1,0.5,12,1),"mm") not found in plot.R source.</message>
        <traceback>M-panel bottom margin changed from 12mm to 4mm</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
    <disposition test="TestPlotR_VContributionRanking::test_underscore_wordwrap_replaces_n_count_lookup" file="tests/test_build_20260507.py" classification="obsolete-test">
      <intended_contract>Discrete x-axis labels apply a uniform underscore-transform at exactly two sites (singleton discrete path + interaction discrete-focal path).</intended_contract>
      <current_test_claim>Transform is gsub("_", "\n", levels(fac)), occurs exactly twice.</current_test_claim>
      <evidence>boost-shap-gii_implement_plan_20260923_101500.md change C6 (T7, locked): separator changed from newline to space to eliminate the "NA" whitespace bug caused by combining the newline transform with the "__NA__" sentinel.</evidence>
      <action>Re-expressed: pattern updated to gsub("_", " ", levels(fac)); cardinality check (==2) preserved; added explicit assertion that the deprecated newline form is absent.</action>
    </disposition>
    <disposition test="TestBootstrapDensityRendering::test_conditional_shared_legend_density_present" file="tests/test_build_20260921.py" classification="obsolete-test">
      <intended_contract>Shared legend distinguishes Permutation Null vs. Trained distributions by color/fill.</intended_contract>
      <current_test_claim>scale_fill_manual/scale_color_manual include name = "Distribution".</current_test_claim>
      <evidence>Tech spec change C3 (T3, locked/approved): "Distribution" legend title removed since the entries are self-explanatory.</evidence>
      <action>Re-expressed: asserts name = NULL on both scales; added explicit assertion that name = "Distribution" is absent.</action>
    </disposition>
    <disposition test="TestPerfPlotStatsAndLegend::test_null_mean_vline_present" file="tests/test_build_20260921b.py" classification="obsolete-test">
      <intended_contract>Null mean vline is visually distinguished from the trained score vline.</intended_contract>
      <current_test_claim>linetype = "dashed" present in the null mean geom_vline styling.</current_test_claim>
      <evidence>Tech spec change C4 (T4, locked/approved): null line changed to solid, differentiated by color alone, matching the trained line's style.</evidence>
      <action>Re-expressed: asserts solid styling (color/linewidth only); added explicit assertion that the dashed form is absent from this construct.</action>
    </disposition>
    <disposition test="TestInteractionLegendAscendingOrder::test_singleton_legend_reverse_unaffected" file="tests/test_build_20260922.py" classification="obsolete-test">
      <intended_contract>(Cycle-3 contract) Singleton discrete V-panel legend retains reverse=TRUE; only the interaction legend goes ascending.</intended_contract>
      <current_test_claim>guide = guide_legend(reverse = TRUE, override.aes = list(alpha = 1)) present, with rationale "F4 is interaction-only."</current_test_claim>
      <evidence>Tech spec change C7 (T8, locked): user explicitly stated "All legends need to go from low (top) to high (bottom) values," directly and intentionally reversing cycle 3's interaction-only scoping.</evidence>
      <action>Re-expressed and renamed to test_singleton_legend_reverse_removed: asserts reverse=TRUE is now absent from the singleton legend; docstring updated to record the supersession.</action>
    </disposition>
    <disposition test="TestBelowAxisLabelPositioning::test_vjust_3_5_present_at_all_five_sites" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>(Cycle-3 contract) Stat labels sit below the x-axis tick marks/labels at vjust=3.5.</intended_contract>
      <current_test_claim>vjust = 3.5 occurs exactly 5 times.</current_test_claim>
      <evidence>Tech spec change C2 (T2, locked/approved): user reported cycle-3's vjust=3.5 placed labels under the tick marks (overcorrection); repositioned to vjust=1.0, between the distribution base and the ticks.</evidence>
      <action>Re-expressed and renamed to test_vjust_1_0_present_at_all_five_sites: value and cardinality updated; added explicit assertions that both 3.5 and the older 1.5 are absent.</action>
    </disposition>
    <disposition test="TestBelowAxisMarginAccommodation::test_perf_panel_bottom_margin_increased" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>(Cycle-3 contract) Performance panel bottom margin=20 accommodates below-tick text at both branches.</intended_contract>
      <current_test_claim>margin(5.5, 5.5, 20, 5.5) occurs exactly twice.</current_test_claim>
      <evidence>Tech spec change C2 (T2/T6, locked/approved): margin reduced to 8 since labels no longer render below the ticks.</evidence>
      <action>Re-expressed and renamed to test_perf_panel_bottom_margin_reduced: value updated to 8, cardinality preserved (2 sites); 20 added to the deprecated-value list.</action>
    </disposition>
    <disposition test="TestBelowAxisMarginAccommodation::test_m_panel_bottom_margin_increased" file="tests/test_build_20260922b.py" classification="obsolete-test">
      <intended_contract>(Cycle-3 contract) M-panel bottom margin=12mm accommodates below-tick text.</intended_contract>
      <current_test_claim>unit(c(1, 0.5, 12, 1), "mm") present; explicitly asserted unit(c(1, 0.5, 4, 1), "mm") absent (calling it deprecated).</current_test_claim>
      <evidence>Tech spec change C2 (T2/T6, locked/approved): margin reduced to 4mm. Note: the pre-existing test's "deprecated" polarity for the 4mm value was factually inverted by this change (4mm is now current, not deprecated).</evidence>
      <action>Re-expressed and renamed to test_m_panel_bottom_margin_reduced: asserts 4mm present; 12mm and 6mm both added to the deprecated-value list (polarity flipped for 4mm).</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>10</tests_created>
    <tests_modified>7</tests_modified>
    <files_created>
      <file path="tests/test_build_20260923.py" test_count="10" coverage_target="Four critique-cycle-4 changes with zero prior coverage: performance panel single-column layout + metric-count-keyed sizing (T1); GII ggsave height increase at both save sites (T5); NA sentinel simplified to bare 'NA' (T7, new coverage distinct from the gsub separator re-expression); continuous-gradient guide_colorbar(reverse=TRUE) (T8); create_ordered_factor NA-last guarantee via genuine behavioral known-answer tests (T8)." />
    </files_created>
    <design_rationale>
      Seven failures were disposed as obsolete-test (100%): each correctly encoded critique-cycle-3's contract, and critique-cycle-4's locked, user-approved decisions (T1-T8) explicitly superseded that contract at the exact sites the tests checked. All re-expressions preserve or strengthen their postconditions per the Meyer 1992 rule: every re-expression keeps the original cardinality/structural check and adds an explicit absence assertion for the newly-deprecated value, which is strictly more specific than the original (which checked presence only, not absence of stale forms).

      A coverage-gap audit (grepping for ncol/fig_w/height=1.275/guide_colorbar/na_sentinel patterns across the full test suite) found four cycle-4 changes with zero existing assertions: the performance panel's single-column layout and simplified sizing formula (T1), the GII ggsave height increase (T5), the continuous-gradient colorbar reversal (T8), and the create_ordered_factor NA-last relocation logic (T8). These are covered by a new file, tests/test_build_20260923.py, following the project's established per-build-date file convention. create_ordered_factor is a small, self-contained pure function with no CLI-arg dependency, so it was extracted via the project's established balanced-brace-scan + Rscript-subprocess technique (precedent: nudge_stat_labels in test_build_20260922.py) for genuine behavioral verification (NA relocated when encoded lowest/middle/already-highest; non-NA ordering unaffected when no sentinel present; custom na_sentinel parameter genuinely wired in) rather than a source-string check, since the NA-last guarantee is a runtime behavior, not a static text pattern.

      The Python-side "__NA__" categorical encoding sentinel (train.py/utils.py, used for missing nominal values during CatBoost training and shadow permutation) was confirmed via cross-reference to be a distinct, unrelated mechanism from the plot.R display-only sentinel changed in this build; no existing Python-side tests required disposition.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>995</total>
    <passed>995</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures />
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>proceed_to_run-local</recommendation>
  </summary>
  <action_items>
    <item priority="P1" target_mode="run-local" description="Visual re-verification of the critique-cycle-4 rendering changes against the three <external_project> run directories (agg/cluster/item), per the standing Session 20/21 sequencing: run-local -> document -> publish v1.7.0." />
  </action_items>
</test_report>
