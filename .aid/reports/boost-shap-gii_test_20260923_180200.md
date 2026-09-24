<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-09-23T18:02:00-04:00" />
  <pre_design_run>
    <total>995</total>
    <passed>995</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures />
  </pre_design_run>
  <failing_test_dispositions />
  <design_phase>
    <tests_created>7</tests_created>
    <tests_modified>0</tests_modified>
    <files_created>
      <file path="tests/test_build_20260923b.py" test_count="7" coverage_target="Metric-ordering factor-preservation fix (build report boost-shap-gii_implement_build_20260923_175000.md): df_boot_long$metric factor(levels = metric_levels) wrapper in the has_boot_perf branch of the model performance plot (plot.R lines 446-471)." />
    </files_created>
    <design_rationale>
      The pre-design run had zero failures, so no disposition ledger was needed. However, first-class analytical review of recent implementation history (the /run-local visual re-verification and its follow-on /implement build) identified a coverage gap: the metric-ordering fix just applied to plot.R had zero existing test coverage. The prior /test cycle (2026-09-23, 17:15) covered two other critique-cycle-5 changes (vjust stat-label positioning and M-panel margin) via source-string re-expression, but the metric-ordering change (C1: R2-to-R-squared rename plus RMSE/MAE/R-squared factor ordering) was never tested, which is why the underlying bug (df_boot_long$metric losing its factor class via a bare ifelse(), causing bind_rows() to silently coerce the combined column to character and lose the level ordering) shipped undetected into visual re-verification.

      A pure source-string assertion would not reliably catch this bug class: the factor(...) call is present at the correct source location both before and after nearby edits, and only the specific presence of the levels = metric_levels argument at one call site, cross-referenced against bind_rows() coercion semantics, distinguishes correct from buggy behavior. Following this project's established convention of preferring genuine behavioral (known-answer) tests over source-string checks wherever a code unit is self-contained enough to extract and execute (precedent: the nudge_stat_labels() helper in test_build_20260922.py), the new test file extracts the exact metric_levels/df_dist construction block from plot.R via literal start/end marker anchoring (the block is a flat statement sequence, not a function definition, so balanced-brace scanning does not apply) and executes it against synthetic fixtures with deliberately scrambled metric input order via an Rscript subprocess.

      Before finalizing the test, the bug mechanism was empirically verified in isolation: bind_rows() combining a factor column with a character column for the same variable coerces the result to character and discards levels, while combining two factor columns with identical levels preserves both the factor type and level ordering. The new test was then confirmed to fail against a reconstruction of the pre-fix code (bare ifelse(), no factor wrapper) and pass against the current fix, establishing that the test is genuinely discriminating rather than a tautology. A complementary fast source-string guard class was added to lock the exact wrapper syntax and explicitly forbid the bare ifelse() regression pattern.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>1002</total>
    <passed>1002</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures />
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>proceed_to_document</recommendation>
  </summary>
  <action_items>
    <item priority="P1" target_mode="run-local" description="Run /run-local to visually re-verify the metric-ordering fix (build report boost-shap-gii_implement_build_20260923_175000.md): confirm the model performance panel now facets in RMSE, MAE, R-squared order (top to bottom) rather than the alphabetical order previously observed, against the three <external_project> child-child run directories." />
  </action_items>
</test_report>
