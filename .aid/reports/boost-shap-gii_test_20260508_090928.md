<test_report>
 <meta project="boost-shap-gii" mode="test" timestamp="2026-05-08T09:09:28-04:00" />
 <pre_design_run>
 <total>625</total>
 <passed>622</passed>
 <failed>2</failed>
 <errors>0</errors>
 <coverage_pct>null</coverage_pct>
 <notes>Pre-design baseline taken from /run-local run_20260508_082707 (18.27s ago, no intervening code changes; deterministic failures). 1 skipped (psutil conditional, by design).</notes>
 <failures>
 <failure test="test_plot_r_argument_validation" file="tests/test_plot_smoke.py" line="16">
 <error_type>AssertionError</error_type>
 <message>'At least 4 arguments must be supplied' not in stderr; actual stderr indicates plot.R reached file-open of 'dummy.yaml' (passed argument validation with 1 arg).</message>
 <traceback>tests/test_plot_smoke.py:16: assert "At least 4 arguments must be supplied" in result.stderr; actual stderr: "Error in file(file, 'rt', encoding = fileEncoding): cannot open the connection" (R got past argument check; assertion message is stale from pre- 4-arg CLI).</traceback>
 </failure>
 <failure test="test_plot_r_discovers_shap_dirs" file="tests/test_plot_smoke.py" line="44">
 <error_type>AssertionError</error_type>
 <message>'Found 1 SHAP output directory' not in result.stdout; plot.R died at config validation due to missing required plot.* keys.</message>
 <traceback>tests/test_plot_smoke.py:44: assert "Found 1 SHAP output directory" in result.stdout; actual stderr: "Error: Missing required plot.* config keys: outcome_max, negate_shap, gii_y_label, gii_y_sublabel, indiv_y_label, indiv_y_sublabel". Test config lacks the post- required plot.* schema.</traceback>
 </failure>
 </failures>
 </pre_design_run>
 <failing_test_dispositions>
 <disposition test="test_plot_r_argument_validation" file="tests/test_plot_smoke.py" classification="obsolete-test">
 <intended_contract>plot.R must reject invocations with insufficient arguments via a non-zero stop exit. The test encodes the principle of argument-validation hardening; the specific minimum-arg count is part of the API surface and changes with refactoring.</intended_contract>
 <current_test_claim>With 1 argument ('dummy.yaml'), result.returncode != 0 AND stderr contains 'At least 4 arguments must be supplied'.</current_test_claim>
 <evidence>plot.R lines 18-21 (post-): `if (length(args) &lt; 1) stop("At least 1 argument must be supplied: CONFIG_PATH", call.=FALSE)`. build change documented in.aid/reports/boost-shap-gii_implement_build_20260507_124302.md and brainstorm 20260507_173959.md (refactor of plot.R from 4-argument CLI to config-driven 1-argument CLI). Old assertion never matches the new error message.</evidence>
 <action>re-express. Invoke with 0 args (matching the post- threshold of length(args) &lt; 1). Preserve `result.returncode != 0`. Update string assertion to "At least 1 argument must be supplied". Add `"CONFIG_PATH" in result.stderr` as a strict strengthening (additional bytes of the error message pinned).</action>
 </disposition>
 <disposition test="test_plot_r_discovers_shap_dirs" file="tests/test_plot_smoke.py" classification="obsolete-test">
 <intended_contract>plot.R must discover shap_analysis subdirectories under RUN_DIR and emit "Found N SHAP output director(y|ies)" for the count, given a valid config.</intended_contract>
 <current_test_claim>With a config containing only paths/execution/shap keys, plot.R outputs "Found 1 SHAP output directory" to stdout when invoked with `[cfg_path, "10", "false", "Label"]`.</current_test_claim>
 <evidence>plot.R lines 49-57: validates `plot.outcome_max`, `plot.negate_shap`, `plot.gii_y_label`, `plot.gii_y_sublabel`, `plot.indiv_y_label`, `plot.indiv_y_sublabel` and stops if any are missing. plot.R lines 107-108: confirms `cat(sprintf("[INFO] Found %d SHAP output director%s to plot.\n"...))` discovery message. plot.R lines 68-69: `if (length(args) >= 2) RUN_DIR &lt;- args[2]` — legacy positional args ("10", "false", "Label") would override RUN_DIR with the literal string "10" (non-existent dir), causing discovery to fail before the assertion target.</evidence>
 <action>re-express. Add the 6 required plot.* keys to the dummy config. Drop the now-meaningless extra positional args ("10", "false", "Label") that would override RUN_DIR with a non-existent path. The core assertion `"Found 1 SHAP output directory" in result.stdout` is preserved verbatim.</action>
 </disposition>
 </failing_test_dispositions>
 <design_phase>
 <tests_created>0</tests_created>
 <tests_modified>2</tests_modified>
 <files_created />
 <files_modified>
 <file path="tests/test_plot_smoke.py" test_count="2" coverage_target="plot.R argument validation and SHAP directory discovery (post- config-driven CLI)" />
 </files_modified>
 <design_rationale>Both failing tests classified as obsolete-test (intended contract changed in, assertions stale). Re-expressions strictly preserve or strengthen postconditions: test 1 retains returncode check, updates message string, AND adds CONFIG_PATH substring check (+1 assertion). Test 2 retains the exact discovery-message assertion verbatim while updating the test setup (config schema and CLI invocation) to match the post- contract. No assertions weakened, no skips/xfails introduced, no try/except suppression. Per Meyer (1992) postcondition-strengthening rule and Bairi et al. (2025) anti-pattern guidance against LLM-test-repair weakening.</design_rationale>
 </design_phase>
 <post_design_run>
 <total>625</total>
 <passed>624</passed>
 <failed>0</failed>
 <errors>0</errors>
 <coverage_pct>null</coverage_pct>
 <notes>Dispatched via execution-agent-sonnet-medium (validated dispatch + return via validate_io.py). 624 passed, 1 skipped (psutil conditional, by design), 0 failures, 0 errors. Wall clock 15.45s. Net change vs pre-design: +2 passed, -2 failed.</notes>
 <failures />
 </post_design_run>
 <summary>
 <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
 <bugs_routed_to_implement>0</bugs_routed_to_implement>
 <recommendation>proceed_to_document</recommendation>
 </summary>
 <action_items>
 <item priority="P3" target_mode="run-local" description="R version mismatch warning (R 4.3.1 vs ggplot2/nanoparquet packages built for 4.3.3). No functional impact; upgrade R to 4.3.3+ if warnings are undesirable." />
 </action_items>
</test_report>
