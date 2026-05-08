<run_local_report>
 <meta project="boost-shap-gii" mode="run-local" timestamp="2026-05-08T08:39:11-04:00" />
 <preflight>
 <sandbox_path>{sandbox_path}</sandbox_path>
 <run_directory>{sandbox_path}/runs/run_20260508_082707</run_directory>
 <environment>
 <name>boost_shap_gii</name>
 <status>exists</status>
 <snapshot_path>{sandbox_path}/runs/run_20260508_082707/environment_snapshot.txt</snapshot_path>
 </environment>
 <external_tools>
 <tool name="Rscript" status="found" path="/usr/local/bin/Rscript" version="4.3.1" />
 <tool name="nanoparquet" status="installed (this session)" version="0.4.2" />
 </external_tools>
 <input_data>
 <item path="tests/" type="test suite" status="valid" size="17 test files, 625 collected items" />
 </input_data>
 <readiness>blocked (pre-install); ready (post-install)</readiness>
 <blockers>
 <blocker>nanoparquet R package was not installed. Resolved by install.packages("nanoparquet") into {user_R_library} with user approval.</blocker>
 </blockers>
 </preflight>
 <execution>
 <command>conda run -n boost_shap_gii python -m pytest tests/ -v --tb=short</command>
 <exit_code>1</exit_code>
 <wall_clock_seconds>20.06</wall_clock_seconds>
 <peak_memory_mb>245</peak_memory_mb>
 <cpu_utilization>80.5%</cpu_utilization>
 <log_files>
 <log type="stdout+stderr" path="{sandbox_path}/runs/run_20260508_082707/pipeline_stdout.log" />
 </log_files>
 <output_manifest path="{sandbox_path}/runs/run_20260508_082707/pipeline_stdout.log" file_count="1" total_size_mb="0.07" />
 </execution>
 <validation>
 <output_accounting>
 <expected_outputs>625 (total collected test items)</expected_outputs>
 <found_outputs>625 (622 passed, 2 failed, 1 skipped)</found_outputs>
 <missing_outputs />
 <unexpected_outputs />
 </output_accounting>
 <log_analysis>
 <errors>2</errors>
 <warnings>2</warnings>
 <anomalies>
 <anomaly source="test_plot_smoke.py" line="16" severity="major">
 test_plot_r_argument_validation: asserts "At least 4 arguments must be supplied" but plot.R (post- refactoring) now requires only 1 argument (config path) and emits "At least 1 argument must be supplied". Test-spec mismatch, not a pipeline bug.
 </anomaly>
 <anomaly source="test_plot_smoke.py" line="44" severity="major">
 test_plot_r_discovers_shap_dirs: creates a config without required plot.* keys (outcome_max, negate_shap, gii_y_label, gii_y_sublabel, indiv_y_label, indiv_y_sublabel). plot.R dies at config validation before discovering SHAP dirs. Test-spec mismatch from, not a pipeline bug.
 </anomaly>
 <anomaly source="R warnings" line="N/A" severity="minor">
 R packages ggplot2 and nanoparquet were built under R 4.3.3 but the local R is 4.3.1. No functional impact; minor version compatibility warning.
 </anomaly>
 </anomalies>
 <key_metrics>
 <metric name="tests_passed" value="622" expected_range="625" status="warn" />
 <metric name="tests_failed" value="2" expected_range="0" status="warn" />
 <metric name="tests_skipped" value="1" expected_range="0-1" status="pass" />
 <metric name="wall_clock_seconds" value="18.27" expected_range="<60" status="pass" />
 </key_metrics>
 </log_analysis>
 <content_examination>
 <finding file="tests/test_plot_smoke.py" category="consistency">
 <observed>Two test assertions reference pre- plot.R interface (4-argument CLI, no plot.* config keys)</observed>
 <expected>Assertions should match the post- config-driven interface (1 argument, plot.* keys required)</expected>
 <assessment>anomalous</assessment>
 <explanation>The build change refactored plot.R from a multi-argument CLI to a config-driven interface. These tests were not updated during Session 9. The nanoparquet import failure masked these mismatches until now.</explanation>
 </finding>
 <finding file="tests/test_indiv_reports_unit.py" category="consistency">
 <observed>test_memory_overflow_raises skipped: No module named 'psutil'</observed>
 <expected>Conditional skip by design (psutil is optional)</expected>
 <assessment>correct</assessment>
 <explanation>The test is decorated with a skipif guard for psutil availability. This is intentional behavior for environments without psutil.</explanation>
 </finding>
 </content_examination>
 <cross_validation>
 <check description="nanoparquet install resolved R-dependent tests" status="consistent">
 Previous 4 failures attributed to nanoparquet: 2 were genuine import failures (now passing), 2 were assertion mismatches masked by the import failure (still failing for different reasons).
 </check>
 <check description="Prior session count vs. current" status="consistent">
 Session 9 reported 620 passed 4 failed 1 skipped. Current: 622 passed 2 failed 1 skipped. Net +2 passed from nanoparquet install.
 </check>
 </cross_validation>
 <critical_assessment>
 <overall_status>pass_with_warnings</overall_status>
 <definitive_errors />
 <concerning_anomalies>
 <anomaly>2 test_plot_smoke.py assertions are stale from refactoring (test-spec drift, not pipeline bugs). Fix required before 625/625 claim.</anomaly>
 </concerning_anomalies>
 <confirmed_correct>
 <item>nanoparquet 0.4.2 installed and functional; 2 previously-env-blocked tests now pass.</item>
 <item>622/625 tests pass; all pipeline logic tests green.</item>
 <item>1 skip (psutil conditional) is by design.</item>
 <item>No new regressions introduced.</item>
 </confirmed_correct>
 <scientific_summary>The pipeline test suite is functionally sound. The 2 remaining failures are test-specification drift from the refactoring of plot.R, not pipeline logic errors. The underlying R script behavior is correct (proper config validation, proper argument checking). The tests need updating to match the refactored interface.</scientific_summary>
 </critical_assessment>
 </validation>
 <summary>
 <pipeline_status>success</pipeline_status>
 <validation_status>pass_with_warnings</validation_status>
 <recommendation>fix_via_implement</recommendation>
 </summary>
 <action_items>
 <item priority="P1" target_mode="implement" description="Update test_plot_smoke.py: (1) test_plot_r_argument_validation should assert 'At least 1 argument must be supplied' and test with zero args. (2) test_plot_r_discovers_shap_dirs should include required plot.* config keys (outcome_max, negate_shap, gii_y_label, gii_y_sublabel, indiv_y_label, indiv_y_sublabel) in the dummy config." />
 <item priority="P2" target_mode="run-local" description="R version mismatch warning (R 4.3.1 vs packages built for 4.3.3). No functional impact; upgrade R to 4.3.3+ if warnings are undesirable." />
 </action_items>
</run_local_report>
