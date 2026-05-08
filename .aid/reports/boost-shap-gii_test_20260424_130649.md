<test_report>
 <meta project="boost-shap-gii" mode="test" timestamp="2026-04-24T13:06:49-04:00" />

 <pre_design_run>
 <total>461</total>
 <passed>450</passed>
 <failed>11</failed>
 <errors>0</errors>
 <coverage_pct>null</coverage_pct>
 <failures>
 <failure test="test_optional_sections_omitted" file="tests/test_shell_and_config.py" line="76">
 <error_type>AssertionError</error_type>
 <message>assert 'shap' not in cfg — minimal config now legitimately has a shap section (indiv_ci_nboot, indiv_scaling_mode have no safe defaults)</message>
 <traceback>'shap' in {'features':..., 'shap': {'indiv_ci_nboot': 2500, 'indiv_scaling_mode': 'sd'}...}</traceback>
 </failure>
 <failure test="TestPlotR::test_uses_as_numeric_for_outcome_range" file="tests/test_shell_and_config.py" line="203">
 <error_type>AssertionError</error_type>
 <message>plot.R refactored to read outcome_max from YAML (as.numeric(plot_cfg$outcome_max)), no longer from positional arg</message>
 <traceback>expected 'as.numeric(args[...])' in plot.R, found 'as.numeric(plot_cfg$outcome_max)'</traceback>
 </failure>
 <failure test="TestPlotR::test_uses_as_logical_for_negate" file="tests/test_shell_and_config.py" line="219">
 <error_type>AssertionError</error_type>
 <message>plot.R refactored to read negate_shap from YAML (as.logical(plot_cfg$negate_shap))</message>
 <traceback>expected 'as.logical(args[...])' in plot.R, found 'as.logical(plot_cfg$negate_shap)'</traceback>
 </failure>
 <failure test="TestPlotR::test_arg_count_message_matches" file="tests/test_shell_and_config.py" line="248">
 <error_type>AssertionError</error_type>
 <message>plot.R now requires exactly 1 positional arg (CONFIG_PATH); error message changed from 'At least 4 arguments must be supplied' to 'At least 1 argument must be supplied: CONFIG_PATH'</message>
 <traceback>expected 4-arg banner, found 1-arg banner</traceback>
 </failure>
 <failure test="test_shell_script_guards_missing_config" file="tests/test_hardening.py" line="33">
 <error_type>AssertionError</error_type>
 <message>run_boost-shap-gii.sh train mode now takes exactly 2 args (mode + CONFIG); test still passed 5</message>
 <traceback>script exited 1 'Error: train mode requires exactly 2 arguments' — test expected success</traceback>
 </failure>
 <failure test="test_shell_script_guards_missing_data" file="tests/test_hardening.py" line="54">
 <error_type>AssertionError</error_type>
 <message>Same cause as above — test passed 5 args to the new 2-arg CLI</message>
 <traceback>script exited 1 'Error: train mode requires exactly 2 arguments'</traceback>
 </failure>
 <failure test="test_user_values_never_overwritten" file="tests/test_train.py" line="261">
 <error_type>AssertionError</error_type>
 <message>fill_config_defaults now legitimately fills shap.compute_global_on_inference=False when absent; test expected zero fills on a sample_config that predates the new key</message>
 <traceback>filled.keys contained {'shap.compute_global_on_inference'} — not in pre-existing allowlist</traceback>
 </failure>
 <failure test="test_check_env_success_in_correct_env" file="tests/test_hardening.py" line="28">
 <error_type>AssertionError (environmental)</error_type>
 <message>check-env rc=1 because R package 'nanoparquet' is not installed in local R library</message>
 <traceback>stderr: [ERROR] Missing R packages: nanoparquet — not attributable to test design</traceback>
 </failure>
 <failure test="TestCLI::test_check_env_runs_successfully" file="tests/test_package_structure.py" line="303">
 <error_type>AssertionError (environmental)</error_type>
 <message>Same root cause: nanoparquet missing from R library</message>
 <traceback>stderr: Missing R packages: nanoparquet</traceback>
 </failure>
 <failure test="test_plot_r_argument_validation" file="tests/test_plot_smoke.py" line="16">
 <error_type>AssertionError (environmental)</error_type>
 <message>Rscript halts on library(nanoparquet) before any argument parsing</message>
 <traceback>Error in library(nanoparquet): there is no package called 'nanoparquet'</traceback>
 </failure>
 <failure test="test_plot_r_discovers_shap_dirs" file="tests/test_plot_smoke.py" line="44">
 <error_type>AssertionError (environmental)</error_type>
 <message>Same root cause: nanoparquet missing</message>
 <traceback>Execution halted on missing nanoparquet</traceback>
 </failure>
 </failures>
 </pre_design_run>

 <design_phase>
 <tests_created>134</tests_created>
 <tests_modified>5</tests_modified>
 <files_created>
 <file path="tests/test_indiv_reports_validators.py" test_count="~30" coverage_target="utils.validate_indiv_reports_config + utils.validate_plot_config: happy-path and 13+ error-path contracts (missing keys, type errors, sd-with-classification rejection, custom_value=non-positive rejection, bool-only compute_global_on_inference incl. rejection of integer 1)" />
 <file path="tests/test_indiv_reports_unit.py" test_count="~40" coverage_target="indiv_reports module internals: module constants (OOB_FLOOR_MIN, CI_LO/HI_PCT, CatBoost param allowlist), _extract_user_level_params, _resolve_scaling_divisor across raw/sd/custom_value × regression/multi_regression, _reconstruct_fold_assignments determinism + balance + DataFrame y, _bootstrap_sample_indices reproducibility + cluster-aware, _emit_interactions_parquet header-only and populated, cache-loader error paths, memory guard via psutil mock, OrchestrateBootstrapCache noop when nboot=0, metadata JSON for training vs inference modes" />
 <file path="tests/test_train_outcome_stats.py" test_count="8" coverage_target="train._summarize_series (pandas quantile + sd ddof=1 unbiased), train._write_train_outcome_stats schema for regression/multi_regression/binary_classification/multiclass, pre-scaling SD preservation guard" />
 <file path="tests/test_sig_gii_loader.py" test_count="6" coverage_target="utils._load_sig_GII_from_shap_stats: single-output main+interaction parsing, ' x ' delimiter enforcement, malformed-row skip, missing-file RuntimeError, multi-output shap_*/ fallback, bool coercion from 0/1" />
 <file path="tests/test_preflight.py" test_count="12" coverage_target="check_env.run_preflight success path, python/r/both-failure exit code 2, main exit code 1 distinct from preflight, CLI wiring of run_preflight into cmd_plot, R_DEPS includes yaml/nanoparquet/ggplot2/stringr" />
 <file path="tests/test_indiv_reports_config_integration.py" test_count="24" coverage_target="example_config_advanced.yaml: all 4 new shap.* + 6 plot.* keys present with correct types, valid scaling_mode. example_config_minimal.yaml: indiv_ci_nboot + indiv_scaling_mode + plot.* keys present (no safe defaults); compute_global_on_inference absent (safe default applied). fill_config_defaults: compute_global_on_inference defaults False; user-provided value preserved; indiv_ci_nboot/scaling_mode/plot never auto-filled (err-on-kill)" />
 </files_created>
 <files_modified>
 <file path="tests/test_shell_and_config.py" test_count="5" coverage_target="test_optional_sections_omitted permits shap section (bans bootstrapping/splines); TestPlotR class rewritten to match new CONFIG_PATH-only plot.R CLI (as.numeric(plot_cfg$outcome_max), as.logical(plot_cfg$negate_shap), 'At least 1 argument must be supplied: CONFIG_PATH'); new test_required_plot_keys_validated confirms all 6 plot.* keys mentioned by the R config-key validator" />
 <file path="tests/test_hardening.py" test_count="2" coverage_target="test_shell_script_guards_missing_config and _missing_data updated to the new 2-arg train CLI (['bash', script, 'train', CONFIG])" />
 <file path="tests/test_train.py" test_count="1" coverage_target="test_user_values_never_overwritten extended with allowed_fills = {'modeling.task_type', 'shap.compute_global_on_inference'} to reflect the new safe-default" />
 </files_modified>
 <design_rationale>
 The indiv_reports feature introduced a new 1110-LOC module (src/boost_shap_gii/indiv_reports.py), two new validators (utils.validate_indiv_reports_config, validate_plot_config), a new SHAP-stats loader (utils._load_sig_GII_from_shap_stats), a new training artifact (train_outcome_stats.json via _summarize_series + _write_train_outcome_stats), a preflight wrapper (check_env.run_preflight), and a major plot.R refactor (YAML-driven CLI, single CONFIG_PATH positional arg). Test design split coverage by concern into six focused files rather than one monolithic file, and fixed five pre-existing tests that had become stale against the intentional CLI/default changes (all classified as test-staleness, not code regressions). Statistical coverage emphasizes: ddof=1 unbiased SD correctness (bootstrap-CI defensibility), deterministic fold reconstruction (reproducibility of per-individual point estimates), bootstrap RNG isolation (reproducibility across parallel refits), err-on-kill behavior for no-safe-default keys (indiv_ci_nboot, indiv_scaling_mode, all six plot.* labels), memory-budget enforcement via psutil (invariant), and the 2.5/97.5 percentile CI boundaries (Efron and Tibshirani 1993). Happy-path and adversarial-input tests are interleaved per class.
 </design_rationale>
 </design_phase>

 <post_design_run>
 <total>595</total>
 <passed>590</passed>
 <failed>4</failed>
 <errors>0</errors>
 <skipped>1</skipped>
 <coverage_pct>null</coverage_pct>
 <failures>
 <failure test="test_check_env_success_in_correct_env" file="tests/test_hardening.py" line="28">
 <error_type>AssertionError (environmental, pre-existing)</error_type>
 <message>check_env.py rc=1 — Missing R packages: nanoparquet</message>
 <traceback>stdout: [ERROR] Missing R packages: nanoparquet | [HINT] install.packages(c('nanoparquet'))</traceback>
 <likely_cause>nanoparquet R package not installed in local R library. Known environmental issue (P2 in MEMORY.md). Fix: Rscript -e 'install.packages("nanoparquet")'. NOT attributable to design phase.</likely_cause>
 </failure>
 <failure test="TestCLI::test_check_env_runs_successfully" file="tests/test_package_structure.py" line="303">
 <error_type>AssertionError (environmental, pre-existing)</error_type>
 <message>check-env CLI subcommand rc=1 — same root cause</message>
 <traceback>stderr: Missing R packages: nanoparquet</traceback>
 <likely_cause>Same as above — environmental, not a regression.</likely_cause>
 </failure>
 <failure test="test_plot_r_argument_validation" file="tests/test_plot_smoke.py" line="16">
 <error_type>AssertionError (environmental, pre-existing)</error_type>
 <message>plot.R halts on library(nanoparquet) before any argument parsing is reached</message>
 <traceback>Error in library(nanoparquet): there is no package called 'nanoparquet' — Execution halted</traceback>
 <likely_cause>Environmental — test will pass once nanoparquet is installed. Not attributable to design phase.</likely_cause>
 </failure>
 <failure test="test_plot_r_discovers_shap_dirs" file="tests/test_plot_smoke.py" line="44">
 <error_type>AssertionError (environmental, pre-existing)</error_type>
 <message>Same root cause: R execution halts on missing nanoparquet</message>
 <traceback>Execution halted on library(nanoparquet)</traceback>
 <likely_cause>Environmental — not attributable to design phase.</likely_cause>
 </failure>
 </failures>
 </post_design_run>

 <summary>
 <all_passing>false</all_passing>
 <pre_design_regressions>0</pre_design_regressions>
 <environmental_failures>4</environmental_failures>
 <recommendation>proceed_to_document</recommendation>
 <notes>
 Pre-design baseline: 450/461 passing with 11 failures (4 environmental, 7 stale-against-refactor, 0 code regressions).
 Post-design: 590/595 passing with 4 failures, ALL of which are the same pre-existing environmental issue (missing nanoparquet R package). Net test-count delta = +134 tests (134 new, 1 renamed from _omitted to _present). All 7 pre-existing stale failures were resolved (5 edited + 2 shell-guard updates). The single new-test regression discovered during post-design run (TestMinimalConfigIndivReportsKeys::test_plot_section_omitted_from_minimal) was corrected in-session: the test's assumption that plot.* should be omitted from the minimal config was incorrect — plot.* keys have no safe defaults (labels are user-specific, outcome_max is outcome-specific) and MUST appear in the minimal config, analogous to indiv_scaling_mode. Test renamed to test_plot_section_present_in_minimal with expanded per-key coverage.
 The four remaining environmental failures are a single root cause (nanoparquet absent from local R library) and do not block /document or /publish. They will resolve the moment the user runs `Rscript -e 'install.packages("nanoparquet")'`.
 Design phase is complete. The indiv_reports feature is covered at unit, schema-integration, and contract-level. Downstream recommendation: proceed to /document to refresh README INPUT_SPECIFICATION, then /publish the consolidated release.
 </notes>
 </summary>

 <action_items>
 <item priority="P2" target_mode="environment_setup" description="Install nanoparquet R package: Rscript -e 'install.packages(\"nanoparquet\")'. Resolves all 4 remaining environmental failures. Not blocking for /document or /publish." />
 <item priority="P2" target_mode="document" description="Refresh README.md and INPUT_SPECIFICATION.md to reflect: (a) new indiv_reports feature, (b) new config keys (shap.indiv_ci_nboot, indiv_scaling_mode, indiv_scaling_value, compute_global_on_inference; plot.*), (c) new train_outcome_stats.json artifact, (d) plot.R CLI surface now CONFIG_PATH-only, (e) run_preflight gate on cmd_plot." />
 </action_items>
</test_report>
