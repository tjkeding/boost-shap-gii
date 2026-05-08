<implement_report>
 <meta project="boost-shap-gii" mode="implement" submodule="build" timestamp="2026-04-24T16:24:19Z" />
 <spec_ref>boost-shap-gii_implement_plan_20260424_084001.md</spec_ref>
 <changes_applied>
 <change id="change-1" status="done" user_decision="n/a">
 <files_modified>
 <file path="example_config_advanced.yaml" lines_changed="42" />
 </files_modified>
 <notes>All 10 new keys added in spec order: 4 shap.* keys and 6 plot.* keys in a newly created plot section. YAML parses cleanly.</notes>
 </change>
 <change id="change-2" status="done" user_decision="n/a">
 <files_modified>
 <file path="example_config_minimal.yaml" lines_changed="13" />
 </files_modified>
 <notes>indiv_scaling_value is retained as a commented-out line so users switching indiv_scaling_mode to custom_value have an in-file prompt. compute_global_on_inference omitted per minimal-config convention (safe default false).</notes>
 </change>
 <change id="change-3" status="done" user_decision="n/a">
 <files_modified>
 <file path="src/boost_shap_gii/utils.py" lines_changed="138" />
 </files_modified>
 <notes>fill_config_defaults gained compute_global_on_inference default via setdefault; validate_indiv_reports_config and validate_plot_config added as module-level public functions. Call-site wiring into predict.py/infer.py/cli.py handled by downstream groups that own those files.</notes>
 </change>
 <change id="change-4" status="done" user_decision="n/a">
 <files_modified>
 <file path="src/boost_shap_gii/train.py" lines_changed="54" />
 </files_modified>
 <notes>y_raw snapshot captured before StandardScaler application to preserve pre-scaling values across task types. Helpers _summarize_series and _write_train_outcome_stats added to the Helper Utilities section. train_outcome_stats.json emitted after training finalization.</notes>
 </change>
 <change id="change-5" status="done" user_decision="n/a">
 <files_modified>
 <file path="src/boost_shap_gii/indiv_reports.py" lines_changed="1110" />
 </files_modified>
 <notes>New module implementing the coupled bootstrap + OOB aggregation pipeline, deterministic fold reconstruction via get_cv_splitter (no new fold-assignment artifact withdrawal), memory budget (0.5 * psutil.virtual_memory.available) with err-on-kill MemoryError, np.nanpercentile CI computation, deployed-ensemble SHAP as point estimate, single shap_stats_global.csv consumption with Singleton filter and " x " effect-column parse, long-format output schema. psutil is lazy-imported inside generate_indiv_reports so module-level import never fails; ImportError with pip-install guidance raised at runtime if the package is unavailable. All terminology neutral ("individual"/"indiv"; no "individual"/"individuals").</notes>
 </change>
 <change id="change-6" status="done" user_decision="n/a">
 <files_modified>
 <file path="src/boost_shap_gii/predict.py" lines_changed="51" />
 <file path="src/boost_shap_gii/utils.py" lines_changed="59" />
 </files_modified>
 <notes>predict.py invokes generate_indiv_reports for training-set coverage post-SHAP. Shared helper _load_sig_GII_from_shap_stats located in utils.py (not predict.py) so infer.py can reuse it per the directive. cache_summary print uses the actual orchestrate_bootstrap_cache return-schema keys (B, total_refits) rather than the spec's illustrative placeholders.</notes>
 </change>
 <change id="change-7" status="done" user_decision="n/a">
 <files_modified>
 <file path="src/boost_shap_gii/infer.py" lines_changed="57" />
 </files_modified>
 <notes>Global SHAP on inference gated behind shap.compute_global_on_inference (default false; breaking change). generate_indiv_reports invoked for the inference set with straight ensemble-averaged CI (inference individuals have no fold assignments, all B iterations are OOB for them). Train-dir artifacts consumed: bootstrap_refits/, model_fold_{k}.cbm, train_outcome_stats.json.</notes>
 </change>
 <change id="change-8" status="done" user_decision="n/a">
 <files_modified>
 <file path="src/boost_shap_gii/scripts/plot.R" lines_changed="347" />
 <file path="src/boost_shap_gii/scripts/run_boost-shap-gii.sh" lines_changed="25" />
 <file path="src/boost_shap_gii/cli.py" lines_changed="12" />
 </files_modified>
 <notes>plot.R CLI reduced to CONFIG_PATH (required) + RUN_DIR (optional); all six plot.* keys validated on entry with fail-loud stop; OUTCOME_RANGE renamed to OUTCOME_MAX; GII panels consume GII_Y_LABEL/GII_Y_SUBLABEL from config. indiv_reports plots added (dot-plus-whisker, signed-rank x-axis, blue-positive/red-negative diverging color anchored to RAW signed SHAP; sign-flip affects y-values only, not color or ordering; below-OOB-floor caption). Parallelized via mclapply with Windows lapply fallback. cli.py plot subcommand reduced to --config + --run-dir; preflight call from preserved.</notes>
 </change>
 <change id="change-9" status="done" user_decision="n/a">
 <files_modified>
 <file path="src/boost_shap_gii/check_env.py" lines_changed="21" />
 <file path="src/boost_shap_gii/cli.py" lines_changed="8" />
 </files_modified>
 <notes>run_preflight added to check_env.py (directly calls check_python + check_r; sys.exit(2) on failure) and wired into the plot subcommand in cli.py. R_DEPS, main, and legacy plot-CLI code left untouched so could layer on cleanly.</notes>
 </change>
 <change id="change-10" status="done" user_decision="n/a">
 <files_modified>
 <file path="INPUT_SPECIFICATION.md" lines_changed="391" />
 </files_modified>
 <notes>New Section 10 (indiv_reports, subsections 10.1-10.5); shap.* table extended with four new keys and cross-reference to Section 10; plot.* config documented; plot subcommand CLI surface updated; directory structure updated with train_outcome_stats.json, bootstrap_refits/, and indiv_reports/ for both train and infer trees; Stage 6 documents the global-SHAP-on-inference gate; inference-individuals-have-no-fold-assignments invariant stated explicitly.</notes>
 </change>
 <change id="change-11" status="done" user_decision="n/a">
 <files_modified>
 <file path="README.md" lines_changed="42" />
 </files_modified>
 <notes>New indiv_reports section placed between Quickstart and Robust Data Loading. Documents per-individual dot-plus-whisker plots with bootstrap CIs, simplified plot CLI surface, and the opt-in shap.compute_global_on_inference gate (breaking change from prior default). Verbatim spec phrase included.</notes>
 </change>
 <change id="change-12" status="done" user_decision="n/a">
 <files_modified>
 <file path="AID_LOG.md" lines_changed="77" />
 </files_modified>
 <notes>Section 7 (Development Session Log) and Section 8 (Version and Release Notes) added; legacy Section 7 (References) renumbered to Section 9 and extended with Breiman 2001, Carpenter &amp; Bithell 2000, and Efron &amp; Tibshirani 1993 citations that support algorithmic decisions. Brainstorm/plan/build reports recorded as audit-trail references.</notes>
 </change>
 </changes_applied>
 <summary>
 <total_changes>12</total_changes>
 <completed>12</completed>
 <skipped>0</skipped>
 <blocked>0</blocked>
 </summary>
 <next_steps>Recommended: run /test to validate all changes. Priority targets: (a) indiv_reports.py algorithmic correctness (scaling modes, bootstrap+OOB aggregation, np.nanpercentile CI bounds, deterministic fold reconstruction, memory-budget err-on-kill path, long-format schema); (b) predict.py and infer.py integration paths (training-set vs inference-set coverage, shap.compute_global_on_inference gate); (c) plot.R end-to-end render under the simplified CLI (GII panels, indiv_reports dot-plus-whisker, below-OOB-floor caption); (d) config validation (validate_indiv_reports_config, validate_plot_config fail-loud paths); (e) preflight (run_preflight sys.exit(2) on missing R deps). Also install nanoparquet in the local R library (Rscript -e 'install.packages("nanoparquet")') to clear the three pre-existing environment-only failures before running the suite.</next_steps>
</implement_report>
