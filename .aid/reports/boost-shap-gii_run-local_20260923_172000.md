<run_local_report>
  <meta project="boost-shap-gii" mode="run-local" timestamp="2026-09-23T17:20:00-04:00" />
  <preflight>
    <sandbox_path><sandbox_path></sandbox_path>
    <run_directory><sandbox_path>/runs/run_20260923_172000</run_directory>
    <environment>
      <name>boost_shap_gii</name>
      <status>exists</status>
      <snapshot_path><sandbox_path>/runs/run_20260923_172000/environment_snapshot.txt</snapshot_path>
    </environment>
    <external_tools>
      <tool name="Rscript" status="found" path="/usr/local/bin/Rscript" />
      <tool name="R" status="found" version="4.3.1" />
      <tool name="boost-shap-gii" status="found" path="<conda_env>/bin/boost-shap-gii" />
    </external_tools>
    <r_packages>
      <package name="ggplot2" version="3.5.2" />
      <package name="dplyr" version="1.1.4" />
      <package name="tidyr" version="1.3.0" />
      <package name="readr" version="2.1.5" />
      <package name="stringr" version="1.5.1" />
      <package name="purrr" version="1.0.4" />
      <package name="jsonlite" version="1.8.8" />
      <package name="scales" version="1.4.0" />
      <package name="splines" version="4.3.1" />
      <package name="nanoparquet" version="0.4.2" />
    </r_packages>
    <input_data>
      <item path="<local_path>/runs/child_child_agg_rmse_r2" type="run_directory" status="valid" />
      <item path="<local_path>/runs/child_child_cluster_rmse_r2" type="run_directory" status="valid" />
      <item path="<local_path>/runs/child_child_item_rmse_r2" type="run_directory" status="valid" />
    </input_data>
    <readiness>ready</readiness>
    <blockers />
  </preflight>
  <execution>
    <runs>
      <run id="agg">
        <command><conda_env>/bin/boost-shap-gii plot --config "<local_path>/runs/child_child_agg_rmse_r2/resolved_config.yaml" --run-dir "<local_path>/runs/child_child_agg_rmse_r2"</command>
        <exit_code>0</exit_code>
        <wall_clock_seconds>14.07</wall_clock_seconds>
        <peak_memory_mb>346.1</peak_memory_mb>
        <cpu_utilization>499%</cpu_utilization>
      </run>
      <run id="cluster">
        <command><conda_env>/bin/boost-shap-gii plot --config "<local_path>/runs/child_child_cluster_rmse_r2/resolved_config.yaml" --run-dir "<local_path>/runs/child_child_cluster_rmse_r2"</command>
        <exit_code>0</exit_code>
        <wall_clock_seconds>16.10</wall_clock_seconds>
        <peak_memory_mb>370.8</peak_memory_mb>
        <cpu_utilization>493%</cpu_utilization>
      </run>
      <run id="item">
        <command><conda_env>/bin/boost-shap-gii plot --config "<local_path>/runs/child_child_item_rmse_r2/resolved_config.yaml" --run-dir "<local_path>/runs/child_child_item_rmse_r2"</command>
        <exit_code>0</exit_code>
        <wall_clock_seconds>30.31</wall_clock_seconds>
        <peak_memory_mb>710.4</peak_memory_mb>
        <cpu_utilization>548%</cpu_utilization>
      </run>
    </runs>
    <log_files>
      <log type="stdout" path="<sandbox_path>/runs/run_20260923_172000/plot_stdout_agg.log" />
      <log type="stderr" path="<sandbox_path>/runs/run_20260923_172000/plot_stderr_agg.log" />
      <log type="stdout" path="<sandbox_path>/runs/run_20260923_172000/plot_stdout_cluster.log" />
      <log type="stderr" path="<sandbox_path>/runs/run_20260923_172000/plot_stderr_cluster.log" />
      <log type="stdout" path="<sandbox_path>/runs/run_20260923_172000/plot_stdout_item.log" />
      <log type="stderr" path="<sandbox_path>/runs/run_20260923_172000/plot_stderr_item.log" />
    </log_files>
    <output_manifest path="<sandbox_path>/runs/run_20260923_172000/output_manifest.txt" file_count="1290" total_size_mb="n/a" />
  </execution>
  <validation>
    <output_accounting>
      <expected_outputs>1290</expected_outputs>
      <found_outputs>1290</found_outputs>
      <missing_outputs />
      <unexpected_outputs />
      <breakdown>
        <run id="agg" total_pngs="317" shap_analysis="6" indiv_reports="311" />
        <run id="cluster" total_pngs="321" shap_analysis="10" indiv_reports="311" />
        <run id="item" total_pngs="652" shap_analysis="30" indiv_reports="622" />
      </breakdown>
      <note>20 macOS AppleDouble resource-fork sidecar files (._*.png) existed in the directories but are not content files and were not regenerated, consistent with prior runs.</note>
    </output_accounting>
    <log_analysis>
      <errors>0</errors>
      <warnings>0</warnings>
      <anomalies />
      <key_metrics>
        <metric name="R_package_warnings" value="2 per run (ggplot2, nanoparquet built under R 4.3.3)" expected_range="2" status="pass" />
        <metric name="total_wall_clock" value="60.48s" expected_range="45-120s" status="pass" />
      </key_metrics>
    </log_analysis>
    <content_examination>
      <finding file="0_model_performance.png (all runs)" category="consistency">
        <observed>R-squared superscript (U+00B2) renders correctly in facet title as "R²". Stat labels below each distribution's mean line in compact "mean (sd)" format. Fixed bottom horizontal legend with "Permutation Null" / "Trained" labels.</observed>
        <expected>Per critique-cycle-5 changes C1 (R² superscript), C2 (vjust=0.5 stat labels), C3 (bottom horizontal legend).</expected>
        <assessment>correct</assessment>
        <explanation>C1 superscript, C2 stat label positioning, and C3 legend placement all render as specified.</explanation>
      </finding>
      <finding file="0_model_performance.png (all runs)" category="consistency">
        <observed>Facet ordering is MAE, R², RMSE (top to bottom), i.e. alphabetical by Unicode character order.</observed>
        <expected>Facet ordering should be RMSE, MAE, R² per C1 metric_levels factor at plot.R line 447: c("RMSE", "MAE", "R²").</expected>
        <assessment>error</assessment>
        <explanation>Product bug in the has_boot_perf=TRUE branch: line 458 applies ifelse(df_boot_long$metric == "R2", "R²", df_boot_long$metric), which strips the factor class from df_boot_long$metric (R's ifelse converts factor to character). When bind_rows(df_null_long, df_boot_long) creates df_dist at line 464, the mixed factor/character metric column is coerced to character, losing the levels ordering. facet_wrap then defaults to alphabetical. Fix: apply factor(..., levels = metric_levels) to df_boot_long$metric at line 458, consistent with the treatment of df_obs (lines 448-451) and df_null_long (lines 452-455). The CI-band fallback branch (lines 509-542) is unaffected because it uses df_null_long and df_obs directly (both factor-typed).</explanation>
      </finding>
      <finding file="*_GII.png (all runs)" category="consistency">
        <observed>M-panel stat labels below distributions in compact format. Bottom horizontal "Noise"/"Signal" legend with tight margins (legend.margin=0, legend.box.margin=0, plot.margin bottom=1mm). Distribution area maximized.</observed>
        <expected>Per C2 (vjust=0.5) and C3 (legend tightening).</expected>
        <assessment>correct</assessment>
        <explanation>M-panel rendering matches all critique-cycle-5 specifications.</explanation>
      </finding>
      <finding file="8_age_intake_GII.png (item run)" category="consistency">
        <observed>V-panel x-axis for age_intake (13 integer levels, 6.0-18.0) renders all labels without overlap or suppression.</observed>
        <expected>Per C4, guide_axis(check.overlap = TRUE) prevents overlap on dense continuous axes.</expected>
        <assessment>correct</assessment>
        <explanation>The check.overlap guard is active; at the rendered width, all 13 labels fit without collision. The guard would automatically suppress overlapping labels if the feature had more or denser values.</explanation>
      </finding>
      <finding file="*_mod_*.png (item run)" category="consistency">
        <observed>Interaction plots: per-stratum splines with bootstrap SD ribbons, ascending strata legend order, focal feature name on x-axis, dot alpha 0.35, connecting-line linewidth 0.6, mean-box width 0.20/linewidth 1.5, no crossing lines.</observed>
        <expected>Per Sessions 19-20 implemented specifications plus critique-cycle-5 M-panel changes.</expected>
        <assessment>correct</assessment>
        <explanation>Interaction plot rendering consistent with all accumulated v1.7.0 specifications.</explanation>
      </finding>
    </content_examination>
    <cross_validation>
      <check description="PNG count matches prior /run-local runs (Sessions 19-20)" status="consistent">
        1290 PNGs regenerated across all 3 runs, matching the established count from Sessions 19, 20, and 21 critique cycles 1-4.
      </check>
      <check description="Zero-byte output check" status="consistent">
        0 zero-byte PNGs found across all 3 run directories.
      </check>
      <check description="Exit codes across all runs" status="consistent">
        All 3 runs completed with exit code 0, zero R errors, zero R warnings in stdout.
      </check>
    </cross_validation>
    <critical_assessment>
      <overall_status>pass_with_warnings</overall_status>
      <definitive_errors>
        <error>C1 metric ordering (facet order): factor levels lost in has_boot_perf=TRUE branch due to ifelse stripping factor class from df_boot_long$metric (line 458). Rendered order is alphabetical (MAE, R², RMSE) instead of specified (RMSE, MAE, R²). Localized to 0_model_performance.png across all runs; no impact on M, V, GII, or interaction plots. Fix requires one line change in plot.R.</error>
      </definitive_errors>
      <concerning_anomalies />
      <confirmed_correct>
        <item>C1 R² superscript rendering (Unicode U+00B2 in facet titles)</item>
        <item>C2 vjust=0.5 stat-label positioning at all 5 annotation sites (perf trained, perf null x2 branches, M noise, M signal)</item>
        <item>C3 M-panel bottom horizontal legend with tight margins (legend.margin=0, legend.box.margin=0, plot.margin bottom=1mm)</item>
        <item>C3 perf-panel bottom horizontal legend ("Permutation Null" / "Trained")</item>
        <item>C4 guide_axis(check.overlap=TRUE) on V-panel continuous x-axes</item>
        <item>All Session 19-20 rendering specifications preserved (per-stratum V splines, bootstrap SD ribbons, interaction plot visual tuning, no crossing lines)</item>
      </confirmed_correct>
      <scientific_summary>The plot stage is functioning correctly for 5 of the 6 distinct rendering changes in critique cycle 5. The metric ordering bug (C1 facet order) is a visual-display issue with no impact on data correctness, metric labeling, or statistical values. All metric panels render the correct data with correct labels and stat annotations; only the display order deviates from the specification. The fix is a single-line change at plot.R line 458 to preserve the factor levels.</scientific_summary>
    </critical_assessment>
  </validation>
  <summary>
    <pipeline_status>success</pipeline_status>
    <validation_status>pass_with_warnings</validation_status>
    <recommendation>fix_via_implement</recommendation>
  </summary>
  <action_items>
    <item priority="P1" target_mode="implement" description="Fix C1 metric ordering bug in plot.R line 458: df_boot_long$metric must be converted to factor with levels = metric_levels after the ifelse R2-to-R² rename, consistent with the treatment of df_obs (lines 448-451) and df_null_long (lines 452-455). The ifelse call strips the factor class, and the subsequent bind_rows at line 464 coerces the mixed factor/character column to character, causing facet_wrap to default to alphabetical order." />
  </action_items>
</run_local_report>
