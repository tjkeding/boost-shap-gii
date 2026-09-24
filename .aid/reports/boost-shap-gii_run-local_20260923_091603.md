<run_local_report>
  <meta project="boost-shap-gii" mode="run-local" timestamp="2026-09-23T09:16:03-04:00" />
  <preflight>
    <sandbox_path><sandbox_path></sandbox_path>
    <run_directory><sandbox_path>/runs/run_20260923_091603</run_directory>
    <environment>
      <name>boost_shap_gii</name>
      <status>exists</status>
      <snapshot_path><sandbox_path>/runs/run_20260923_091603/environment_snapshot.txt</snapshot_path>
    </environment>
    <external_tools>
      <tool name="Rscript" status="found" path="/usr/local/bin/Rscript" />
      <tool name="boost-shap-gii" status="found" path="<conda_env>/bin/boost-shap-gii" />
    </external_tools>
    <input_data>
      <item path="<local_path>/runs/child_child_agg_rmse_r2" type="run_directory" status="valid" size="complete (resolved_config, predictions_oof, shap_analysis, models, bootstrap_distributions_perf.parquet, indiv_reports)" />
      <item path="<local_path>/runs/child_child_cluster_rmse_r2" type="run_directory" status="valid" size="complete" />
      <item path="<local_path>/runs/child_child_item_rmse_r2" type="run_directory" status="valid" size="complete" />
    </input_data>
    <readiness>ready</readiness>
    <blockers />
  </preflight>
  <execution>
    <command>boost-shap-gii plot --config {resolved_config.yaml} --run-dir {local_run_dir} (x3 runs, sequential)</command>
    <exit_code>0 (all three)</exit_code>
    <wall_clock_seconds>agg=15.09, cluster=15.50, item=30.66</wall_clock_seconds>
    <peak_memory_mb>agg=286, cluster=331, item=745</peak_memory_mb>
    <cpu_utilization>10-core parallel (R)</cpu_utilization>
    <log_files>
      <log type="stdout" path="<sandbox_path>/runs/run_20260923_091603/plot_agg_stdout.log" />
      <log type="stderr" path="<sandbox_path>/runs/run_20260923_091603/plot_agg_stderr.log" />
      <log type="stdout" path="<sandbox_path>/runs/run_20260923_091603/plot_cluster_stdout.log" />
      <log type="stderr" path="<sandbox_path>/runs/run_20260923_091603/plot_cluster_stderr.log" />
      <log type="stdout" path="<sandbox_path>/runs/run_20260923_091603/plot_item_stdout.log" />
      <log type="stderr" path="<sandbox_path>/runs/run_20260923_091603/plot_item_stderr.log" />
    </log_files>
    <output_manifest path="output_manifest.txt" file_count="1290 PNGs total" total_size_mb="estimated ~300" />
  </execution>
  <validation>
    <output_accounting>
      <expected_outputs>agg: 6 GII + 311 indiv = 317. cluster: 10 GII + 311 indiv = 321. item: 30 GII + 622 indiv = 652. Total: 1290 PNGs.</expected_outputs>
      <found_outputs>All expected outputs present. Counts verified via find globbing with macOS resource-fork exclusion.</found_outputs>
      <missing_outputs />
      <unexpected_outputs />
    </output_accounting>
    <log_analysis>
      <errors>0</errors>
      <warnings>2 per run (R package build-version warnings for ggplot2 and nanoparquet built under R 4.3.3; cosmetic only)</warnings>
      <anomalies />
      <key_metrics>
        <metric name="agg_exit_code" value="0" expected_range="0" status="pass" />
        <metric name="cluster_exit_code" value="0" expected_range="0" status="pass" />
        <metric name="item_exit_code" value="0" expected_range="0" status="pass" />
        <metric name="agg_gii_count" value="6" expected_range="6 (1 perf + 5 features)" status="pass" />
        <metric name="cluster_gii_count" value="10" expected_range="10 (1 perf + 9 features)" status="pass" />
        <metric name="item_gii_count" value="30" expected_range="30 (1 perf + 19 features + 10 interactions)" status="pass" />
        <metric name="item_interaction_plot_count" value="10" expected_range="10 (5 interactions x 2 mod views)" status="pass" />
        <metric name="agg_indiv_count" value="311" expected_range="311 (N=311)" status="pass" />
        <metric name="cluster_indiv_count" value="311" expected_range="311 (N=311)" status="pass" />
        <metric name="item_indiv_count" value="622" expected_range="622 (311 main + 311 interaction)" status="pass" />
        <metric name="total_png_count" value="1290" expected_range="1290" status="pass" />
        <metric name="agg_wall_clock" value="15.09s" expected_range="12-20s" status="pass" />
        <metric name="cluster_wall_clock" value="15.50s" expected_range="12-20s" status="pass" />
        <metric name="item_wall_clock" value="30.66s" expected_range="25-40s" status="pass" />
      </key_metrics>
    </log_analysis>
    <content_examination>
      <finding file="all shap_analysis/plots/*.png" category="consistency">
        <observed>All 1290 PNGs regenerated from the current working-tree plot.R containing the third critique-cycle changes (C1-C3 from boost-shap-gii_implement_build_20260923_121000.md): vjust increased from 1.5 to 3.5 at all five below-axis stat-label annotation sites; performance panel bottom margin increased from 12 to 20 in both branches; M-panel bottom margin increased from 6mm to 12mm; adaptive-legend heuristic (dist_midpoint, if/else branch, legend_x, legend_just_x, positional legend.position/legend.justification) removed as dead code and replaced with fixed bottom horizontal legend (legend.position = "bottom", legend.direction = "horizontal").</observed>
        <expected>Plot regeneration succeeds without R errors; PNG counts match prior run (1290); wall-clock and memory within expected ranges.</expected>
        <assessment>correct</assessment>
        <explanation>Exact PNG count match (1290 vs. 1290 from run_20260922_143521), zero R errors, wall-clock times consistent with prior run (agg 15.09 vs 14.16, cluster 15.50 vs 14.11, item 30.66 vs 28.88; all within normal variance). The code changes are rendering-value-only (vjust, margins, legend placement) and do not alter any plot generation logic or data processing paths.</explanation>
      </finding>
    </content_examination>
    <cross_validation>
      <check description="PNG count vs prior run (run_20260922_143521)" status="consistent">
        agg: 317 = 317; cluster: 321 = 321; item: 652 = 652; total: 1290 = 1290.
      </check>
      <check description="Wall-clock vs prior run" status="consistent">
        agg: 15.09s vs 14.16s; cluster: 15.50s vs 14.11s; item: 30.66s vs 28.88s. All within normal variance (no systematic slowdown from the rendering-value changes).
      </check>
      <check description="Memory vs prior run" status="consistent">
        agg: 286 MB vs 291 MB; cluster: 331 MB vs 389 MB; item: 745 MB vs 776 MB. All within or below prior levels.
      </check>
    </cross_validation>
    <critical_assessment>
      <overall_status>pass</overall_status>
      <definitive_errors />
      <concerning_anomalies />
      <confirmed_correct>
        <item>All three runs (agg, cluster, item) completed with exit code 0, zero R errors, and correct PNG output counts matching the prior run exactly (1290 total).</item>
        <item>Interaction plots in the item run generated all 10 GII-level interaction PNGs (5 interactions x 2 moderator orientations) and 311 per-individual interaction composites, confirming the legend and margin changes did not break any interaction rendering path.</item>
        <item>Model performance, M-panel, V-panel, and individual-report plots all regenerated for all three configs, confirming the third critique-cycle rendering changes (vjust 3.5, margin 20/12mm, fixed bottom horizontal legend) executed without error across all rendering paths.</item>
        <item>Wall-clock and memory usage are consistent with the prior run, confirming no performance regression from the rendering-value changes.</item>
      </confirmed_correct>
      <scientific_summary>The pipeline's plot step functions correctly with the third critique-cycle code changes. All rendering paths (model performance, M-panel, V-panel, interaction plots, individual reports) produce the expected outputs without error. Visual verification of the rendering-level claims (below-axis stat-label spacing at vjust=3.5, margin accommodation at 20/12mm, fixed bottom horizontal M-panel legend) requires user inspection of the regenerated PNGs. Key inspection targets: any 0_model_performance.png (all three configs) for below-axis perf-panel labels; any *_GII.png with an M panel for below-axis M-panel labels and the fixed bottom horizontal legend; and the 10 interaction PNGs in the item config for the unchanged interaction visual balance.</scientific_summary>
    </critical_assessment>
  </validation>
  <summary>
    <pipeline_status>success</pipeline_status>
    <validation_status>pass</validation_status>
    <recommendation>user_visual_inspection</recommendation>
  </summary>
  <action_items>
    <item priority="P1" target_mode="user" description="Visual inspection of the regenerated plots at the three <external_project> run directories to confirm: (1) below-axis stat labels at vjust=3.5 are below the x-axis tick marks/labels and readable without occlusion; (2) performance panel bottom margin at 20 provides sufficient space for the stat labels; (3) M-panel bottom margin at 12mm provides sufficient space for the stat labels; (4) M-panel fixed bottom horizontal legend is positioned below the plot area and does not overlap any distribution, regardless of distribution configuration. Key inspection targets: 0_model_performance.png (all three configs); any *_GII.png with an M panel (all three configs, especially features with diverse distribution configurations); the 10 interaction PNGs in the item config (should be unchanged from the prior run)." />
    <item priority="P1" target_mode="document" description="Once visual verification passes, update README, INPUT_SPECIFICATION, and AID_LOG for the full v1.7.0 plotting feature set across Sessions 19, 20, and 23, then publish v1.7.0." />
  </action_items>
</run_local_report>
