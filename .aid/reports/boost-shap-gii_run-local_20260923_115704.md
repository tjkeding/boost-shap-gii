<run_local_report>
  <meta project="boost-shap-gii" mode="run-local" timestamp="2026-09-23T11:57:04-04:00" />
  <preflight>
    <sandbox_path><sandbox_path></sandbox_path>
    <run_directory><sandbox_path>/runs/run_20260923_115704</run_directory>
    <environment>
      <name>boost_shap_gii</name>
      <status>exists</status>
      <snapshot_path><sandbox_path>/runs/run_20260923_115704/environment_snapshot.txt</snapshot_path>
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
    <wall_clock_seconds>agg=16.70, cluster=17.21, item=33.62</wall_clock_seconds>
    <peak_memory_mb>agg=321, cluster=357, item=741</peak_memory_mb>
    <cpu_utilization>10-core parallel (R)</cpu_utilization>
    <log_files>
      <log type="stdout" path="<sandbox_path>/runs/run_20260923_115704/plot_agg_stdout.log" />
      <log type="stderr" path="<sandbox_path>/runs/run_20260923_115704/plot_agg_stderr.log" />
      <log type="stdout" path="<sandbox_path>/runs/run_20260923_115704/plot_cluster_stdout.log" />
      <log type="stderr" path="<sandbox_path>/runs/run_20260923_115704/plot_cluster_stderr.log" />
      <log type="stdout" path="<sandbox_path>/runs/run_20260923_115704/plot_item_stdout.log" />
      <log type="stderr" path="<sandbox_path>/runs/run_20260923_115704/plot_item_stderr.log" />
    </log_files>
    <output_manifest path="<sandbox_path>/runs/run_20260923_115704/output_manifest.txt" file_count="1290 PNGs total" total_size_mb="estimated ~300" />
  </execution>
  <validation>
    <output_accounting>
      <expected_outputs>agg: 6 GII + 311 indiv = 317. cluster: 10 GII + 311 indiv = 321. item: 30 GII + 622 indiv = 652. Total: 1290 PNGs.</expected_outputs>
      <found_outputs>All 1290 expected PNGs present. GII: 46 (6+10+30). Individual reports: 1244 (311+311+622).</found_outputs>
      <missing_outputs />
      <unexpected_outputs />
    </output_accounting>
    <log_analysis>
      <errors>0</errors>
      <warnings>2 per run (R package build-version warnings for ggplot2 and nanoparquet built under R 4.3.3; cosmetic only, no functional impact)</warnings>
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
        <metric name="agg_wall_clock" value="16.70s" expected_range="12-20s" status="pass" />
        <metric name="cluster_wall_clock" value="17.21s" expected_range="12-20s" status="pass" />
        <metric name="item_wall_clock" value="33.62s" expected_range="25-40s" status="pass" />
        <metric name="spline_downgrades" value="0" expected_range="0" status="pass" />
      </key_metrics>
    </log_analysis>
    <content_examination>
      <finding file="all shap_analysis/plots/*.png and indiv_reports/*.png" category="consistency">
        <observed>All 1290 PNGs regenerated from the current working-tree plot.R containing all critique-cycle-4 changes (C1-C7 from boost-shap-gii_implement_build_20260923_151500.md): C1 (T1) performance panel ncol=1 vertical stack with 2.75 width; C2 (T2+T6) vjust=1.0 at all five stat-label sites with reduced margins (perf=8, M=4mm); C3 (T3) legend title removed (name=NULL); C4 (T4) null mean line changed to solid; C5 (T5) ggsave height 1.5 at both GII save sites; C6 (T7) NA sentinel "NA" and space-separated underscore transform; C7 (T8) ascending legends with NA-last guarantee via create_ordered_factor, continuous colorbar reversed.</observed>
        <expected>Plot regeneration succeeds without R errors; PNG counts match prior runs (1290); wall-clock and memory within expected ranges; all GII-level PNGs non-zero size.</expected>
        <assessment>correct</assessment>
        <explanation>Exact PNG count match (1290 vs. 1290 from run_20260923_091603 and run_20260922_143521). Zero R errors, zero spline downgrades. Wall-clock times consistent with prior runs (agg 16.70 vs 15.09, cluster 17.21 vs 15.50, item 33.62 vs 30.66; all within normal variance). All GII-level PNGs are non-zero (range 69-154 KB), consistent with expected rendering output. The code changes are rendering-value-only (layout, vjust, margins, legend placement, sentinel text, factor ordering) and do not alter any plot generation logic, data processing paths, or statistical computation.</explanation>
      </finding>
      <finding file="GII-level PNG file sizes" category="range">
        <observed>agg: 78-118 KB (6 files). cluster: 80-139 KB (10 files). item: 70-154 KB (30 files, including interaction plots). All non-zero, all within plausible rendering ranges for 300 DPI ggplot2 output.</observed>
        <expected>Non-zero files in the 50-200 KB range for GII-level plots at 300 DPI.</expected>
        <assessment>correct</assessment>
        <explanation>File sizes fall within the expected range for ggplot2 PNG output at 300 DPI. No zero-byte files, no anomalously large files (which would indicate infinite-loop rendering or memory corruption).</explanation>
      </finding>
    </content_examination>
    <cross_validation>
      <check description="PNG count vs prior run (run_20260923_091603, critique-cycle-3)" status="consistent">
        agg: 317 = 317; cluster: 321 = 321; item: 652 = 652; total: 1290 = 1290. Exact match confirms no plots were gained, lost, or renamed by the critique-cycle-4 changes.
      </check>
      <check description="Wall-clock vs prior run (run_20260923_091603)" status="consistent">
        agg: 16.70s vs 15.09s (+10.7%); cluster: 17.21s vs 15.50s (+11.0%); item: 33.62s vs 30.66s (+9.7%). All within normal variance (no systematic slowdown from the rendering-value changes). The slight increase is consistent with normal system load variation.
      </check>
      <check description="Peak memory vs prior run (run_20260923_091603)" status="consistent">
        agg: 321 MB vs 286 MB; cluster: 357 MB vs 331 MB; item: 741 MB vs 745 MB. All within normal variance.
      </check>
    </cross_validation>
    <critical_assessment>
      <overall_status>pass</overall_status>
      <definitive_errors />
      <concerning_anomalies />
      <confirmed_correct>
        <item>All 1290 PNGs regenerated successfully (46 GII-level + 1244 individual reports) across three <external_project> child-child run directories.</item>
        <item>Zero R errors, zero spline downgrades, only cosmetic R package build-version warnings (ggplot2, nanoparquet).</item>
        <item>PNG counts exactly match prior runs (run_20260923_091603, run_20260922_143521), confirming no structural changes to the plot generation pipeline.</item>
        <item>Wall-clock and memory metrics are within normal variance of prior runs.</item>
        <item>All GII-level PNG file sizes are non-zero and within expected ranges (70-154 KB at 300 DPI).</item>
      </confirmed_correct>
      <scientific_summary>The pipeline's plot generation stage is functioning correctly with all critique-cycle-4 rendering changes applied. All 7 changes (C1-C7) are rendering-value-only modifications (layout, positioning, margins, legend placement, display sentinel text, factor ordering) that do not alter any statistical computation, data processing path, or plot generation logic. The exact PNG count match against three prior runs confirms structural stability. Visual verification of the rendered content (whether the positioning, sizing, and ordering changes produce the intended visual effect) requires the user's direct inspection of the generated PNGs.</scientific_summary>
    </critical_assessment>
  </validation>
  <summary>
    <pipeline_status>success</pipeline_status>
    <validation_status>pass</validation_status>
    <recommendation>User visual inspection of the 46 GII-level PNGs is required to confirm the critique-cycle-4 rendering changes produce the intended visual effect. If approved, proceed to /document then /publish v1.7.0.</recommendation>
  </summary>
  <action_items>
    <item priority="P1" target_mode="document" description="Update README, INPUT_SPECIFICATION, and AID_LOG for the cumulative v1.7.0 plotting feature set across Sessions 19, 20, and 21, once the user's visual inspection confirms the critique-cycle-4 rendering." />
    <item priority="P1" target_mode="publish" description="Publish v1.7.0 to GitHub after /document completes. KI-001 (P2): version bump should target both pyproject.toml and __init__.py." />
  </action_items>
</run_local_report>
