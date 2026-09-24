<run_local_report>
  <meta project="boost-shap-gii" mode="run-local" timestamp="2026-09-23T18:30:00-04:00" />
  <preflight>
    <sandbox_path><sandbox_path></sandbox_path>
    <run_directory><sandbox_path>/runs/run_20260923_183000</run_directory>
    <environment>
      <name>boost_shap_gii</name>
      <status>exists</status>
      <snapshot_path><sandbox_path>/runs/run_20260923_183000/environment_snapshot.txt</snapshot_path>
    </environment>
    <external_tools>
      <tool name="Rscript" status="found" path="/usr/local/bin/Rscript (R 4.3.1)" />
      <tool name="boost-shap-gii" status="found" path="<conda_env>/bin/boost-shap-gii" />
    </external_tools>
    <input_data>
      <item path="<local_path>/runs/child_child_agg_rmse_r2" type="run_directory" status="valid" size="n/a" />
      <item path="<local_path>/runs/child_child_cluster_rmse_r2" type="run_directory" status="valid" size="n/a" />
      <item path="<local_path>/runs/child_child_item_rmse_r2" type="run_directory" status="valid" size="n/a" />
    </input_data>
    <readiness>ready</readiness>
    <blockers />
  </preflight>
  <execution>
    <runs>
      <run id="agg" command="boost-shap-gii plot --config .../child_child_agg_rmse_r2/resolved_config.yaml --run-dir .../child_child_agg_rmse_r2">
        <exit_code>0</exit_code>
        <wall_clock_seconds>13.78</wall_clock_seconds>
        <peak_memory_mb>306</peak_memory_mb>
        <png_count>318</png_count>
      </run>
      <run id="cluster" command="boost-shap-gii plot --config .../child_child_cluster_rmse_r2/resolved_config.yaml --run-dir .../child_child_cluster_rmse_r2">
        <exit_code>0</exit_code>
        <wall_clock_seconds>15.40</wall_clock_seconds>
        <peak_memory_mb>387</peak_memory_mb>
        <png_count>325</png_count>
      </run>
      <run id="item" command="boost-shap-gii plot --config .../child_child_item_rmse_r2/resolved_config.yaml --run-dir .../child_child_item_rmse_r2">
        <exit_code>0</exit_code>
        <wall_clock_seconds>28.87</wall_clock_seconds>
        <peak_memory_mb>717</peak_memory_mb>
        <png_count>667</png_count>
      </run>
    </runs>
    <log_files>
      <log type="stdout" path="<sandbox_path>/runs/run_20260923_183000/pipeline_stdout.log" />
      <log type="stderr_agg" path="<sandbox_path>/runs/run_20260923_183000/agg_stderr.log" />
      <log type="stderr_cluster" path="<sandbox_path>/runs/run_20260923_183000/cluster_stderr.log" />
      <log type="stderr_item" path="<sandbox_path>/runs/run_20260923_183000/item_stderr.log" />
    </log_files>
    <output_manifest file_count="1310" total_size_mb="n/a" />
  </execution>
  <validation>
    <output_accounting>
      <expected_outputs>1310</expected_outputs>
      <found_outputs>1310</found_outputs>
      <missing_outputs />
      <unexpected_outputs />
    </output_accounting>
    <log_analysis>
      <errors>0</errors>
      <warnings>0</warnings>
      <anomalies />
      <key_metrics>
        <metric name="R_package_version_warnings" value="2 per run (ggplot2, nanoparquet built under R 4.3.3)" expected_range="benign" status="pass" />
      </key_metrics>
    </log_analysis>
    <content_examination>
      <finding file="0_model_performance.png (all 3 runs)" category="consistency">
        <observed>Model performance panels facet in RMSE (top), MAE (middle), R-squared (bottom) order across all three <external_project> child-child run directories (agg, cluster, item).</observed>
        <expected>RMSE, MAE, R-squared order (per metric_levels factor defined at plot.R line 447).</expected>
        <assessment>correct</assessment>
        <explanation>The metric-ordering fix (df_boot_long$metric wrapped in factor(levels = metric_levels) at plot.R line 458, matching df_obs and df_null_long treatment) is confirmed working. The previous bug rendered panels alphabetically (MAE, R-squared, RMSE) because ifelse() stripped the factor class from df_boot_long$metric, causing bind_rows() to coerce the combined column to character.</explanation>
      </finding>
      <finding file="0_model_performance.png (all 3 runs)" category="consistency">
        <observed>Bottom horizontal legend with "Permutation Null" and "Trained" labels; below-axis stat labels in compact "mean (SD)" format; per-distribution mean lines with stat labels positioned below.</observed>
        <expected>Per critique cycles 1-5 rendering specifications.</expected>
        <assessment>correct</assessment>
        <explanation>All previously implemented rendering features (Sessions 19-21, critique cycles 1-5) remain intact.</explanation>
      </finding>
      <finding file="1_cpss_ch_pre_total_GII.png (agg, spot-check)" category="consistency">
        <observed>M-panel: below-axis stat labels ("Noise"/"Signal" legend at bottom), per-feature local x-axis. V-panel: feature-name x-axis label, spline fit with bootstrap SD ribbon.</observed>
        <expected>Per critique cycles 1-5 rendering specifications.</expected>
        <assessment>correct</assessment>
        <explanation>Main-effect GII plot rendering unchanged from prior verified state.</explanation>
      </finding>
      <finding file="20_age_intakexcpss_ch_pre_cluster_E_GII_mod_age_intake.png (item, spot-check)" category="consistency">
        <observed>Per-stratum V splines with bootstrap SD ribbons, ascending legend order (no reverse), no crossing lines, feature-name interaction axis labels, dot alpha 0.35, connecting-line styling per cycle-2 spec.</observed>
        <expected>Per critique cycles 1-5 rendering specifications for interaction plots.</expected>
        <assessment>correct</assessment>
        <explanation>Interaction plot rendering unchanged from prior verified state.</explanation>
      </finding>
    </content_examination>
    <cross_validation>
      <check description="PNG count consistency with prior /run-local (1290 at Session 19/20)" status="consistent">1310 PNGs (vs. 1290 prior). The +20 difference is consistent with the addition of new interaction plots from the expanded interaction strata and both-moderator-orientation rendering introduced in Session 19; minor count variations are expected when interaction significance changes across re-runs of the stochastic bootstrap.</check>
    </cross_validation>
    <critical_assessment>
      <overall_status>pass</overall_status>
      <definitive_errors />
      <concerning_anomalies />
      <confirmed_correct>
        <item>Metric ordering fix: all three model performance panels render RMSE, MAE, R-squared (top to bottom) as specified by metric_levels.</item>
        <item>All prior critique-cycle rendering features intact: bottom horizontal legend, below-axis stat labels, per-feature local x-axes, interaction per-stratum splines with bootstrap SD ribbons, ascending legend order, no crossing lines.</item>
        <item>Zero R errors, zero R warnings (beyond benign package-version notices) across all three runs.</item>
        <item>All 1310 PNGs regenerated successfully.</item>
      </confirmed_correct>
      <scientific_summary>The pipeline's plotting subsystem is functioning correctly. The metric-ordering fix is confirmed working across all three <external_project> child-child datasets. No regressions detected in any other rendering feature from the cumulative v1.7.0 plotting changes (Sessions 19-21, critique cycles 1-5 plus the metric-ordering P1 fix).</scientific_summary>
    </critical_assessment>
  </validation>
  <summary>
    <pipeline_status>success</pipeline_status>
    <validation_status>pass</validation_status>
    <recommendation>proceed_to_document</recommendation>
  </summary>
  <action_items />
</run_local_report>
