<run_local_report>
  <meta project="boost-shap-gii" mode="run-local" timestamp="2026-09-23T20:00:00-04:00" />
  <preflight>
    <sandbox_path><sandbox_path></sandbox_path>
    <run_directory><sandbox_path>/runs/run_20260923_200000</run_directory>
    <environment>
      <name>boost_shap_gii</name>
      <status>exists</status>
      <snapshot_path><sandbox_path>/runs/run_20260923_200000/environment_snapshot.txt</snapshot_path>
    </environment>
    <external_tools>
      <tool name="Rscript" status="found" path="/usr/local/bin/Rscript" />
      <tool name="boost-shap-gii" status="found" path="<conda_env>/bin/boost-shap-gii" />
    </external_tools>
    <input_data>
      <item path="<local_path>/runs/child_child_agg_rmse_r2/resolved_config.yaml" type="config" status="valid" size="present" />
      <item path="<local_path>/runs/child_child_cluster_rmse_r2/resolved_config.yaml" type="config" status="valid" size="present" />
      <item path="<local_path>/runs/child_child_item_rmse_r2/resolved_config.yaml" type="config" status="valid" size="present" />
    </input_data>
    <readiness>ready</readiness>
    <blockers />
  </preflight>
  <execution>
    <runs>
      <run dataset="agg" exit_code="0" wall_clock_seconds="13.91" peak_memory_mb="327">
        <command><conda_env>/bin/boost-shap-gii plot --config <local_path>/runs/child_child_agg_rmse_r2/resolved_config.yaml --run-dir <local_path>/runs/child_child_agg_rmse_r2</command>
        <png_count>318</png_count>
        <log_files>
          <log type="stdout" path="<sandbox_path>/runs/run_20260923_200000/agg_stdout.log" />
          <log type="stderr" path="<sandbox_path>/runs/run_20260923_200000/agg_stderr.log" />
        </log_files>
      </run>
      <run dataset="cluster" exit_code="0" wall_clock_seconds="14.10" peak_memory_mb="343">
        <command><conda_env>/bin/boost-shap-gii plot --config <local_path>/runs/child_child_cluster_rmse_r2/resolved_config.yaml --run-dir <local_path>/runs/child_child_cluster_rmse_r2</command>
        <png_count>325</png_count>
        <log_files>
          <log type="stdout" path="<sandbox_path>/runs/run_20260923_200000/cluster_stdout.log" />
          <log type="stderr" path="<sandbox_path>/runs/run_20260923_200000/cluster_stderr.log" />
        </log_files>
      </run>
      <run dataset="item" exit_code="0" wall_clock_seconds="28.82" peak_memory_mb="729">
        <command><conda_env>/bin/boost-shap-gii plot --config <local_path>/runs/child_child_item_rmse_r2/resolved_config.yaml --run-dir <local_path>/runs/child_child_item_rmse_r2</command>
        <png_count>667</png_count>
        <log_files>
          <log type="stdout" path="<sandbox_path>/runs/run_20260923_200000/item_stdout.log" />
          <log type="stderr" path="<sandbox_path>/runs/run_20260923_200000/item_stderr.log" />
        </log_files>
      </run>
    </runs>
    <output_manifest path="<sandbox_path>/runs/run_20260923_200000/output_manifest.txt" file_count="1310" total_size_mb="n/a" />
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
      <warnings>6</warnings>
      <anomalies>
        <anomaly source="all stderr logs" line="n/a" severity="minor">
          R package version warnings (ggplot2 and nanoparquet built under R 4.3.3 vs. runtime R 4.3.1). Benign; no functional impact.
        </anomaly>
      </anomalies>
    </log_analysis>
    <cross_validation>
      <check description="PNG count matches prior run (Session 21 metric-ordering verification)" status="consistent">
        Prior run: 1310 PNGs (318 + 325 + 667). Current run: 1310 PNGs (318 + 325 + 667). Identical.
      </check>
      <check description="All three runs exit 0" status="consistent">
        agg: 0, cluster: 0, item: 0. No R errors.
      </check>
    </cross_validation>
    <critical_assessment>
      <overall_status>pass</overall_status>
      <definitive_errors />
      <concerning_anomalies />
      <confirmed_correct>
        <item>All three datasets completed plot regeneration with exit code 0 and identical PNG counts to the prior metric-ordering verification run.</item>
        <item>Zero R errors in any stderr log. Only benign package-version warnings.</item>
      </confirmed_correct>
      <scientific_summary>
        Plot regeneration completed successfully across all three <external_project> child-child datasets. The 1310 PNGs are ready for visual inspection of the white-background occlusion fix (implement build boost-shap-gii_implement_build_20260923_190500.md, changes C1 and C2). The user should verify: (1) performance panel stat labels (trained mean+SD, permutation-null mean+SD) now render as opaque white boxes that occlude the geom_vline mean lines beneath them, and (2) M-panel stat labels (noise mean+SD, signal mean+SD) now render as opaque white boxes with visible clearance below the density distributions.
      </scientific_summary>
    </critical_assessment>
  </validation>
  <summary>
    <pipeline_status>success</pipeline_status>
    <validation_status>pass</validation_status>
    <recommendation>User visual inspection of the white-background fix. On approval, proceed to /document then /publish v1.7.0.</recommendation>
  </summary>
  <action_items>
    <item priority="P1" target_mode="user" description="Visually inspect the regenerated model_performance and main_effects PNGs across all three datasets to confirm: (1) performance panel stat labels render as opaque white boxes occluding the vertical mean lines, and (2) M-panel stat labels render as opaque white boxes with clearance below the density distributions." />
  </action_items>
</run_local_report>
