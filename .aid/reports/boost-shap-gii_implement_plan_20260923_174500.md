<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-09-23T17:45:00-04:00" />
  <input_reports>
    <report path="boost-shap-gii_run-local_20260923_172000.md" mode="run-local" key_items="1" />
  </input_reports>
  <changes>
    <change id="C1" priority="P1" source_item="run-local action_items[0]">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Apply factor(levels = metric_levels) to df_boot_long$metric after the ifelse R2-to-R² rename, consistent with the identical treatment of df_obs (lines 448-451) and df_null_long (lines 452-455). Currently, ifelse strips the factor class, and the subsequent bind_rows at line 464 coerces the mixed factor/character column to character, causing facet_wrap to default to alphabetical order (MAE, R², RMSE) instead of the specified order (RMSE, MAE, R²).</description>
      <spec>
At plot.R line 458, replace:

    df_boot_long$metric <- ifelse(df_boot_long$metric == "R2", "R²", df_boot_long$metric)

with:

    df_boot_long$metric <- factor(
      ifelse(df_boot_long$metric == "R2", "R²", df_boot_long$metric),
      levels = metric_levels
    )

This is the exact same pattern used at lines 448-451 (df_obs) and 452-455 (df_null_long). The metric_levels variable is already defined at line 447 as c("RMSE", "MAE", "R²") and is in scope. After this change, bind_rows at line 464 will receive two factor columns with identical levels, preserving the level ordering in the resulting df_dist tibble, and facet_wrap(~ metric, ...) will render panels in RMSE, MAE, R² order.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - single-line change to a variable assignment within an existing if-branch; no control flow, no new dependencies, no signature changes; pattern already validated at lines 448-455</risk>
      <rollback>Revert the factor() wrapper, restoring the bare ifelse assignment at line 458.</rollback>
    </change>
  </changes>
  <execution_order>C1</execution_order>
</implement_plan>
