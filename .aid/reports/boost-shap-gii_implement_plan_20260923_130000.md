<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-09-23T13:00:00-04:00" />
  <input_reports>
    <report path="(orchestrator-level visual critique analysis, Session 21 critique cycle 5)" mode="visual-critique" key_items="5" />
  </input_reports>
  <assumptions>
    <assumption>None. All five topics (T1-T5) were proposed by the orchestrator and accepted by the user via bare `/implement` invocation without modification.</assumption>
  </assumptions>
  <changes>
    <change id="C1" priority="P1" source_item="T1: Perf panel metric ordering + R-squared superscript">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Force performance panel facet order to RMSE (top), MAE (middle), R-squared (bottom), and display "R2" as "R²" (Unicode superscript). Currently facet_wrap orders alphabetically (MAE, R2, RMSE). The fix converts the `metric` column to a factor with explicit levels in all data frames used by the performance plot, after renaming "R2" to "R²".</description>
      <spec>
Insert a metric-renaming and factor-ordering block immediately after `df_obs` is fully constructed (after line 444, before the `if (has_boot_perf)` block at line 446). This block must apply to `df_obs` unconditionally, since both the has_boot_perf and the CI-band fallback branches use `df_obs`. Additionally, `df_null_long` must be renamed and factored before the fallback branch uses it.

Concretely, insert after line 444 (`left_join(df_null_stats, by = "metric")`):

```r
    # Enforce metric display order: RMSE, MAE, R² (Unicode superscript)
    metric_levels <- c("RMSE", "MAE", "R²")
    df_obs$metric <- factor(
      ifelse(df_obs$metric == "R2", "R²", df_obs$metric),
      levels = metric_levels
    )
    df_null_long$metric <- factor(
      ifelse(df_null_long$metric == "R2", "R²", df_null_long$metric),
      levels = metric_levels
    )
```

Then, inside the `if (has_boot_perf)` block (currently lines 446-456), the `df_boot_long` rename must happen before `df_boot_stats` is computed (so the join key matches the already-renamed `df_obs$metric`), and `df_dist` must inherit the factor. Insert immediately after `df_obs <- df_obs %>% left_join(df_boot_stats, by = "metric")` at line 450, but BEFORE the `df_dist <- bind_rows(...)` at line 452:

Actually, the cleaner approach: rename `df_boot_long$metric` right after the existing `df_boot_stats` join (line 450), then `df_dist` (line 452-456) will inherit the renamed metric from both `df_null_long` and `df_boot_long`. Concretely:

Rename `df_boot_long$metric` BEFORE `df_boot_stats` is computed. Insert immediately after line 446 (`if (has_boot_perf) {`):

```r
      df_boot_long$metric <- ifelse(df_boot_long$metric == "R2", "R²", df_boot_long$metric)
```

This ensures the `group_by(metric)` in `df_boot_stats` produces keys that match the already-renamed `df_obs$metric` for the `left_join`.

The `df_dist` assignment at lines 452-456 inherits the renamed metric from both `df_null_long` and `df_boot_long`, and the `mutate(source = factor(...))` call only factors `source`, not `metric`, so the existing `metric` factor levels propagate correctly.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - display-only change; no data processing or statistical logic affected</risk>
      <rollback>Remove the metric-renaming and factor-ordering block; revert df_boot_long rename line</rollback>
    </change>

    <change id="C2" priority="P1" source_item="T2: Stat label spacing (perf + M panels)">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Increase vertical separation between the M(SD) stat labels and the x-axis tick marks/labels. Currently vjust=1.0 at all 5 annotation sites places the text top at the panel bottom, putting the text body in the same vertical zone as the tick labels. Change vjust from 1.0 to 0.5 at all 5 sites, which centers the text at the panel bottom (top half inside the panel near the density floor, bottom half extending slightly below), creating a visible gap above the tick marks.</description>
      <spec>
At all 5 annotation sites, change `vjust = 1.0` to `vjust = 0.5`:

1. Line 473 (perf panel has_boot_perf branch, trained annotation):
   `vjust = 1.0, hjust = 0.5, size = 1.5, color = "black"` → `vjust = 0.5, hjust = 0.5, size = 1.5, color = "black"`

2. Line 477 (perf panel has_boot_perf branch, null annotation):
   same pattern

3. Line 515 (perf panel CI-band fallback branch, null annotation):
   same pattern

4. Line 639 (M-panel, noise annotation):
   same pattern (this is inside an `annotate("text", ...)` call)

5. Line 642 (M-panel, signal annotation):
   same pattern

Use replace_all since the exact string `vjust = 1.0, hjust = 0.5, size = 1.5, color = "black"` occurs exactly 5 times and all 5 must change.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - rendering-value-only; no logic affected</risk>
      <rollback>Revert vjust from 0.5 to 1.0 at all 5 sites</rollback>
    </change>

    <change id="C3" priority="P1" source_item="T3+T4: M-panel legend tightening and size maximization">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Reduce whitespace around the M-panel's bottom horizontal legend to maximize the distribution plot area. Three edits: (a) reduce legend.margin from margin(1,1,1,1) to margin(0,0,0,0); (b) reduce plot.margin bottom from 4mm to 1mm (the vjust reduction in C2 places stat labels closer to the panel, needing less below-panel accommodation); (c) add legend.box.margin = margin(0,0,0,0) to eliminate the default box-margin ggplot2 adds around the legend grob.</description>
      <spec>
In the M-panel theme block (lines 646-664), make three edits:

1. Line 653: `legend.margin = margin(1, 1, 1, 1),` → `legend.margin = margin(0, 0, 0, 0),`

2. Line 664: `plot.margin = unit(c(1, 0.5, 4, 1), "mm")` → `plot.margin = unit(c(1, 0.5, 1, 1), "mm")`

3. After the existing `legend.margin` line (now `margin(0, 0, 0, 0),`), insert a new line:
   `legend.box.margin = margin(0, 0, 0, 0),`

All three edits are within the same theme() call for the M-panel (p1).
      </spec>
      <dependencies>C2 (vjust reduction enables the bottom margin reduction)</dependencies>
      <risk>low - rendering-value-only; the M-panel density area grows but no data changes</risk>
      <rollback>Revert legend.margin to margin(1,1,1,1), plot.margin bottom to 4mm, remove legend.box.margin line</rollback>
    </change>

    <change id="C4" priority="P1" source_item="T5: V-panel x-axis label overlap prevention">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Prevent x-axis tick label overlap on continuous V-panel features with many integer values (e.g., age_intake with 13 levels from 6-18). Add `guide = guide_axis(check.overlap = TRUE)` to all continuous `scale_x_continuous()` calls in the V-panel. This is ggplot2's built-in mechanism that automatically suppresses labels that would overlap, keeping the visible subset legible. Three sites: singleton continuous (line 996), interaction continuous (line 851), and the interaction continuous per-stratum spline path also uses the same x_scale variable (line 851 assigns it, line 863 applies it).</description>
      <spec>
Two edits:

1. Line 851 (interaction continuous focal x_scale):
   `x_scale <- scale_x_continuous()` → `x_scale <- scale_x_continuous(guide = guide_axis(check.overlap = TRUE))`

2. Line 996 (singleton continuous):
   `scale_x_continuous() +` → `scale_x_continuous(guide = guide_axis(check.overlap = TRUE)) +`
      </spec>
      <dependencies>none</dependencies>
      <risk>low - ggplot2 built-in; labels are suppressed only when they would overlap, no data or statistical change</risk>
      <rollback>Remove guide = guide_axis(check.overlap = TRUE) from both sites</rollback>
    </change>
  </changes>
  <execution_order>C1, C2, C3, C4</execution_order>
</implement_plan>
