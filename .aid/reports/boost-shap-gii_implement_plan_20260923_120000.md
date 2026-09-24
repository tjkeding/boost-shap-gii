<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-09-23T12:00:00-04:00" />
  <input_reports>
    <report path="brainstorm_history/boost-shap-gii_brainstorm_20260922_184900.md" mode="brainstorm" key_items="3" />
  </input_reports>
  <changes>
    <change id="C1" priority="P1" source_item="AI-1 (T1: performance panel stat labels overlap x-axis tick marks)">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Increase the vertical displacement (vjust) of below-axis stat labels in the model performance panel from 1.5 to 3.5, and increase the bottom margin from 12 to 20, in both the has_boot_perf branch and the CI-band fallback branch. This pushes the "mean (SD)" annotations below the x-axis tick label zone and prevents device-boundary clipping.</description>
      <spec>
        5 edits in plot.R:
        1. Line 469: change `vjust = 1.5` to `vjust = 3.5` (trained mean+SD geom_text in has_boot_perf branch)
        2. Line 473: change `vjust = 1.5` to `vjust = 3.5` (null mean+SD geom_text in has_boot_perf branch)
        3. Line 487: change `plot.margin = margin(5.5, 5.5, 12, 5.5)` to `plot.margin = margin(5.5, 5.5, 20, 5.5)` (has_boot_perf theme)
        4. Line 511: change `vjust = 1.5` to `vjust = 3.5` (null mean+SD geom_text in fallback branch)
        5. Line 525: change `plot.margin = margin(5.5, 5.5, 12, 5.5)` to `plot.margin = margin(5.5, 5.5, 20, 5.5)` (fallback theme)
      </spec>
      <dependencies>none</dependencies>
      <risk>low - mechanical value substitutions at well-identified annotation sites; no logic changes</risk>
      <rollback>Revert all five values: vjust back to 1.5, margins back to 12.</rollback>
    </change>

    <change id="C2" priority="P1" source_item="AI-2 (T2: M-panel stat labels overlap x-axis tick marks)">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Increase the vertical displacement (vjust) of below-axis stat labels in the M-panel (importance magnitude density) from 1.5 to 3.5, and increase the bottom margin from 6mm to 12mm. Same root cause and fix strategy as C1, applied to the M-panel's annotate() calls and margin declaration.</description>
      <spec>
        3 edits in plot.R:
        1. Line 645: change `vjust = 1.5` to `vjust = 3.5` (noise annotate in M-panel)
        2. Line 648: change `vjust = 1.5` to `vjust = 3.5` (signal annotate in M-panel)
        3. Line 670: change `plot.margin = unit(c(1, 0.5, 6, 1), "mm")` to `plot.margin = unit(c(1, 0.5, 12, 1), "mm")` (M-panel theme)
      </spec>
      <dependencies>none</dependencies>
      <risk>low - mechanical value substitutions; no logic changes</risk>
      <rollback>Revert: vjust back to 1.5, bottom margin back to 6.</rollback>
    </change>

    <change id="C3" priority="P1" source_item="AI-3 (T3: replace adaptive M-panel legend with fixed bottom horizontal legend)">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Remove the adaptive inset legend placement heuristic (dist_midpoint comparison, if/else branch, legend_x and legend_just_x variables) and replace the positional legend.position/legend.justification theme entries with a fixed bottom horizontal legend (legend.position = "bottom", legend.direction = "horizontal"). The adaptive code becomes dead code once the legend is outside the plot area.</description>
      <spec>
        2 edit regions in plot.R:

        Region A: Remove lines 619-626 (the adaptive heuristic block):
          619:    dist_midpoint <- (noise_m + signal_m) / 2
          620:    if (dist_midpoint < local_xmax / 2) {
          621:      legend_x <- 0.95
          622:      legend_just_x <- 1
          623:    } else {
          624:      legend_x <- 0.05
          625:      legend_just_x <- 0
          626:    }

        Region B: Replace lines 653-654 (theme legend entries):
          653:        legend.position = c(legend_x, 0.95),
          654:        legend.justification = c(legend_just_x, 1),
        with:
          653:        legend.position = "bottom",
          654:        legend.direction = "horizontal",

        The remaining legend theme entries (legend.background, legend.key.size, legend.text, legend.title, legend.margin on lines 655-659) are retained as-is; they apply correctly to a bottom-positioned legend.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - removes an 8-line heuristic block and replaces two theme values; all downstream references to legend_x/legend_just_x are eliminated by this change. The nudge_stat_labels helper (line 617) is unaffected as it controls annotation x-positions, not legend placement.</risk>
      <rollback>Restore the 8-line adaptive heuristic block and the two positional theme entries.</rollback>
      <test_note>The 7 behavioral tests for the adaptive-legend logic added in tests/test_build_20260922b.py during critique cycle 2 will require obsolete-test re-expression in the subsequent /test phase, since the adaptive heuristic is being removed as dead code.</test_note>
    </change>
  </changes>
  <execution_order>C1, C2, C3 (no interdependencies; all target distinct line ranges in plot.R and may be executed in any order)</execution_order>
</implement_plan>
