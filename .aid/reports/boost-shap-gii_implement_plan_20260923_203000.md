<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-09-23T20:30:00-04:00" />
  <input_reports>
    <report path="user-direct-instruction" mode="run-local-visual-feedback" key_items="3" />
  </input_reports>
  <changes>
    <change id="C1" priority="P1" source_item="user-visual-feedback-m-panel-stat-labels">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Revert the M-panel stat label vjust from 1.5 back to 0.5, increase label.padding from unit(0.15, "lines") to unit(0.5, "lines"), and revert the bottom margin from 3mm to 1mm. The vjust=1.5 pushed labels into the x-axis tick label zone (confirmed by visual inspection of rendered GII plots), defeating the purpose of the fix. Reverting to vjust=0.5 centers the labels on the panel bottom edge (matching the performance panel's confirmed-working approach), and the larger label.padding creates a visually effective white rectangle that occludes the density distribution tails above. The 3mm bottom margin was added solely to accommodate the vjust=1.5 below-panel positioning and is no longer needed.</description>
      <spec>
Three edits in src/boost_shap_gii/scripts/plot.R, all within the M-panel (p1) construction block:

Edit 1 (noise label, line 657): Change `vjust = 1.5` to `vjust = 0.5`.
Edit 2 (noise label, line 658): Change `label.padding = unit(0.15, "lines")` to `label.padding = unit(0.5, "lines")`.
Edit 3 (signal label, line 661): Change `vjust = 1.5` to `vjust = 0.5`.
Edit 4 (signal label, line 662): Change `label.padding = unit(0.15, "lines")` to `label.padding = unit(0.5, "lines")`.
Edit 5 (bottom margin, line 685): Change `plot.margin = unit(c(1, 0.5, 3, 1), "mm")` to `plot.margin = unit(c(1, 0.5, 1, 1), "mm")`.

After all edits, the two annotate blocks should read:
```
      annotate("label", x = stat_pos$x1, y = -Inf,
               label = sprintf("%.2f (%.2f)", noise_m, noise_sd),
               vjust = 0.5, hjust = 0.5, size = 1.5, color = "black",
               fill = "white", label.size = NA, label.padding = unit(0.5, "lines")) +
      annotate("label", x = stat_pos$x2, y = -Inf,
               label = sprintf("%.2f (%.2f)", signal_m, signal_sd),
               vjust = 0.5, hjust = 0.5, size = 1.5, color = "black",
               fill = "white", label.size = NA, label.padding = unit(0.5, "lines")) +
```

And the margin line should read:
```
        plot.margin = unit(c(1, 0.5, 1, 1), "mm")
```

The performance panel's three stat-label sites (lines 486-489, 491-494, 530-533) are NOT modified; they remain at vjust=0.5 with label.padding=unit(0.15, "lines") as confirmed working by the user.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - reverting vjust to the performance panel's confirmed-working value, increasing padding for visibility, and reverting margin to the pre-C2 value</risk>
      <rollback>Restore vjust=1.5, label.padding=unit(0.15, "lines"), and plot.margin 3mm at the three sites</rollback>
    </change>
  </changes>
  <execution_order>C1</execution_order>
</implement_plan>
