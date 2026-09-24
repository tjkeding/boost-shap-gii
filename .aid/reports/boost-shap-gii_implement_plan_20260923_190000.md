<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-09-23T19:00:00-04:00" />
  <assumptions>
    <assumption>label.padding = unit(0.15, "lines") provides a compact white halo without visible box bulk at size 1.5 text. The ggplot2 default is 0.25 lines; 0.15 is a modest reduction to keep labels tight. If visual inspection reveals the padding is too wide or too narrow, this is the single parameter to adjust.</assumption>
    <assumption>For the M-panel, increasing vjust from 0.5 to 1.5 (combined with the existing coord_cartesian(clip = "off")) pushes the labels entirely below the panel edge, clearing the density tails. The bottom margin increase from 1mm to 3mm provides canvas room for the labels to render without ggsave clipping.</assumption>
  </assumptions>
  <input_reports>
    <report path="user_instruction" mode="direct" key_items="2" />
  </input_reports>
  <changes>
    <change id="C1" priority="P1" source_item="user instruction (1): performance panel stat labels need white background">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Switch all performance panel stat labels from geom_text (transparent background) to geom_label (white filled background, no border) so the labels occlude the vertical mean lines beneath them. Three edit sites: two in the has_boot_perf branch (trained mean+SD at line 486 and null mean+SD at line 490) and one in the CI-band fallback branch (null mean+SD at line 528). Layer ordering is already correct (labels are added after geom_vline in all three sites); the only change is adding the white fill via geom_label.</description>
      <spec>
Site 1 (has_boot_perf, trained label, line 486-488): Replace:

        geom_text(data = df_obs, aes(x = boot_mean, y = -Inf,
                  label = sprintf("%.2f (%.2f)", boot_mean, boot_sd)),
                  vjust = 0.5, hjust = 0.5, size = 1.5, color = "black") +

with:

        geom_label(data = df_obs, aes(x = boot_mean, y = -Inf,
                   label = sprintf("%.2f (%.2f)", boot_mean, boot_sd)),
                   vjust = 0.5, hjust = 0.5, size = 1.5, color = "black",
                   fill = "white", label.size = NA, label.padding = unit(0.15, "lines")) +

Site 2 (has_boot_perf, null label, line 490-492): Replace:

        geom_text(data = df_obs, aes(x = null_mean, y = -Inf,
                  label = sprintf("%.2f (%.2f)", null_mean, null_sd)),
                  vjust = 0.5, hjust = 0.5, size = 1.5, color = "black") +

with:

        geom_label(data = df_obs, aes(x = null_mean, y = -Inf,
                   label = sprintf("%.2f (%.2f)", null_mean, null_sd)),
                   vjust = 0.5, hjust = 0.5, size = 1.5, color = "black",
                   fill = "white", label.size = NA, label.padding = unit(0.15, "lines")) +

Site 3 (CI-band fallback, null label, line 528-530): Replace:

        geom_text(data = df_obs, aes(x = null_mean, y = -Inf,
                  label = sprintf("%.2f (%.2f)", null_mean, null_sd)),
                  vjust = 0.5, hjust = 0.5, size = 1.5, color = "black") +

with:

        geom_label(data = df_obs, aes(x = null_mean, y = -Inf,
                   label = sprintf("%.2f (%.2f)", null_mean, null_sd)),
                   vjust = 0.5, hjust = 0.5, size = 1.5, color = "black",
                   fill = "white", label.size = NA, label.padding = unit(0.15, "lines")) +
      </spec>
      <dependencies>none</dependencies>
      <risk>low - purely visual change; geom_label is a drop-in replacement for geom_text with additional fill/border parameters; no data flow, no control flow, no signature changes</risk>
      <rollback>Revert each geom_label back to geom_text and remove the fill, label.size, and label.padding arguments.</rollback>
    </change>
    <change id="C2" priority="P1" source_item="user instruction (2): M-panel stat labels need white background plus more vertical clearance">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Switch both M-panel stat labels from annotate("text", ...) to annotate("label", ...) with white fill, and increase their vjust from 0.5 to 1.5 to push them entirely below the panel edge (clearing the density tails). Also increase the bottom margin from 1mm to 3mm to provide canvas room for the labels to render below the panel without ggsave clipping.</description>
      <spec>
Site 1 (noise label, line 652-654): Replace:

      annotate("text", x = stat_pos$x1, y = -Inf,
               label = sprintf("%.2f (%.2f)", noise_m, noise_sd),
               vjust = 0.5, hjust = 0.5, size = 1.5, color = "black") +

with:

      annotate("label", x = stat_pos$x1, y = -Inf,
               label = sprintf("%.2f (%.2f)", noise_m, noise_sd),
               vjust = 1.5, hjust = 0.5, size = 1.5, color = "black",
               fill = "white", label.size = NA, label.padding = unit(0.15, "lines")) +

Site 2 (signal label, line 655-657): Replace:

      annotate("text", x = stat_pos$x2, y = -Inf,
               label = sprintf("%.2f (%.2f)", signal_m, signal_sd),
               vjust = 0.5, hjust = 0.5, size = 1.5, color = "black") +

with:

      annotate("label", x = stat_pos$x2, y = -Inf,
               label = sprintf("%.2f (%.2f)", signal_m, signal_sd),
               vjust = 1.5, hjust = 0.5, size = 1.5, color = "black",
               fill = "white", label.size = NA, label.padding = unit(0.15, "lines")) +

Site 3 (bottom margin, line 680): Replace:

        plot.margin = unit(c(1, 0.5, 1, 1), "mm")

with:

        plot.margin = unit(c(1, 0.5, 3, 1), "mm")
      </spec>
      <dependencies>none</dependencies>
      <risk>low - purely visual changes; annotate("label") is a drop-in replacement for annotate("text") with additional fill/border parameters; margin increase is a single value change in the theme; no data flow, no control flow, no signature changes</risk>
      <rollback>Revert annotate("label") back to annotate("text"), remove fill/label.size/label.padding arguments, restore vjust to 0.5, and restore bottom margin to 1mm.</rollback>
    </change>
  </changes>
  <execution_order>C1, C2</execution_order>
</implement_plan>
