<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-09-23T22:00:00-04:00" />
  <assumptions>
    <assumption>Legend text size is increased from 3.8 to 4.2 to accommodate the longer label strings; the user will verify readability via /run-local.</assumption>
  </assumptions>
  <input_reports>
    <report path="(direct user instruction + orchestrator recommendation, this session)" mode="user-feedback" key_items="1" />
  </input_reports>
  <changes>
    <change id="C1" priority="P1" source_item="user visual feedback: M-panel stat-label spatial collision problem is unfixable with annotation-based approach; switch to legend-integrated stats">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Replace the M-panel's spatial annotate("label") stat labels with legend-integrated stats. Eight critique cycles have demonstrated that placing two stat labels at data-dependent x-positions at the panel bottom edge cannot be made collision-free when the noise and signal distributions are close together. The legend-integrated approach embeds the mean(SD) values directly into the legend labels (e.g., "Noise: 0.45 (0.12)"), eliminating all spatial collision, annotation positioning, margin hacking, and white-background occlusion complexity.</description>
      <spec>
Six edit sites within plot.R:

(A) Remove the nudge_stat_labels function definition (lines 378-394). This function computed collision-avoiding x-positions for the two stat labels and has exactly one call site (line 638), which is also being removed. Dead code after this change.

(B) Move and redefine the noise_label and signal_label assignments (currently lines 636-637, before the mean/SD computation). Change from static strings to stat-embedded strings. The assignments must come AFTER the mean/SD computation (lines 630-633) and BEFORE the scale_*_manual() calls that reference them (lines 647-652). Replace:
    signal_label &lt;- "Signal"
    noise_label &lt;- "Noise"
with:
    noise_label &lt;- sprintf("Noise: %.2f (%.2f)", noise_m, noise_sd)
    signal_label &lt;- sprintf("Signal: %.2f (%.2f)", signal_m, signal_sd)

(C) Remove the stat_pos assignment (line 638):
    stat_pos &lt;- nudge_stat_labels(noise_m, signal_m, local_xmax)

(D) Remove the two annotate("label") calls (lines 655-662, the 8 lines starting with annotate("label", x = stat_pos$x1, ...) through the second annotate's closing paren + plus sign).

(E) Remove coord_cartesian(clip = "off") + (line 663). With the annotate labels removed, no element extends outside the panel boundary and clip-off is no longer needed.

(F) In the theme() block:
  - Remove axis.text.x = element_text(margin = margin(t = 5)), (line 680). The tick-label gap was added solely to accommodate the now-removed annotation labels.
  - Revert plot.margin bottom from 3mm to 1mm (line 686): unit(c(1, 0.5, 3, 1), "mm") to unit(c(1, 0.5, 1, 1), "mm"). The extra margin was added to accommodate the now-removed below-panel labels.
  - Increase legend.text size from 3.8 to 4.2 (line 671) to accommodate the longer stat-embedded legend strings.

Post-edit verification:
  - annotate("label" count in M-panel block: 0
  - nudge_stat_labels definition: absent
  - nudge_stat_labels call: absent
  - stat_pos reference: absent
  - coord_cartesian(clip = "off") in M-panel block: absent
  - axis.text.x in M-panel theme: absent
  - bottom margin 3mm: absent
  - bottom margin 1mm: present (1 occurrence, M-panel)
  - legend.text size 4.2: present (1 occurrence, M-panel)
  - sprintf("Noise: in noise_label assignment: present
  - sprintf("Signal: in signal_label assignment: present
  - Performance panel stat labels (geom_label at y=-Inf, vjust=0.5, label.padding=0.15): UNCHANGED (3 sites)
      </spec>
      <dependencies>none</dependencies>
      <risk>low - eliminates 8 critique cycles of positioning complexity; the legend is a fixed layout element with zero collision risk; performance panel is untouched</risk>
      <rollback>Restore annotate("label") calls, nudge_stat_labels function, stat_pos assignment, coord_cartesian(clip="off"), axis.text.x margin, 3mm bottom margin, and revert legend labels to static "Noise"/"Signal" strings</rollback>
    </change>
  </changes>
  <execution_order>C1</execution_order>
</implement_plan>
