<plotting_critique>
  <meta project="boost-shap-gii" mode="run-local-feedback" timestamp="2026-09-21T15:00:00-04:00" />
  <source>Visual inspection of /run-local output from three <external_project> child-child runs (agg, cluster, item) generated with v1.7.0 working-tree code.</source>
  <findings>
    <finding id="F1" target="plot.R: model performance panel" severity="minor" scope="all runs">
      <title>Model performance plot: replace p-value annotation with mean+SD statistics</title>
      <description>Remove the p-value text annotation (currently rendered via geom_text at ~line 487-489). The permutation p-values are effectively zero for all metrics in these runs and provide no additional information. Replace with: (a) a vertical mean line for the null/permutation distribution (grey, dashed), complementing the existing observed-score vertical line (blue, solid); (b) mean and SD statistics embedded directly into the existing figure legend labels for both distributions, e.g., "Noise (M = 0.31, SD = 0.08)" for the grey box and "Signal (M = 1.42, SD = 0.19)" for the blue box. Reposition the legend leftward/centrally within the panel so the extended labels remain fully legible and are not clipped by the panel edge.</description>
      <current_behavior>geom_text annotation showing "p = 0.000" in the upper region of each metric facet. Single vertical line at observed score (blue). Legend labels are unadorned "Signal"/"Noise" (or equivalent).</current_behavior>
      <requested_behavior>No p-value annotation. Two vertical lines: observed score (blue, solid, existing) and null mean (grey, dashed, new). Legend labels carry mean and SD statistics for both distributions. Legend repositioned centrally enough that extended labels are not clipped.</requested_behavior>
    </finding>
    <finding id="F2" target="plot.R: M (left) panel of singleton GII plots" severity="minor" scope="all runs">
      <title>M panel: per-feature x-axis scaling with legend-embedded statistics</title>
      <description>The current global shared x-axis (consistent within a run for cross-feature visual comparison) causes small-effect features to render with extremely thin, far-left distributions that are difficult to interpret. Replace the global x-axis with per-feature limits computed from each feature's own signal and noise distribution range (with a small expansion factor). Embed the M point estimate and SD into the existing figure legend labels: e.g., "Noise (M = 0.31, SD = 0.08)" for the grey box, "Signal (M = 1.42, SD = 0.19)" for the blue box. Reposition the legend leftward/centrally so the extended labels are legible. The V (right) panel is unchanged and retains the global axis.</description>
      <current_behavior>Global x-axis range applied to all features' M panels within a run (set via get_global_x_limit_dir). Legend labels unadorned.</current_behavior>
      <requested_behavior>Per-feature x-axis range on the M panel. Legend labels carry M point estimate and SD for signal and noise distributions. Legend repositioned centrally. V panel retains global axis.</requested_behavior>
      <rationale>The M panel's primary analytical purpose is showing signal-vs-noise separation (distribution overlap, shape) per feature. Cross-feature magnitude comparison is already served by the ranked GII output and numeric values in shap_stats_global.csv. Per Cleveland (1993), positional encoding on a common scale is effective only when the signal-to-range ratio is sufficient; when it collapses, a numeric annotation becomes more informative.</rationale>
    </finding>
    <finding id="F3" target="plot.R: interaction plots (all types)" severity="minor" scope="item run">
      <title>Remove all vertical red crossing lines from interaction plots</title>
      <description>The vertical red dashed crossing lines (geom_vline + annotate text markers indicating where moderator-stratified SHAP lines cross) produce visual clutter in interaction plots without proportionate interpretive value. Remove crossing-line rendering from all interaction plot paths: the interaction discrete site (~line 858-860), the singleton discrete site (~lines 985-986), and the nominal site (~line 1066) if applicable. This supersedes the v1.7.0 C2 partial fix (which gated crossing lines behind a nominal-type check) by removing them entirely regardless of moderator type.</description>
      <current_behavior>Red dashed vertical lines at crossing points with text annotations showing the x-coordinate, rendered for non-nominal moderator types (post-C2).</current_behavior>
      <requested_behavior>No crossing lines or crossing-point annotations in any interaction plot, regardless of moderator type.</requested_behavior>
    </finding>
    <finding id="F4" target="plot.R: interaction plots (continuous/ordinal moderators)" severity="minor" scope="item run">
      <title>Interaction plots: reduce default strata to 3, add feature-specific axis labels</title>
      <description>Two sub-items:
      (a) Reduce the default max_interaction_strata from 5 to 3 (Aiken and West 1991 convention for interaction visualization). With 5 overlaid spline lines, the moderator-stratified curves become visually jumbled and difficult to interpret, especially for small-effect interactions. Three strata (corresponding to tertile-like quantile bins) remain within the preattentive color discrimination limit (Healey 1996) while preserving the ability to detect non-linear moderation via the middle stratum. The config parameter (plot.max_interaction_strata) remains user-overridable. Keep the existing range notation for strata labels (e.g., "[2.1, 5.3]"); do NOT substitute with "low/medium/high" labels. Update example_config_advanced.yaml default comment from 5 to 3.
      (b) Replace generic axis labels with actual feature names. The x-axis currently reads "Feature Value" and the moderator reference reads "Moderator Value". Replace with the actual feature names parsed from the interaction effect name (split on the "x" delimiter: e.g., "age_intakexcpss_ch_pre_cluster_E" yields primary="age_intake" and moderator="cpss_ch_pre_cluster_E"). The pipeline is project-agnostic, so names must be derived from existing train/predict outputs (the interaction effect column in shap_stats_global.csv or the plot filename convention), not hardcoded.</description>
      <current_behavior>max_interaction_strata defaults to 5. Axis labels use generic "Feature Value" and "Moderator Value".</current_behavior>
      <requested_behavior>max_interaction_strata defaults to 3 (config-overridable). Strata labels use range notation. Axes display actual feature names parsed from the interaction effect name.</requested_behavior>
    </finding>
  </findings>
  <action_items>
    <item priority="P1" target_mode="implement" finding_ref="F1" description="Model performance plot: remove p-value annotation, add null mean vertical line, embed mean+SD in legend labels, reposition legend" />
    <item priority="P1" target_mode="implement" finding_ref="F2" description="M panel: replace global x-axis with per-feature limits, embed M+SD in legend labels, reposition legend" />
    <item priority="P1" target_mode="implement" finding_ref="F3" description="Remove all crossing-line rendering (geom_vline + annotate) from all interaction plot paths" />
    <item priority="P1" target_mode="implement" finding_ref="F4" description="Reduce max_interaction_strata default from 5 to 3, replace generic axis labels with parsed feature names, update example_config_advanced.yaml" />
  </action_items>
  <next_steps>Route all four findings through /implement (plan + build), then /test, /run-local (re-verify against <external_project>), /document, /publish v1.7.0.</next_steps>
</plotting_critique>
