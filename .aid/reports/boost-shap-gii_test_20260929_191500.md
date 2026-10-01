<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-09-29T19:15:00Z" />
  <pre_design_run>
    <total>1068</total>
    <passed>1053</passed>
    <failed>15</failed>
    <errors>0</errors>
    <coverage_pct></coverage_pct>
    <failures>
      <failure test="TestPlotR_VContributionRanking::test_v_contribution_formula_present" file="tests/test_build_20260507.py" line="283">
        <error_type>AssertionError</error_type>
        <message>V-contribution formula missing from plot.R</message>
        <traceback>assert "count_k * (mean_SHAP_k - grand_mean_SHAP)^2" in plot_r_source or "count_k * (mean_SHAP_k - grand_mean)^2" in plot_r_source</traceback>
      </failure>
      <failure test="TestCapacityHeuristicFix::test_uniform_capacity_formula_present_at_both_sites" file="tests/test_build_20260921.py" line="74">
        <error_type>AssertionError</error_type>
        <message>Expected 'capacity &lt;- n_lev + 0.5' at exactly 2 sites; found 0.</message>
        <traceback>occurrences = plot_r_source.count("capacity &lt;- n_lev + 0.5"); assert occurrences == 2</traceback>
      </failure>
      <failure test="TestStrataCapDecoupling::test_call_site_passes_max_interaction_strata" file="tests/test_build_20260921.py" line="158">
        <error_type>AssertionError</error_type>
        <message>Call site must pass MAX_INTERACTION_STRATA through to stratify_moderator().</message>
        <traceback>assert "stratify_moderator(df_ori$mod_value_enc, df_ori$mod_raw, mod_type, SPLINE_DISC_THRESH, MAX_INTERACTION_STRATA)" in plot_r_source</traceback>
      </failure>
      <failure test="TestStrataCapDecoupling::test_post_stratification_cap_nominal_branch_present" file="tests/test_build_20260921.py" line="165">
        <error_type>AssertionError</error_type>
        <message>if (identical(mod_type, "nominal")) not in plot.R source</message>
        <traceback>assert 'if (identical(mod_type, "nominal"))' in plot_r_source</traceback>
      </failure>
      <failure test="TestConnectingLinesAndFixedWidths::test_fixed_widths_and_visibility_styling_present" file="tests/test_build_20260921.py" line="211">
        <error_type>AssertionError</error_type>
        <message>Exact string 'color = strata_colors[[s]], width = 0.20, linewidth = 1.5' not in plot.R source</message>
        <traceback>assert "color = strata_colors[[s]], width = 0.20, linewidth = 1.5" in plot_r_source</traceback>
      </failure>
      <failure test="TestConnectingLinesAndFixedWidths::test_connecting_geom_line_present" file="tests/test_build_20260921.py" line="239">
        <error_type>AssertionError</error_type>
        <message>Exact string 'color = strata_colors[[s]], linewidth = 0.6, alpha = 1.0' not in plot.R source</message>
        <traceback>assert "color = strata_colors[[s]], linewidth = 0.6, alpha = 1.0" in plot_r_source</traceback>
      </failure>
      <failure test="TestMPanelPerFeatureAxis::test_local_xmax_per_feature_computed" file="tests/test_build_20260921b.py" line="246">
        <error_type>AssertionError</error_type>
        <message>Exact string 'scale_x_continuous(limits = c(0, local_xmax), expand = c(0, 0))' not in plot.R source</message>
        <traceback>assert "scale_x_continuous(limits = c(0, local_xmax), expand = c(0, 0))" in plot_r_source</traceback>
      </failure>
      <failure test="TestMPanelPerFeatureAxis::test_legend_integrated_stats_replace_spatial_annotation" file="tests/test_build_20260921b.py" line="253">
        <error_type>AssertionError</error_type>
        <message>Legend-integrated stats pattern not in plot.R source</message>
        <traceback>assert 'labels = c("Noise" = noise_label, "Signal" = signal_label)' in plot_r_source</traceback>
      </failure>
      <failure test="TestInteractionFeatureNameAxisLabels::test_per_orientation_labels_assigned_in_loop" file="tests/test_build_20260921b.py" line="260">
        <error_type>AssertionError</error_type>
        <message>Per-orientation label assignment in loop not in plot.R source</message>
        <traceback>assert "legend_title &lt;- ori$mod_name" in plot_r_source</traceback>
      </failure>
      <failure test="TestInteractionFeatureNameAxisLabels::test_interaction_axis_uses_focal_label" file="tests/test_build_20260921b.py" line="267">
        <error_type>AssertionError</error_type>
        <message>Interaction axis focal label not in plot.R source</message>
        <traceback>assert "labs(y = NULL, x = focal_label)" in plot_r_source</traceback>
      </failure>
      <failure test="TestInteractionLegendAscendingOrder::test_interaction_legend_reverse_removed" file="tests/test_build_20260922.py" line="105">
        <error_type>AssertionError</error_type>
        <message>'scale_color_manual(values = strata_colors, name = legend_title,' not in plot.R source</message>
        <traceback>assert "scale_color_manual(values = strata_colors, name = legend_title," in plot_r_source</traceback>
      </failure>
      <failure test="TestInteractionLegendAscendingOrder::test_singleton_legend_reverse_removed" file="tests/test_build_20260922.py" line="115">
        <error_type>AssertionError</error_type>
        <message>'scale_color_manual(values = get_red_blue_palette(n_lev), name = legend_title,' not in plot.R source</message>
        <traceback>assert "scale_color_manual(values = get_red_blue_palette(n_lev), name = legend_title," in plot_r_source</traceback>
      </failure>
      <failure test="TestContinuousGradientColorbarReversed::test_guide_colorbar_reverse_wired_in" file="tests/test_build_20260923.py" line="236">
        <error_type>AssertionError</error_type>
        <message>'scale_color_gradient(low = "#b2182b", high = "#2166ac", name = legend_title,' not in plot.R source</message>
        <traceback>assert 'scale_color_gradient(low = "#b2182b", high = "#2166ac", name = legend_title,' in plot_r_source</traceback>
      </failure>
      <failure test="TestMPanelLegendIntegratedStats::test_scale_manuals_reference_the_markdown_labels" file="tests/test_build_20260924.py" line="91">
        <error_type>AssertionError</error_type>
        <message>Expected markdown-label mapping in all 3 M-panel aesthetic channels (fill, color, alpha); found 0 of 3.</message>
        <traceback>assert plot_r_source.count('labels = c("Noise" = noise_label, "Signal" = signal_label)') == 3</traceback>
      </failure>
      <failure test="TestPlotRInteractionDecomposition::test_bootstrap_ribbon_helpers_present" file="tests/test_shell_and_config.py" line="216">
        <error_type>AssertionError</error_type>
        <message>'bootstrap_group_mean_sd &lt;- function' not in plot.R source</message>
        <traceback>assert "bootstrap_group_mean_sd &lt;- function" in content</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
    <disposition test="test_v_contribution_formula_present" file="tests/test_build_20260507.py" classification="obsolete-test">
      <intended_contract>V-contribution-ranked top-5 nominal selection must weight by count times squared deviation from grand-mean SHAP, not raw count alone. Test's own historical design accepted either a descriptive comment or the operative code as the anchor.</intended_contract>
      <current_test_claim>Asserts the literal conceptual-notation string "count_k * (mean_SHAP_k - grand_mean_SHAP)^2" (or its unsuffixed variant) appears anywhere in plot.R.</current_test_claim>
      <evidence>git show HEAD:src/boost_shap_gii/scripts/plot.R line 1003 held this string only as a standalone comment (not executable code); the comment was lost during the implement-build's Phase 1/Phase 2 file reassembly. The operative R expression "n() * (mean(shap_value, na.rm = TRUE) - grand_mean_shap)^2" (and the ") ^ 2" spacing variant, both pre-existing in HEAD) is byte-identical in current plot.R at all 3 call sites (lines 791, 831, 931).</evidence>
      <action>re-express: check the operative formula directly (grand_mean_shap baseline plus squared-deviation weighting), which is a strengthening over the prior comment-or-code anchor since it now requires the executable code to implement the contract. User declined a documentation-restoration follow-up to /implement.</action>
    </disposition>
    <disposition test="test_uniform_capacity_formula_present_at_both_sites" file="tests/test_build_20260921.py" classification="obsolete-test">
      <intended_contract>Both discrete axis-limit sites (interaction discrete-focal, singleton discrete) must use the uniform capacity formula n_lev + 0.5, with no branching heuristic.</intended_contract>
      <current_test_claim>Counts occurrences of the literal assignment "capacity &lt;- n_lev + 0.5"; expects exactly 2.</current_test_claim>
      <evidence>The Phase 1/Phase 2 restructuring (change C1) inlined the formerly-separate capacity variable directly into each limits=c(...) call, threaded through Phase-1-returned struct fields: "limits = c(0.5, od$n_lev + 0.5))" (line 1059) and "limits = c(0.5, sd_data$n_lev + 0.5))" (line 1222). No intermediate assignment statement exists in current source, but the formula is applied identically at both sites.</evidence>
      <action>re-express: count occurrences of "n_lev + 0.5" (== 2) plus explicit presence checks for both fully-qualified limits=c(...) call sites, which verifies wiring more directly than the old intermediate-variable check.</action>
    </disposition>
    <disposition test="test_call_site_passes_max_interaction_strata" file="tests/test_build_20260921.py" classification="obsolete-test">
      <intended_contract>The stratify_moderator() call site must pass MAX_INTERACTION_STRATA through alongside the moderator type.</intended_contract>
      <current_test_claim>Asserts the literal call "stratify_moderator(df_ori$mod_value_enc, df_ori$mod_raw, mod_type, SPLINE_DISC_THRESH, MAX_INTERACTION_STRATA)".</current_test_claim>
      <evidence>Current plot.R line 768 uses "mod_type_ori" in place of "mod_type" (a disambiguating rename introduced when Phase 1's restructuring unified previously-separate scopes into one shared worker function); MAX_INTERACTION_STRATA is still passed through unchanged in the same position.</evidence>
      <action>re-express: substitute mod_type_ori for mod_type in the asserted literal; no other change.</action>
    </disposition>
    <disposition test="test_post_stratification_cap_nominal_branch_present" file="tests/test_build_20260921.py" classification="obsolete-test">
      <intended_contract>The post-stratification cap's nominal branch must gate on the moderator's type equal to "nominal".</intended_contract>
      <current_test_claim>Asserts the literal 'if (identical(mod_type, "nominal"))'.</current_test_claim>
      <evidence>Current plot.R line 784 uses "mod_type_ori" (same disambiguating rename as above); the branch logic, cap threshold, and keep_strata computation (line 795) are unchanged.</evidence>
      <action>re-express: substitute mod_type_ori for mod_type; other assertions in this test body (n_strata &gt; MAX_INTERACTION_STRATA, keep_strata line) confirmed unchanged and left as-is.</action>
    </disposition>
    <disposition test="test_fixed_widths_and_visibility_styling_present" file="tests/test_build_20260921.py" classification="obsolete-test">
      <intended_contract>Per-stratum mean ticks and SD errorbars on discrete-focal interaction plots must use fixed widths/linewidths (0.20/1.5 for mean ticks, 0.05/0.5 alpha-0.6 for SD errorbars), colored per stratum.</intended_contract>
      <current_test_claim>Asserts literal "color = strata_colors[[s]], width = 0.20, linewidth = 1.5" and "color = strata_colors[[s]], alpha = 0.6, width = 0.05, linewidth = 0.5".</current_test_claim>
      <evidence>Current plot.R lines 1114 and 1108 use "od$strata_colors[[s]]" (Phase 2 reading the per-feature color mapping from the Phase-1-returned "od" list); the numeric width/linewidth/alpha values are unchanged.</evidence>
      <action>re-express: add the od$ prefix to strata_colors in both literals; the two "not in" assertions in the same test body (deprecated 0.15/1.2 sizing) confirmed still absent and left as-is.</action>
    </disposition>
    <disposition test="test_connecting_geom_line_present" file="tests/test_build_20260921.py" classification="obsolete-test">
      <intended_contract>Per-stratum group means must be connected by a geom_line when more than one focal level exists, styled at linewidth 0.6/alpha 1.0, colored per stratum.</intended_contract>
      <current_test_claim>Asserts literal "color = strata_colors[[s]], linewidth = 0.6, alpha = 1.0".</current_test_claim>
      <evidence>Current plot.R line 1119 uses "od$strata_colors[[s]]"; the nrow(means_s) &gt; 1 guard and geom_line( call (lines 1116-1117) are unchanged, and the deprecated linewidth-0.8 form remains absent.</evidence>
      <action>re-express: add the od$ prefix to strata_colors in the asserted literal.</action>
    </disposition>
    <disposition test="test_local_xmax_per_feature_computed" file="tests/test_build_20260921b.py" classification="obsolete-test">
      <intended_contract>Each M-panel feature must compute its own local x-axis maximum from its noise/signal SHAP distributions and apply it as the axis limit (replacing a formerly global x-max).</intended_contract>
      <current_test_claim>Asserts literal "local_xmax &lt;- max(...)" (confirmed present, unprefixed) and literal "scale_x_continuous(limits = c(0, local_xmax), expand = c(0, 0))".</current_test_claim>
      <evidence>local_xmax remains an unprefixed Phase-1-local variable (line 645, confirmed present). Phase 2's axis-limit call (line ~1009) now reads it via the Phase-1-returned list: "scale_x_continuous(limits = c(0, p1d$local_xmax), expand = c(0, 0))".</evidence>
      <action>re-express: add the p1d$ prefix to local_xmax in the axis-limit literal only; the local_xmax computation assertion is unchanged.</action>
    </disposition>
    <disposition test="test_legend_integrated_stats_replace_spatial_annotation" file="tests/test_build_20260921b.py" classification="obsolete-test">
      <intended_contract>The M-panel legend must embed mean+SD statistics directly into the "Noise"/"Signal" legend entries via markdown (replacing the superseded spatial-annotation architecture, which must remain fully absent).</intended_contract>
      <current_test_claim>Multiple assertions: noise_label/signal_label sprintf markdown construction present; scale_manual labels mapping present; deprecated plain-string labels, nudge_stat_labels, stat_pos, and annotate("label"/"text", ...) calls all absent.</current_test_claim>
      <evidence>All assertions except one confirmed unchanged via direct grep: noise_label/signal_label sprintf forms present verbatim (lines 647-648, unprefixed Phase-1-local); nudge_stat_labels, stat_pos, and both annotate() forms confirmed fully absent (zero grep matches). Only the scale_manual labels-mapping literal fails, because Phase 2 now reads "p1d$noise_label"/"p1d$signal_label" (lines 1011, 1013, 1015) instead of the bare Phase-1-local names.</evidence>
      <action>re-express: add the p1d$ prefix to noise_label/signal_label in the labels=c(...) mapping literal only; all other assertions in this test body confirmed unchanged and left as-is.</action>
    </disposition>
    <disposition test="test_per_orientation_labels_assigned_in_loop" file="tests/test_build_20260921b.py" classification="obsolete-test">
      <intended_contract>Each interaction-plot orientation must assign its own legend title (from the moderator's name) and focal axis label (from the focal feature's name) inside the per-orientation loop.</intended_contract>
      <current_test_claim>Asserts literals "legend_title &lt;- ori$mod_name" and "focal_label &lt;- ori$focal_name".</current_test_claim>
      <evidence>Current plot.R line 732 uses "legend_title_ori &lt;- ori$mod_name" (same disambiguating-rename pattern as mod_type_ori, needed because Phase 1 now shares this scope across what were previously separate invocations); line 733 "focal_label &lt;- ori$focal_name" is unchanged.</evidence>
      <action>re-express: substitute legend_title_ori for legend_title in the first literal; leave the focal_label literal unchanged.</action>
    </disposition>
    <disposition test="test_interaction_axis_uses_focal_label" file="tests/test_build_20260921b.py" classification="obsolete-test">
      <intended_contract>The interaction plot's x-axis label must be the per-orientation focal feature name (not a generic "Feature Value" string).</intended_contract>
      <current_test_claim>Asserts literal "labs(y = NULL, x = focal_label)".</current_test_claim>
      <evidence>Current plot.R line 1147 uses "labs(y = NULL, x = od$focal_label)" (Phase 2 reading the Phase-1-returned focal_label field via the "od" list).</evidence>
      <action>re-express: add the od$ prefix to focal_label in the asserted literal.</action>
    </disposition>
    <disposition test="test_interaction_legend_reverse_removed" file="tests/test_build_20260922.py" classification="obsolete-test">
      <intended_contract>The interaction plot's discrete-stratum legend must render in ascending order (guide_legend without reverse = TRUE), colored per stratum, titled with the per-orientation legend title.</intended_contract>
      <current_test_claim>Asserts literal "scale_color_manual(values = strata_colors, name = legend_title,".</current_test_claim>
      <evidence>Current plot.R line 1072 uses "od$strata_colors"/"od$legend_title" (both Phase-1-returned, read via the "od" list in Phase 2); the co-asserted guide_legend(override.aes = ...) call (line 1073) is unchanged.</evidence>
      <action>re-express: add od$ prefix to both strata_colors and legend_title in the asserted literal.</action>
    </disposition>
    <disposition test="test_singleton_legend_reverse_removed" file="tests/test_build_20260922.py" classification="obsolete-test">
      <intended_contract>The singleton discrete V-panel's legend must also render in ascending order, matching the interaction legend's treatment (T8 critique extension).</intended_contract>
      <current_test_claim>Asserts literal "scale_color_manual(values = get_red_blue_palette(n_lev), name = legend_title,".</current_test_claim>
      <evidence>Current plot.R line 1219 uses "get_red_blue_palette(sd_data$n_lev)" and "sd_data$legend_title" (Phase 2 reading both fields via the "sd_data" list); the co-asserted guide_legend(override.aes = ...) call (line 1220) is unchanged, and the deprecated reverse=TRUE form remains absent.</evidence>
      <action>re-express: add sd_data$ prefix to both n_lev and legend_title in the asserted literal.</action>
    </disposition>
    <disposition test="test_guide_colorbar_reverse_wired_in" file="tests/test_build_20260923.py" classification="obsolete-test">
      <intended_contract>The continuous singleton V-panel's color gradient legend must render low-to-high top-to-bottom via guide_colorbar(reverse = TRUE).</intended_contract>
      <current_test_claim>Asserts literal 'scale_color_gradient(low = "#b2182b", high = "#2166ac", name = legend_title,'.</current_test_claim>
      <evidence>Current plot.R line 1193 uses "sd_data$legend_title"; the co-asserted "guide = guide_colorbar(reverse = TRUE)" call (line 1194) is unchanged.</evidence>
      <action>re-express: add sd_data$ prefix to legend_title in the asserted literal.</action>
    </disposition>
    <disposition test="test_scale_manuals_reference_the_markdown_labels" file="tests/test_build_20260924.py" classification="obsolete-test">
      <intended_contract>All three M-panel aesthetic channels (fill, color, alpha) must independently wire the markdown-embedded noise_label/signal_label into their labels= argument.</intended_contract>
      <current_test_claim>Asserts plot_r_source.count('labels = c("Noise" = noise_label, "Signal" = signal_label)') == 3, plus static color-literal presence checks for scale_fill_manual/scale_color_manual/scale_alpha_manual and nudge_stat_labels/stat_pos absence.</current_test_claim>
      <evidence>Current plot.R lines 1011, 1013, 1015 use 'labels = c("Noise" = p1d$noise_label, "Signal" = p1d$signal_label)' at exactly 3 sites (confirmed via grep, one per aesthetic channel); the three static color-literal lines (1010, 1012, 1014) and the nudge_stat_labels/stat_pos absence are unchanged.</evidence>
      <action>re-express: add the p1d$ prefix to noise_label/signal_label inside the counted literal; all other assertions in this test body confirmed unchanged and left as-is.</action>
    </disposition>
    <disposition test="test_bootstrap_ribbon_helpers_present" file="tests/test_shell_and_config.py" classification="obsolete-test">
      <intended_contract>Both bootstrap-ribbon helper functions (spline SD and group mean SD) must be defined in plot.R.</intended_contract>
      <current_test_claim>Asserts literals "bootstrap_spline_sd &lt;- function" and "bootstrap_group_mean_sd &lt;- function".</current_test_claim>
      <evidence>Per boost-shap-gii_implement_plan_20260929_140000.md change C3 (user-approved), bootstrap_group_mean_sd was intentionally replaced with an analytical-SE function "group_mean_sd &lt;- function" (line 371; formula sd(subset_y) / sqrt(length(subset_y)), Efron 1979), dropping the B (bootstrap-iteration-count) parameter. bootstrap_spline_sd (line 320) is unchanged in name, with an added max_subsample_n parameter per change C2.</evidence>
      <action>re-express: substitute "group_mean_sd &lt;- function" for the deprecated "bootstrap_group_mean_sd &lt;- function" literal, reflecting the approved API rename (not an incidental refactor side effect).</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>0</tests_created>
    <tests_modified>15</tests_modified>
    <files_created>
      <file path="tests/test_build_20260507.py" test_count="1" coverage_target="V-contribution formula re-expressed to check operative code (grand_mean_shap squared-deviation weighting) instead of the now-orphaned descriptive comment" />
      <file path="tests/test_build_20260921.py" test_count="4" coverage_target="mod_type_ori rename; od$/sd_data$ struct-field prefixes on strata_colors; inlined uniform capacity formula (n_lev + 0.5) wiring at both axis-limit call sites" />
      <file path="tests/test_build_20260921b.py" test_count="4" coverage_target="p1d$ prefix on local_xmax/noise_label/signal_label; legend_title_ori rename; od$ prefix on focal_label" />
      <file path="tests/test_build_20260922.py" test_count="2" coverage_target="od$/sd_data$ prefixes on strata_colors/legend_title/n_lev in ascending-order legend wiring" />
      <file path="tests/test_build_20260923.py" test_count="1" coverage_target="sd_data$ prefix on legend_title in the reversed-colorbar wiring" />
      <file path="tests/test_build_20260924.py" test_count="1" coverage_target="p1d$ prefix on noise_label/signal_label in the 3-channel markdown-label count check" />
      <file path="tests/test_shell_and_config.py" test_count="1" coverage_target="bootstrap_group_mean_sd to group_mean_sd rename, reflecting the approved analytical-SE replacement (change C3)" />
    </files_created>
    <design_rationale>All 15 failures traced via git show HEAD comparison and direct grep against current plot.R to a single root cause class: the approved compute-parallel/render-sequential restructuring (change C1) threaded values through Phase-1-returned structured list fields (od$, sd_data$, p1d$) and introduced two disambiguating variable renames (mod_type -&gt; mod_type_ori, legend_title -&gt; legend_title_ori) plus one intentional, user-approved API rename (bootstrap_group_mean_sd -&gt; group_mean_sd, change C3). No product-bug or ambiguous dispositions were found. Every re-expression is a one-line literal substitution reflecting the new-but-equivalent source text; no assertion was removed, weakened, skipped, or replaced with a tautology, and every co-assertion in each modified test body was independently confirmed unchanged before being left untouched.</design_rationale>
  </design_phase>
  <post_design_run>
    <total>1068</total>
    <passed>1068</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct></coverage_pct>
    <failures></failures>
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>proceed_to_document</recommendation>
  </summary>
  <action_items>
  </action_items>
</test_report>
