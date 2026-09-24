<plotting_critique_report>
  <meta project="boost-shap-gii" mode="ad_hoc_critique_review" timestamp="2026-09-18T12:00:00-04:00" />
  <context>
    Ad hoc critique session conducted after the v1.7.0 plotting-fix /run-local validation (all three <external_project>
    plotting runs passing with zero errors). The user reviewed the generated figures directly and raised 8 critiques
    plus one cross-cutting statistical-design question, discussed sequentially. This report compiles the full session:
    each critique's grounding (code inspection, data inspection, or both), the current disposition, and the explicit
    decisions locked in-session versus items still awaiting the user's call before routing to /implement.
  </context>

  <critique id="C1" title="Stale background diagnostic shell">
    <observation>A background shell (task bvwk717vw) from an earlier spline-crash diagnostic investigation was still
    resident, idle (0 CPU time accumulated), never explicitly terminated after its diagnostic purpose was superseded.</observation>
    <root_cause>Orchestrator hygiene gap: the diagnostic was launched via background Bash during investigation of the
    original cluster-run spline crash, and was not stopped once the investigation concluded.</root_cause>
    <status>resolved</status>
    <resolution>Stopped via TaskStop during this session. No further action.</resolution>
  </critique>

  <critique id="C2" title="Model performance plot: asymmetric distribution representation">
    <observation>In 0_model_performance.png, the null-distribution panel (gray) is a genuine kernel density estimate
    over many permutation draws. The observed-score panel (blue) is a geom_rect spanning [ci_low, ci_high] plus a
    single vertical line at the point estimate — a mean+CI summary, not a distribution.</observation>
    <root_cause>plot.R:459-485 (renders both). compute_bootstrap_ci() in utils.py:1065+ internally generates n_boot
    resampled scores, computes percentile CI bounds from them, and discards the full score vector — only
    (base_score, lower, upper) is returned and persisted to performance_final.csv. The null side has a genuine
    persisted-distribution artifact (permutation_null_distributions.parquet); the observed side has no equivalent.</root_cause>
    <status>recommended_fix (not yet decided by user)</status>
    <recommendation>Persist the observed-score bootstrap draws to a new artifact (e.g.,
    performance_bootstrap_distributions.parquet), analogous to the existing null-distribution artifact, so plot.R can
    density-plot both distributions symmetrically. Requires a Python-side change (predict.py / infer.py /
    compute_bootstrap_ci signature or a parallel persistence call) plus a plot.R rendering change.</recommendation>
    <target_mode>implement (requires user confirmation before scoping)</target_mode>
  </critique>

  <critique id="C3" title="age_intake routed to discrete (group-means) path instead of spline">
    <observation>age_intake (declared type continuous) is rendered via the group-means/jitter/error-bar discrete
    singleton path rather than a spline, in the singleton GII plot.</observation>
    <root_cause>plot.R:909: is_discrete override fires when n_distinct(feature_value) &lt;= SPLINE_DISC_THRESH (15).
    Verified directly against microdata_GII.parquet: age_intake has n_distinct=13 (integer ages 6-18, n=311). Traced
    to the raw source (<external_project>/data/<raw_source_file>, the exact file preprocess.py reads):
    the raw REDCap field "age" is already whole-number valued at collection (float64 dtype, zero fractional values,
    880 raw rows, values 6-20 plus 24). preprocess.py:959-960 does a direct rename with no rounding/truncation step.
    Confirmed: not a preprocessing artifact — age was captured at whole-year resolution at the source.</root_cause>
    <status>open_decision — NOT locked</status>
    <governing_constraint>example_config_advanced.yaml:124: discrete_threshold &gt;= n_knots + degree + 2
    (Wood 2017, GAM ch. 4). With this project's config (n_knots=4, degree=3), the hard floor is 9. At 15, the margin
    above the floor is 6; at a proposed threshold of 10, the margin shrinks to 1. INPUT_SPECIFICATION.md:514-517,
    726-729 documents that features near discrete_threshold exhibit spline/group-means method switching across
    bootstrap resamples with correspondingly wide CIs — an expected, tolerated side effect, not a bug, mitigated by
    the stability gate (stab_thresh).</governing_constraint>
    <analysis>Lowering the shared discrete_threshold to 10 was considered and NOT recommended: it only leaves a
    1-unit safety margin above the Wood-2017 floor, age_intake (13) would sit just 3 above the new cutoff (still
    squarely in the "near-threshold" instability zone), and the change would be global (affecting every feature in
    every config), not scoped to this one variable. Once C7 (moderator-stratification decoupling) and C8 (per-stratum
    mean-tick legibility) are fixed, the only remaining consequence of age_intake's discrete classification is the
    singleton plot's group-means-vs-spline choice — which is a defensible representation on its own merits for a
    coarse, whole-year-resolution clinical age measure, not an obvious defect.</analysis>
    <recommendation>Keep discrete_threshold at 15 (package default, statistically defensible margin) and document
    age_intake's discrete/group-means treatment as an intentional, justified choice in the relevant report/methods
    text. If continuous/spline treatment for age_intake specifically is still wanted after C7/C8 land, prefer a
    <external_project>-project-scoped config change to a modest value (e.g., 12, preserving a 3-unit margin above the
    floor) over the package default, and never to 10.</recommendation>
    <decision_pending>User has not yet given a final verdict; this is the primary open item to resolve before
    documentation/publication.</decision_pending>
  </critique>

  <critique id="C4a" title="Discrete singleton panel: excessive blank space before legend">
    <observation>In plots with few discrete levels (e.g., 3-level features), roughly half the panel width is blank
    space between the last data point and the legend.</observation>
    <root_cause>plot.R:991: capacity &lt;- if(n_lev==2) 7 else if(n_lev==3) 6 else if(n_lev==4) 5.5 else n_lev+0.5.
    For a 3-level feature, capacity=6 while data occupies only x=1-3 (scale_x_continuous(limits=c(0.5,6)),
    plot.R:1030) — roughly half the panel reserved as empty axis space before the legend renders outside the panel.
    This heuristic does not materially affect higher-n_lev features (e.g., age_intake at n_lev=13 gets capacity=13.5,
    negligible padding) — it specifically affects the 2/3/4-level cases where the padding constants were hand-tuned.</root_cause>
    <status>recommended_fix (not yet decided)</status>
    <recommendation>Replace the hardcoded per-n_lev capacity constants with a formula that scales legend/label-width
    reservation more proportionately (or reserve legend space via plot layout/theme mechanisms rather than inflating
    the data x-axis range).</recommendation>
    <target_mode>implement</target_mode>
  </critique>

  <critique id="C4b" title="Spurious red crossing lines for nominal features and __NA__ boundaries">
    <observation>Red dashed "crossing" lines are drawn between every adjacent level pair (by arbitrary factor order)
    wherever the sign of mean SHAP flips — including transitions into/out of the synthetic __NA__ missing-data
    placeholder level, and for genuinely unordered nominal features where adjacent-level order is arbitrary.</observation>
    <root_cause>plot.R:1002-1009 (singleton discrete path) computes cross_points over ALL adjacent df_means pairs
    with zero gating on m_type (nominal vs. ordinal/binary) or on whether either side is the __NA__ placeholder
    (recoded at plot.R:630-637 for any discrete feature). Precedent already exists elsewhere in the codebase for the
    correct policy: the interaction moderator path (plot.R:815-816) explicitly gates crossover annotations with
    if (strat$ordered &amp;&amp; n_strata &gt;= 2) — i.e., only draws them when the stratifying variable is ordered.
    The singleton path has no equivalent gate, and neither path excludes __NA__ from crossing consideration.</root_cause>
    <status>recommended_fix (not yet decided)</status>
    <recommendation>Apply the interaction path's existing ordered-gate pattern to the singleton discrete path (suppress
    crossing lines for nominal/unordered features); additionally, exclude any adjacent pair involving the __NA__
    placeholder from crossing-point consideration in both paths, since missingness is not a point on any ordinal
    scale regardless of the feature's orderedness.</recommendation>
    <target_mode>implement</target_mode>
  </critique>

  <critique id="C5" title="THQ item plots lack short descriptive labels in filenames">
    <observation>Item-level THQ plot filenames use raw codes (e.g., 13_thqch13_GII.png, 16_thqch8_GII.png) rather
    than the short descriptive labels already defined for these items.</observation>
    <root_cause>A THQ short-name mapping already exists (<external_project>/analysis/report/formatting.py:50-69, e.g.,
    "thqch13": "Sexual abuse/assault"), but boost-shap-gii's plot.R has no mechanism to consume any label lookup for
    filenames — clean_name is derived directly from the raw feature name (str_replace_all(feat_name, ...)).</root_cause>
    <status>resolved — descoped by explicit user decision</status>
    <decision>User confirmed this is out of scope for boost-shap-gii (a general-purpose, feature-set-agnostic
    package per its own CLAUDE.md mandate) and belongs in <external_project>'s own post-processing step instead.</decision>
    <target_mode>none (routes to <external_project> project, outside this repo)</target_mode>
  </critique>

  <critique id="C6" title="Nominal feature x-axis labels overlap; sample-size annotations unwanted">
    <observation>event_type (8 raw levels, top-5 truncated to 5 by V-contribution) renders with long category labels
    (e.g., "unintentional_injury") overlapping illegibly on the x-axis.</observation>
    <root_cause>Level labels for nominal features with &gt;5 levels get an N-count suffix
    ("{level}\n(N={n_k})", plot.R:975-978), but no line-break exists within multi-word level names themselves. The
    panel-2 theme block (plot.R:1040-1062, now shifted after C1/C2 edits) has no axis.text.x size or wrapping
    override — it inherits theme_minimal(base_size=7)'s default (~5.6pt), with no rotation, producing collisions
    with 5 long labels packed into scale_x_continuous(breaks=1:n_lev, ...).</root_cause>
    <status>partially locked</status>
    <decision locked="true">Remove sample-size (N=) annotations from x-axis labels entirely — subgroup Ns will be
    documented elsewhere (e.g., a companion table) and are not needed on the plots themselves. This applies to both
    the singleton nominal top-5 path (plot.R:972-978) and the interaction discrete-focal top-5 path
    (plot.R:761-765).</decision>
    <remaining_open_item>Removing the N-suffix shortens labels but does not by itself resolve collision for
    genuinely long level names (e.g., "unintentional_injury" alone is 21 characters). A sizing and/or word-wrap
    treatment (e.g., str_replace_all(label, "_", "\n"), or a reduced axis.text.x size, or label rotation) is still
    needed and was not finalized to a specific implementation.</remaining_open_item>
    <target_mode>implement</target_mode>
  </critique>

  <critique id="C6_aside" title="V-contribution top-5 level-selection criterion (raised during C6 discussion)">
    <observation>event_type's top-5 V-contribution selection excludes sexual_abuse (N=126, the largest subgroup by
    far) in favor of smaller subgroups with larger mean-SHAP deviation from the grand mean.</observation>
    <root_cause_verified>Recomputed directly against microdata_GII.parquet: grand mean SHAP = -0.0625.
    sexual_abuse: N=126, mean_shap=-0.111 (close to grand mean, contribution=0.294, ranks 6th of 8).
    traumatic_death: N=34, mean_shap=-0.711 (large deviation, contribution=14.3, ranks 1st). The ranking formula
    (n_k * (mean_shap_k - grand_mean)^2) is the standard ANOVA between-group sum-of-squares decomposition and is
    deliberately built to mirror the feature's V (variability) statistic elsewhere in the GII framework — confirmed
    functioning as designed, not a bug.</root_cause_verified>
    <status>decided</status>
    <decision>User elected to keep the current V-weighted selection criterion (statistically consistent with how V is
    computed elsewhere in the pipeline). User flagged a documentation concern: excluding a large, clinically salient
    subgroup (sexual_abuse) without explanation may draw reviewer scrutiny.</decision>
    <recommendation_carried_forward>When this reaches /document, add explicit language (methods note and/or a plot
    caption/footnote convention) stating that levels are selected by ANOVA-style V-contribution rather than sample
    size, and that a level's absence indicates its mean SHAP does not deviate materially from the feature's grand
    mean — not that it lacks data.</recommendation_carried_forward>
    <target_mode>document</target_mode>
  </critique>

  <critique id="C7" title="Interaction moderator stratification: excessive spline lines and crossing lines for low-cardinality continuous moderators">
    <observation>Interaction plots using a low-cardinality continuous variable as the moderator (e.g., age_intake x
    cpss_ch_pre_cluster_E, moderator=age_intake) render one full spline line, confidence ribbon, and independent
    crossing-point detection PER UNIQUE MODERATOR VALUE — e.g., 13 overlapping spline lines and a dozen-plus red
    dashed crossing lines for age_intake (13 unique ages), plus stacked/overlapping per-stratum spline-equation text
    annotations at the bottom of the panel.</observation>
    <root_cause>stratify_moderator() (plot.R:255-304) reuses SPLINE_DISC_THRESH (disc_thresh) as BOTH the per-feature
    spline-numerical-stability cutoff (its documented, Wood-2017-anchored purpose — see C3) AND the
    per-moderator-stratum-count cap: when n_unique &lt;= disc_thresh, method_out="natural_levels" creates one stratum
    per unique value; only when n_unique &gt; disc_thresh does it fall back to method_out="quantile_bins" with a
    sensible cap (n_bins = min(floor(n_valid/disc_thresh), n_unique)). These are two distinct concerns that should not
    share one cutoff: a value can be "few enough to visualize as its own feature's discrete group means" while being
    "far too many to serve as a stratifying moderator with its own full spline sub-analysis."</root_cause>
    <status>recommended_fix (not yet decided; user's question about the age_intake threshold implicitly validated
    this as the higher-leverage fix, but no explicit implementation approval was given)</status>
    <recommendation>Decouple the interaction-moderator stratum cap from SPLINE_DISC_THRESH: introduce a separate,
    smaller cap (e.g., a new config field, or a fixed small constant such as 4-5) governing how many strata
    stratify_moderator() will ever produce for a continuous-typed moderator, routing to quantile_bins whenever
    natural-level count would exceed that cap — independent of whether SPLINE_DISC_THRESH itself changes. This fixes
    the worst manifestation of the age_intake problem (C3) regardless of the C3 threshold decision.</recommendation>
    <target_mode>implement</target_mode>
  </critique>

  <critique id="C8" title="Discrete-focal interaction panels: illegible/invisible per-stratum group means">
    <observation>In plot 24 (child_race_ethnicity x cpss_ch_pre_sleep_problems), per-stratum group-mean tick marks at
    each focal category are extremely small and hard to distinguish from the scatter cloud; some focal levels show
    multiple small mean ticks (one per moderator stratum present at that level). A related, more severe instance
    (identified during the C9/age_intake discussion) is plot 20's age_intake-as-focal orientation, where the
    moderator (cpss_ch_pre_cluster_E) resolves to ~12-13 quantile-bin strata, shrinking the mean-tick width to the
    point of near-total invisibility.</observation>
    <root_cause>plot.R:829-860 (discrete-focal branch). Mean-tick width and inter-stratum offset both scale as
    1/n_strata: offset_width=0.6, x_offset spacing = offset_width/n_strata; mean-tick width = 0.3/n_strata; SD
    errorbar width = 0.15/n_strata. As n_strata grows, marks shrink linearly and become imperceptible against
    size-0.9 scatter points. This branch has no crossing-line logic at all (unlike the continuous-focal branch,
    C4b/C7's sibling code at plot.R:815-828), so plot 24 shows no red lines — only the illegible tick problem.</root_cause>
    <status>open_decision — design question raised by user, not resolved</status>
    <user_question>Should per-stratum group means be connected with a line across focal levels (a categorical
    "profile plot" style) to make direction/pattern traceable, analogous to how the continuous-focal branch already
    connects per-stratum spline predictions with geom_line (plot.R:810-812)?</user_question>
    <analysis>A connecting-line treatment would directly address both symptoms: it replaces isolated, hard-to-see
    ticks with a traceable line per stratum, and inherently communicates directional pattern across levels the way
    the user is asking. It does not by itself fix the width-shrinks-with-n_strata problem if raw tick marks are
    retained alongside the line — that scaling issue would need to be decoupled from n_strata (e.g., a fixed minimum
    width, or capping displayed strata, which C7's proposed stratum cap would also mitigate as a side effect for the
    interaction-plot case specifically).</analysis>
    <options>
      <option id="8a">Add a per-stratum geom_line connecting group means across focal x-levels (profile-plot style),
      retaining the existing small tick marks as point indicators along the line.</option>
      <option id="8b">Same as 8a, but replace the tick marks with the connecting line entirely (cleaner, less
      cluttered, loses the individual-level SD/CI errorbar granularity unless retained separately).</option>
      <option id="8c">Decouple tick/offset width from n_strata (e.g., fixed minimum visual width regardless of
      stratum count) without adding a connecting line — addresses legibility but not the "hard to trace direction"
      concern.</option>
    </options>
    <recommendation>8a or 8b in combination with C7's stratum-count cap (which would also reduce n_strata in the
    interaction-plot context generally, indirectly widening tick marks). Final choice between 8a/8b left to user
    preference on visual density.</recommendation>
    <target_mode>implement (pending user's choice of option)</target_mode>
  </critique>

  <cross_cutting_note>
    Per the user's own observation at the close of the critique session: the large majority of issues found (C4a, C4b,
    C6, C7, C8, and the age_intake classification question in C3) concentrate in the item-level run
    (child_child_item_rmse_r2), which has substantially more nominal/ordinal-coded features than the agg or cluster
    runs. This is consistent with the root causes identified above — every finding traces to code paths specific to
    discrete/nominal/ordinal feature handling (singleton discrete path, moderator stratification, top-5 level
    selection), none to the continuous-feature spline paths validated earlier this session.
  </cross_cutting_note>

  <summary>
    <resolved_this_session>C1 (stale shell, stopped), C5 (descoped to <external_project>), C6-N-annotation-removal
    (locked decision), C6_aside/V-contribution criterion (kept as-is, documentation note carried forward)</resolved_this_session>
    <open_decisions_requiring_user_call>
      <item>C3: discrete_threshold — keep at 15 (recommended) vs. project-scoped reduction (max ~12, never 10)</item>
      <item>C8: connecting-line treatment for discrete-focal group means — option 8a, 8b, or 8c</item>
    </open_decisions_requiring_user_call>
    <ready_for_implement_pending_confirmation>C2 (persist observed-score bootstrap distribution), C4a (blank-space
    capacity heuristic), C4b (ordered-gate + __NA__ exclusion for crossing lines), C6 (label wrap/size for long
    nominal names), C7 (decouple moderator-stratum cap from SPLINE_DISC_THRESH)</ready_for_implement_pending_confirmation>
    <routes_to_document>C6_aside (V-contribution selection methods note)</routes_to_document>
  </summary>
</plotting_critique_report>
