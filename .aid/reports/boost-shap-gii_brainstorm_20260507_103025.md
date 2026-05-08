<brainstorm_report>
 <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-05-07T10:30:25-04:00" />

 <context_files>
 <file path=".aid/reports/boost-shap-gii_cr_20260424_192405.md" relevance="Source CR report enumerating findings all 25 critical-review findings plus indiv_reports notes N#1-N#2; primary input to this brainstorm." />
 <file path="src/boost_shap_gii/indiv_reports.py" relevance="++N#1+N#2 unified tensor-shape fix targets this module." />
 <file path="src/boost_shap_gii/train.py" relevance=" (outcome-distribution thresholds), (shadow-model leakage), (categorical fillna) target this module." />
 <file path="src/boost_shap_gii/predict.py" relevance=" (categorical fillna) targets predict.py:137." />
 <file path="src/boost_shap_gii/infer.py" relevance=" (categorical fillna) targets infer.py:156; inference CI redesign targets bootstrap workflow in this module." />
 <file path="src/boost_shap_gii/shap_utils.py" relevance=" pooled BH, ddof=1, density-gate, energy-gate target this module." />
 <file path="src/boost_shap_gii/utils.py" relevance=" compute_bootstrap_ci degenerate fallback targets this module." />
 <file path="src/boost_shap_gii/scripts/plot.R" relevance=" spline parity, OUTCOME_MAX scaling fix, V-driver weighted selection target this script." />
 <file path="example_config_advanced.yaml" relevance=" NEGATE_SHAP comment, discrete_threshold validation message anchor target this config." />
 <file path="INPUT_SPECIFICATION.md" relevance=" CI-scale asymmetry documentation, determinism caveat note target this doc." />
 <file path=".gitignore" relevance="Operational fix: indiv_reports.py allowlist entry needed (allowlist pattern at lines 14-27)." />
 </context_files>

 <topics>

 <topic id="topic" title="++N#1+N#2 — Unified Tensor-Shape Fix in indiv_reports.py">
 <summary>The CR identified a memory guard miscalculation (×8 for float64 used where buffers are ×4 float32), a multiclass shape mismatch (point_shap_int allocated 3D and int_iter_folds 4D, neither carrying the class dimension that _compute_interaction_values returns), and an interaction projection that mis-shapes the actual N×B×F×F float32 tensor. CR notes N#1 and N#2 describe related indiv_reports concerns subsumed by the same fix.</summary>
 <research>SHAP interaction values for multiclass tasks return a 4D tensor (N, C, F, F) where C is the number of output classes (Lundberg &amp; Lee 2017; Fujimoto et al. 2006). Memory budgeting for float32 tensors requires sizeof(float32)=4 bytes, not 8.</research>
 <approaches>
 <approach label="Unified tensor-shape fix" feasibility="high" risk="medium">
 <description>Rewrite indiv_reports.py shape handling to: (a) introduce an _output_dim helper deriving C (number of output classes) from the trained estimator; (b) add bias-trim handling in _shap_interaction_single so the per-replicate tensor is properly slimmed of bias columns; (c) expand point_shap_int allocation from (N, F, F) to (N, C, F, F) and int_iter_folds from (K, N, F, F) to (K, N, C, F, F); (d) update the memory guard to use ×4 (float32) and multiply by C for multi-output tasks: bytes = N × B × F × F × 4 × n_outputs.</description>
 <pros>Fixes both the memory guard miscalculation and the multiclass shape error in a single module rewrite. Preserves the float32 efficiency of the existing buffers.</pros>
 <cons>Changes call signatures between indiv_reports.py internals; downstream consumers (parquet writers, plotting microdata producers) must be audited to handle the new class axis.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directives: (1) add `_output_dim` helper to indiv_reports.py returning C from the trained CatBoost estimator (n_classes_ for multiclass, 1 for regression/multi_regression coordinate-wise treatment); (2) modify `_shap_interaction_single` to apply bias-trim returning a tensor of shape (N, C, F, F) where C=1 collapses for non-multiclass; (3) reallocate `point_shap_int` to shape (N, C, F, F) and `int_iter_folds` to shape (K, N, C, F, F); (4) replace memory guard `bytes = N * B * F * F * 8` with `bytes = N * B * F * F * 4 * n_outputs`. Audit downstream consumers (parquet writers, plot.R microdata) to handle the new class axis; add class-stratified output paths for multiclass results. /implement target.</decision>
 </topic>

 <topic id="topic" title=" — Inference-Mode CI Methodological Mismatch">
 <summary>K refits per bootstrap iteration b shared the SAME bootstrap sample, but the deployed K-fold ensemble averages over K DIFFERENT fold splits. CI inferential target did not match point-estimate inferential target.</summary>
 <research>Bootstrap-of-CV (Efron 1983; Davison &amp; Hinkley 1997 ch. 5) restores matching inferential targets: each bootstrap replicate b draws s_b once and partitions s_b via a fresh K-fold split, refitting K models without HP re-tuning, averaging SHAP across K. Basic/reverse-percentile intervals (Davison &amp; Hinkley 1997 §5.2.1) guarantee containment of the point estimate within the CI by structural midpoint-symmetry.</research>
 <approaches>
 <approach label="Bootstrap-of-CV with basic/reverse-percentile intervals" feasibility="high" risk="medium">
 <description>For each bootstrap iteration b: (1) draw bootstrap sample s_b; (2) generate a fresh K-fold split on s_b; (3) train K CatBoost models on the K fold-train portions of s_b using the original fold-specific hyperparameters (no re-tuning); (4) compute SHAP from each of the K models; (5) average SHAP across K to produce one ensemble-replicate per b. Across the B bootstrap replicates, compute the basic/reverse-percentile interval [2·hat − q_hi, 2·hat − q_lo] where q_lo and q_hi are the percentile bounds of the bootstrap distribution and hat is the original ensemble point estimate.</description>
 <pros>Restores matching inferential target between point and CI (both are now ensemble-level on this cohort). Basic/reverse-percentile intervals guarantee point-estimate containment. HP transfer (no re-tuning per b) keeps cost tractable.</pros>
 <cons>Training-mode CIs remain single-fold (typically wider) while inference-mode CIs are ensemble-level (typically narrower). The two are not directly comparable across modes; this asymmetry must be documented for users.</cons>
 <statistical_considerations>The bootstrap distribution now operates at the ensemble level, capturing training-data resampling uncertainty for the K-fold ensemble averaging the deployed pipeline performs. The asymmetry between training-mode (single-fold) and inference-mode (ensemble) CIs is principled, not pathological.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directives: (1) modify infer.py bootstrap loop so each iteration b draws a bootstrap sample once, then partitions via a fresh K-fold split, refitting K models with original fold-specific HPs (no re-tuning), averaging SHAP across K to produce one ensemble-replicate; (2) compute CI as basic/reverse-percentile [2·hat − q_hi, 2·hat − q_lo] using the original point-estimate hat as the centering anchor; (3) add documentation paragraph to INPUT_SPECIFICATION.md explicitly framing the training/inference CI-scale asymmetry: training-mode CIs reflect single-fold variability (wider); inference-mode CIs reflect K-fold ensemble variability on a single bootstrap-resampled cohort (narrower); the two are not directly comparable across modes. /implement target.</decision>
 </topic>

 <topic id="topic" title=" — Pooled BH-FDR Across Effects">
 <summary>Original implementation applied stratified BH per stratum without verifiable PRDS; CR raised the question of whether BY (Benjamini-Yekutieli) is conservative-correct under arbitrary dependence.</summary>
 <research>Pooled BH applied to a single union of null distributions (Benjamini &amp; Hochberg 1995) is FDR-correct under independence and weak dependence; pooling across all effects rather than per-stratum is the standard SHAP-significance approach when the family is conceptually unified.</research>
 <approaches>
 <approach label="Pooled empirical FDR (BH) across effects" feasibility="high" risk="low">
 <description>Apply BH to the pooled p-values across all effects rather than stratifying. Do not apply any cross-family correction across (M, V, GII): each family receives its own independent BH call.</description>
 <pros>Statistically principled; matches the deployed pipeline's inferential intent. Avoids the conservatism of per-stratum BH while retaining FDR control under typical dependence.</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directives: (1) at shap_utils.py lines 905 and 908, retain the existing fdr_bh call but apply across the pooled set of all F effects rather than stratifying by feature type; (2) preserve three independent FDR calls — one each for sig_M, sig_V, sig_GII — with NO cross-family correction. /implement target.</decision>
 </topic>

 <topic id="topic" title=" — GII Composite Justification (Quarantined)">
 <summary>GII = sqrt(M × V) lacked literature precedent or empirical calibration. CR raised three paths: empirical calibration study against known-truth, demote to exploratory, or decision-theoretic anchoring.</summary>
 <research>The GII composite represents a conjunction of two utilities: feature-level model-output magnitude (M) and dose-response informativeness (V). Decision-theoretic framing as a Cobb-Douglas geometric mean (Cobb &amp; Douglas 1928) anchors the composite to multi-attribute utility theory. Hill (1910) dose-response framework supports the V-component as a measure of feature-value-driven prediction variation; Goldstein et al. (2015) ICE plots support the visual interpretation.</research>
 <approaches>
 <approach label="Decision-theoretic framing in public docs + quarantined calibration study" feasibility="high" risk="low">
 <description>Public-repo documentation receives ONLY the decision-theoretic framing: GII as Cobb-Douglas geometric mean of magnitude utility (M) and trend-informativeness utility (V); Hill (1910) dose-response anchor; Goldstein et al. (2015) ICE anchor. No reference to any calibration study, no "in prep" citation, no "see supplemental." The empirical calibration study and the end-to-end determinism evaluation are unified into a single quarantined sub-project living entirely outside the GitHub repo. Specifics of the unified simulation study are deferred to a future brainstorm session.</description>
 <pros>Public-facing pipeline retains decision-theoretic justification for the GII formula. Calibration concern is addressed via a separate study without entangling the main codebase. AID_LOG.md (transparency-only) may reference the quarantined study as part of disclosed development process.</pros>
 <cons>Calibration-study specifics (synthetic data design, recovery metrics, sample sizes) are deferred to next-session brainstorm; this brainstorm session does not specify those details.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directives: (1) update INPUT_SPECIFICATION.md, README.md, and the docstrings of GII-computing functions in shap_utils.py with the decision-theoretic framing: "GII is a Cobb-Douglas geometric mean of magnitude utility M and trend-informativeness utility V; conceptually, a feature is globally important when it both drives model output (M) and exhibits dose-response variation in feature values (V). Hill 1910; Goldstein et al. 2015"; (2) NO reference to calibration study, simulation, "in prep," or "see supplemental" anywhere in any public-repo file (.gitignore-tracked); (3) AID_LOG.md MAY reference the quarantined unified simulation study as transparency disclosure. The unified + quarantined simulation study (calibration + determinism drift) is OUT OF SCOPE for this brainstorm. /implement target for the documentation insertion only.</decision>
 </topic>

 <topic id="topic" title=" — V-Component Sample SD (ddof=1) and Length Guard">
 <summary>V-component uses np.std at six sites; default ddof=0 is population SD, but unbiased sample SD requires ddof=1.</summary>
 <research>Sample SD with ddof=1 is the unbiased estimator for population variance from a sample (Fisher 1925). The numerical divergence from R's default sd (which uses ddof=1) is a contributing source of the plot.R-vs-Python spline mismatch.</research>
 <approaches>
 <approach label="ddof=1 + len&lt;2 guard at all six np.std sites" feasibility="high" risk="low">
 <description>Change np.std(signal) to np.std(signal, ddof=1) at all six V-component computation sites. Add a `if len(signal) &lt; 2: return np.nan` guard before each call to avoid divide-by-zero on degenerate inputs.</description>
 <pros>Fixes the unbiased-SD inconsistency; closes the Python-side component of spline divergence.</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directives: at shap_utils.py lines 223, 229, 251, 264, 303, 360, change `np.std(signal)` to `np.std(signal, ddof=1)`. Before each call, add `if len(signal) &lt; 2: return np.nan`. /implement target.</decision>
 </topic>

 <topic id="topic" title=" — SS-ANOVA Decomposition for 2D Spline (Closed)">
 <summary>CR raised concern that the bivariate spline lacks SS-ANOVA decomposition; the user clarified that SHAP φ_ij is already a pure pairwise-interaction signal with main effects (φ_ii, φ_jj) decomposed by SHAP construction.</summary>
 <research>SHAP interaction values (Lundberg &amp; Lee 2017) decompose feature contributions into main effects (diagonal entries φ_ii) and pairwise interactions (off-diagonal entries φ_ij), so additional SS-ANOVA decomposition is unnecessary.</research>
 <approaches>
 <approach label="Closed as not-a-finding" feasibility="high" risk="low">
 <description>SHAP φ_ij signal has already had main effects controlled for at the SHAP construction layer. The 2D spline operates on this already-decomposed signal; no further decomposition is needed.</description>
 <pros>No code or documentation change.</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Closed. Zero change. No implementation directive.</decision>
 </topic>

 <topic id="topic" title=" — Stratified-Shadow Type-I Error Analysis (Closed by Extension)">
 <summary>CR raised concern that stratified-shadow lacks Type-I error analysis; closed by extension to (pooled BH controls Type-I error simultaneously).</summary>
 <research>Pooled BH-FDR controls Type-I error under independence and weak dependence (Benjamini &amp; Hochberg 1995); the lock to pooled BH directly resolves the concern by treating the family as a single hypothesis space.</research>
 <approaches>
 <approach label="Closed by extension to " feasibility="high" risk="low">
 <description> pooled BH-FDR resolves Type-I error control simultaneously across the M/V/GII families. requires no separate change.</description>
 <pros>No additional implementation work.</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Closed by extension. Zero change. No implementation directive.</decision>
 </topic>

 <topic id="topic" title=" — No Joint M/V/GII Multiplicity Correction">
 <summary>CR raised concern that sig_M, sig_V, sig_GII are independently FDR-corrected without joint multiplicity correction; user determined that no cross-family correction is appropriate.</summary>
 <research>The M, V, and GII families measure conceptually distinct properties of the same effects (mean magnitude, dose-response variability, conjunction). Joint correction across families is not standard practice in SHAP-significance pipelines; the user judges independent FDR per family as the correct approach.</research>
 <approaches>
 <approach label="Independent FDR per family, no cross-family correction" feasibility="high" risk="low">
 <description>Three independent BH calls — one for sig_M, one for sig_V, one for sig_GII — without any joint correction across families.</description>
 <pros>Preserves the conceptual distinction between magnitude, variability, and conjunction. Matches the deployed pipeline's inferential intent.</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directive: existing three independent significance flags (sig_M, sig_V, sig_GII) at shap_utils.py lines 918-946 are retained as-is. Zero code change.</decision>
 </topic>

 <topic id="topic" title=" — Outcome-Distribution Threshold Citation Anchor">
 <summary>CR flagged outcome-distribution thresholds in _diagnose_outcome_distribution as heuristic; user determined the thresholds themselves are appropriate but the citation anchor needs minimal clarification.</summary>
 <research>Skewness and kurtosis thresholds for distributional diagnostics: Groeneveld &amp; Meeden (1984) for skewness; Joanes &amp; Gill (1998) for kurtosis.</research>
 <approaches>
 <approach label="Terse citation anchor update only" feasibility="high" risk="low">
 <description>Add a concise literature citation to the docstring of _diagnose_outcome_distribution at train.py near lines 252-255. No threshold value changes.</description>
 <pros>Minimal change; closes the citation gap without altering pipeline behavior.</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directive: add Groeneveld &amp; Meeden (1984) for skewness and Joanes &amp; Gill (1998) for kurtosis as citation anchors in the docstring of _diagnose_outcome_distribution at train.py near lines 252-255. Keep the docstring update terse (single sentence per citation, no extended discussion). NO threshold value changes. /implement target.</decision>
 </topic>

 <topic id="topic" title=" — plot.R Spline Parity with Python LSQUnivariateSpline">
 <summary>plot.R uses lm(bs) for spline visualization while the Python pipeline uses scipy LSQUnivariateSpline for V-computation. The two diverge in basis construction and fitting (lm uses OLS minimization on B-spline basis vs LSQ minimization on a different basis), producing visualizations inconsistent with the V-statistic shown beneath them.</summary>
 <research>Python's _get_adaptive_knots_and_degree uses np.percentile-based knot construction with boundary exclusion and degree downgrade to 1 if fewer than 4 knots (shap_utils.py:146-164). R's splines::splineDesign supports fixed interior knots and matches the scipy LSQ approach at the basis-construction level.</research>
 <approaches>
 <approach label="Replace lm(bs) with R-native LSQ-equivalent" feasibility="high" risk="medium">
 <description>In plot.R lines 118-148 (calc_v_spline_pred function), replace lm(ys ~ bs(xs, knots, degree)) with an R implementation of LSQUnivariateSpline using the same adaptive-knot construction from Python: np.percentile-based knots, boundary exclusion, degree downgrade to 1 if fewer than 4 knots. Use splines::splineDesign or splines::spline.des with fixed interior knots to mirror the scipy LSQ behavior.</description>
 <pros>Closes the R/Python spline divergence; visualization will match the V-statistic beneath it.</pros>
 <cons>Requires careful porting of the adaptive-knot logic from Python to R; numerical parity must be verified during /test.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directives: (1) port Python's `_get_adaptive_knots_and_degree` from shap_utils.py:146-164 to plot.R as a helper function using R's quantile with type=7 (default, matches numpy's default percentile method) for knot construction; (2) implement boundary exclusion and degree downgrade to 1 if fewer than 4 unique interior knots; (3) replace lm(bs) at calc_v_spline_pred (plot.R:118-148) with splines::splineDesign-based LSQ fit using the ported knot/degree logic. /implement target.</decision>
 </topic>

 <topic id="topic" title=" — Cluster-Bootstrap Equal-Size Assertion (Closed)">
 <summary>CR flagged the cluster-bootstrap equal-size assertion at shap_utils.py:730-734 as brittle for unbalanced clusters in real data. User determined cluster_ids is set internally at shap_utils.py:1096 from the K-fold structure and is K-balanced by construction.</summary>
 <research>cluster_ids is not user-supplied data; it is a K-balanced internal grouping derived from the K-fold split, so the equal-size assertion is satisfied by construction. The CR's concern reflects a context gap (CR did not have visibility into the internal-only nature of cluster_ids), not a real brittleness.</research>
 <approaches>
 <approach label="Closed as not-a-finding" feasibility="high" risk="low">
 <description>cluster_ids is set internally and K-balanced by construction. The equal-size assertion is correct.</description>
 <pros>No code or documentation change.</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Closed. Zero change. No implementation directive.</decision>
 </topic>

 <topic id="topic" title=" — Phase-2 Shadow-Model Early-Stopping Leakage">
 <summary>Phase-2 shadow CatBoost fit uses eval_set=pool_val_full for early stopping, leaking the outer-validation pool into the shadow model's stopping criterion.</summary>
 <research>Boruta-style stratified shadow features (Kursa &amp; Rudnicki 2010) require the shadow model to be trained on the same training data as the real model with no leakage from validation/test pools. CatBoost oblivious-tree training with ordered boosting (Prokhorenkova et al. 2018) is well-defined for fixed-iteration training without eval_set.</research>
 <approaches>
 <approach label=" — Fixed shadow_iterations ceiling at 2 × tuned_iters" feasibility="high" risk="low">
 <description>Replace the eval_set-driven early stopping with a fixed iteration ceiling: shadow_iterations = 2 * tuned_iters. Remove the eval_set=pool_val_full argument from the phase-2 shadow CatBoost fit. The 2× ceiling preserves the original rationale that shadow models need additional iterations to converge with 2p shadow features added to the feature space.</description>
 <pros>Eliminates outer-validation-pool leakage; preserves the principled shadow-iteration upper bound.</pros>
 <cons>Loses adaptive iteration count for shadow training; may train slightly longer than necessary on some datasets but never under-trains.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directives: at train.py lines 910-950, (1) set shadow CatBoost's iterations parameter to 2 * tuned_iters as a fixed ceiling; (2) remove the eval_set=pool_val_full argument from the phase-2 shadow fit call at lines 947-948; (3) do NOT use any early-stopping criterion in phase-2 shadow training. /implement target.</decision>
 </topic>

 <topic id="topic" title=" — multi_regression OUTCOME_MAX Scaling Mismatch">
 <summary>train.py applies StandardScaler z-scaling to multi_regression targets; SHAP values are produced on the z-scaled space. plot.R divides SHAP values by raw OUTCOME_MAX (a non-z-scaled scalar) and multiplies by 100, creating a unit mismatch.</summary>
 <research>StandardScaler z-scaling produces SHAP values in standard-deviation units of the original target. Dividing by OUTCOME_MAX is a percentage-of-max transform that is incompatible with z-scaled SHAP. The correct behavior is to plot z-scaled SHAP on the z-scale.</research>
 <approaches>
 <approach label="Remove OUTCOME_MAX rescaling in plot.R for multi_regression" feasibility="high" risk="low">
 <description>Remove the line at plot.R:385 `df_m$shap_value &lt;- (df_m$shap_value OUTCOME_MAX) * 100` for multi_regression. SHAP values remain on z-scaled space and are plotted as such.</description>
 <pros>Eliminates the unit mismatch; z-scaled SHAP is the natural unit for multi_regression with MultiRMSE loss.</pros>
 <cons>Plot y-axis units change for multi_regression users; requires a brief documentation note that multi_regression SHAP is on z-scale.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directives: (1) at plot.R:385, remove the line `df_m$shap_value &lt;- (df_m$shap_value OUTCOME_MAX) * 100` for the multi_regression code path; (2) ensure non-multi_regression code paths retain their existing scaling; (3) confirm predict.py inverse-transform at lines 271-278 applies to predictions only (NOT to SHAP values), which is the correct existing behavior; (4) add a brief INPUT_SPECIFICATION.md note that multi_regression SHAP values are in z-scaled units of the target. /implement target.</decision>
 </topic>

 <topic id="topic" title=" — CatBoost Bitwise Determinism Caveat (Documentation Note)">
 <summary>CatBoost is bit-exact only at thread_count=1; multi-thread execution introduces non-deterministic floating-point order-of-operations drift. The pipeline runs multi-threaded for tractability.</summary>
 <research>CatBoost (Prokhorenkova et al. 2018), unlike LightGBM (deterministic=true) and XGBoost (partial determinism support), does not provide a multi-thread bitwise-determinism flag. The expected magnitude of drift between independent multi-thread runs is empirically small relative to the shadow-bootstrap noise floor.</research>
 <approaches>
 <approach label="Brief documentation note" feasibility="high" risk="low">
 <description>Add a terse note to INPUT_SPECIFICATION.md (or README.md) stating that bit-exact determinism is not guaranteed under multi-thread execution, and that the magnitude of numerical drift is assumed to fall well below the shadow-bootstrap noise floor. No config flag, no code change.</description>
 <pros>Sets correct user expectations without imposing a single-thread performance penalty. Defers any quantitative drift evaluation to the -unified simulation study.</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directive: add one or two terse sentences to INPUT_SPECIFICATION.md (preferred) noting that CatBoost bit-exact reproducibility is not guaranteed under multi-thread execution, and that the expected magnitude of numerical drift is assumed to fall well below the shadow-bootstrap noise floor. NO config flag, NO code change. /implement target.</decision>
 </topic>

 <topic id="topic" title=" — Degenerate compute_bootstrap_ci Fallback">
 <summary>compute_bootstrap_ci returns (base_score, base_score, base_score) when all bootstrap iterations are NaN/dropped, silently presenting a zero-width CI as if valid.</summary>
 <research>NaN-propagation conventions in numpy (via nanmean, nanpercentile, nanmedian) treat NaN as "missing" rather than "valid zero." JSON serialization of NaN as null is the standard convention for "CI undefined." Existing pipeline V-spline failures already return NaN; the fallback should follow the same convention.</research>
 <approaches>
 <approach label="Return (base_score, NaN, NaN) with explicit warning" feasibility="high" risk="low">
 <description>Replace the degenerate fallback at utils.py:665-666 with `return (base_score, np.nan, np.nan)` and emit an explicit warning identifying the effect for which all bootstrap iterations were dropped.</description>
 <pros>Aligns with the pipeline's existing NaN-on-failure convention. NaN serializes to JSON null with correct "CI undefined" semantics.</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directives: at utils.py:665-666, change the degenerate fallback to `return (base_score, np.nan, np.nan)`; emit a warning identifying the effect for which all bootstrap iterations were dropped. Audit downstream consumers to confirm NaN propagation is handled (existing `nanmean`/`nanpercentile` paths already accommodate this). /implement target.</decision>
 </topic>

 <topic id="topic" title=" — Inference-Mode cluster_ids Semantics (Closed)">
 <summary>CR raised semantic concerns about cluster_ids in inference mode; user determined the existing inline comment at shap_utils.py:974-987 is sufficient and the behavior is statistically principled (K-fold ensemble mean as point estimate, cluster-level bootstrap as the correct resampling unit).</summary>
 <research>cluster_ids in inference mode serves a dual role: K-fold ensemble replicate grouping key and bootstrap resampling unit. The existing inline comment at lines 974-975 documents the inference-mode microdata averaging behavior.</research>
 <approaches>
 <approach label="Status quo, zero change" feasibility="high" risk="low">
 <description>Behavior is correct and existing inline comment is sufficient. The CR concern reflects naming/documentation rather than behavior.</description>
 <pros>No code or documentation change.</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Closed. Zero change. No implementation directive.</decision>
 </topic>

 <topic id="topic" title=" — Top-5 Nominal V-Driver Weighted Selection">
 <summary>plot.R selects top-5 nominal levels by frequency for visualization, but V_nominal is computed as frequency-weighted between-group SD (ANOVA SS), so the appropriate selection criterion is V-contribution per level: N_k × (mean_k − grand_mean)². Frequency-only selection can hide V-driving levels that are moderately frequent but extreme in mean SHAP.</summary>
 <research>V_nominal is computed via groupby.transform('mean') then np.std on the per-observation series, producing the frequency-weighted between-group SD. The per-level contribution to V is exactly the ANOVA between-group sum-of-squares term: N_k × (mean_k − grand_mean)². Top-K selection by this contribution is the principled criterion for V-driving levels.</research>
 <approaches>
 <approach label="V-driver weighted top-5 selection (nominal-only)" feasibility="high" risk="low">
 <description>In plot.R lines 481-485, replace the frequency-only top-5 selection with a V-contribution-ranked selection: per level k, compute count_k × (mean_SHAP_k − grand_mean_SHAP)²; select the 5 levels with the highest V-contribution; display these 5 levels in the plot. Applies to NOMINAL features only — ordinal features must retain their inherent ordering and cannot be permutation-truncated.</description>
 <pros>Selection criterion now matches the V-statistic being visualized. Preserves the most V-informative levels rather than the most frequent ones.</pros>
 <cons>None.</cons>
 <statistical_considerations>The V-contribution is the ANOVA between-group SS contribution per level; ranking by this contribution exactly matches the per-level contribution to V_nominal.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directives: (1) in plot.R lines 481-485, replace the frequency-based top-5 filter with V-contribution-ranked selection: for each level k, compute `contribution_k = count_k * (mean_SHAP_k - grand_mean_SHAP)^2`; select the 5 levels with highest contribution; (2) display the selected 5 levels in the plot, with N_k annotated below each level for transparency; (3) restrict this selection to NOMINAL features ONLY — ordinal features retain their inherent ordering and are NOT subject to permutation truncation. /implement target.</decision>
 </topic>

 <topic id="topic" title=" — Categorical fillna with Distinct Sentinels">
 <summary>Current categorical handling uses fillna("__NA__") indiscriminately for both training-time NaN and inference-time unseen levels. NaN at inference may carry SHAP signal (refusal-to-answer is informative), but unseen levels are out-of-distribution and should receive prior-mean fallback.</summary>
 <research>CatBoost's prior-mean fallback (Prokhorenkova et al. 2018) handles unseen categorical levels at inference by assigning the prior mean of the target. NaN-handling at training preserves the SHAP signal of "missing" as a meaningful category (e.g., refusal to answer in survey data may itself predict the outcome). Distinguishing these two cases requires distinct sentinels at inference.</research>
 <approaches>
 <approach label="Tier-1/Tier-2 validation + distinct __NA__/__UNSEEN__ sentinels" feasibility="high" risk="low">
 <description>At predict.py:137 and infer.py:156, mirror the existing ordinal validation pattern (lines 148-169): tier-1 (ValueError if more than 50% of unique values are unknown), tier-2 (warning if more than 10% of observations are unknown). Define a new `_label_nominal` function: NaN → "__NA__", unseen-level → "__UNSEEN__", in-distribution → preserved. train.py:651 unchanged (training never produces "__UNSEEN__"). The sentinels are distinct so CatBoost can handle them independently — "__NA__" carries training-time signal; "__UNSEEN__" routes to prior-mean fallback.</description>
 <pros>Distinguishes informative-missing from out-of-distribution. Tier-1 fail-loud guard against silent OOD inference. Tier-2 surfaces problematic but non-fatal cases as warnings.</pros>
 <cons>Tier-1 ValueError is a breaking change for workflows where the inference dataset contains a high fraction of novel levels; users must either retrain with expanded codebook or accept OOD-data violation.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directives: (1) define `_label_nominal(v, levels)` helper that returns "__NA__" if pd.isna(v), "__UNSEEN__" if v not in levels, else v unchanged; (2) at predict.py:137 and infer.py:156, replace existing fillna logic with `X[c] = df_raw[c].apply(lambda v: _label_nominal(v, levels)).astype(str).astype("category")`; (3) at the same sites, add tier-1/tier-2 validation mirroring the ordinal pattern at predict.py:148-169 — tier-1 raises ValueError if more than 50% of unique values are unknown, tier-2 emits warning if more than 10% of observations are unknown; (4) train.py:651 remains unchanged (training data never produces "__UNSEEN__"). Downstream impact: "__UNSEEN__" propagates through SHAP buffers and microdata parquets; CatBoost prior-mean fallback handles it correctly; plot.R will display "__UNSEEN__" as a literal categorical level, with V-driver selection naturally handling its inclusion or exclusion based on V-contribution. /implement target.</decision>
 </topic>

 <topic id="topic" title=" — NEGATE_SHAP Plot-Only Semantic Documentation">
 <summary>NEGATE_SHAP is a config flag whose semantic role was undocumented; user clarified the flag should ONLY affect plot.R rendering, not M/V/GII calculations or microdata SHAP values.</summary>
 <research>Code audit confirms NEGATE_SHAP negate_shap is consumed ONLY in plot.R (lines 382, 652-659, 812-818) and validated as plot.negate_shap (config-block-scoped). M, V, GII, sig_*, bootstrap CIs, and microdata parquets are sign-invariant.</research>
 <approaches>
 <approach label="Single-line config comment" feasibility="high" risk="low">
 <description>Add a single comment "ONLY affects plot.R rendering" to example_config_advanced.yaml under the plot.negate_shap entry. No other documentation change. No source-code change.</description>
 <pros>Minimal change; clarifies the semantic at the user's primary touchpoint (config file).</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directive: at example_config_advanced.yaml under the `plot.negate_shap` entry, add a single comment "# ONLY affects plot.R rendering". No additional documentation, no source-code change. /implement target.</decision>
 </topic>

 <topic id="topic" title=" — Density-Gate discrete_threshold Hard-Error Validation">
 <summary>The user-configurable discrete_threshold gates spline-vs-group-means routing. Default is 10, which is barely above the spline-stability lower bound (n_knots + degree + 2 = 9 for default config). Users may set it below this lower bound, producing near-degenerate spline fits.</summary>
 <research>Spline stability requires at least n_knots + degree + 2 unique x values to support the basis (Wood 2017 Generalized Additive Models, ch. 4). Below this threshold, the basis is rank-deficient and fits become unstable.</research>
 <approaches>
 <approach label="Hard-error config validation + Wood 2017 anchor" feasibility="high" risk="low">
 <description>At config-load, raise a ValueError if user-supplied `discrete_threshold &lt; n_knots + degree + 2`. Add a one-line config comment anchoring the constraint to Wood (2017) GAM ch. 4.</description>
 <pros>Matches the project's "err on kill" doctrine. Surfaces the constraint at config-load before any pipeline work, allowing the user to make a deliberate within-bounds choice.</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directives: (1) in utils.py config validation, after parsing shap.splines.discrete_threshold, n_knots, and degree, raise ValueError if `discrete_threshold &lt; n_knots + degree + 2` with message identifying the violation and citing Wood (2017) GAM ch. 4; (2) at example_config_advanced.yaml under `shap.splines.discrete_threshold`, add a single comment anchoring the lower bound: "# Must satisfy: discrete_threshold >= n_knots + degree + 2 (Wood 2017 GAM ch. 4)". /implement target.</decision>
 </topic>

 <topic id="topic" title=" — Energy-Gate 1.001 Tolerance Anchor">
 <summary>The energy-gate stability check uses `if tv_signal &gt; 1.001 * tv_reference: return False`; the 1.001 multiplier is a floating-point tolerance whose magnitude (0.1%) exceeds machine epsilon by orders of magnitude. The existing comment is brief but does not explicitly frame the constant as an empirical balance margin.</summary>
 <research>Pure machine epsilon for float64 (≈2.2e-16) compounded through N splev operations produces relative errors around 1e-10 to 1e-8 for typical N — well below 1e-3. The 0.1% tolerance is an empirical balance: tight enough to catch genuine spline overshoot, loose enough that splev rounding plus diff/sum cancellation does not produce false fails. Higham (2002) Accuracy and Stability of Numerical Algorithms ch. 1 discusses the role of empirical tolerances in numerical software.</research>
 <approaches>
 <approach label="Brief docstring anchor to Higham (2002)" feasibility="high" risk="low">
 <description>Expand the existing comments in `_check_spline_energy_stability_1d/2d` to frame 0.1% as an empirical balance margin between false-pass and false-fail rates of the energy gate; anchor to Higham (2002) ch. 1.</description>
 <pros>Documentation honesty about the empirical nature; minimal change proportional to a "minor" finding.</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directive: in shap_utils.py, expand the existing comments at lines 178-180 (1D) and 209-211 (2D) to frame 0.1% as an empirical balance margin between false-pass and false-fail rates of the energy gate; anchor to Higham (2002) Accuracy and Stability of Numerical Algorithms ch. 1. NO code change beyond the docstring expansion. /implement target.</decision>
 </topic>

 <topic id="topic" title=" — Color-Blind Accessibility (Not-a-Finding)">
 <summary>RdBu palette in plot.R was flagged as potentially problematic for severe deuteranopia/protanopia; user determined that RdBu is ColorBrewer-rated colorblind-safe for 5+ classes and is the appropriate divergent palette for signed-SHAP rendering.</summary>
 <research>ColorBrewer's RdBu sequence is rated colorblind-safe for 5+ classes (Brewer 1999). Sequential palettes (viridis, cividis) are not divergent and would not preserve the signed-SHAP zero-centered semantic.</research>
 <approaches>
 <approach label="Closed as not-a-finding" feasibility="high" risk="low">
 <description>RdBu is appropriate for the signed-SHAP rendering use case. No change.</description>
 <pros>None applicable.</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Closed. Zero change. No implementation directive.</decision>
 </topic>

 <topic id="topic" title=" — MAD/IQR/SD Scale-Estimator Fallback (Already Resolved)">
 <summary>CR flagged MAD/IQR/SD scale-estimator fallback chain as undocumented; code audit confirms the fallback rationale, trigger conditions, and selected-estimator surfacing are all documented and visible in the diagnostic message.</summary>
 <research>Existing implementation in train.py:_diagnose_outcome_distribution: docstring lines 213-215 explain the fallback chain; lines 239-241 explain the trigger condition (MAD = 0 with high zero-inflation); lines 244-249 set `scale_method` to "MAD", "IQR", or "SD"; lines 284-297 emit explicit messages stating which method was used. Citations: Huber (1981), Maronna et al. (2006).</research>
 <approaches>
 <approach label="Already resolved by existing code" feasibility="high" risk="low">
 <description>No additional change needed. CR recommendations (log selected estimator + candidate values, document fallback triggers) are already satisfied by the existing diagnostic warning.</description>
 <pros>None applicable.</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Closed. Zero change. No implementation directive.</decision>
 </topic>

 <topic id="topic" title=" + — Unified Quarantined Simulation Study (Out-of-Scope This Session)">
 <summary> (end-to-end determinism integration test) and (GII calibration study) both require simulated data and simulated results. They are unified into a single quarantined sub-project living entirely outside the GitHub repo. Specifics of the unified simulation study are deferred to a future brainstorm session.</summary>
 <research>The quarantine architecture (lock): no reference to the simulation study in any public-repo file; AID_LOG.md may reference the study as transparency disclosure; the sub-project lives entirely outside the GitHub working tree.</research>
 <approaches>
 <approach label="Defer to next-session brainstorm with quarantine architecture preserved" feasibility="high" risk="low">
 <description> + are formally OUT OF SCOPE for the current brainstorm. The unified simulation study (covering GII calibration AND end-to-end determinism drift evaluation) will be brainstormed in a separate, dedicated session. The public repo receives ONLY the -theoretic framing (locked under). Determinism handling in the public repo is the brief documentation note (locked under) — no code or quantitative drift evaluation in the public repo.</description>
 <pros>Preserves the quarantine architecture from the lock; avoids public-repo entanglement with a sub-project that has its own design questions.</pros>
 <cons>Specifics of calibration metrics, synthetic data design, drift tolerance, and sample sizes are not specified in this brainstorm.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Out-of-scope for this brainstorm. Implementation directives: NONE in this cycle. Next-session brainstorm: design the unified simulation study (calibration + determinism drift) with quarantine architecture intact. NO public-repo file shall reference the simulation study, "in prep," "see supplemental," or any similar forward citation. AID_LOG.md MAY reference it as transparency disclosure. The simulation sub-project lives outside the GitHub working tree.</decision>
 </topic>

 <topic id="topic" title="Operational — gitignore Allowlist + Bundled Release Sequencing">
 <summary>indiv_reports.py is not in the.gitignore allowlist (allowlist pattern: line 2 `*` ignores everything; lines 14-27 whitelist specific source files). The new module is silently excluded from version control. The bundled release plan from prior cycles is reaffirmed: single bundled release after this brainstorm's remediation lands.</summary>
 <research>The.gitignore allowlist pattern requires explicit whitelisting of every tracked source file. The current allowlist covers all pre-existing source files but missed indiv_reports.py.</research>
 <approaches>
 <approach label="Single bundled release with gitignore fix" feasibility="high" risk="medium">
 <description>(1) Add `!src/boost_shap_gii/indiv_reports.py` to.gitignore allowlist after line 24 (alphabetical with existing source list). (2) Run /implement plan + build for all locked findings (topics 1-23 with code/docs targets). (3) Run /test. (4) Run /publish for a single bundled release covering pre-existing uncommitted work plus all this-cycle remediation.</description>
 <pros>Cleanest narrative; the indiv_reports module ships only after its critical findings are remediated.</pros>
 <cons>Larger change set in a single release; mitigated by the /test phase between /implement and /publish.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Locked. Implementation directives: (1) at.gitignore, insert `!src/boost_shap_gii/indiv_reports.py` after line 24 (alphabetical with the existing source allowlist); (2) audit other current source files for missing allowlist entries during /implement/build; (3) sequence: /implement plan → /implement build → /test → review → /publish, single bundled release. /implement target.</decision>
 </topic>

 </topics>

 <action_items>
 <item priority="P0" target_mode="implement" description=" (++N#1+N#2): Unified tensor-shape fix in indiv_reports.py — add _output_dim helper, expand point_shap_int to (N,C,F,F), int_iter_folds to (K,N,C,F,F), update memory guard to 4 × n_outputs bytes, audit downstream consumers (parquet writers, plot.R microdata) for class-axis handling." />
 <item priority="P0" target_mode="implement" description="Topic 2: Bootstrap-of-CV with basic/reverse-percentile intervals in infer.py; add training/inference CI-scale asymmetry paragraph to INPUT_SPECIFICATION.md." />
 <item priority="P0" target_mode="implement" description="Topic 3: Pooled BH-FDR across all effects at shap_utils.py:905 and:908; preserve three independent FDR calls (sig_M, sig_V, sig_GII)." />
 <item priority="P0" target_mode="implement" description=" (docs portion): Insert decision-theoretic GII framing (Cobb-Douglas geometric mean of M and V; Hill 1910; Goldstein et al. 2015) into INPUT_SPECIFICATION.md, README.md, and shap_utils.py GII function docstrings. NO simulation/calibration references in any public-repo file." />
 <item priority="P1" target_mode="implement" description="Topic 5: Change np.std(signal) to np.std(signal, ddof=1) at shap_utils.py lines 223, 229, 251, 264, 303, 360; add len(signal)&lt;2 → np.nan guard before each call." />
 <item priority="P1" target_mode="implement" description="Topic 9: Add Groeneveld &amp; Meeden (1984) and Joanes &amp; Gill (1998) citations to _diagnose_outcome_distribution docstring at train.py near lines 252-255 (terse, single sentence per citation)." />
 <item priority="P1" target_mode="implement" description="Topic 10: Port Python adaptive-knot logic from shap_utils.py:146-164 to plot.R; replace lm(bs) at calc_v_spline_pred (plot.R:118-148) with splines::splineDesign-based LSQ fit." />
 <item priority="P1" target_mode="implement" description="Topic 12: Set shadow CatBoost iterations = 2 * tuned_iters as fixed ceiling at train.py:910-950; remove eval_set=pool_val_full from phase-2 shadow fit at lines 947-948." />
 <item priority="P1" target_mode="implement" description="Topic 13: Remove `df_m$shap_value &lt;- (df_m$shap_value OUTCOME_MAX) * 100` at plot.R:385 for the multi_regression code path; add brief INPUT_SPECIFICATION.md note that multi_regression SHAP is in z-scaled units." />
 <item priority="P1" target_mode="implement" description="Topic 14: Add brief CatBoost determinism caveat note to INPUT_SPECIFICATION.md (one or two sentences); no config flag, no code change." />
 <item priority="P2" target_mode="implement" description="Topic 15: Replace utils.py:665-666 fallback with `return (base_score, np.nan, np.nan)` plus explicit warning identifying the affected effect." />
 <item priority="P2" target_mode="implement" description="Topic 17: Replace plot.R:481-485 frequency-only top-5 selection with V-contribution-ranked selection: count_k * (mean_SHAP_k - grand_mean_SHAP)^2; nominal-only; annotate N_k below each level. Ordinal features unaffected." />
 <item priority="P2" target_mode="implement" description="Topic 18: Add `_label_nominal` helper distinguishing NaN→__NA__ from unseen→__UNSEEN__ in predict.py and infer.py; mirror ordinal tier-1 (>50% unique unknown → ValueError) and tier-2 (>10% obs unknown → warning) at predict.py:137 and infer.py:156." />
 <item priority="P2" target_mode="implement" description="Topic 19: Add comment `# ONLY affects plot.R rendering` to example_config_advanced.yaml under plot.negate_shap." />
 <item priority="P2" target_mode="implement" description="Topic 20: Add config-load ValueError in utils.py if discrete_threshold &lt; n_knots + degree + 2; add comment to example_config_advanced.yaml citing Wood (2017) GAM ch. 4." />
 <item priority="P2" target_mode="implement" description="Topic 21: Expand comments at shap_utils.py:178-180 and:209-211 framing 0.1% as empirical balance margin; anchor to Higham (2002) ch. 1." />
 <item priority="P0" target_mode="implement" description=" operational: Insert `!src/boost_shap_gii/indiv_reports.py` to.gitignore after line 24 (alphabetical); audit other source files for missing allowlist entries during /implement/build." />
 <item priority="P1" target_mode="test" description="Post-/implement: run /test to verify all changes (especially + tensor shape, inference CI, R-Python spline parity, phase-2 shadow leakage) pass with the existing 461-test suite plus any new regression tests for this cycle's fixes." />
 <item priority="P1" target_mode="publish" description="After /test passes: single bundled /publish release covering pre-existing uncommitted work (fillna patch, indiv_reports module, AID_LOG refresh, INPUT_SPECIFICATION Section 9, Huber guidance) plus this brainstorm's remediation." />
 <item priority="P2" target_mode="brainstorm" description="Next-session brainstorm: Design unified + quarantined simulation study — synthetic data construction, GII calibration recovery metrics (Kendall's tau, AUROC for is_relevant), end-to-end determinism drift tolerance and sample sizes. Sub-project lives outside the GitHub working tree." />
 </action_items>

 <next_steps>Run /implement to execute the plan and build phases against the locked directives in this report. The /implement plan phase will translate each topic's decision directive into a concrete implementation tech-spec; the build phase will execute the code/docs changes. After /implement build completes, run /test to verify the changes against the existing 461-test suite plus any new regression tests for this cycle. After /test passes, run /publish for a single bundled release. The unified + quarantined simulation study is deferred to a separate next-session brainstorm.</next_steps>
</brainstorm_report>
