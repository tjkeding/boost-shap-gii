<brainstorm_report>
 <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-04-23T19:19:51Z" />
 <context_files>
 <file path="src/boost_shap_gii/infer.py" relevance="Target module for per-individual SHAP report extension; current output structure and run_shap_pipeline invocation examined." />
 <file path="src/boost_shap_gii/shap_utils.py" relevance="Existing population-level bootstrap logic (_run_bootstrap_pipeline), microdata emission (_process_and_save_microdata), SHAP computation primitives (_run_shap_for_slice), orchestration entry point (run_shap_pipeline)." />
 <file path="src/boost_shap_gii/scripts/plot.R" relevance="Existing plot generator with NEGATE_SHAP sign-flip and Y_AXIS_LABEL CLI args; PLOT_DIR convention at shap_analysis/plots/; microdata_GII.parquet consumer at line 279." />
 <file path="src/boost_shap_gii/predict.py" relevance="Training-time evaluation; co-owner of symmetric per-individual CI generation alongside infer.py." />
 <file path="src/boost_shap_gii/check_env.py" relevance="R package preflight infrastructure; to be extended with ggplot2 stack and promoted to mandatory-gate on every CLI entry point." />
 <file path="example_config_advanced.yaml" relevance="Target for new config keys (shap.indiv_ci_nboot, shap.indiv_scaling_mode, shap.indiv_scaling_value, shap.compute_global_on_inference, and the migrated plot.* parameters)." />
 </context_files>

 <topics>

 <topic id="topic" title="SHAP variant selection — path-dependent vs interventional">
 <summary>Whether to offer interventional (marginal) SHAP as an alternative to the pipeline's current path-dependent (observational, tree_path_dependent) computation, motivated by possible causal-language benefits for outcome-prediction use cases.</summary>
 <research>
Chen, Covert, Lundberg &amp; Lee (2022, JMLR), "Algorithms to Estimate Shapley Value Feature Attributions," formalizes the "true to the model" (interventional, marginal) vs "true to the data" (path-dependent, observational) distinction as a choice of subset-conditioning scheme, not a causal-identification decision. Interventional SHAP breaks feature correlations by sampling from the marginal distribution when conditioning on a subset; path-dependent SHAP respects the joint training distribution by following tree paths. Both are descriptive attributions of model behavior; neither establishes causal relationships between features and the outcome without a causal-identification framework (e.g., backdoor criterion satisfied via a DAG). Ghassemi, Oakden-Rayner &amp; Beam (2021, Lancet Digital Health) caution specifically against treating SHAP values as evidence of causal effect or as a basis for use-case-specific use-case decisions absent an experimental validation step. Lundberg et al.'s own work on TreeSHAP notes that path-dependent attributions are natively supported by CatBoost's ShapInteractionValues API, whereas interventional attributions require shap.TreeExplainer with a separate background dataset and produce different attribution structure for correlated features.
 </research>
 <approaches>
 <approach id="topic" label="Stay with path-dependent (CatBoost native)" feasibility="high" risk="low">
 <description>Retain current path-dependent SHAP as the sole computation path.</description>
 <pros>On-manifold fidelity for correlated features (critical for use-case-specific composites/scale items that are functionally linked); no background-dataset choice required; respects functional dependencies among features; consistent with the paradigm under which CatBoost itself was trained; existing GII/M/V bootstrap/spline/significance infrastructure is already validated on this output structure; CatBoost native API.</pros>
 <cons>Off-manifold perturbation questions (what would the prediction be if feature X were changed independently?) are not answerable by this attribution; some ML-methodology reviewers prefer interventional as the default for feature-importance claims.</cons>
 <statistical_considerations>Path-dependent attributions for correlated features split credit across the correlated set; in datasets with many scale items, this produces small per-item attributions that sum to a meaningful group-level signal. Interventional attributions would concentrate credit on individual items in ways that reflect model-specific extrapolation, potentially misleading in the presence of strong feature correlation.</statistical_considerations>
 </approach>
 <approach id="topic" label="Add interventional SHAP as config option" feasibility="medium" risk="medium">
 <description>Introduce a config flag selecting between path-dependent and interventional computation.</description>
 <pros>Flexibility across use cases; aligns with some reviewers' methodological preferences.</pros>
 <cons>Interventional SHAP via shap.TreeExplainer does not natively produce interaction values in the same structure as CatBoost's ShapInteractionValues; the entire GII decomposition (singleton Phi[i,i] vs interaction Phi[i,j]+Phi[j,i]) would require re-derivation and re-validation; ripples through the bootstrap, spline-fitting, Boruta-shadow, and significance machinery; doubled test surface; does not license causal claims regardless.</cons>
 <statistical_considerations>Does NOT materially change causal interpretability. "Interventional" in the SHAP sense refers to the subset-conditioning scheme, not to interventions on the data-generating process. Causal claims still require a separate identification framework (DAG + backdoor criterion) or an experimental design.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="topic">Path-dependent SHAP retained as the sole computation path. CatBoost's robustness to correlated features during training is the reason the user selected CatBoost initially; path-dependent SHAP preserves this robustness downstream in attribution. Switching to interventional would require reconstruction of the entire GII framework without a corresponding causal-identification benefit. Causal interpretation is deferred to the planned multi-arm experimental study (control baseline-intervention model-informed-intervention), where the model-informed arm is the experimental intervention that licenses causal language.</decision>
 </topic>

 <topic id="topic" title="Framing posture for the per-individual report feature">
 <summary>Choice among three framings for how the feature is documented and described: (a) pure model-explanation, (b) individual-case inspection tool, (c) inspection tool.</summary>
 <research>
Ghassemi, Oakden-Rayner &amp; Beam (2021) explicitly argue SHAP should not be positioned as evidence of causal effect mechanism in use-case-specific ML. Rudin (2019, Nature Machine Intelligence) argues more broadly that post-hoc explanations should not be conflated with interpretable models for high-stakes use-case-specific decisions. Covert &amp; Lee (2021) raise the "false precision" concern: deterministic SHAP point estimates without uncertainty quantification encourage overconfident per-individual claims. The hypothesis-generating framing is established in genomics (e.g., GWAS-to-followup workflows) where ML attributions flag candidate mechanisms for downstream experimental validation without claiming causality.
 </research>
 <approaches>
 <approach id="topic" label="(a) Pure model-explanation" feasibility="high" risk="low">
 <description>Present the feature as a diagnostic tool for understanding model behavior only; no use-case-specific-decision language.</description>
 <pros>Maximally defensible under peer review; aligns with Rudin's position; zero causal-inference risk.</pros>
 <cons>Undersells the user's actual applied-domain intent; inconsistent with the user's stated experimental study program.</cons>
 <statistical_considerations>None beyond standard SHAP validity.</statistical_considerations>
 </approach>
 <approach id="topic" label="(b) Hypothesis-generating individual-case inspection tool" feasibility="high" risk="low">
 <description>Present the feature as surfacing candidate predictor features for a user to weigh alongside use-case-specific judgment, explicitly non-causal. Causal interpretation deferred to downstream experimental study.</description>
 <pros>Matches the user's experimental study plan (experimental identification in the model-informed-intervention arm); aligns with hypothesis-generating framing from genomics/ML-for-applied-domains literature; defensible under peer review.</pros>
 <cons>Requires explicit documentation language and user-facing disclaimers to prevent misinterpretation.</cons>
 <statistical_considerations>Per-individual CIs with honest labeling ("bootstrap CI under training-sample uncertainty, does not reflect distribution shift or model-class uncertainty") are essential to avoid Covert &amp; Lee's false-precision trap.</statistical_considerations>
 </approach>
 <approach id="topic" label="(c) inspection tool" feasibility="low" risk="high">
 <description>Present the feature as directly licensing use-case-specific modifications based on per-individual SHAP profiles.</description>
 <pros>Strong translational narrative.</pros>
 <cons>Directly contradicts Ghassemi et al. 2021; not defensible without prospective experimental validation; would need to be retracted if the downstream experimental study failed to validate.</cons>
 <statistical_considerations>Causal claims require experimental identification; SHAP values are observational attributions under a fitted model.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="topic">Hypothesis-generating individual-case inspection tool framing adopted. Causal language is eased in user-facing documentation and docstrings: per-individual SHAP profiles "surface candidate predictor features for user consideration" rather than "drive use-case decisions." The multi-arm experimental study (control baseline-intervention model-informed-intervention) is the vehicle for causal claims once completed. This framing survives peer review and does not overstate what the pipeline provides.</decision>
 </topic>

 <topic id="topic" title="Outcome-scale translation for per-individual SHAP values">
 <summary>Mechanism for translating per-individual SHAP values into outcome-scale units (e.g., "% symptom change") suitable for applied interpretation.</summary>
 <research>
Standard effect-size conventions: Cohen's-d-like scaling (divide by outcome SD) is the most common normalization in behavioral and use-case-specific research; Jacobson &amp; Truax (1991) established a minimum-meaningful-difference threshold as an alternative anchor when the outcome has a known use-case-specific-significance threshold. The existing pipeline's plot.R uses a user-provided theoretical-maximum outcome value to scale GII magnitudes, which is effectively a range-based scaling equivalent to custom_value with value = outcome_max.
 </research>
 <approaches>
 <approach id="topic" label="Three-mode mutually exclusive scaling" feasibility="high" risk="low">
 <description>Config exposes shap.indiv_scaling_mode in {raw, sd, custom_value} with shap.indiv_scaling_value providing the divisor for the custom_value mode. raw emits unscaled SHAP; sd divides by SD(training outcome); custom_value divides by the user-provided value.</description>
 <pros>Minimal config surface; mirrors existing plot.R convention for GII plots (single user-provided scaling value); mutually exclusive selection avoids ambiguity; custom_value mode subsumes minimum-meaningful-difference, theoretical-max, and observed-range scalings without enumeration.</pros>
 <cons>User is responsible for selecting the appropriate scaling value for their outcome; pipeline does not validate that the custom_value makes use-case-specific sense.</cons>
 <statistical_considerations>SD is computed from TRAINING outcomes only (cached to training artifact during predict.py) so that inference-time CIs are comparable across inference runs and do not drift with inference-sample composition.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="topic">Three-mode scaling adopted. Config keys: shap.indiv_scaling_mode (string, required) and shap.indiv_scaling_value (number, used only when mode=custom_value; ignored otherwise). SD(training outcome) cached in training artifact (e.g., extend target_scaler.json) and read by infer.py from train_dir.</decision>
 </topic>

 <topic id="topic" title="Per-individual uncertainty quantification">
 <summary>How to quantify per-individual uncertainty on SHAP values and predicted outcomes, given that the current pipeline emits deterministic point estimates only.</summary>
 <research>
Covert &amp; Lee (2021) frame the "false precision" problem: deterministic SHAP point estimates invite overinterpretation. Efron &amp; Tibshirani (1993) establish bootstrap percentile CIs as the standard for nonparametric uncertainty when analytic standard errors are unavailable. Breiman (2001) introduced out-of-bag (OOB) aggregation for random-forest feature importance, providing an efficient mechanism for per-observation CIs from a shared bootstrap pool. Efron &amp; Tibshirani recommend B ≥ 1000 for stable percentile-CI tails; Carpenter &amp; Bithell (2000) report B ≥ 200 as acceptable minimum in practice. No closed-form standard error exists for TreeSHAP attributions; bootstrap is the only defensible path to inference.
 </research>
 <approaches>
 <approach id="topic" label="None — deterministic point estimate" feasibility="high" risk="high">
 <description>Report point estimates only.</description>
 <pros>Zero computational cost.</pros>
 <cons>Creates Covert &amp; Lee false-precision risk; not defensible for use-case-specific-aid framing.</cons>
 <statistical_considerations>Incompatible with framing commitments.</statistical_considerations>
 </approach>
 <approach id="topic" label="K-fold spread as stability range" feasibility="high" risk="medium">
 <description>Report the min/max of SHAP values across the K fold models as a "fold stability range," not a CI.</description>
 <pros>Zero additional compute; honest labeling avoids overstatement.</pros>
 <cons>Not a formal CI; only K samples; tail precision poor; insufficient for publication-grade per-individual uncertainty.</cons>
 <statistical_considerations>Variance of K-fold spread converges to sample variance of K fold estimates; does not extend to formal inference.</statistical_considerations>
 </approach>
 <approach id="topic" label="Bootstrap K fold models with replacement" feasibility="high" risk="medium">
 <description>Draw B bootstrap resamples of size K from existing K fold models; compute mean SHAP per resample.</description>
 <pros>Cheap extension of.</pros>
 <cons>Statistically equivalent to: bootstrap distribution variance converges to sample variance of the K fold estimates. Cosmetically a CI, informationally the fold range.</cons>
 <statistical_considerations>Rejected — does not extract new information beyond K fold models.</statistical_considerations>
 </approach>
 <approach id="topic" label="Training-set bootstrap with model retraining + OOB aggregation" feasibility="medium" risk="low">
 <description>Draw B bootstrap resamples of the training set (cluster-aware when cluster_ids present); refit the CatBoost model under each resample with hyperparameters fixed to final-tuned values; compute SHAP for all training + all inference individuals under each refit. For training individual i, aggregate only bootstrap iterations where i was OOB (per-individual OOB rate ~1/e ≈ 0.368). For inference individuals, aggregate all B iterations (always OOB by definition). Compute 2.5th/97.5th percentile CIs from the aggregated distributions. Cache B refit models to disk during predict.py; infer.py loads cached models from train_dir/bootstrap_refits/ without per-inference-run refitting.</description>
 <pros>Statistically defensible individual-level CI capturing training-sample uncertainty; OOB mechanism avoids per-individual LOO refit explosion; shared bootstrap pool means only B refits total (not B × N_train); symmetric treatment of training and inference individuals; Breiman-OOB is a recognized standard from the random-forest literature.</pros>
 <cons>Compute cost: B CatBoost refits + B SHAP computations; disk cost: B model files (~500 MB–2.5 GB at B=500). Does not capture model-class uncertainty, hyperparameter-tuning uncertainty, or distribution-shift — labeling must be explicit about scope.</cons>
 <statistical_considerations>Target of inference is E[phi_j(x*; f-hat)] over training-set re-draws from the same population, with x* held fixed. Per-individual variance is endogenously individual-specific (each x* projects through each refit independently); no homoscedasticity assumption across individuals. individuals in sparsely-sampled feature regions yield larger bootstrap variance, which is statistically correct (extrapolation instability). Assumes: (i) training sample is representative of target population; (ii) x* is drawn from the same population; (iii) x* lies near or within training support; (iv) HP held fixed is an explicit conservative choice. Per-individual OOB floor: individuals with OOB count &lt; some floor (e.g., 25) emit NaN CI bounds with oob_count preserved for diagnostics.</statistical_considerations>
 </approach>
 <approach id="topic" label="Per-individual LOO bootstrap" feasibility="low" risk="low">
 <description>Remove each training individual from the data before running B bootstrap refits specific to that individual. Guarantees no in-bag contamination.</description>
 <pros>Statistically cleanest guarantee against in-bag influence on attribution.</pros>
 <cons>Cost: N_train × B refits. For N_train = 500 and B = 100 = 50,000 CatBoost fits. Infeasible.</cons>
 <statistical_considerations>Rejected on computational grounds; OOB aggregation in provides equivalent no-in-bag guarantee at O(B) cost rather than O(N × B).</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="topic">Training-set bootstrap with model retraining + OOB aggregation adopted, bootstrap-by-default (not opt-in). Config key: shap.indiv_ci_nboot (required, no default, user MUST specify). Inline comment in config specifies minimum recommended B=500 with guidance to increase to 1000-2000 for large cohorts rare-event outcomes tail-percentile precision. HP always fixed to final-tuned values. Symmetric: both training and inference individuals receive CIs. Bootstrap cache: B refit models persisted to train_dir/bootstrap_refits/boot_{b}.cbm + bootstrap indices to train_dir/bootstrap_refits/boot_indices.npz during predict.py for reproducibility. infer.py reads the cache from the existing train_dir pointer (config.paths.output_dir); errors loudly if cache absent. Per-individual OOB count floor: individuals with OOB count &lt; floor (concrete value set at implementation time, e.g., 25) emit NaN CI bounds with oob_count column preserved.</decision>
 </topic>

 <topic id="topic" title="Plot design for per-individual reports">
 <summary>Visual design for per-individual main-effects and interactions plots, given the uncertainty decision (per-individual CIs) and scaling decision.</summary>
 <research>
Dot-plus-whisker plots are the standard format for displaying point estimates with interval uncertainty in use-case-specific and behavioral research (see Cumming &amp; Finch 2005, American Psychologist, on CI visualization). Bar plots obscure CI information when whiskers are overlaid. Diverging color palettes (dark-neutral-dark) are the recommended choice for signed continuous variables (Brewer 2003, ColorBrewer). The existing plot.R uses a blue-positive red-negative convention for SHAP, diverging from the SHAP library default (red-positive blue-negative); retaining the pipeline-specific convention preserves within-pipeline visual consistency.
 </research>
 <approaches>
 <approach id="topic" label="User-defined feature groupings (grouped x-axis)" feasibility="medium" risk="medium">
 <description>Config-specified feature groupings produce a grouped x-axis (facet by group, within-group ordering by |SHAP|).</description>
 <pros>meaningful organization by scale/domain.</pros>
 <cons>User must know sig_GII features ahead of time to enumerate them correctly (circular — sig_GII is computed by the pipeline); large config surface; tedious authoring burden; post-hoc group assignment often arbitrary.</cons>
 <statistical_considerations>None.</statistical_considerations>
 </approach>
 <approach id="topic" label="Signed-rank dot-plus-whisker (no grouping)" feasibility="high" risk="low">
 <description>Dot-plus-whisker plot; x-axis ordered by this individual's raw signed SHAP value (most positive left, most negative right); y-axis in -scaled units; horizontal reference line at y=0; diverging color gradient (dark blue = most positive raw SHAP, white = near zero, dark red = most negative raw SHAP).</description>
 <pros>Monotonic signed ordering makes three readings immediately visible: features pushing prediction up (left, blue), features indistinguishable from null (center, faint, CIs crossing zero), features pushing prediction down (right, red). Zero config burden for users. Generalizes cleanly to interactions plot.</pros>
 <cons>Cross-individual visual comparison harder (each individual's plot has a different x-axis ordering), but this is appropriate because the plot is per-individual.</cons>
 <statistical_considerations>CI whiskers crossing y=0 is the visual indicator that a feature's contribution is not distinguishable from null for that individual; anchoring to signed SHAP (not |SHAP|) preserves directional information in the x-axis ordering.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="topic">Signed-rank dot-plus-whisker adopted. Same design for main-effects and interactions plots (both filtered schema: main-effects plot to sig_GII=True subset of the main-effects table, interactions plot to the full sig_GII=True-filtered interactions table). Color palette: blue-positive red-negative diverging, anchored to RAW signed SHAP. Sign-flip behavior (shared with existing GII plots via plot.negate_shap): y-axis values flip; color gradient and x-axis rank ordering do NOT flip (they remain anchored to raw signed SHAP). This yields a coherent visual where color+position express model-internal direction and y-axis expresses the user-semantic direction.</decision>
 </topic>

 <topic id="topic" title="Output scope and file organization">
 <summary>Which individuals receive reports; directory structure; plot-generator placement.</summary>
 <approaches>
 <approach id="topic" label="All individuals, aggregated parquets + per-individual PNGs, plot.R as generator" feasibility="high" risk="low">
 <description>All inference individuals (and all training individuals, symmetric decision) receive reports. Top-level directory indiv_reports/ (parallel to shap_analysis/) in BOTH train_dir and infer_dir. Aggregated long-format parquets (main_effects, interactions, predictions) + nested plots/ subdirectory with per-individual PNGs. Plot generator is the existing plot.R, extended to auto-discover indiv_reports/*.parquet and emit per-individual plots alongside the existing GII plots. Same CLI entry point (boost-shap-gii plot); no new subcommand.</description>
 <pros>Matches user's original feature request literally; zero scope arbitration; aggregated parquets form a tidy canonical data store; per-individual PNGs are separable from data; plot.R as sole plot generator centralizes visualization logic and preserves pipeline's R-for-plotting convention; no new CLI subcommand.</pros>
 <cons>At very large inference N, per-individual PNG emission adds filesystem overhead; deferred as future optimization if needed.</cons>
 <statistical_considerations>N/A.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="topic">Directory collision check completed: existing plots/ directory is nested inside shap_analysis/ (plot.R line 183), so the new indiv_reports/plots/ at a distinct parent is structurally non-colliding. Final structure for both train_dir and infer_dir: {root}/indiv_reports/{main_effects.parquet, interactions.parquet, predictions.parquet, indiv_reports_metadata.json, plots/{id}_main_effects.png, plots/{id}_interactions.png}.</decision>
 </topic>

 <topic id="topic" title="Architectural placement and infer.py global-SHAP behavior">
 <summary>(a) Where the per-individual bootstrap + OOB-aggregation + scaling + emission logic lives in the Python codebase; (b) whether infer.py continues to emit population-level shap_analysis/ on inference data.</summary>
 <approaches>
 <approach id="topic" label="Extend infer.py inline" feasibility="high" risk="medium">
 <description>Add ~200 LOC to existing infer.py for the per-individual logic.</description>
 <pros>Fits existing control flow.</pros>
 <cons>Breaks symmetric-training decision from Topic 4: predict.py also needs to emit training-individual CIs, so inline extension of infer.py alone fails the scope. Logic duplication if both predict.py and infer.py extended inline.</cons>
 <statistical_considerations>N/A.</statistical_considerations>
 </approach>
 <approach id="topic" label="New module indiv_reports.py" feasibility="high" risk="low">
 <description>New module src/boost_shap_gii/indiv_reports.py encapsulates bootstrap-refit loop, OOB aggregation, CI percentile computation, scaling, parquet emission. Called from both predict.py (training-individual CIs, OOB-aggregated) and infer.py (inference-individual CIs, all iterations). Calls into existing shap_utils.py primitives (e.g., _run_shap_for_slice) and existing train.py model-fitting code; no duplication.</description>
 <pros>Single-responsibility module; distinct bootstrap mechanism (model-refit) from shap_utils.py's row-resample bootstrap; cleaner isolation for unit testing; avoids further bloat of shap_utils.py (currently 1230 lines); called symmetrically from both predict.py and infer.py.</pros>
 <cons>One additional module in src/.</cons>
 <statistical_considerations>N/A.</statistical_considerations>
 </approach>
 <approach id="topic" label="Extend shap_utils.py" feasibility="medium" risk="medium">
 <description>Add new functions alongside _run_bootstrap_pipeline in shap_utils.py.</description>
 <pros>Keeps all SHAP-related code in one file.</pros>
 <cons>shap_utils.py already large; new logic is semantically distinct (per-individual vs population-level) and mechanistically distinct (model-refit vs matrix-row-resample); would add ~300-500 LOC to an already-stressed module.</cons>
 <statistical_considerations>N/A.</statistical_considerations>
 </approach>
 <approach id="topic" label="infer.py shap_analysis: remove entirely" feasibility="high" risk="low">
 <description>Remove the run_shap_pipeline call from infer.py. No shap_analysis/ directory emitted on inference.</description>
 <pros>Simplest; avoids degenerate statistics at small inference N; eliminates redundancy for users whose inference use case is use-case-specific rather than external-validation.</pros>
 <cons>Eliminates the external-validation distribution-shift diagnostic option for users running inference on large external cohorts.</cons>
 <statistical_considerations>GII is a function of (model, data). Training-time GII is the canonical model-characterization. Inference-time GII is a distinct quantity describing model behavior on new data — potentially diagnostic of distribution shift at large N, statistically meaningless at small N.</statistical_considerations>
 </approach>
 <approach id="topic" label="infer.py shap_analysis: config-gated, default off" feasibility="high" risk="low">
 <description>Add config key shap.compute_global_on_inference defaulted to false. When false, infer.py skips run_shap_pipeline; when true, current behavior preserved.</description>
 <pros>Preserves distribution-shift diagnostic option for users who want it; default-off protects small-N inference users from degenerate outputs; minimal config surface (one boolean).</pros>
 <cons>One additional config key.</cons>
 <statistical_considerations>User opt-in is the right gate: a user running large-N external validation can set true; default users (including small-N use-case-specific workflows) get the appropriate minimal output.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="topic + topic">New module src/boost_shap_gii/indiv_reports.py created, called from both predict.py and infer.py. shap_analysis/ emission on inference is config-gated via new shap.compute_global_on_inference key (default false); default inference runs emit only predictions_ensemble.csv + performance files + indiv_reports/ + inference_metadata.json. Users running large-N external validation may opt in by setting the config key to true.</decision>
 </topic>

 <topic id="topic" title="Table schemas for indiv_reports/">
 <summary>Column specifications for main_effects.parquet, interactions.parquet, predictions.parquet, and the accompanying metadata JSON.</summary>
 <approaches>
 <approach id="topic" label="Three-file long-format schema + metadata JSON" feasibility="high" risk="low">
 <description>
main_effects.parquet columns (one row per individual × feature; ALL features, unfiltered):
 id (str), feature (str), feature_value_raw (str), feature_type (str: nominal/ordinal/continuous), shap_value_raw (float), shap_value_scaled (float), shap_value_ci_lo (float), shap_value_ci_hi (float), oob_count (int), sig_GII (bool).

interactions.parquet columns (one row per individual × interaction; HARD-FILTERED to sig_GII=True):
 id (str), feature_a (str), feature_b (str), feature_a_value_raw (str), feature_b_value_raw (str), feature_a_type (str), feature_b_type (str), shap_value_raw (float), shap_value_scaled (float), shap_value_ci_lo (float), shap_value_ci_hi (float), oob_count (int).

predictions.parquet columns (one row per individual):
 id (str), y_pred_raw (float), y_pred_scaled (float), y_pred_ci_lo (float), y_pred_ci_hi (float), y_pred_oob_count (int), y_true (float, NaN when outcome not present in inference data).

indiv_reports_metadata.json:
 {scaling_mode, scaling_divisor, n_boot, oob_count_floor, outcome_name, negate_shap, timestamp}.

Additional provenance:
 - SD(training outcome) cached during predict.py (extend target_scaler.json or a new training artifact) so infer.py reads the scaling divisor from train_dir rather than recomputing on inference data.
 - sig_GII values for each feature broadcast from shap_analysis/shap_stats_global.csv (training-time output) rather than recomputed.
 </description>
 <pros>Long format supports arbitrary downstream analysis without schema migration; metadata JSON avoids redundant constants per row; sig_GII inheritance from training-time analysis preserves the global-significance concept appropriately (sig_GII is a property of the model, not of individual individuals); raw + scaled columns provide both model-output and user-semantic values for downstream flexibility.</pros>
 <cons>NaN propagation for individuals below OOB floor must be handled consistently by consumers.</cons>
 <statistical_considerations>oob_count exposure per row enables consumers to audit CI reliability; NaN CI bounds for below-floor individuals prevent misleading narrow intervals.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="topic">Three-file long-format schema + metadata JSON adopted. sig_M and sig_V not included in the per-individual table (sig_GII alone conveys both-significant status discussion). SD(training outcome) cached to training artifact during predict.py; inferred by infer.py from train_dir.</decision>
 </topic>

 <topic id="topic" title="Cross-cutting: plot.R configuration migration">
 <summary>All plot.R parameters (except CONFIG_PATH and optional RUN_DIR) migrated from positional CLI args to YAML config keys.</summary>
 <approaches>
 <approach id="topic" label="Full config migration" feasibility="high" risk="low">
 <description>
CLI args retained: args[1]=CONFIG_PATH (required), args[2]=RUN_DIR (optional override for config.paths.output_dir).

Migrated to config under plot:
 plot.outcome_max (was OUTCOME_RANGE; renamed outcome_range → outcome_max per user directive).
 plot.negate_shap (was NEGATE_SHAP).
 plot.gii_y_label (was Y_AXIS_LABEL; explicitly scoped to GII plots).
 plot.gii_y_sublabel (NEW).
 plot.indiv_y_label (NEW).
 plot.indiv_y_sublabel (NEW).

Behavior changes:
 - No plot titles are emitted anywhere (neither GII plots nor indiv plots).
 - No default values in R source; if any required plot.* key is missing from config, plot.R errors loudly with the missing key name.
 - Label text is rendered verbatim from config; no programmatic composition (no auto-appended CI annotations, no individual-ID substitution in titles, etc.).
 </description>
 <pros>All plot configuration in one authoritative location (the config YAML); R does not parse user-supplied strings from the CLI; parameters reusable across invocations; explicit fail-loud behavior on missing keys surfaces misconfiguration immediately.</pros>
 <cons>Users must populate six plot.* keys in the config (or omit the plot step entirely).</cons>
 <statistical_considerations>N/A.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="topic">Full config migration adopted. CLI retains CONFIG_PATH (required) + RUN_DIR (optional). All six plot.* keys become required when the plot subcommand is invoked (fail-loud if missing). No defaults, no programmatic title/label composition.</decision>
 </topic>

 <topic id="topic" title="Cross-cutting: R package preflight promoted to mandatory gate">
 <summary>R package validation elevated from optional check-env utility to mandatory preflight on every CLI entry point, reflecting plot.R's new central role in the pipeline.</summary>
 <approaches>
 <approach id="topic" label="Preflight on every entry point" feasibility="high" risk="low">
 <description>check_env.py extended with all required R packages (current: nanoparquet; add: ggplot2 + any additional dependencies used by the extended plot.R — scales, dplyr, patchwork or gridExtra as applicable). The check_env function is called as a preflight step at the beginning of every CLI entry point (train, predict, infer, plot, check-env). If any R package is missing, the CLI fails fast with a clear error listing each missing package and its install command.</description>
 <pros>R packages now central to the pipeline (plot.R is the sole plot generator); preflight prevents failures deep in an inference pipeline from propagating to unhelpful error messages at plot time; fail-fast surface is short and explicit.</pros>
 <cons>Small startup cost (~1 second) on every CLI invocation; acceptable given the alternative of mid-pipeline R-package failures.</cons>
 <statistical_considerations>N/A.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="topic">R package preflight promoted to mandatory gate on every CLI entry point. check_env.py extended with required packages (concrete list finalized at implementation time based on plot.R imports).</decision>
 </topic>

 </topics>

 <action_items>
 <item priority="P1" target_mode="implement" description="Create new module src/boost_shap_gii/indiv_reports.py containing: (a) bootstrap-refit loop resampling the training set with replacement (cluster-aware when cluster_ids present), (b) per-iteration CatBoost refit with HP fixed to final-tuned values, (c) SHAP evaluation under each refit for all training + inference individuals, (d) OOB aggregation (OOB-only for training individuals, all-B for inference individuals), (e) 2.5/97.5 percentile CI computation on scaled values with NaN emission below the per-individual OOB floor, (f) scaling application (raw/sd/custom_value), (g) emission of main_effects.parquet, interactions.parquet, predictions.parquet, and indiv_reports_metadata.json to {run_dir}/indiv_reports/." />
 <item priority="P1" target_mode="implement" description="Extend predict.py to (a) orchestrate the bootstrap-refit run after training model evaluation completes, (b) persist B refit models to train_dir/bootstrap_refits/boot_{b}.cbm + bootstrap row indices to train_dir/bootstrap_refits/boot_indices.npz for reproducibility, (c) call indiv_reports module for training-individual CI emission, (d) cache SD(training outcome) to target_scaler.json (or a new training artifact) for downstream infer.py consumption." />
 <item priority="P1" target_mode="implement" description="Extend infer.py to (a) load cached bootstrap refit models from train_dir/bootstrap_refits/, (b) compute SHAP at inference individuals under each cached refit, (c) call indiv_reports module for inference-individual CI emission, (d) config-gate the existing run_shap_pipeline call with new shap.compute_global_on_inference key (default false), (e) error loudly if bootstrap cache is absent when indiv_ci_nboot is set in config." />
 <item priority="P1" target_mode="implement" description="Add new config keys to example_config_advanced.yaml and example_config_minimal.yaml: shap.indiv_ci_nboot (required, no default, inline comment specifying minimum recommended 500 with guidance for 1000-2000 on large cohorts rare-event outcomes tail-percentile precision), shap.indiv_scaling_mode (required string in {raw, sd, custom_value}), shap.indiv_scaling_value (number used only when scaling_mode=custom_value), shap.compute_global_on_inference (bool, default false), plot.outcome_max (renamed from current outcome_range), plot.negate_shap, plot.gii_y_label, plot.gii_y_sublabel, plot.indiv_y_label, plot.indiv_y_sublabel. Update utils.py config-defaulting logic where applicable." />
 <item priority="P1" target_mode="implement" description="Migrate plot.R from positional CLI args to config-driven parameters. Retain only args[1]=CONFIG_PATH and optional args[2]=RUN_DIR. All other parameters read from config.plot.*. Remove plot title emission throughout. Add fail-loud behavior for missing required plot.* config keys. Extend plot.R with a new section that auto-discovers indiv_reports/*.parquet files and emits per-individual main-effects and interactions plots (dot-plus-whisker, signed-rank x-ordering, diverging blue-positive/red-negative color anchored to raw SHAP, y-axis in -scaled units, y-axis values flip when plot.negate_shap=true with color and ordering NOT flipped)." />
 <item priority="P1" target_mode="implement" description="Extend check_env.py with the full R-package dependency list required by plot.R (current: nanoparquet; add: ggplot2 and any additional packages used by the extended plot.R). Promote check_env to a mandatory preflight on every CLI entry point (train, predict, infer, plot, check-env). Fail fast on missing packages with clear install instructions." />
 <item priority="P1" target_mode="test" description="Design test suite for indiv_reports.py covering: (a) OOB aggregation correctness (training individuals use OOB iterations only; inference individuals use all iterations), (b) per-individual OOB floor NaN emission behavior, (c) scaling-mode correctness for raw/sd/custom_value, (d) reproducibility given cached bootstrap indices, (e) cluster-aware bootstrap structure preservation, (f) parquet schema conformance to the specification, (g) metadata JSON content correctness, (h) SD(training) caching and inference-time read-back." />
 <item priority="P2" target_mode="document" description="Update INPUT_SPECIFICATION.md with new config keys and their semantics. Update README.md with a brief section on the indiv_reports feature positioned under the hypothesis-generating inspection framing (no causal-intervention language). Update AID_LOG.md if the feature materially changes the pipeline workflow." />
 </action_items>

 <next_steps>
Recommended sequencing: the user has a pending /publish for the existing Categorical-fillna patch + Sessions 5/6 uncommitted work. The indiv_reports feature designed in this brainstorm is a substantial new capability (~800-1200 LOC across new module + extensions to predict.py, infer.py, plot.R, check_env.py, and config schemas) that warrants its own implement + test + publish cycle. Suggested sequencing: (1) /publish the current patch + Session 5/6 work now (already tested, ready), (2) /implement plan for this brainstorm as a new work unit, (3) /test, (4) /publish the indiv_reports feature as a distinct release. Alternative: bundle all into a single implement + publish cycle if the user prefers a single release vehicle. User decision required before proceeding.
 </next_steps>
</brainstorm_report>
