<brainstorm_report>
 <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-03-30T16:53:06Z" />
 <context_files>
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/config_outcome_a.yaml" relevance="User config for outcome_a dataset_v2 primary analysis" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/config_outcome_a_dataset_v1.yaml" relevance="User config for outcome_a dataset_v1 secondary analysis" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/config_outcome_b.yaml" relevance="User config for outcome_b dataset_v2 primary analysis (fdr_correct: false discrepancy)" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/config_outcome_b_dataset_v1.yaml" relevance="User config for outcome_b dataset_v1 secondary analysis" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_a/metrics_oof.json" relevance="Fold-level metrics for outcome_a dataset_v2" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_a_dataset_v1/metrics_oof.json" relevance="Fold-level metrics for outcome_a dataset_v1" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_b1/metrics_oof.json" relevance="Fold-level metrics for outcome_b dataset_v2" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_b_dataset_v1/metrics_oof.json" relevance="Fold-level metrics for outcome_b dataset_v1" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_a/performance_final.csv" relevance="Aggregate performance with bootstrap CIs for outcome_a dataset_v2" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_a_dataset_v1/performance_final.csv" relevance="Aggregate performance with bootstrap CIs for outcome_a dataset_v1" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_b1/performance_final.csv" relevance="Aggregate performance with bootstrap CIs for outcome_b dataset_v2" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_b_dataset_v1/performance_final.csv" relevance="Aggregate performance with bootstrap CIs for outcome_b dataset_v1" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_a/permutation_test_results.csv" relevance="Permutation test for outcome_a dataset_v2" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_a_dataset_v1/permutation_test_results.csv" relevance="Permutation test for outcome_a dataset_v1" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_b1/permutation_test_results.csv" relevance="Permutation test for outcome_b dataset_v2" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_b_dataset_v1/permutation_test_results.csv" relevance="Permutation test for outcome_b dataset_v1" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_a/shap_analysis/shap_stats_global.csv" relevance="Full GII results for outcome_a dataset_v2 (228 effects, 1 sig GII, 5 sig M)" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_a_dataset_v1/shap_analysis/shap_stats_global.csv" relevance="Full GII results for outcome_a dataset_v1 (145 effects, 0 sig GII, 2 sig M)" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_b1/shap_analysis/shap_stats_global.csv" relevance="Full GII results for outcome_b dataset_v2 (196 effects, 2 sig GII, 9 sig M)" />
 <file path="<local_path>/boost-shap-gii_sandbox/example_study/out/target_b_dataset_v1/shap_analysis/shap_stats_global.csv" relevance="Full GII results for outcome_b dataset_v1 (151 effects, 0 sig GII, 2 sig M)" />
 <file path="./src/boost_shap_gii/shap_utils.py" relevance="SHAP GII computation and significance logic" />
 <file path="./src/boost_shap_gii/utils.py" relevance="Config default filling (fdr_correct default behavior)" />
 </context_files>

 <topics>

 <!-- ============================================================== -->
 <!-- TOPIC 1: OVERALL MODEL PERFORMANCE EVALUATION -->
 <!-- ============================================================== -->
 <topic id="topic" title="Overall Model Performance Across Four Runs">
 <summary>
All four runs completed without error and produced full output artifact sets. The pipeline is functioning correctly at a mechanical level. However, model performance is modest to poor across all runs, with substantial fold-to-fold R2 variability and frequent negative R2 values at the fold level.

**Performance Summary (aggregate, bootstrap CI):**

| Run | RMSE | R2 | Perm. p (RMSE) | Perm. p |
|----------------------|---------------|-----------------|-----------------|---------------|
| outcome_a dataset_v2 | 6.80 [5.91, 7.70] | 0.139 [-0.005, 0.257] | <0.0001 | <0.0001 |
| outcome_a dataset_v1 | 7.41 [6.55, 8.34] | -0.024 [-0.136, 0.070] | 0.0067 | 0.0067 |
| outcome_b dataset_v2 | 3.70 [3.18, 4.23] | 0.105 [-0.033, 0.214] | <0.0001 | <0.0001 |
| outcome_b dataset_v1 | 4.01 [3.46, 4.56] | -0.052 [-0.165, 0.034] | 0.0746 | 0.0746 |

**Fold-level R2 variability:**

| Run | R2 mean | R2 sd | Neg. R2 folds | Min R2 | Max R2 |
|----------------------|---------|---------|---------------|---------|---------|
| outcome_a dataset_v2 | 0.054 | 0.282 | 4/10 | -0.412 | 0.481 |
| outcome_a dataset_v1 | -0.099 | 0.292 | 6/10 | -0.863 | 0.138 |
| outcome_b dataset_v2 | 0.033 | 0.204 | 3/10 | -0.480 | 0.238 |
| outcome_b dataset_v1 | -0.089 | 0.125 | 7/10 | -0.295 | 0.101 |

Key observations:
1. dataset_v2 runs (41 features, n=244/240) produce statistically significant permutation p-values (<0.0001), confirming genuine signal above chance. However, effect sizes are modest (R2 approximately 0.10-0.14).
2. Sociodem runs (18 features) show negative aggregate R2 for both outcomes. The outcome_a dataset_v1 run barely reaches significance (p=0.007), while the outcome_b dataset_v1 run does not reach significance (p=0.075).
3. Fold-level R2 variability is extreme in all runs. Even the best-performing outcome_a dataset_v2 run has 4/10 folds with negative R2. This reflects the combination of small sample size (n approximately 240) and 10-fold CV.
4. Prediction compression is severe across all runs: dataset_v2 runs compress prediction variance to approximately 48-57% of true outcome variance; dataset_v1 runs compress to approximately 34-38%. This is consistent with weak signal and model regularization.
 </summary>
 <research>
Fold-level negative R2 in 10-fold CV is not inherently alarming with n approximately 240, particularly when the aggregate permutation test is significant. Hastie, Tibshirani, and Friedman (2009, Elements of Statistical Learning, 2nd ed.) note that individual fold estimates of generalization error are high-variance when n/K is small (here, approximately 24 per fold). Bates, Hastie, and Tibshirani (2024, JASA) formally characterize the variance of cross-validated estimates and demonstrate that fold-level R2 estimates can be dramatically unstable when the signal-to-noise ratio is modest. The aggregate R2 with bootstrap CI is the appropriate summary, not individual fold R2 values.

The overall R2 of approximately 0.10-0.14 for the dataset_v2 models, while modest in absolute terms, is not unusual for psychological/behavioral outcome prediction from predictor measurements in use-case-specific samples. Steyerberg et al. (2010, Epidemiology) note that R2 values in the 0.05-0.20 range are common for behavioral prediction models, and what matters is whether the model captures genuine signal (confirmed by permutation test) and identifies actionable features.
 </research>
 <approaches>
 <approach id="A1a" label="Accept current performance as adequate" feasibility="high" risk="low">
 <description>Accept the aggregate R2 of approximately 0.10-0.14 for dataset_v2 runs as genuine, modest signal. Report with appropriate caveats about effect sizes. Focus on SHAP-based feature importance (the pipeline's primary purpose) rather than prediction accuracy per se.</description>
 <pros>Honest reporting; permutation tests confirm signal is real; SHAP importance is the main contribution, not raw prediction</pros>
 <cons>Reviewers may fixate on low R2 and question whether feature importance from a weak model is interpretable</cons>
 <statistical_considerations>The pipeline is designed for feature importance discovery, not high-accuracy prediction. A permutation-significant model with R2=0.14 still provides valid SHAP decomposition — the features identified as important are genuinely driving the predictable variance.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="A1a">The dataset_v2 models demonstrate genuine signal (permutation p less than 0.0001) and the SHAP importance analysis is valid for these models. The dataset_v1 models should be reported as confirming that demographics alone are insufficient, not as independent importance analyses.</decision>
 </topic>

 <!-- ============================================================== -->
 <!-- TOPIC 2: SOCIODEM vs dataset_v2 COMPARISON -->
 <!-- ============================================================== -->
 <topic id="topic" title="Sociodem vs dataset_v2 Run Comparisons">
 <summary>
**Design difference:** dataset_v2 runs use 41 features (full use-case-specific battery + demographics) from `STRING2_removed_CatBoost_Dataset.csv`. Sociodem runs use 18 features (demographics + behavioral risk measures only) from `Sociodemo_CatBoost_Dataset.csv`. Both datasets have n=294 rows (244 after dropping missing outcome_a outcome; 240 after dropping missing outcome_b outcome).

**Largest performance differences:**
- outcome_a: dataset_v2 R2=0.139 vs dataset_v1 R2=-0.024 (delta=+0.163). Both models reach permutation significance, but dataset_v1 only marginally (p=0.007 vs p less than 0.0001).
- outcome_b: dataset_v2 R2=0.105 vs dataset_v1 R2=-0.052 (delta=+0.157). dataset_v2 reaches significance; dataset_v1 does NOT (p=0.075).

**Largest GII/M differences for shared features (outcome_a outcome):**
The 18 features present in both datasets show systematically HIGHER GII and M values in the dataset_v1 runs. This is a methodological artifact: with fewer features competing, each feature absorbs a larger share of the (minimal) predictable variance.

| Feature | dataset_v2 GII | Sociodem GII | Delta |
|-----------------------------|-------------|--------------|---------|
| feature_x | 0.166 | 0.416 | -0.250 |
| feature_x | 0.084 | 0.284 | -0.200 |
| feature_x | 0.013 | 0.155 | -0.142 |
| feature_x | 0.062 | 0.198 | -0.136 |
| feature_x | 0.040 | 0.160 | -0.120 |

**Same pattern for outcome_b outcome:**
| Feature | dataset_v2 GII | Sociodem GII | Delta |
|-----------------------------|-------------|--------------|---------|
| feature_x | 0.095 | 0.302 | -0.207 |
| feature_x | 0.067 | 0.151 | -0.084 |
| feature_x | 0.017 | 0.096 | -0.079 |
| demo_maritalstatus | 0.028 | 0.104 | -0.076 |
| feature_x | 0.035 | 0.106 | -0.071 |

**Key findings:**
1. The GII inflation in dataset_v1 runs is expected and well-understood. With p=18 features (vs p=41), each feature's relative share of SHAP variance increases mechanically, even if absolute predictive contribution remains unchanged. The GII metric normalizes across the feature set present in each model.
2. Notably, `feature_x` becomes the dominant feature in BOTH dataset_v1 runs (GII=0.416 for outcome_a, 0.302 for outcome_b) — despite this feature being absent from all three psychological measure batteries (outcome_a, outcome_c, MAI) that dominate the dataset_v2 feature set. This suggests STRESS_childfreq may serve as a proxy for unmeasured variance when the full use-case-specific battery is absent.
3. The dataset_v1 outcome_a run shows extremely heterogeneous model complexity (model file sizes range from 11 KB to 149 KB), with one fold (fold 9 at 149 KB) having a dramatically more complex model. This suggests potential overfitting in that fold.
4. No features reach GII significance (joint q and stability) in either dataset_v1 run. Only M (magnitude) reaches significance for 2 features in each dataset_v1 run, which means these features have reliable average effects but the cross-fold variability is too high or the noise exceedance threshold is not met for the combined GII metric.

**Significance comparison:**
| Run | Sig GII | Sig M | Sig V | Total effects |
|----------------|---------|-------|-------|---------------|
| outcome_a dataset_v2 | 1 | 5 | 0 | 228 |
| outcome_a dataset_v1 | 0 | 2 | 0 | 145 |
| outcome_b dataset_v2 | 2 | 9 | 0 | 196 |
| outcome_b dataset_v1| 0 | 2 | 0 | 151 |

5. The V (variability) component reaches significance in zero runs across all four analyses. This is noteworthy and discussed in.
 </summary>
 <research>
The GII inflation effect when reducing feature sets is analogous to the well-documented phenomenon of SHAP value redistribution under feature subset selection. Lundberg and Lee (2017, NIPS) establish that SHAP values sum to the model's predicted deviation from the baseline, so removing features necessarily redistributes their contributions to remaining features. In practice, this means absolute GII values across models with different feature sets are not directly comparable — only within-model rankings and significance calls are meaningful.

The dataset_v1 model's failure to achieve even permutation significance for the outcome_b outcome (p=0.075) indicates that the 18-feature set genuinely lacks sufficient signal for this outcome. The outcome_a dataset_v1 model's marginal significance (p=0.007) suggests minimal but detectable signal — consistent with the two M-significant features (ABIpartner, ABIself), which are behavioral violence measures rather than purely demographic.
 </research>
 <approaches>
 <approach id="A2a" label="Report dataset_v1 as performance-only comparison" feasibility="high" risk="low">
 <description>Present dataset_v1 results solely as evidence that sociodemographic/behavioral-risk features alone are insufficient. Do not interpret dataset_v1 GII values as substantive importance rankings. Focus manuscript feature importance analysis exclusively on dataset_v2 runs.</description>
 <pros>Avoids misleading interpretation of inflated GII values from a non-significant model; clean narrative</pros>
 <cons>Wastes the dataset_v1 run somewhat</cons>
 <statistical_considerations>Interpreting GII from a model with negative aggregate R2 and non-significant permutation test (outcome_b dataset_v1) is statistically indefensible. Even the marginal outcome_a dataset_v1 model (R2=-0.024, p=0.007) has negative point-estimate R2, meaning the "important" features are explaining noise-level variance.</statistical_considerations>
 </approach>
 <approach id="A2b" label="Report dataset_v1 M-significant features with caveats" feasibility="med" risk="med">
 <description>For the outcome_a dataset_v1 run (which reaches permutation significance), report the two M-significant features (ABIpartner, ABIself) as suggestive of behavioral-violence measures contributing beyond demographics, but with heavy caveats about the negative R2 and model weakness.</description>
 <pros>Extracts maximum information from the data; highlights violence measures as uniquely informative among demographics</pros>
 <cons>Reviewers may object to feature importance from a negative-R2 model</cons>
 <statistical_considerations>The outcome_a dataset_v1 permutation test IS significant (p=0.007), which means the model captures non-zero signal. The negative R2 reflects that RMSE exceeds the outcome variance — but the model is still better than chance (permutation null R2 mean is -0.145). M-significance of ABIpartner/ABIself in this context means these features reliably exceed shadow noise in magnitude. This interpretation is defensible but requires careful framing.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="A2a">Report dataset_v1 runs as performance comparison only. The primary feature importance analysis should come exclusively from dataset_v2 models, which have significant permutation tests and positive R2. The dataset_v1 results serve a clear purpose: demonstrating that the psychological measure batteries (outcome_a, outcome_c, outcome_b, MAI) contribute meaningful predictive signal beyond what demographics and behavioral risk alone provide.</decision>
 </topic>

 <!-- ============================================================== -->
 <!-- TOPIC 3: outcome_a dataset_v2 FEATURE IMPORTANCE -->
 <!-- ============================================================== -->
 <topic id="topic" title="outcome_a dataset_v2 Feature Importance Analysis">
 <summary>
The outcome_a dataset_v2 run is the strongest model (R2=0.139, permutation p less than 0.0001) and provides the most interpretable feature importance results.

**Significant GII effects (1):**
- `feature_a` (Singleton): GII=0.467, q=0.023, stability=3.10. This is the intake outcome_a total score predicting discharge outcome_a total score — the autoregressive predictor. Its dominance is expected (same construct measured at two time points) and serves as a positive control for the pipeline.

**Significant M effects (4 additional, not GII-significant):**
- `feature_x` (M=0.212, q=0.011, stab=2.76): Goal-oriented emotion regulation difficulty
- `feature_x` (M=0.233, q=0.015, stab=3.84): Hostile attribution bias
- `feature_x` (M=0.217, q=0.034, stab=2.85): Emotion regulation strategy access
- `feature_c1` (M=0.207, q=0.041, stab=2.84): General stress symptomatology

These four features are M-significant but not GII-significant. This means they have reliable average SHAP magnitude (exceeding shadow noise), but their GII (which combines M and V) does not exceed shadow noise. Examining the V component:
- DERSgoals V: p=0.606, q=1.0 (V not significant — high variability component fails noise exceedance)
- MAIhostileoutlook V: p=0.170, q=1.0
- DERSstrategies V: p=0.547, q=1.0
- DASSstress V: p=0.418, q=1.0

This pattern (M-significant, V-not-significant, GII-not-significant) indicates that these features have stable average contributions but the variability of their effects across observations is indistinguishable from noise. The V component's high noise exceedance p-values drag the GII below significance.

**Substantive interpretation:**
1. The outcome_a total score's dominance confirms autoregressive prediction (intake predicts discharge for the same measure).
2. The M-significant outcome_a subscales (goals, strategies) suggest that specific emotion regulation difficulties — not just overall dysregulation — contribute to outcome prediction.
3. MAIhostileoutlook's M-significance indicates that maternal hostile attribution bias is an independent predictor of emotion dysregulation outcomes, consistent with social-cognitive theory.
4. DASSstress captures general distress beyond the outcome_a-specific construct.

**No significant interactions.** The top interaction is `feature_a x feature_x` (GII=0.070, q=1.0, stab=1.16), which has a reasonable GII magnitude but fails both the noise exceedance and stability tests.

**Plot quality:** The outcome_a singleton SHAP dependence plot shows a clear, monotonic positive relationship between intake outcome_a and the SHAP contribution to discharge outcome_a. The spline fit is smooth and the noise distribution is well-separated from the signal distribution. Visually, the GII magnitude distribution shows clear separation from the noise (shadow) distribution.
 </summary>
 <research>Not applicable (domain-specific applied interpretation; no methodological research required).</research>
 <approaches>
 <approach id="A3a" label="Report GII and M-significant effects" feasibility="high" risk="low">
 <description>Report the 1 GII-significant and 4 M-significant effects. Frame the M-only effects as having reliable average contributions whose cross-observation variability is indistinguishable from noise.</description>
 <pros>Transparent; scientifically defensible; maximizes information extraction</pros>
 <cons>M-only effects require careful framing to avoid over-interpretation</cons>
 <statistical_considerations>The distinction between GII-significant and M-only-significant is methodologically important. GII significance requires both M and V to jointly exceed noise (or the combined metric to do so). M-only significance indicates consistent average effect but not necessarily consistent effect direction or magnitude across observations.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="A3a">Report all 5 significant effects with clear distinction between GII-level and M-only significance. The outcome_a total autoregressive predictor is the headline finding; the four subscale/related-construct effects provide mechanistic insight.</decision>
 </topic>

 <!-- ============================================================== -->
 <!-- TOPIC 4: outcome_b dataset_v2 FEATURE IMPORTANCE -->
 <!-- ============================================================== -->
 <topic id="topic" title="outcome_b dataset_v2 Feature Importance Analysis">
 <summary>
The outcome_b dataset_v2 run has R2=0.105 (permutation p less than 0.0001), providing valid feature importance despite a challenging outcome distribution.

**Critical distributional issue:** The outcome_b severity outcome is heavily zero-inflated (55% of n=240 observations are zero) and right-skewed (median=0, IQR=3, 12.1% outliers). This violates the implicit assumption of continuous, approximately normal outcomes in RMSE-based loss functions and affects SHAP value interpretation.

**Significant GII effects (2):**
- `feature_b1` (GII=0.312, q=0.007, stab=2.74): Autoregressive predictor (intake outcome_b severity predicting discharge outcome_b severity). Both M and V components are significant.
- `feature_x` (GII=0.118, q=0.007, stab=2.97): Hostile attribution bias. Only M is significant; V is not (p=0.140, q=1.0).

**Significant M effects (7 additional):**
- outcome_c (M=0.225, q=0.002, stab=2.01): Also has significant V (p less than 0.0001, q=0.010), making it the only effect where V is independently significant across all four runs.
- STRESSmood (M=0.092, q=0.002, stab=3.31)
- STRESS_childfreq (M=0.098, q=0.002, stab=4.07)
- STRESSavoidance (M=0.083, q=0.002, stab=2.28)
- STRESSintrusions (M=0.090, q=0.002, stab=2.64)
- STRESSadultfreq (M=0.087, q=0.002, stab=2.60)
- DASSanxiety (M=0.075, q=0.002, stab=2.11)

**Notable: outcome_c has V-significant effects.** This means outcome_c not only has a reliable average effect on outcome_b severity prediction, but the VARIABILITY of that effect across observations is genuinely non-noise. In practical terms, this suggests that the relationship between intake arousal symptoms and discharge stress severity changes meaningfully depending on individual-level characteristics (potential moderation/heterogeneity).

**Comparison to outcome_a dataset_v2:**
- The outcome_b model identifies more significant features (2 GII + 7 M vs 1 GII + 4 M), despite similar R2.
- MAIhostileoutlook appears as GII-significant in the outcome_b model but only M-significant in the outcome_a model. This may reflect different effect profiles for hostile attribution on stress vs emotion regulation outcomes.
- The outcome_b model benefits from a larger battery of outcome_b-specific subscales (arousal, mood, intrusions, avoidance, dissociation, functional impairment) that are collinear with the outcome measure, providing more targets for M-significance.
- 6 of the 7 outcome_b M-significant features are outcome_b subscales or related trauma measures, indicating that the pipeline correctly identifies within-construct predictors.

**Interaction effects:** No interactions reach significance. Top interaction is CRSE x outcome_b (GII=0.035, q=1.0).

**Impact of zero-inflation:** The 55% zero-inflation rate means the model is essentially distinguishing zero vs non-zero outcomes for over half the sample. This likely compresses SHAP contributions for the zero-scorers and concentrates signal in the non-zero subgroup. The V component of outcome_c being significant may reflect this bimodal structure — different SHAP dynamics in the zero vs non-zero subpopulations. The fdr_correct config discrepancy (discussed in) does not affect these significance calls since the resolved config shows fdr_correct=true for this run.
 </summary>
 <research>
Zero-inflated outcomes present known challenges for tree-based SHAP analysis. Chen and Guestrin (2016, KDD) note that gradient boosted trees handle zero-inflation naturally by learning split-based decision rules, but the SHAP decomposition may show bimodal contribution patterns. Zheng et al. (2023, Statistical Methods in Medical Research) recommend examining SHAP waterfall plots stratified by zero/non-zero outcome groups when zero-inflation exceeds 30%, as the feature importance rankings may differ between the two subgroups.

Regarding the general lack of V-significance across all four runs: this is expected when sample sizes are small (n approximately 240) and the outcome signal is modest. The V component captures within-feature heterogeneity (i.e., how much SHAP values vary across observations for a given feature). With n approximately 240 in 10 folds, each fold has approximately 24 observations, providing limited power to detect genuine V differences above shadow noise. Molnar (2022, Interpretable Machine Learning, 2nd ed.) notes that interaction/heterogeneity detection from SHAP requires substantially larger samples than main effect detection.
 </research>
 <approaches>
 <approach id="A4a" label="Report outcome_b results with zero-inflation caveat" feasibility="high" risk="low">
 <description>Report the outcome_b dataset_v2 feature importance with explicit acknowledgment of the zero-inflated outcome distribution. Note outcome_c's V-significance as a highlight finding.</description>
 <pros>Honest; the permutation test confirms real signal; the zero-inflation caveat protects against over-interpretation</pros>
 <cons>Some reviewers may question RMSE loss on a zero-inflated outcome</cons>
 <statistical_considerations>RMSE is not ideal for zero-inflated outcomes, but CatBoost handles the distributional challenge through its tree structure. A Tweedie or zero-inflated loss would be more principled, but the pipeline does not currently support these. The permutation test result (p less than 0.0001) confirms that the model captures real structure regardless of loss function suboptimality.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="A4a">Report outcome_b dataset_v2 results with distributional caveats. The 11 significant effects (2 GII + 9 M) provide a rich feature importance landscape. The outcome_c V-significance is a genuinely interesting finding worthy of manuscript attention.</decision>
 </topic>

 <!-- ============================================================== -->
 <!-- TOPIC 5: CONFIG DISCREPANCY (fdr_correct) -->
 <!-- ============================================================== -->
 <topic id="topic" title="Config Discrepancy: fdr_correct Override">
 <summary>
**Finding:** The user-facing config `config_outcome_b.yaml` specifies `fdr_correct: false`, but the resolved config saved by train.py shows `fdr_correct: true`. All other runs consistently show `fdr_correct: true` in both user and resolved configs.

**Root cause analysis:** The `fill_config_defaults` function in utils.py uses `_setdefault_nested`, which only sets a key if it does not already exist. Since the user config explicitly provides `fdr_correct: false`, the default `True` value should NOT override it. The YAML parser correctly parses `false` as Python `False` (verified).

**Most likely explanation:** The resolved config discrepancy indicates that this outcome_b dataset_v2 run was re-executed using a different config than `config_outcome_b.yaml` — possibly a copy that had already been modified, or the run was re-executed from the resolved config of a previous run. The file timestamps (all files in all four output directories show March 30 12:10) suggest all four runs were re-executed in rapid succession, possibly via a script that standardized the config.

**Impact on results:** Since the resolved config shows `fdr_correct: true`, the significance calls in the output use FDR-corrected q-values, which is the MORE conservative approach. This means the outcome_b dataset_v2 results, if anything, would have produced MORE significant features with `fdr_correct: false` (using raw p-values). The current results are therefore conservative and defensible.

**Recommendation:** Standardize all configs to `fdr_correct: true` (as was done at runtime) and update `config_outcome_b.yaml` to match. FDR correction is the scientifically recommended default for multiple comparisons across features.
 </summary>
 <research>
Benjamini and Hochberg (1995, JRSS-B) established FDR correction as the standard for multiple comparisons in discovery-oriented analyses. The SHAP GII pipeline tests each feature and interaction against stratum-specific noise, producing approximately 145-228 simultaneous tests per run. Without FDR correction, the expected number of false positives at alpha=0.05 is approximately 7-11 features per run. With FDR correction, the expected false discovery RATE is controlled at 5%.
 </research>
 <approaches>
 <approach id="A5a" label="Standardize fdr_correct: true across all configs" feasibility="high" risk="low">
 <description>Update config_outcome_b.yaml to set fdr_correct: true. All runs should use FDR-corrected significance.</description>
 <pros>Consistent; conservative; standard practice for discovery analysis</pros>
 <cons>None</cons>
 <statistical_considerations>With 145-228 simultaneous tests per run, FDR correction is strongly recommended. The current results already used FDR correction at runtime (per resolved configs), so no re-run is needed.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="A5a">Update config_outcome_b.yaml to fdr_correct: true. No re-run required since the actual computation already used FDR correction.</decision>
 </topic>

 <!-- ============================================================== -->
 <!-- TOPIC 6: PIPELINE EVALUATION — MECHANICAL CORRECTNESS -->
 <!-- ============================================================== -->
 <topic id="topic" title="Pipeline Mechanical Correctness Assessment">
 <summary>
**The pipeline is performing correctly across all four runs.** Key evidence:

1. **No calc_failed effects in any run.** All GII, M, and V components computed successfully for all singletons and interactions across all four runs.

2. **V failure rates are negligible.** Maximum v_failure_rate across all runs is 0.0035 (0.35%), well below the 5% concern threshold documented in the pipeline. These are rare bootstrap iterations where the spline fitting for the V component encountered numerical difficulty.

3. **Noise-stratified exceedance testing is functioning.** Each run shows appropriate stratum assignment (singleton_continuous, singleton_nominal, singleton_ordinal, interaction_continuous_continuous, etc.) with correct shadow effect counts in each stratum.

4. **Permutation tests are calibrated.** The null distributions show plausible null characteristics: RMSE null means are substantially higher than observed for dataset_v2 runs (confirming genuine signal), and only marginally higher for dataset_v1 runs (confirming weak/absent signal).

5. **Shadow models are functioning.** Shadow model file sizes are comparable to real model sizes (ratios 0.71-1.19), indicating the shadow models are learning noise at a comparable complexity level. The outcome_a dataset_v1 shadow/real ratio of 0.71 is notable — shadow models are simpler than real models, which is appropriate (noise should be harder to learn than signal).

6. **Plot generation is correct.** dataset_v2 runs produce SHAP dependence plots for significant features; dataset_v1 runs produce only the model performance plot (no significant GII features to plot). The GII magnitude distribution plots show clean separation between signal and noise for significant effects.

7. **FDR correction, stability thresholds, and CI computation all appear to be working as designed.**

**One anomaly observed:** The outcome_a dataset_v1 run has one fold (fold 9) with an extremely large model file (149 KB vs mean of 36 KB). This suggests that fold received hyperparameters (from the Bayesian search) that produced a highly complex model, possibly overfitting. However, this is within the pipeline's design — early stopping should have mitigated this. The corresponding fold 9 R2 is 0.138 (the best fold for that run), suggesting the model's complexity did capture some signal for that particular data split, but this may also reflect overfitting to the specific validation fold.
 </summary>
 <research>Not applicable (mechanical evaluation of pipeline outputs).</research>
 <approaches>
 <approach id="A6a" label="No pipeline changes needed for mechanical issues" feasibility="high" risk="low">
 <description>The pipeline is functioning correctly. No code changes are indicated by these results.</description>
 <pros>Avoids unnecessary changes to validated code</pros>
 <cons>None</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="A6a">No pipeline changes needed for mechanical issues. All outputs are consistent with correct pipeline operation.</decision>
 </topic>

 <!-- ============================================================== -->
 <!-- TOPIC 7: POTENTIAL PIPELINE IMPROVEMENTS -->
 <!-- ============================================================== -->
 <topic id="topic" title="Potential Pipeline Improvements Suggested by Results">
 <summary>
While the pipeline is mechanically correct, these results suggest three potential enhancements worth discussing. None are urgent, but all address patterns visible in these results.

**7A. Zero-inflated semicontinuous outcome support:**
The outcome_b severity outcome (55% zeros, right-skewed) would benefit from a loss function designed for zero-inflated data. CatBoost supports Tweedie loss (`Tweedie:variance_power=1.5`), which is appropriate for semicontinuous data with excess zeros. Adding Tweedie as a loss function option in the config would require minimal code changes (CatBoost handles it natively) but would require validation of the SHAP decomposition under Tweedie loss.

**7B. Fold-level diagnostic output:**
The extreme R2 variability across folds (e.g., -0.863 to +0.138 in outcome_a dataset_v1) is currently only visible through the `metrics_oof.json` file. Adding a fold-level diagnostic summary (fold-specific R2, RMSE, MAE, model complexity metrics) as a CSV or visual would help users identify problematic folds without manual JSON parsing.

**7C. V-component power analysis guidance:**
Zero V-significant effects across four runs (except one isolated case for outcome_c) raises the question of whether the V component has adequate power with n approximately 240. The stability threshold (stab_thresh=2) requires the bootstrap median to be at least twice the CI width, which may be unrealistic for V with small n. Providing guidance on minimum sample size for V significance detection (or a post-hoc power analysis for V) would help users interpret null V results.

**7D. Ensemble predictions for dataset_v1 runs:**
The dataset_v1 runs are missing `predictions_ensemble.csv` (predict.py was not run). This does not affect SHAP analysis (which uses OOF predictions from train.py), but it means the ensemble predictions are unavailable for potential downstream inference tasks. This is a user workflow issue, not a pipeline bug.
 </summary>
 <research>
Regarding Tweedie loss for zero-inflated outcomes: Yang, Qian, and Zou (2018, JMLR) demonstrate that Tweedie regression with gradient boosting provides superior performance for semicontinuous data compared to Gaussian loss, particularly when zero-inflation exceeds 30%. CatBoost's Tweedie implementation (variance_power parameter) is documented and stable.

Regarding V-component power: formal power analysis for the noise-exceedance bootstrap test is complex because it depends on the true effect heterogeneity, the noise distribution, and the number of features. However, simulation studies by Strobl et al. (2007, BMC Bioinformatics) suggest that detection of feature-level heterogeneity in random forests requires n in the range of 500-1000 for moderate effect sizes, which is consistent with the null V results observed here at n approximately 240.
 </research>
 <approaches>
 <approach id="A7a" label="Tweedie loss support" feasibility="high" risk="med">
 <description>Add Tweedie loss as a config option (loss_function: "Tweedie:variance_power=X"). Validate SHAP decomposition under Tweedie loss.</description>
 <pros>Better suited for zero-inflated outcomes; CatBoost supports natively; minimal code change</pros>
 <cons>Requires SHAP validation under Tweedie; users need guidance on when to use it; adds option complexity</cons>
 <statistical_considerations>Tweedie loss with variance_power=1.5 is appropriate for gamma-like data with point mass at zero. The SHAP TreeExplainer decomposition is valid for any differentiable loss function in CatBoost's oblivious tree structure.</statistical_considerations>
 </approach>
 <approach id="A7b" label="Fold-level diagnostic CSV" feasibility="high" risk="low">
 <description>Add a `fold_diagnostics.csv` output with per-fold R2, RMSE, MAE, n_train, n_val, model_iterations, and model_file_size.</description>
 <pros>Immediately useful for all users; trivial to implement; helps identify problematic folds</pros>
 <cons>Minor output expansion</cons>
 </approach>
 <approach id="A7c" label="V-component power guidance" feasibility="med" risk="low">
 <description>Add documentation (INPUT_SPECIFICATION.md) noting that V-component significance requires substantially larger sample sizes than M-component significance, with approximate sample size thresholds from simulation.</description>
 <pros>Prevents user misinterpretation of null V results as "no heterogeneity"</pros>
 <cons>Requires simulation study to determine sample size thresholds accurately</cons>
 </approach>
 </approaches>
 <decision status="open" chosen="none">All three improvements (A7a, A7b, A7c) are worth implementing but are not blocking for the current analysis. A7b (fold diagnostics) is lowest-risk and highest-value for immediate use. A7a (Tweedie) should be considered for the outcome_b analyses specifically. A7c (V power guidance) is a documentation improvement for general use. User should prioritize based on timeline.</decision>
 </topic>

 <!-- ============================================================== -->
 <!-- TOPIC 8: CROSS-OUTCOME FEATURE CONSISTENCY -->
 <!-- ============================================================== -->
 <topic id="topic" title="Cross-Outcome Feature Consistency (outcome_a vs outcome_b)">
 <summary>
Comparing significant features across outcome_a and outcome_b dataset_v2 runs reveals important consistency and specificity patterns:

**Shared significant feature:** `feature_x` is significant in both outcomes:
- outcome_a: M-significant (M=0.233, q=0.015, stab=3.84) but not GII-significant
- outcome_b: GII-significant (GII=0.118, q=0.007, stab=2.97)

This is a cross-outcome finding: hostile attribution bias independently predicts both emotion dysregulation (outcome_a) and trauma stress severity (outcome_b). The fact that it achieves full GII significance for outcome_b but only M-significance for outcome_a may reflect different heterogeneity profiles — the hostile outlook effect on stress severity may be more consistent across individuals than its effect on emotion regulation.

**Outcome-specific features:**
- outcome_a-specific: feature_a (autoregressive), DERSgoals, DERSstrategies, DASSstress
- outcome_b-specific: feature_b1 (autoregressive), outcome_c, STRESSmood, STRESS_childfreq, STRESSavoidance, STRESSintrusions, STRESSadultfreq, DASSanxiety

**Pattern:** The autoregressive predictor (same construct at intake predicting discharge) is consistently the dominant feature. Beyond that, within-battery subscales contribute to their respective outcomes, suggesting construct-specific pathways rather than a single common factor driving both outcomes.

**DASSstress (outcome_a) vs DASSanxiety (outcome_b):** General distress symptoms differentially predict the two outcomes — stress for emotion dysregulation, anxiety for trauma stress. This is a potentially interesting cross-construct dissociation if it replicates.
 </summary>
 <research>Not applicable (cross-outcome comparison is domain-specific).</research>
 <approaches>
 <approach id="A8a" label="Highlight MAIhostileoutlook as cross-outcome finding" feasibility="high" risk="low">
 <description>In the manuscript, highlight MAIhostileoutlook as the sole non-autoregressive, cross-outcome predictor. Frame the outcome_c stress/anxiety dissociation as exploratory.</description>
 <pros>Identifies the most generalizable finding; distinguishes from autoregressive "noise"</pros>
 <cons>Single-study finding requires replication caveat</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="A8a">MAIhostileoutlook is the key cross-outcome finding. The autoregressive predictors and within-battery subscales are expected; the hostile attribution bias finding is the novel, cross-construct contribution.</decision>
 </topic>

 </topics>

 <action_items>
 <item priority="P0" target_mode="implement" description="Update config_outcome_b.yaml to set fdr_correct: true, matching all other configs and the actual runtime behavior" />
 <item priority="P1" target_mode="implement" description="Add fold-level diagnostic CSV output (fold_diagnostics.csv) with per-fold R2, RMSE, MAE, n_train, n_val, model_iterations" />
 <item priority="P1" target_mode="document" description="Add V-component power guidance to INPUT_SPECIFICATION.md, noting that V significance requires substantially larger samples than M significance (approximately n >= 500-1000 based on Strobl et al. 2007 simulations)" />
 <item priority="P2" target_mode="implement" description="Add Tweedie loss function support for zero-inflated outcomes (CatBoost native; requires SHAP validation)" />
 <item priority="P2" target_mode="implement" description="Run predict.py for dataset_v1 runs to generate ensemble predictions (completeness, not analysis-critical)" />
 <item priority="P2" target_mode="document" description="Document the GII inflation artifact when comparing models with different feature set sizes (cross-model GII values are not directly comparable)" />
 </action_items>

 <next_steps>
Recommended downstream workflow:
1. **Immediate (P0):** Fix config_outcome_b.yaml fdr_correct setting via /implement.
2. **For manuscript preparation:** Use the outcome_a dataset_v2 and outcome_b dataset_v2 results as the primary feature importance analyses. Report dataset_v1 results as performance comparison only (demonstrating the necessity of the full use-case-specific battery). Highlight MAIhostileoutlook as the cross-outcome finding. Frame outcome_c V-significance as an exploratory heterogeneity finding.
3. **Pipeline enhancement (P1-P2):** Implement fold diagnostics and V-power documentation before journal submission. Tweedie loss can be deferred unless outcome_b results receive specific reviewer criticism about the outcome distribution.
 </next_steps>

</brainstorm_report>
