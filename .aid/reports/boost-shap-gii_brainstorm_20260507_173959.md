<brainstorm_report>
 <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-05-07T21:39:59Z" />

 <context_files>
 <file path="brainstorm_history/boost-shap-gii_brainstorm_20260507_103025.md" relevance="Prior brainstorm where ( Cobb-Douglas anchor for public repo) and (+ unified quarantined simulation study deferred) were locked; current brainstorm executes the deferred sub-project design." />
 <file path="memory/MEMORY.md" relevance="Project state at Session 9 close: pipeline post-CR-remediation; dispositions resolved; pending /run-local + /document + /publish cycle." />
 <file path="memory/session_history.md" relevance="Session 1-9 chronological history; CR-remediation cycle scope and 18 build changes." />
 <file path="memory/reference_milgram_hpc.md" relevance="Yale Milgram HPC environment (Python 3.14, pandas 3.0.1, R 4.4.1-foss-2022b, GCCcore-12.2.0, nanoparquet for R parquet I/O)." />
 <file path="src/boost_shap_gii/shap_utils.py" relevance="GII = sqrt(M × V) Cobb-Douglas anchor; M = mean(|SHAP|); V = LSQ-spline dose-response variability; the methodology under calibration." />
 <file path="src/boost_shap_gii/train.py" relevance="Phase-2 shadow leakage closure (no eval_set, no early_stopping_rounds in shadow fit); shadow-feature null reference for significance flags." />
 <file path="src/boost_shap_gii/infer.py" relevance="Bootstrap-of-CV CI machinery (Efron 1983; Davison and Hinkley 1997 ch. 5); reference for D5 drift anchor." />
 <file path="INPUT_SPECIFICATION.md" relevance="Decision-theoretic GII framing in Section 3 (Cobb-Douglas; Hill 1910; Goldstein et al. 2015 ICE); CatBoost multi-thread non-determinism note." />
 </context_files>

 <topics>

 <topic title="Hierarchical primary endpoint structure (M-anchor, V-novelty, GII-conjunction)">
 <summary>Three components produced by the boost-shap-gii pipeline (M, V, GII) play distinct scientific roles. M is the field-consensus global-importance metric (mean(|SHAP|)); V is the pipeline's primary novel contribution (dose-response variability); GII is the conjunction (sqrt(M × V) Cobb-Douglas composite). The calibration study's primary endpoints are organized hierarchically by scientific role rather than treated as undifferentiated co-primaries.</summary>

 <research>Williamson and Feng (2020, ICML) report Kendall's tau = 0.71 for SPVIM rank recovery, establishing a concrete reference point for "field-consensus" rank-recovery quality on SHAP-comparable importance estimators. Janitza and Griebel (2013, BMC Bioinformatics) frame relevance discrimination as AUROC over the importance-score threshold. Boulesteix et al. (2020, Statistics in Medicine) argue for neutrally pre-specified success criteria to prevent investigator-driven outcome selection. The hierarchical-endpoint framing is consistent with Morris et al. (2019, Statistics in Medicine, ADEMP) requirement that performance measures be declared as primary vs. secondary in advance.</research>

 <approaches>
 <approach label="GII as sole primary endpoint" feasibility="high" risk="med">
 <description>Calibrate only GII recovery; M and V reported as supporting diagnostics.</description>
 <pros>Simple test structure; one primary outcome.</pros>
 <cons>Cannot diagnose whether GII failure reflects M, V, or composite-form issues; conflates three distinct scientific claims into one endpoint; fails to satisfy ADEMP requirement that estimands be specifically declared.</cons>
 </approach>
 <approach label="Hierarchical primary structure (M-anchor, V-novelty, GII-conjunction)" feasibility="high" risk="low">
 <description>Three primary endpoints with distinct scientific roles. M-anchor is the field-consensus calibration target (failure means the pipeline is broken at the most basic global-importance level). V-novelty is the primary novelty test (failure means V is not a defensible novel contribution). GII-conjunction is the primary conjunction test (failure means the composite advantage over either M-only or V-only is not demonstrated).</description>
 <pros>Each endpoint maps to a distinct scientific claim; failure modes are diagnostic rather than conflated; aligns with ADEMP's primary-vs-secondary discipline; supports separate threshold tightness per endpoint role.</pros>
 <cons>Three endpoints means more pre-registered thresholds; requires explicit logic for "all three pass" vs partial-pass dispositions.</cons>
 <statistical_considerations>Multiple-testing across three endpoints handled by Holm-Bonferroni at the 12-family adequacy level (4 DGP families × 3 endpoints).</statistical_considerations>
 </approach>
 </approaches>

 <decision status="decided" chosen="">User pushback on undifferentiated co-primaries clarified that M is field-consensus, V is novel, GII is conjunction. The hierarchical structure operationalizes this scientific asymmetry. Locked.</decision>
 </topic>

 <topic title="DGP design and full calibration matrix">
 <summary>Synthetic data-generating processes covering linear, nonlinear, interaction, and mixed-type signal structures, with controlled correlation, sparsity, signal-to-noise ratio, sample size, and outcome type. Real-data anchor via UCI Bike Sharing for external validity check. Full factorial matrix produces 432 cells.</summary>

 <research>Friedman (1991) introduced the canonical Friedman1 nonlinear-additive DGP (10 sin(pi X1 X2) + 20(X3-0.5)^2 + 10 X4 + 5 X5) used widely as a DGP benchmark. Strobl et al. (2008, BMC Bioinformatics) formalized correlated-feature DGPs with relevant-irrelevant correlation blocks for variable importance simulation. Liu et al. (NeurIPS 2021, XAI-Bench) established controlled DGP benchmarks for SHAP-comparable methods. Hooker and Mentch (2019, arXiv:1905.03151) showed that unrestricted permutation forces extrapolation in correlated DGPs, motivating careful ground-truth definition. Sklar's theorem and Gaussian copulas provide a principled construction for mixed continuous/ordinal/nominal feature distributions with controlled marginal distributions.</research>

 <approaches>
 <approach label="Sparse, scenario-by-scenario DGP set" feasibility="high" risk="high">
 <description>Hand-pick 4-6 representative scenarios; no factorial design.</description>
 <pros>Cheap to compute; quick turnaround.</pros>
 <cons>Niessl et al. 2022 (WIREs DMKD) showed that benchmark results are highly variable across design choices; sparse scenario selection is itself a researcher-degrees-of-freedom problem.</cons>
 </approach>
 <approach label="Full factorial 4 × 3 × 2 × 3 × 3 × 2 = 432 cells" feasibility="high" risk="low">
 <description>4 DGP families (linear-additive, nonlinear-additive, Friedman1-extended, mixed-type-with-interaction) × 3 correlation levels (rho = 0.0, 0.5, 0.85) × 2 sparsity (sparse 5/25, dense 15/15) × 3 SNR (10, 3, 1) × 3 sample sizes (n = 200, 500, 2000) × 2 outcome types (regression, multiclass K=3). Mixed-type construction via Gaussian copula. Multiclass via latent-threshold equal-prevalence cuts.</description>
 <pros>Full factorial enables every two-way and three-way interaction test; ADEMP-defensible; aligns with Boulesteix neutral-comparison rigor.</pros>
 <cons>Total runs depend on K Monte Carlo replication count; State A (K=1000) implies 432,000 runs.</cons>
 <statistical_considerations>Block-correlation structure within feature blocks (relevant-irrelevant) per Strobl 2008. Ordinal: 5 levels via quantile cuts on continuous latent. Nominal: 3-level and 10-level mix. Real-data anchor: UCI Bike Sharing (n=17,379, regression, mixed-type).</statistical_considerations>
 </approach>
 <approach label="OFAT + selected interaction cells" feasibility="high" risk="med">
 <description>17 cells: 1 anchor + 11 OFAT variations + 5 interaction cells.</description>
 <pros>Boulesteix-style focused design; lower compute envelope.</pros>
 <cons>OFAT misses interactions outside the chosen 5; reviewers may flag this regardless of justification.</cons>
 </approach>
 </approaches>

 <decision status="decided" chosen="">Full factorial 432-cell matrix locked. DGP equations: linear-additive Y = sum(beta_i X_i) + epsilon, beta ~ Uniform(0.5, 2.0); nonlinear-additive g_i in {sin(pi X), 5(X-0.5)^2, exp(X)-1, tanh(2X-1), |X-0.5|}; Friedman1-extended (10 sin(pi X1 X2) + 20(X3-0.5)^2 + 10 X4 + 5 X5 + epsilon); mixed-type-with-interaction. Multiclass: K=3 latent-threshold equal-prevalence. Feature mix: 30 features total; sparse 5/25 = 14 cont + 10 ord + 6 nom; dense 15/15. Correlation: block structure (relevant-irrelevant blocks per Strobl 2008), Gaussian copula. Real-data anchor: UCI Bike Sharing. Monte Carlo K = 1000 (per State A locked in); numpy default_rng (PCG64).</decision>
 </topic>

 <topic title="Ground-truth definitions for M, V, GII">
 <summary>Each endpoint requires an analytically grounded ground-truth target. M-truth uses oracle Shapley value as primary and Sobol S_i as secondary. V-truth uses ALE amplitude as primary (matching the pipeline's V definition for ordinal/nominal levels) and Sobol T_i - S_i as secondary. GII-truth = sqrt(M_truth × V_truth). is_relevant binary label for AUROC/PR-AUC discrimination.</summary>

 <research>Sobol (2001) defined first-order S_i and total-effect T_i variance decomposition indices. Owen (2014, SIAM/ASA JUQ) proved the bracketing theorem S_i &lt;= Shapley_i &lt;= T_i for variance-explained games, establishing that Sobol indices provide computable bounds on Shapley importance even though they are not exact equivalents. Saltelli (2002, Computer Physics Communications) provided efficient Sobol estimators via quasi-random sampling. Apley and Zhu (2020, JRSS-B) defined ALE plots; ALE amplitude (L1 or L2 norm of the ALE curve) captures dose-response informativeness aligned with the pipeline's V definition. Williamson and Feng (2020, ICML) defined population-level variable importance as predictiveness contrast with all features minus target subset, providing a theory-grounded M target avoiding circular SHAP-ground-truth issues. Janzing, Minorics, Bloebaum (2020, AISTATS) formalized observational-vs-interventional Shapley distinction; for additive DGPs with independent inputs, the two coincide. Liu et al. (2021, NeurIPS) used exact Shapley for controlled DGPs with binary is_relevant labels in XAI-Bench.</research>

 <approaches>
 <approach label="Sobol S_i T_i as primary M V truth" feasibility="high" risk="med">
 <description>Use Sobol indices as primary ground truth for both M and V.</description>
 <pros>Closed-form for additive DGPs; well-understood by sensitivity analysis literature.</pros>
 <cons>Owen 2014 shows Sobol indices bracket but do not coincide with Shapley value; for SHAP-based pipeline calibration, Sobol primary risks measuring the wrong target.</cons>
 </approach>
 <approach label="Oracle Shapley + ALE amplitude (recommended)" feasibility="high" risk="low">
 <description>M-truth primary = oracle Shapley value via permutation at 10,000 coalitions evaluated against analytical f(X). M-truth secondary = Sobol S_i via Saltelli sampling at 50,000 evaluations. V-truth primary = ALE amplitude (L1 norm of ALE curve, 200-point X-grid for continuous; SD of E[f(X) | X_cat = level] weighted by frequency for ordinal/nominal, matching the pipeline's V definition). V-truth secondary = T_i - S_i. GII-truth = sqrt(M_truth_primary × V_truth_primary). Single is_relevant binary label since M/V/GII partitions coincide for. Multiclass: per-class binary indicator regression P(Y_class = k | X), per-class metrics averaged across K=3.</description>
 <pros>Oracle Shapley directly aligns with the pipeline's SHAP target; ALE amplitude directly aligns with V's pipeline definition; avoids circular SHAP-on-SHAP validation; Owen 2014 bracketing theorem provides theoretical justification for Sobol secondary.</pros>
 <cons>Permutation-Shapley at 10,000 coalitions is more compute than Sobol; ALE on ordinal/nominal requires careful handling.</cons>
 <statistical_considerations>Implementation: SALib for Sobol; custom permutation Shapley per Strumbelj and Kononenko (2014); custom ALE on 200-point grid; scipy.stats multivariate_normal for Gaussian copula; numpy quantile cuts for multiclass latent threshold.</statistical_considerations>
 </approach>
 </approaches>

 <decision status="decided" chosen="">Locked. Oracle Shapley is the principled primary M-truth; ALE amplitude matches the pipeline's V definition operationally; Sobol indices serve as secondary cross-checks per Owen's bracketing theorem.</decision>
 </topic>

 <topic title="Recovery metrics and pre-specified numeric thresholds">
 <summary>Six metrics across three endpoint roles, with locked numeric thresholds per ADEMP/Pawel et al. 2024 pre-registration discipline. Per-DGP-family + aggregate adequacy rule. Holm-Bonferroni at family-level for adequacy declaration. K=1000 MC reps adequate (MC SE bounded well below threshold gaps).</summary>

 <research>Williamson and Feng (2020) report Kendall's tau = 0.71 as a concrete reference for adequate rank recovery in SHAP-comparable importance settings. Janitza and Griebel (2013) use AUROC over the importance-score ROC curve as the canonical discrimination metric. Davis and Goadrich (2006, ICML) established that PR-AUC is preferred under sparse positive class. Benjamini and Yekutieli (2001, Annals of Statistics) provided BY-FDR control under arbitrary dependence with conservative penalty c(m) = sum(1/k for k=1..m). Genovese and Wasserman (2002, JRSS-B) showed empirical FDR converges to nominal q under independence; under positive dependence BY remains conservative. Fithian and Lei (2022, Annals of Statistics) demonstrated BY uniform conservatism. Pawel, Kook, Reeve (2024, Biometrical Journal) showed questionable research practices in simulation studies generate spurious superiority claims; pre-registering thresholds before code execution is the primary countermeasure.</research>

 <approaches>
 <approach label="Universal threshold suite (no endpoint-role differentiation)" feasibility="high" risk="med">
 <description>Same Kendall's tau AUROC thresholds for M, V, GII.</description>
 <pros>Simpler reporting.</pros>
 <cons>Ignores hierarchical endpoint structure from; reviewers will flag uniform thresholds as failing to operationalize the field-consensus novelty conjunction asymmetry.</cons>
 </approach>
 <approach label="Hierarchical thresholds matched to endpoint role (recommended)" feasibility="high" risk="low">
 <description>M-anchor (field consensus): tau &gt;= 0.70 at n=2000 SNR=10; tau &gt;= 0.55 at n=500 SNR=3; AUROC &gt;= 0.85 at n &gt;= 500. V-novelty (primary novel test): tau &gt;= 0.60 at n=2000 SNR=10; tau &gt;= 0.45 at n=500 SNR=3; AUROC &gt;= 0.75. GII-conjunction (composite advantage): tau(GII) &gt;= max(tau(M), tau(V)) at n=2000 SNR=10 in and; AUROC(GII) &gt;= max(AUROC(M), AUROC(V)) - 0.02 at n=500 SNR=3; median delta-Kendall's tau (GII - best of M,V) &gt;= +0.05 in and at n=2000 SNR=10. PR-AUC: M &gt;= 0.70, V &gt;= 0.60 in sparse 5/25. FDR coverage: empirical FDR &lt;= 1.05 × q at each rho in {0.0, 0.5, 0.85} for sig_M, sig_V, sig_GII independently. TDR: &gt;= 0.50 at n=500 SNR=3; &gt;= 0.70 at n=2000 SNR=10 in dense 15/15.</description>
 <pros>Operationalizes hierarchical endpoint structure; M anchor is field-tied (Williamson and Feng 2020 reference); V novel target is principled; GII conjunction tests composite-advantage claim; FDR coverage tolerance accommodates Monte Carlo noise; TDR floors gate detectability.</pros>
 <cons>More numeric thresholds to pre-register; more places where failure can occur.</cons>
 <statistical_considerations>Per-DGP-family adequacy at the (n=500, SNR=3) anchor; aggregate adequacy at (n=2000, SNR=10). Holm-Bonferroni at 12-family level (4 DGP × 3 endpoints) for the calibration-adequacy declaration; cell-level results reported in supplement without correction. Monte Carlo SE at K=1000: tau ~ 0.025, AUROC ~ 0.005, empirical FDR ~ 0.007; all well below threshold gaps.</statistical_considerations>
 </approach>
 </approaches>

 <decision status="decided" chosen="">Hierarchical thresholds locked exactly as specified.</decision>
 </topic>

 <topic title="Multi-thread determinism drift study ">
 <summary>Quantify pairwise drift in M, V, GII point estimates and sig flag stability between independent CatBoost runs at fixed random_seed and dataset, varying only thread_count. Two hardwares (M1 Max + Yale Milgram) reflecting actual deployment. Three thread counts. Two cells. 50 reps per condition. Cross-hardware reported as diagnostic only, not gated.</summary>

 <research>CatBoost GitHub Issue #1587 documents user-reproducible divergence in model outputs when thread_count varies even with fixed random_seed, rsm=1, random_strength=0; root cause is floating-point non-associativity (FPNA) in parallel histogram reductions. LightGBM's deterministic=true flag (Parameters Documentation v4.6) provides reproducibility guarantees only at training slowdown cost and only on same system/binary/compiler. XGBoost is described as "mostly deterministic" with multithreading. H2O GBM is "deterministic up to floating-point rounding errors" arising from out-of-order atomic additions. Groves et al. (2024, SC'24 Workshops) introduced FPNA metrics: scalar relative difference (Vs) and elementwise relative mean absolute variation (ERMV) with count variability (Vc); FP64 accumulation differences range 10^-15 to 10^-13 at the leaf level. Bouthillier et al. (2021, MLSys) advocate reporting multiple variance sources. Goldwasser et al. (2024, ICML 2025; arXiv:2401.15800) propose SPRT-SHAP and RankSHAP for ranking stability significance testing. Nogueira, Sechidis, Brown (2018, JMLR) provide corrected-Jaccard set-stability metrics. Critical literature gap: no published study quantifies SHAP-derived index drift between multi-thread CatBoost runs.</research>

 <approaches>
 <approach label="Architectural prevention only (thread_count=1 in pipeline default)" feasibility="high" risk="med">
 <description>No drift study; recommend users fix thread_count=1.</description>
 <pros>Cheapest; matches H2O LightGBM prevention guidance.</pros>
 <cons>Does not quantify the drift the user actually faces; users will inevitably use multi-thread for speed; no defensible reproducibility claim across thread counts.</cons>
 </approach>
 <approach label="Empirical drift quantification with two-hardware design (recommended)" feasibility="high" risk="low">
 <description>K_drift = 50 independent runs per (thread_count × cell × hardware). thread_count in {1, 4, 8} on M1 Max (8 P-cores, no E-core mixing) and {1, 4, max_available_per_node} on Milgram. Two cells: = Friedman1-extended dense 15/15 rho=0.5 SNR=3 n=500 regression (modal real-world); = mixed-type-with-interaction sparse 5/25 rho=0.5 SNR=3 n=500 multiclass (mixed-type stress). Drift metrics: per-feature relative absolute difference (RAD) on M, V, GII independently with median, 95th-percentile, max aggregates; pairwise Kendall's tau on rankings (median, 5th-percentile); sig flag flip rate at q=0.05 for sig_M, sig_V, sig_GII; top-5 corrected Jaccard (Nogueira 2018) reported only. Anchors: bootstrap-of-CV CI half-width (B=200) and shadow-feature point estimate. Pass criteria (locked): median RAD &lt; 0.10 × bootstrap CI half-width AND &lt; 0.20 × shadow point estimate; median Kendall's tau &gt;= 0.95; sig flip rate &lt;= 0.05 at q=0.05. Cross-hardware drift: reported as a diagnostic; not gated (: cross-system reproducibility fundamentally limited).</description>
 <pros>Fills the literature gap ( explicit gap finding); quantifies drift relative to two pipeline-internal anchors (statistical-uncertainty floor and shadow-noise floor); deployment-realistic two-hardware design; failure modes (RAD-only, ranking-only, sig-flip-only) have distinct dispositions in the report.</pros>
 <cons>2-hardware design requires manual scheduling on both M1 Max and Milgram; cross-hardware drift may exceed thresholds ( expectation) but is reported diagnostically only.</cons>
 <statistical_considerations>1225 pairwise comparisons per (thread × cell × hardware) at K_drift=50. Failure-mode dispositions (locked): RAD failure -&gt; thread_count caveat; rank tau failure -&gt; rank-stability caveat; sig flip failure -&gt; BY-FDR family-call architecture flagged for revision (hardest failure).</statistical_considerations>
 </approach>
 </approaches>

 <decision status="decided" chosen="">Locked. M1 Max thread grid {1, 4, 8} chosen to avoid E-core mixing (M1 Max has 8 performance + 2 efficiency cores; mixing introduces heterogeneous-core variance orthogonal to FPNA). Milgram thread grid set at build time to {1, 4, max_available_per_node}. Cross-hardware diagnostic-only.</decision>
 </topic>

 <topic title="Compute envelope and execution platform">
 <summary>Three design states (most ideal less ideal but OK not great but adequate) compared. State A (full factorial × K=1000) chosen for maximum defensibility. Wall time ~28 days on M1 Max single machine; ~1-3 days end-to-end on Yale Milgram HPC with SLURM array. HPC pivot accepted; deployment is local-only within the HPC (no cloud).</summary>

 <research>Per-run wall time estimate ~10-80 seconds on M1 Max thread_count=1 depending on n in {200, 500, 2000}; average ~40 seconds, +20% joblib overhead = 45 seconds effective. State A on M1 Max: 432,000 × 45 8 P-cores ~= 28 days continuous (sustained M1 Max thermal throttling 5-15% absorbed in overhead). State A on Milgram (assumed 24-core node, 100 concurrent SLURM tasks): 432 array tasks × 1000 reps 24 cores ~= 31 minutes per task; ~10 hours pure compute; 1-3 days end-to-end with queue. Niessl et al. 2022 caution that compute pruning choices are themselves researcher degrees of freedom. State A defensibility justified by avoiding fractional-design OFAT confounding.</research>

 <approaches>
 <approach label="State A: Full factorial × K=1000 on M1 Max" feasibility="low" risk="high">
 <description>432,000 runs × ~45s 8 cores = 28 days continuous on laptop.</description>
 <pros>No HPC pivot needed.</pros>
 <cons>4 weeks of laptop unavailability; thermal sustainability unverified for 28-day window; impractical.</cons>
 </approach>
 <approach label="State A: Full factorial × K=1000 on Milgram HPC (recommended)" feasibility="high" risk="low">
 <description>SLURM array of 432 tasks × 1000 reps 24-core nodes; 1-3 days end-to-end including queue. Drift study runs separately on M1 Max + Milgram. Real-data anchor (UCI Bike Sharing) on either host.</description>
 <pros>State A defensibility preserved at 10-25× speedup; Milgram environment already configured per project memory; M1 Max free for daily work.</pros>
 <cons>Manual deployment workflow; result back-copy required.</cons>
 <statistical_considerations>SLURM template: 432 array tasks, 2-hour walltime per task (4× safety margin over 31-min estimate), 16 GB memory per task, 24 cores per task. Disk: ~5 GB total parquet outputs. Memory peak per task: ~8 GB at n=2000 multiclass; well under 128 GB node memory.</statistical_considerations>
 </approach>
 <approach label="State B: Full factorial × K=500 on M1 Max" feasibility="med" risk="med">
 <description>216,000 runs × 45s 8 cores = 14 days; MC SE 0.036.</description>
 <pros>Full factorial preserved.</pros>
 <cons>Still 2 weeks of laptop unavailability; MC SE 0.036 vs threshold gaps as small as 0.05 means borderline cells require Tier-2.</cons>
 </approach>
 <approach label="State C: OFAT + 5 interactions × K=500" feasibility="high" risk="med">
 <description>9,000 runs × 45s 8 cores = 13 hours; weekend feasible.</description>
 <pros>Single overnight; high MC precision per cell.</pros>
 <cons>Limited interaction coverage; reviewers may flag OFAT vs full factorial.</cons>
 </approach>
 </approaches>

 <decision status="decided" chosen="">State A on Milgram HPC locked. User explicitly chose maximum defensibility and accepted HPC pivot. Milgram deployment per existing project conventions.</decision>
 </topic>

 <topic title="Comparison baselines (Categories 1, 2, 3b)">
 <summary>Three baseline categories cover (1) within-pipeline composite-form ablations, (2) within-pipeline component-estimator ablations, (3) external feature-importance methods. Categories 1+2 are essentially free (computed from same SHAP outputs). Category 3 doubles per-cell wall time; recommended scope is 3b (RF Gini + SPVIM) excluding permutation importance (compute-prohibitive at K=1000) and SAGE (theoretical redundancy with SPVIM).</summary>

 <research>Breiman (2001, Machine Learning) introduced permutation importance and Random Forest Gini importance. Williamson and Feng (2020) introduced SPVIM with Kendall's tau = 0.71 reference for ICU mortality SHAP-comparable application. Covert, Lundberg, Lee (2020, NeurIPS) introduced SAGE (Shapley Additive Global Explanations) as a Shapley-additive global importance method. Goldwasser et al. (2024) introduced SPRT-SHAP and RankSHAP for sequential significance testing of feature rankings, which is more principled than vanilla permutation importance and supports lower K. Cobb and Douglas (1928) established the geometric mean as the default Cobb-Douglas composite at alpha = beta. Hill (1910) established the geometric mean's foundations in statistical aggregation. Goldstein et al. (2015, Journal of Computational and Graphical Statistics, ICE) provided dose-response visualization conceptually aligned with the V component.</research>

 <approaches>
 <approach label="Categories 1+2 only (within-pipeline ablations)" feasibility="high" risk="med">
 <description>No external comparators.</description>
 <pros>Free compute; clean ablation isolation.</pros>
 <cons>Boulesteix neutral-comparison criteria require external comparators; without them, study is self-validation.</cons>
 </approach>
 <approach label="Categories 1+2+3b (RF Gini + SPVIM)" feasibility="high" risk="low">
 <description>All three categories. Category 1 composite forms: M-only, V-only, arithmetic mean (M+V)/2, max(M,V), Cobb-Douglas alpha=0.7/beta=0.3 and alpha=0.3/beta=0.7, geometric mean sqrt(M×V) (current). Category 2 component-estimator ablations: M with mean(|SHAP|) (current) vs median(|SHAP|); V with LSQ spline (current) vs simple variance(|SHAP|) vs ALE-amplitude-on-fitted-model. Category 3b: RF Gini importance + SPVIM. Permutation importance excluded due to K=1000 compute prohibitive (100 perms × refit per replicate). SAGE excluded for theoretical redundancy with SPVIM.</description>
 <pros>Categories 1+2 directly test the Cobb-Douglas anchor and component-estimator choices; Category 3b includes the most common practitioner baseline (RF Gini) and the Shapley-theoretical baseline (SPVIM with published tau=0.71 reference); SAGE redundancy avoids overhead without information loss; Goldwasser 2024 SPRT-SHAP can be cited for permutation deferral rationale.</cons>
 <cons>Adds ~50% per-cell compute on Milgram; total wall-time ~1.5-4 days end-to-end.</cons>
 <statistical_considerations>16-method × 432-cell × 1000-rep matrix produces parquet outputs ~5 GB. Pre-specified Category 3 thresholds: GII tau &gt;= RF Gini tau at modal cell (n=500 SNR=3) is soft success criterion; GII tau within 0.05 of SPVIM tau is parity criterion; GII tau &lt; SPVIM tau by &gt; 0.10 at modal cell triggers substantive-finding discussion in report.</statistical_considerations>
 </approach>
 <approach label="Categories 1+2+3c (RF Gini + SPVIM + SAGE)" feasibility="med" risk="low">
 <description>Full Shapley-family comparison.</description>
 <pros>Maximum external comparator coverage.</pros>
 <cons>SAGE adds ~50% additional compute over with redundant Shapley-theoretical content (both SPVIM and SAGE are Shapley-additive); doubles per-cell compute total.</cons>
 </approach>
 </approaches>

 <decision status="decided" chosen="">Categories 1+2+3b locked. Permutation importance deferred to Tier-2 supplementary if reviewers request it. SAGE excluded for redundancy.</decision>
 </topic>

 <topic title="Quarantine and portability architecture">
 <summary>Self-contained local directory at {simulation_local_dir}/ with no git remote of any kind. Manual two-way copy to Milgram for execution. Pipeline-under-test pinned to specific commit hash from upcoming /publish cycle, installed via pip on Milgram. AID_LOG disclosure deferred until post-results decision on publication.</summary>

 <research>Boulesteix, Wilson, Hapfelmeier (2017/2018, Biometrical Journal; doi:10.1002/bimj.201700129) established author-conflict disclosure as a minimum for developer-authored simulation studies. Siepe, Bartos, Morris, Boulesteix, Heck, Pawel (2024, Psychological Methods) developed ADEMP-PreReg requiring DGP, method, performance evaluation criteria locked before result generation. Kapoor and Narayanan et al. (2024, Science Advances) REFORMS checklist requires explicit pre-specification of evaluation metrics and identification of leakage paths. Hofman et al. (2023, arXiv:2311.18807) two-phase pre-registration: Phase A locks features, metrics, baselines; Phase B locks final model and confirms test-data independence. Methodology-implementation independence (the harness lives outside the public pipeline repo) is a Boulesteix-style structural guarantee that the calibration is not optimized to the pipeline's idiosyncrasies post-hoc.</research>

 <approaches>
 <approach label="Private GitHub repo for harness" feasibility="high" risk="med">
 <description>Separate private GitHub repo; clone on Milgram via git clone.</description>
 <pros>Cloud backup; standard git workflow.</pros>
 <cons>User explicitly rejected: "I do NOT want to publish any of this simulation study to a GitHub repo." Even private repos involve cloud publication.</cons>
 </approach>
 <approach label="Self-contained local directory, manual two-way copy (locked)" feasibility="high" risk="low">
 <description>Simulation harness lives entirely in {simulation_local_dir}/. No git remote. User manually copies to Milgram (any tool of choice; no script encodes transfer); SLURM writes to results/ on server filesystem; user manually copies results/ back to local. Local analysis runs on M1 Max against back-copied results. No network transfer commands in the harness (no rsync, scp, git push, curl to remote endpoints). Allowed network calls: pip install git+... for pinned pipeline at one-time env setup, and sklearn.datasets.fetch_openml for UCI Bike Sharing one-time first run.</description>
 <pros>Complete user control; no cloud publication; methodology-implementation independence preserved by directory separation alone.</pros>
 <cons>User assumes responsibility for local backup discipline; no remote redundancy for the 1-3 days of HPC compute investment.</cons>
 <statistical_considerations>Pipeline version pinning: lock to specific commit hash from upcoming /publish cycle in config/pipeline_commit.txt. Installed on Milgram via pip install --no-cache-dir --force-reinstall git+https://github.com/tjkeding/boost-shap-gii.git@&lt;commit_hash&gt;.</statistical_considerations>
 </approach>
 </approaches>

 <decision status="decided" chosen="">Self-contained local directory locked. Directory structure includes README.md, ADEMP_PreReg.md (locked locally before execution but not submitted to OSF D9.6.B), environment.yml (cross-platform osx-arm64 + linux-64), requirements_pip.txt, config/ (matrix_cells.yaml, seed_schedule.yaml, thresholds.yaml, pipeline_commit.txt), src/ (dgp/, ground_truth/, pipeline_wrap/, metrics/, analysis/), scripts/ (slurm_submit_calibration.sh, slurm_submit_drift_milgram.sh, run_drift_local_m1max.sh, run_realdata_anchor.sh, aggregate_results.sh), results/ (created at run time, populated server-side, manually back-copied), reports/ (generated post-analysis: study report, plots, tables). All synthetic data generated on-the-fly from seeded numpy default_rng(PCG64); no data files copied between local and Milgram.</decision>
 </topic>

 <topic title="Deliverables, sequencing, and disposition">
 <summary>Local-only working report; disposition deferred until results review. No OSF submission. Internal ADEMP-PreReg discipline only (locked locally before execution, not publicly registered). Sequencing: complete /run-local + /document + /publish on public repo first; then build harness pinned to /publish commit; then execute on Milgram; then back-copy and analyze; then decide on any AID_LOG update or publication path.</summary>

 <research>Morris, White, Crowther (2019, Statistics in Medicine, ADEMP) requires aims, data-generating mechanisms, estimands, methods, performance measures locked in advance. Siepe et al. (2024) ADEMP-PreReg recommends OSF submission. Pawel, Kook, Reeve (2024, Biometrical Journal; arXiv:2203.13076) demonstrated questionable research practices in simulation studies generate spurious superiority claims; pre-registration is the primary countermeasure. Boulesteix (editorial 2024, Biometrical Journal; doi:10.1002/bimj.202400031) consolidates community consensus that simulation-based and real-data benchmarking are complementary. The user's choice to defer publication-disposition until post-results is consistent with hypothesis-driven internal validation; the absence of OSF registration weakens external pre-registration claim but internal ADEMP discipline (locked thresholds and matrix before execution) preserves the core methodological argument.</research>

 <approaches>
 <approach label="OSF Registries pre-registration submitted before execution" feasibility="high" risk="low">
 <description>Submit ADEMP_PreReg.md to OSF Registries before Phase 1 execution; receive DOI; cite in AID_LOG and study report.</description>
 <pros>Strongest defensibility against post-hoc threshold adjustment claims; complete Boulesteix neutral-comparison fulfillment.</pros>
 <cons>Public methodology disclosure even though code stays private; commits to a publication path before results are seen.</cons>
 </approach>
 <approach label="Internal pre-registration only (locked candidate)" feasibility="high" risk="low">
 <description>ADEMP_PreReg.md produced and version-locked locally before Phase 1 execution. No OSF submission. Internal discipline preserved (thresholds locked before code generation outputs); external public-timestamp claim absent. Report disposition decided post-results: options include (a) keep local-only never publish, (b) supplementary material with future publications citing the pipeline, (c) standalone methodology paper. No commitment now.</description>
 <pros>Methodological discipline preserved; defers publication-path decision until informed by results; matches user's explicit choice that this is not a public-facing study.</pros>
 <cons>Reviewer scrutiny of post-hoc threshold concerns weaker than with OSF timestamp; some defensibility lost.</cons>
 </approach>
 </approaches>

 <decision status="decided" chosen="">Locked. Sequencing: (1) /run-local (install nanoparquet, clear 4 environmental test failures, confirm 625/625 green); (2) /document (update README.md, INPUT_SPECIFICATION.md, AID_LOG.md, docstrings.aid/ to reflect 18 build changes from Session 9); (3) /publish (bundled GitHub release; capture commit hash for pipeline_commit.txt); (4) Simulation harness build (a /implement plan + build cycle for harness in {simulation_local_dir}/, registered as separate project via /new-project; followed by /test for harness; followed by ADEMP-PreReg document finalization); (5) Manual deployment + Phase 1-4 execution on Milgram; (6) Manual back-copy + local analysis + report drafting (/writing-draft → /writing-cr → /writing-revise → /writing-ref-verif as appropriate); (7) Post-results decisions on disposition and downstream public-repo updates per pre-registered rules. Downstream feedback rules locked: all-pass triggers AID_LOG note + README "Methodological validation" subsection; Tier-2-confirmed-boundary triggers INPUT_SPECIFICATION.md scope-condition documentation; hard failure triggers new /cr cycle on public repo. Study report structure: 9 sections (Background; ADEMP recap; Methods; Tier-1 results; Drift results; Real-data results; Comparison baselines; Tier-2 contingency; Conclusions/limitations/downstream).</decision>
 </topic>

 </topics>

 <action_items>
 <item priority="P1" target_mode="run-local" description="Install nanoparquet R package; clear 4 environmental test failures; confirm 625/625 green test suite. Pre-existing P1 from MEMORY.md." />
 <item priority="P1" target_mode="document" description="Update public-repo README.md, INPUT_SPECIFICATION.md, AID_LOG.md, docstrings.aid/ to reflect 18 Session 9 build changes. AID_LOG update at this stage does NOT mention the simulation study (study not yet built). Pre-existing P1 from MEMORY.md." />
 <item priority="P1" target_mode="publish" description="Bundled GitHub release of public repo. Capture commit hash for simulation harness config/pipeline_commit.txt. Pre-existing P1 from MEMORY.md, now sequenced as the strict prerequisite to simulation-harness build." />
 <item priority="P1" target_mode="new-project" description="Register {simulation_local_dir}/ as a separate project via /new-project once /publish completes. Project gets its own CLAUDE.md, project memory, and grants file." />
 <item priority="P1" target_mode="implement" description="Plan + build the simulation harness in {simulation_local_dir}/. Scope: directory structure decision; locked configs (matrix_cells.yaml = 432 cells; seed_schedule.yaml = PCG64 seed table; thresholds.yaml = numeric thresholds; pipeline_commit.txt = pinned commit from /publish); src/ modules (DGP + multiclass; Sobol via SALib + permutation Shapley + ALE oracles; pipeline_wrap calling boost-shap-gii as installed library; metrics for Kendall tau/AUROC/PR-AUC/FDR coverage/TDR; analysis for cell+family+aggregate threshold checks); scripts/ (SLURM submission for calibration and drift; M1 Max drift runner; UCI Bike Sharing real-data runner; local aggregation). environment.yml dual-platform conda-forge spec with explicit version pins." />
 <item priority="P1" target_mode="test" description="Test cycle for the simulation harness: smoke tests on small cells; ground-truth oracle verification (Sobol against analytical S_i/T_i for closed-form DGPs; permutation-Shapley convergence; ALE amplitude on synthetic single-feature monotone signals); reproducibility tests at fixed seeds; cross-platform sanity tests." />
 <item priority="P1" target_mode="implement" description="Finalize ADEMP_PreReg.md before any Phase 1 execution. Locked thresholds, matrix, seed policy, baseline list, downstream feedback rules. Internal pre-registration only; no OSF submission." />
 <item priority="P2" target_mode="run-local" description="Phase 1 (Tier-1 calibration) execution on Milgram via SLURM array; ~1-3 days end-to-end. Manual two-way copy by user. Phase 2 (drift) on M1 Max + Milgram. Phase 3 (UCI Bike Sharing) on either host. Phase 4 (Tier-2 contingency) conditional on Phase 1 results per pre-registered triggers." />
 <item priority="P2" target_mode="writing-draft" description="Study report drafting (9-section structure decision) after results back-copy and local analysis. Sequenced /writing-draft → /writing-cr → /writing-revise → /writing-ref-verif as appropriate." />
 <item priority="P2" target_mode="document" description="Post-results AID_LOG.md update on public repo (only if user decides to disclose study existence). Wording determined at decision time based on user's publication choice and study findings (all-pass boundary failure)." />
 </action_items>

 <next_steps>Sequencing is strict: /run-local first, then /document, then /publish on the public boost-shap-gii repo. Only after /publish completes and the release commit hash is captured does the simulation-harness build begin (via /new-project + /implement + /test). The brainstorm report itself completes the design phase; no further design decisions remain. Recommend proceeding to /run-local on the public repo as the immediate next mode invocation.</next_steps>

</brainstorm_report>
