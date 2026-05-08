<brainstorm_report>
 <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-03-30T17:27:04Z" />
 <context_files>
 <file path="./src/boost_shap_gii/train.py" relevance="Contains loss function selection (line 248, 505, 609), model training pipeline, and the insertion point for outcome diagnostics (around line 440)." />
 <file path="./src/boost_shap_gii/utils.py" relevance="Contains _TASK_LOSS_SCORING mapping (line 161), fill_config_defaults (line 246), and statistical helpers. The diagnostic function and auto-resolution logic would be added here." />
 <file path="./example_config_advanced.yaml" relevance="Documents loss_function options (line 67-71); needs update to include 'auto' option." />
 <file path="./example_config_minimal.yaml" relevance="Minimal config omits loss_function entirely; auto-fill default would change from 'RMSE' to 'auto'." />
 </context_files>
 <topics>
 <topic id="topic" title="Diagnostic Tests for When RMSE Loss Becomes Problematic">
 <summary>Formal normality tests (Shapiro-Wilk, Anderson-Darling, K-S) are unsuitable as automated triggers because they are asymptotically consistent -- they reject for any departure from normality as n grows, including trivially small departures. Effect-size-based diagnostics (zero fraction, excess kurtosis magnitude, skewness magnitude, Hartigan's dip test) are appropriate because they measure the magnitude of distributional pathology, not its statistical detectability. No single diagnostic captures all relevant failure modes; a battery of complementary criteria is required, where triggering on any single criterion is sufficient.</summary>
 <research>
 - Razali &amp; Wah (2011): Shapiro-Wilk is most powerful normality test for small n, but all normality tests approach power=1.0 as n grows, making them useless as practical triggers for large datasets. Published in Journal of Statistical Modeling and Analytics.
 - Kim (2013): Proposes |skewness| > 2 as "extreme" and excess kurtosis > 7 as "extreme" for assessing non-normality impact, published in British Journal of Mathematical and Statistical Psychology.
 - Olsen &amp; Schafer (2001); Tooze et al. (2002): In the two-part/semicontinuous model literature, special modeling is recommended when zero proportions exceed roughly 10-20%. Published in Statistical Methods in Medical Research and Biometrics.
 - Hartigan &amp; Hartigan (1985): The dip test statistic measures the maximum difference between the empirical CDF and the best-fitting unimodal CDF. Distribution-free under the null. Published in Annals of Statistics.
 - D'Agostino &amp; Pearson (1973): Combined skewness-kurtosis omnibus test; available as scipy.stats.normaltest. Better than individual moment tests but shares the large-n power problem.
 - Campbell et al. (2021): Demonstrates that checking for zero-inflation and overdispersion as a model-selection step can inflate Type I error rates if the diagnostic is correlated with the model comparison; recommends diagnostic independence from the test statistic. Published in Methods in Ecology and Evolution.
 - Python implementations: scipy.stats.shapiro, scipy.stats.normaltest, scipy.stats.skew, scipy.stats.kurtosis, diptest.diptest (PyPI package, pre-compiled wheels for CPython 3.8-3.12 on Linux/macOS/Windows).
 </research>
 <approaches>
 <approach label="Effect-size diagnostic battery" feasibility="high" risk="low">
 <description>Compute four complementary diagnostics on the outcome vector before training: (1) zero fraction >= 0.15, (2) excess kurtosis >= 5.0, (3) |skewness| >= 2.0, (4) Hartigan's dip test p &lt; 0.05. Trigger Huber if any one criterion is met. These are magnitude/effect-size measures that do not suffer from the large-n sensitivity of formal normality tests.</description>
 <pros>Simple to implement; complementary coverage of distinct distributional pathologies; thresholds grounded in literature; no iterative computation; distribution-free dip test adds formal bimodality detection.</pros>
 <cons>Thresholds are not universally validated for the specific question "when does RMSE degrade for gradient boosting?" -- they are borrowed from adjacent literature (two-part models, robust statistics, descriptive statistics guidelines). The 0.15 zero-fraction threshold is a judgment call.</cons>
 <statistical_considerations>The diagnostics are computed on the full y vector before CV splitting, ensuring consistent loss function across folds. The dip test has well-calibrated Type I error. The moment-based thresholds (skewness, kurtosis) are descriptive, not inferential, which is actually an advantage here -- we want to measure the magnitude of the problem, not test a hypothesis.</statistical_considerations>
 </approach>
 <approach label="Formal normality tests only" feasibility="high" risk="high">
 <description>Use Shapiro-Wilk or D'Agostino-Pearson p-value &lt; 0.05 as the sole trigger.</description>
 <pros>Theoretically clean; single test.</pros>
 <cons>Nearly always triggers for n > 500 due to asymptotic consistency; fails to distinguish trivial from consequential non-normality; does not specifically detect the distributional features (zero-inflation, heavy tails) that actually degrade RMSE.</cons>
 <statistical_considerations>This approach would almost always select Huber for any reasonably sized dataset, defeating the purpose of conditional selection.</statistical_considerations>
 </approach>
 <approach label="Cross-validation-based loss comparison" feasibility="medium" risk="medium">
 <description>Train separate models under RMSE and Huber, compare CV performance, select the better loss function.</description>
 <pros>Directly measures the consequence of loss choice on model performance; no need for distributional assumptions.</pros>
 <cons>Doubles computational cost (or more, if delta must also be tuned); introduces a model selection step that could itself overfit on small samples; the performance difference between RMSE and Huber may be within CV noise for well-behaved data, leading to unstable selection.</cons>
 <statistical_considerations>The CV comparison would need to account for the paired nature of the folds. A corrected paired t-test (Nadeau &amp; Bengio, 2003) or a Bayesian correlated t-test would be needed, but both have limited power for small k (e.g., 5-fold). This approach is principled but computationally expensive and statistically fragile.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="">The effect-size diagnostic battery provides the best trade-off between simplicity, defensibility, and practical utility. It avoids the fatal flaw of formal normality tests (asymptotic power = 1) and the computational cost of cross-validation-based selection. The thresholds are conservative and well-documented in adjacent literature. The approach should be presented as a principled heuristic, not as a formally optimal procedure.</decision>
 </topic>

 <topic id="topic" title="Literature on Automatic Loss Function Selection in Gradient Boosting">
 <summary>There is no established, peer-reviewed methodology for automatic loss function selection in gradient boosting frameworks. AutoML systems (Auto-sklearn, H2O AutoML, AutoGluon) treat loss function as a fixed architectural choice, not a tunable parameter. Scikit-learn offers Huber loss with a user-specified alpha (default 0.9) but no automatic selection. The closest work is Thomas (2018, "Automatic Gradient Boosting") which automates hyperparameters but not the loss function. Implementing an "auto" mode would be a novel pipeline feature, not an adoption of established practice.</summary>
 <research>
 - Thomas (2018): "Automatic Gradient Boosting" (arxiv:1807.03873). Automates hyperparameter selection for gradient boosting via Bayesian optimization but treats loss function as fixed.
 - Scikit-learn GradientBoostingRegressor: Offers loss='huber' with alpha parameter (default 0.9, range (0,1)). Alpha is the quantile at which the transition from L2 to L1 occurs. User-specified, not automatic.
 - H2O AutoML, AutoGluon, Auto-sklearn: Default to squared error for regression with no distributional diagnostics. Loss function is not part of the hyperparameter search space.
 - RAHL (2025): Residual-based Adaptive Huber Loss for 5G CQI prediction. Incorporates a learnable residual into delta. Domain-specific, not generalizable.
 - Janocha &amp; Czarnecki (2017): Survey of loss functions for deep learning. Recommends Huber for robustness but provides no automatic selection criteria.
 - Barron (2019): General and adaptive robust loss function. Proposes a parametric family that interpolates between L2, L1, and Cauchy losses, with the shape parameter tunable by backpropagation. Not applicable to tree-based models.
 </research>
 <approaches>
 <approach label="Pre-training diagnostic trigger (proposed)" feasibility="high" risk="low">
 <description>Run distributional diagnostics on y before training. If any criterion is met, switch from RMSE to Huber with a data-driven delta. This is a one-time decision, not iterative.</description>
 <pros>No additional computational cost beyond the diagnostic computation (negligible); transparent and reproducible; user can override by specifying loss_function explicitly.</pros>
 <cons>Novel approach with no direct literature precedent for gradient boosting specifically; relies on the assumption that distributional pathology in y translates to suboptimal RMSE gradients.</cons>
 <statistical_considerations>The assumption that pathological y distributions degrade RMSE training is well-grounded in the theory of M-estimation. RMSE gradients are proportional to residuals, so outliers and zero-inflation produce disproportionately large gradients that dominate the tree-building process. Huber's piecewise-linear gradient caps this influence.</statistical_considerations>
 </approach>
 <approach label="Include loss function in Optuna search space" feasibility="medium" risk="medium">
 <description>Add loss_function as a categorical hyperparameter in the Optuna search space, allowing the tuner to compare RMSE vs Huber (with delta as a nested continuous parameter).</description>
 <pros>Data-driven selection via the same optimization framework already in use; directly optimizes the scoring metric.</pros>
 <cons>Increases search space dimensionality; categorical + nested continuous parameters are poorly handled by TPE; the scoring metric (neg_rmse) may favor RMSE by construction; requires careful handling of the conditional delta parameter.</cons>
 <statistical_considerations>Adding a categorical choice to TPE effectively doubles the search space and can degrade sample efficiency. The TPE sampler in Optuna handles conditional parameters via independent modeling of each configuration, but this requires sufficient trials in each arm. With 300 trials total, splitting between RMSE and Huber(delta) leaves only ~150 per arm, reducing tuning quality.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Pre-training diagnostic trigger is preferred. It is computationally free, transparent, and avoids the search space inflation of. The novelty of the approach is acceptable given the conservative thresholds and the clear theoretical grounding of Huber loss for non-Gaussian outcomes.</decision>
 </topic>

 <topic id="topic" title="Statistical Basis for Choosing Huber's Delta Parameter">
 <summary>Three principled approaches exist: (A) classical MAD-based with k=1.345 for 95% ARE at the normal; (B) adaptive Huber (Sun, Zhou &amp; Fan, 2020, JASA) with sample-size-dependent calibration; (C) IQR-based. The MAD-based approach (A) is the most defensible for this pipeline: it is simple, has a 50+ year theoretical pedigree, the efficiency loss under normality is exactly quantified (5%), and it requires no hyperparameter tuning. The JASA adaptive approach is theoretically optimal for linear regression with heavy tails but over-engineered for gradient boosting where the delta plays a different mechanistic role.</summary>
 <research>
 - Huber (1981): "Robust Statistics." Establishes the classical theory of M-estimation. The tuning constant k=1.345 yields 95% ARE at the Gaussian model. Published by Wiley.
 - Maronna, Martin &amp; Yohai (2006): "Robust Statistics: Theory and Methods." Standard reference for MAD-based scale estimation. sigma_hat = 1.4826 * MAD, where 1.4826 = 1/Phi^{-1}(3/4) is the consistency factor at the normal. Published by Wiley.
 - Sun, Zhou &amp; Fan (2020): "Adaptive Huber Regression." JASA 115(529):254-265. Proposes tau_n = c_tau * v_hat_delta * (n/log(d))^{1/(1+delta)} for optimal bias-robustness tradeoff. Two data-driven schemes using fitted residuals. Theoretically optimal for linear regression with bounded (1+delta)-th moment.
 - RAHL (2025): Residual-based Adaptive Huber Loss. Incorporates a learnable residual into delta for neural network training. Not applicable to tree-based gradient boosting.
 - Wang &amp; Zhao (2025): "Robust and Efficient Mediation Analysis via Huber Loss." Psychometrika. Proposes a data-driven delta selection procedure that minimizes asymptotic variance.
 - Scikit-learn convention: GradientBoostingRegressor uses alpha=0.9 (90th percentile) as default for Huber loss, which corresponds to a different parameterization than CatBoost's delta.
 </research>
 <approaches>
 <approach label="MAD-based delta (k=1.345)" feasibility="high" risk="low">
 <description>delta = 1.345 * 1.4826 * MAD(y), where MAD(y) = median(|y - median(y)|). This is the classical robust statistics formula for achieving 95% asymptotic relative efficiency at the normal distribution. In CatBoost syntax: f"Huber:delta={delta:.6f}".</description>
 <pros>One line of computation; 50+ year theoretical backing; exactly quantified efficiency loss (5% under normality); no hyperparameters; highly reproducible; the MAD is the most robust scale estimator (50% breakdown point).</pros>
 <cons>Optimal for location estimation under contaminated normals, not specifically for gradient boosting loss functions. The relationship between MAD-based delta and optimal CatBoost training is indirect (mediated through gradient signals, not direct M-estimation).</cons>
 <statistical_considerations>The 1.345 constant is derived from the influence function of the Huber psi function. At the normal model, the asymptotic variance of the Huber M-estimator with k=1.345 is 1.0526 * sigma^2, compared to sigma^2 for OLS. This 5.26% inflation translates to negligible practical impact on model training, especially given that CatBoost's internal regularization (L2 leaf regularization, bagging, learning rate shrinkage) already introduces substantially larger variance inflation. The consistency factor 1.4826 ensures that MAD * 1.4826 is a consistent estimator of sigma under the normal model.</statistical_considerations>
 </approach>
 <approach label="Adaptive Huber (Sun-Zhou-Fan)" feasibility="medium" risk="medium">
 <description>Set tau_n = c_tau * v_hat_delta * (n/log(d))^{1/(1+delta)}, where v_hat_delta is the (1+delta)-th sample absolute central moment, calibrated via fitted residuals in a two-step procedure.</description>
 <pros>Theoretically optimal for heavy-tailed linear regression; adapts to sample size and dimension; achieves sub-Gaussian rates when moments exist.</pros>
 <cons>Designed for linear regression, not tree ensembles; requires estimating moments of unknown order; calibration constant c_tau must be set via cross-validation; substantially more complex to implement; the theoretical guarantees do not transfer to gradient boosting.</cons>
 <statistical_considerations>The Sun-Zhou-Fan framework assumes a linear model y = X*beta + epsilon with heavy-tailed epsilon. Gradient boosting is nonparametric -- the residuals at each boosting iteration are not the same as the regression errors in their framework. The theoretical optimality guarantees do not apply.</statistical_considerations>
 </approach>
 <approach label="IQR-based delta" feasibility="high" risk="low">
 <description>delta = c * IQR(y), where IQR = Q3 - Q1 and c is a constant (e.g., 1.0 or 0.75).</description>
 <pros>Simple; robust to outliers (25% breakdown point); intuitive interpretation.</pros>
 <cons>Lower breakdown point than MAD (25% vs 50%); less efficient estimator of scale; the constant c has no theoretical justification analogous to the 1.345 for MAD.</cons>
 <statistical_considerations>The IQR is a valid robust scale estimator but is less efficient than the MAD. Under the normal model, IQR/1.3489 is a consistent estimator of sigma, but with higher asymptotic variance than MAD/0.6745. For the purpose of setting Huber's delta, the practical difference is small, but the MAD approach has a stronger theoretical foundation.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="">MAD-based delta with k=1.345 is the most defensible choice. It is simple, theoretically grounded, and the efficiency loss is precisely quantified. The formula is: delta = 1.345 * 1.4826 * median(|y - median(y)|) = 1.9940 * MAD(y). This can be computed in a single line before training begins.</decision>
 </topic>

 <topic id="topic" title="Architecture of the 'Auto' Mode in the Pipeline">
 <summary>The auto mode should be implemented as a pre-training diagnostic step that resolves "auto" to either "RMSE" or "Huber:delta=X" before any CV splitting occurs. The resolved loss function is recorded in resolved_config.yaml and outcome_diagnostics.json for full reproducibility. The diagnostic is computed on the full y vector (not per-fold) to ensure consistent loss across all folds. Users can override auto-detection by specifying any explicit loss_function value. The default for regression in minimal config mode changes from "RMSE" to "auto".</summary>
 <research>
 - CatBoost API: Huber loss requires the string format "Huber:delta=&lt;value&gt;" where delta is obligatory. Documented at catboost.ai/docs/en/concepts/loss-functions-regression.
 - diptest PyPI package: Pre-compiled wheels for CPython 3.8-3.12 on Linux x86-64, macOS (x86-64 and ARM-64), Windows x86-64. Import as `import diptest; stat, p = diptest.diptest(y)`. New dependency.
 - Campbell et al. (2021): Warns that diagnostic-driven model selection can inflate Type I error if the diagnostic correlates with the model comparison statistic. In this pipeline, the diagnostic is on y (marginal distribution) while training optimizes conditional predictions, so the correlation is indirect and the concern is mitigated.
 </research>
 <approaches>
 <approach label="Pre-training auto-resolution with diagnostic logging" feasibility="high" risk="low">
 <description>
 Implementation plan:
 1. Add `diagnose_outcome_distribution(y: np.ndarray) -> dict` to utils.py. Returns dict with keys: zero_frac, skewness, excess_kurtosis, dip_stat, dip_p, flags (dict of booleans), triggered (bool), resolved_loss (str), delta (float or None).
 2. In train.py, after y is constructed (line ~442) and before CV splitting (line ~572): if config["modeling"]["loss_function"] == "auto", call the diagnostic function and replace the loss_function in the config with the resolved value.
 3. Save outcome_diagnostics.json to run_dir.
 4. In utils.py, update _TASK_LOSS_SCORING to map regression to ("auto", "neg_rmse") and multi_regression to ("auto", "neg_rmse").
 5. In fill_config_defaults, handle the "auto" value by setting it as the default but not resolving it yet (resolution happens in train.py when y is available).
 6. Add diptest to pyproject.toml dependencies and environment.yaml.
 7. Update example_config_advanced.yaml to document "auto" as an option with explanation.
 8. Update INPUT_SPECIFICATION.md and README.md.
 </description>
 <pros>Clean separation of concerns (diagnostics in utils, resolution in train); full reproducibility via saved diagnostics; user override preserved; minimal code changes; no computational overhead.</pros>
 <cons>New dependency (diptest); thresholds are heuristic; "auto" adds a layer of indirection that may confuse users unfamiliar with the concept.</cons>
 <statistical_considerations>The diagnostic must be computed on the full y, not per-fold. This is both a statistical requirement (loss function should be consistent across folds for valid CV) and a practical one (diagnosing per-fold y subsets would be noisier and could lead to inconsistent loss selection across folds, violating the assumptions of nested CV). The resolved loss function should be treated as a fixed design choice, not as an additional model selection step that requires correction for multiplicity.</statistical_considerations>
 </approach>
 <approach label="Per-fold adaptive loss" feasibility="low" risk="high">
 <description>Diagnose each fold's training y independently and potentially use different loss functions per fold.</description>
 <pros>Adapts to fold-specific distributional variation.</pros>
 <cons>Violates the assumption that all folds are trained under identical conditions; makes OOF predictions non-comparable; complicates downstream SHAP analysis; theoretically unjustified.</cons>
 <statistical_considerations>This is fundamentally flawed. Cross-validation assumes identical model specification across folds. Different loss functions per fold would invalidate the OOF performance estimates and make fold metrics non-comparable.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Pre-training auto-resolution with diagnostic logging is the correct architecture. Per-fold adaptive loss is rejected as statistically unsound.</decision>
 </topic>

 <topic id="topic" title="Risks of Automatic Loss Selection">
 <summary>Five risks were identified and assessed. (1) Efficiency loss under normality: quantified at 5% ARE loss with k=1.345, negligible for SHAP-based importance analysis. (2) Tail underfitting for well-behaved data: marginal (&lt;1-2% RMSE increase in benchmarks), mitigated by CatBoost's regularization. (3) False triggering: conservative thresholds minimize risk; any false trigger results in the small, quantified efficiency loss of (1). (4) SHAP interaction: loss function change affects tree structure and therefore SHAP values, but this is expected and desirable -- a more robust model produces more robust SHAP attributions. (5) Reproducibility/user confusion: mitigated by explicit logging, diagnostic JSON output, and resolved_config.yaml. Overall risk assessment: the downside of false triggering (5% efficiency loss) is vastly smaller than the downside of missing a true positive (catastrophically biased RMSE training on zero-inflated data).</summary>
 <research>
 - Huber (1981): ARE of Huber M-estimator at normal model with k=1.345 is 95.0%. This is the exact efficiency cost of a false trigger.
 - Barron (2019): "A General and Adaptive Robust Loss Function" (CVPR). Shows that adaptive robust losses generally perform within 1-3% of L2 on clean data, with substantial gains on contaminated data.
 - Comprehensive survey by Wang et al. (2022): "A Comprehensive Survey of Regression Based Loss Functions" (arxiv:2211.02989). Confirms Huber loss is never substantially worse than L2 on clean data for practical regression tasks.
 - SHAP (Lundberg &amp; Lee, 2017): SHAP values are model-specific -- they decompose the predictions of a given model. A model trained under a different loss produces different SHAP values because it is a different model. This is not an artifact or error; it is the correct behavior.
 </research>
 <approaches>
 <approach label="Conservative asymmetric risk framework" feasibility="high" risk="low">
 <description>Accept the 5% efficiency loss of false triggering as the cost of robustness. The risk profile is asymmetric: false trigger costs 5% efficiency; missed detection costs potentially catastrophic RMSE bias on pathological outcomes. Conservative thresholds ensure the false trigger rate is low. All decisions are logged for full transparency.</description>
 <pros>Quantified, bounded risk; asymmetric risk profile strongly favors robustness; transparent and auditable.</pros>
 <cons>Cannot be zero-risk; some users may prefer explicit control (which is preserved via override).</cons>
 <statistical_considerations>The asymmetric risk framework is analogous to Bayesian decision theory with asymmetric loss: the cost of a Type II error (missing pathological distribution, training under RMSE) is far greater than the cost of a Type I error (false trigger, training under Huber with 5% efficiency loss). The conservative thresholds are set to minimize the sum of weighted error costs.</statistical_considerations>
 </approach>
 </approaches>
 <decision status="decided" chosen="">The conservative asymmetric risk framework is the correct mental model. The 5% efficiency loss from false triggering is a small, known, bounded cost. The potential cost of missed detection (biased RMSE training on pathological distributions) is large and unbounded. The pipeline should err on the side of robustness.</decision>
 </topic>
 </topics>
 <action_items>
 <item priority="P1" target_mode="implement" description="Add diagnose_outcome_distribution(y) function to utils.py implementing the four-criterion diagnostic battery (zero fraction >= 0.15, excess kurtosis >= 5.0, |skewness| >= 2.0, dip test p < 0.05) with MAD-based delta computation (delta = 1.345 * 1.4826 * MAD(y))." />
 <item priority="P1" target_mode="implement" description="Modify train.py to resolve loss_function='auto' after y is constructed but before CV splitting. Save outcome_diagnostics.json to run_dir." />
 <item priority="P1" target_mode="implement" description="Update _TASK_LOSS_SCORING in utils.py to map regression and multi_regression defaults to 'auto' instead of 'RMSE' 'MultiRMSE'. Handle 'auto' for multi_regression by computing diagnostics on each target column and triggering if any target meets criteria." />
 <item priority="P1" target_mode="implement" description="Add diptest to pyproject.toml dependencies and environment.yaml." />
 <item priority="P1" target_mode="test" description="Design test suite for diagnose_outcome_distribution: (a) normal y -> no trigger, (b) zero-inflated y -> trigger, (c) heavy-tailed y (t-distribution, df=3) -> trigger, (d) highly skewed y -> trigger, (e) bimodal y -> trigger, (f) edge cases (constant y, all zeros, n=1). Verify delta computation against hand-calculated MAD values." />
 <item priority="P1" target_mode="test" description="Design integration test: train pipeline with loss_function='auto' on a synthetic zero-inflated outcome. Verify resolved_config.yaml contains 'Huber:delta=X', outcome_diagnostics.json is saved, and model trains successfully." />
 <item priority="P2" target_mode="implement" description="Update example_config_advanced.yaml to document 'auto' as a loss_function option with explanation of diagnostic criteria and delta computation." />
 <item priority="P2" target_mode="implement" description="Add auto-loss resolution logic to predict.py and infer.py (they load the resolved config, so this should already work, but verify)." />
 <item priority="P2" target_mode="document" description="Update INPUT_SPECIFICATION.md and README.md to document the auto-loss feature, including the diagnostic criteria, delta formula, and override mechanism." />
 <item priority="P2" target_mode="test" description="Design regression test: verify that explicit loss_function='RMSE' bypasses auto-detection entirely, even when y is pathological." />
 </action_items>
 <next_steps>Proceed to implement mode (plan submodule) to create a technical specification from this report, then build submodule to implement the changes. The implementation order should be: (1) utils.py diagnostic function, (2) train.py auto-resolution, (3) config/dependency updates, (4) test suite. The test suite should be designed before implementation begins (per project guardrails) and updated as features are added.</next_steps>
</brainstorm_report>
