# AI Development Log

This document discloses the use of AI-assisted development tools in the creation of the **boost-shap-gii** analysis pipeline, in accordance with emerging best practices for transparency in scientific software development.

---

## 1. Purpose

This document provides a structured disclosure of AI tool usage during the development of the boost-shap-gii pipeline. The disclosure follows the AI Disclosure (AID) Framework (Weaver, 2025) and adheres to recommendations for responsible AI use in scientific computing (Bridgeford et al., 2025; Nussberger et al., 2024; Jamieson et al., 2024). The intent is to ensure that reviewers, collaborators, and end users can assess the nature and extent of AI involvement in the development process.

## 2. Scope

AI assistance was utilized for **analysis pipeline development**, encompassing:

- Code architecture and design
- Statistical methodology review and validation
- Implementation of pipeline modules
- Test suite development and validation
- Documentation updates and refinement

AI was **not** used for:

- Running analyses on real data
- Interpreting scientific results from pipeline outputs
- Making domain-specific methodological decisions (e.g., selection of covariates, outcome definitions, or study-specific analytical choices)

The boost-shap-gii pipeline is a general-purpose tool for gradient boosting with SHAP-based global importance indices. Its application to specific datasets and scientific questions is conducted independently by the researcher.

## 3. Tools Used

Development utilized **Claude Code** (Anthropic), employing two model tiers:

| Model | Tasks |
|-------|-------|
| Claude Opus 4 | Critical review of statistical methods, brainstorming sessions, code quality audits, risk assessment, and architectural decisions |
| Claude Sonnet 4 | Code scaffolding under direction, test implementation support, documentation updates, and file management |

This dual-model approach ensured that analytical depth (Opus) was applied to decisions with statistical or methodological consequences, while implementation efficiency (Sonnet) was used for well-specified coding tasks under explicit human direction.

## 4. Development Workflow

The pipeline was developed through an iterative, mode-based workflow with the following stages:

1. **Brainstorm** -- Structured discussion of design decisions, trade-offs, and alternative approaches. Every brainstorm session produced a report with explicit decision records (accepted, rejected, deferred).

2. **Critical Review (CR)** -- Formal review of the codebase for statistical correctness, robustness, reproducibility, and defensive coding practices. Each finding was classified by severity (P0/P1/P2) and required explicit human triage (accept, reject, or modify).

3. **Implement (Plan + Build)** -- Implementation proceeded in two sub-phases: (a) a technical specification mapping each approved change to specific code modifications with risk assessment, and (b) execution of the specification. All plans required human approval before code generation began.

4. **Test** -- Comprehensive test suite development (705 tests across 19 test files) covering unit, integration, edge-case, and statistical invariant tests. Tests were designed prior to implementation where feasible (test-first methodology).

5. **Clean** -- Code quality review for consistency, style, and maintainability.

6. **Document** -- Updating and maintaining user-facing documentation (README.md) and machine-readable technical specifications (INPUT_SPECIFICATION.md).

Key properties of this workflow:

- All decisions required **explicit human approval** before implementation.
- The pipeline was developed with a **test-first** approach; 705 tests validate statistical correctness, edge-case handling, and integration behavior.
- Every statistical and algorithmic choice was subjected to **formal critical review**, with findings documented and triaged individually.

## 5. Human Oversight

The researcher maintained full oversight and decision authority throughout the development process:

- **(a)** Defined all statistical methodology, including the GII formulation (magnitude and variability components), SHAP interpretation strategy, Boruta-inspired shadow feature calibration, bootstrap confidence interval methodology, and permutation testing procedures.

- **(b)** Triaged every critical review finding with explicit accept/reject/modify decisions, documented in brainstorm reports with rationale for each determination.

- **(c)** Approved all implementation plans (technical specifications) before any code generation was executed.

- **(d)** Validated all test results and ensured test coverage aligned with the statistical guarantees required by the pipeline.

- **(e)** Made all domain-specific decisions, including the choice of CatBoost as the gradient boosting framework, the design of the config-driven architecture, the selection of cross-validation strategies, and the formulation of the GII as a composite importance metric.

## 6. Audit Trail

A complete record of the structured development process is available in the `.aid/reports/` directory within this repository. The audit trail includes:

- **Brainstorm reports** -- Records of design discussions, decision rationale, and trade-off analyses.
- **Critical review reports** -- Formal findings with severity classifications and human triage decisions.
- **Implementation plans** -- Technical specifications mapping approved changes to code modifications.
- **Implementation build reports** -- Records of executed changes with deviation notes.
- **Test reports** -- Test suite results and coverage summaries.
- **Code quality reviews** -- Clean-pass reports on style and consistency.
- **Documentation reports** -- Records of documentation updates and revisions.

The project-level configuration file used to guide AI interactions is preserved as `.aid/project_claude.md`.

Raw session transcripts are excluded for privacy reasons. The structured reports above capture all substantive technical decisions, rationale, and implementation details.

## 7. Development Session Log

### Session 2026-04-24 -- indiv_reports feature and A1 patch (bundled Option B release)

**Date:** 2026-04-24

**Session scope:**

- New module `src/boost_shap_gii/indiv_reports.py`: per-individual SHAP reports with coupled-bootstrap confidence intervals (CIs). Implements Option E with a shared bootstrap sample per iteration across all K fold refits, producing estimand-matched CIs for both training individuals (OOF single-model SHAP) and inference individuals (ensemble-mean SHAP).
- Bundled A1 patch (pandas-3.0 Categorical `fillna` compatibility): applied to `train.py`, `predict.py`, and `infer.py`; 19 regression tests added (`tests/test_categorical_fillna_bugfix.py`).
- Config schema additions: six new `shap.*` keys (`indiv_ci_nboot`, `indiv_scaling_mode`, `indiv_scaling_value`, `compute_global_on_inference`) and six new `plot.*` keys (`outcome_max`, `negate_shap`, `gii_y_label`, `gii_y_sublabel`, `indiv_y_label`, `indiv_y_sublabel`).
- New training artifact: `train_outcome_stats.json` (written unconditionally by `train.py` for regression tasks; empty stats dict for classification tasks).
- `predict.py` update: orchestrates bootstrap-refit cache and invokes `generate_indiv_reports` for training-mode individual reports.
- `infer.py` update: invokes `generate_indiv_reports` for inference-mode individual reports; default behavior change (see Breaking Changes).
- `plot.R` extension: per-individual dot-plus-whisker plot rendering with signed-rank x-ordering; CLI simplified to CONFIG_PATH + optional RUN_DIR (positional args 2-4 removed).
- `check_env.py` / `cli.py`: environment preflight (`run_preflight()`) elevated to a mandatory gate called at every CLI entry point.
- Documentation updates: `INPUT_SPECIFICATION.md` Section 10 (per-individual reports), `README.md` usage notes, `AID_LOG.md` (this entry).

**LLM tools used:**

- Claude Opus 4.7 (Anthropic): used for brainstorm sessions, implementation plan structuring support, algorithmic design review, plan-discipline verification.
- Claude Sonnet 4.6 (Anthropic): used for code scaffolding under direction, config edits, documentation updates, file management.

LLM tool use carries no claim to scientific credit under project policy. All algorithmic decisions, scope determinations, and plan approvals were made by the researcher prior to any code-generation step.

**Key algorithmic decisions (researcher-approved):**

- *Option E coupled bootstrap:* per iteration b, one shared bootstrap sample s_b is drawn from the full training set; all K fold refits for iteration b use s_b. This binds fold refits per iteration so that ensemble replicates capture between-fold covariance correctly.
- *Estimand-matched point and CI:* the point estimate for each individual is the deployed-product SHAP (OOF single-model for training individuals assigned to fold k_i; ensemble-mean across K models for inference individuals). Bootstrap refits are consumed only for CI bounds, not for the point estimate, preventing the systematic point-outside-CI risk that arises when single-model variance is applied to an ensemble estimator.
- *OOB floor = 50:* training individuals whose OOB count is below 50 emit NaN CI bounds with `oob_count` preserved in the output schema. Below-floor plots render point estimates only with an in-plot caption.
- *Minimum recommended B = 2500; peer-review runs B = 5000:* at B = 2500 and K = 10, inference CIs are Efron-tier (B >= 1000 effective); training OOB CIs are near-Efron (approximately 0.368 x 2500 = 920 effective). At B = 5000, both sides clear the Efron threshold (Efron & Tibshirani, 1993).
- *Fold-assignment reconstruction:* training-individual fold assignments are reconstructed deterministically at predict-time via `get_cv_splitter(config, y_for_split)` (matching `predict.py`'s existing reconstruction pattern) rather than persisted as a new artifact by `train.py`.
- *Path-dependent SHAP retained:* raw path-dependent SHAP interactions are used for the individual interaction reports, consistent with the global GII computation in `shap_utils.py`.
- *Per-individual inspection framing:* the feature is framed as a hypothesis-generating tool for individual-level SHAP inspection, not a diagnostic or prescriptive output.
- *Three-mode scaling:* `raw` (unscaled), `sd` (divide by training-outcome SD from `train_outcome_stats.json`; regression only), `custom_value` (user-supplied divisor; any task type).
- *Dot-plus-whisker plot format with signed-rank x-ordering:* per-individual plots render features ordered by signed SHAP value, with whisker extent encoding the bootstrap CI range.

**Test metrics:**

- Pre-session: 461 tests across 16 test files (458 passing; 3 failing on missing `nanoparquet` R package, environment-only).
- Post-build target (before /test phase): 461 + tests for `indiv_reports.py` and supporting changes. Exact post-test count populated in end-session report after /test phase completes.

**Breaking changes:**

1. `infer.py` no longer emits population-level `shap_analysis/` by default. Users who require global SHAP on inference data must set `shap.compute_global_on_inference: true` in their config. Rationale: global GII on small inference sets produces degenerate results; opt-in is safer than opt-out.
2. `plot.R` CLI simplified: positional args 2-4 (`OUTCOME_RANGE`, `NEGATE_SHAP`, `Y_AXIS_LABEL`) removed. These values are now read from the config file. Users invoking `plot.R` directly with the old 4-arg signature will see a fail-loud error from the script's argument-count check.
3. All configs must include the six new `shap.*` keys and six new `plot.*` keys. Existing configs without them will fail `validate_indiv_reports_config()` / `validate_plot_config()` with precise error messages (err-on-kill per project philosophy).

**Audit trail references (.aid/reports/):**

- Brainstorm: `boost-shap-gii_brainstorm_20260423_191951.md` (10 topics locked; 5 decisions deferred; Decision 5 HP-source lock = Option E)
- Implementation plan: `boost-shap-gii_implement_plan_20260424_084001.md` (12 changes, 15 plan-discipline resolutions)
- Implementation build reports: `boost-shap-gii_implement_build_20260424_*.md` (one per agent dispatch group)

---

### Session 2026-05-07 -- CR-remediation cycle (critical review + remediation implementation)

**Date:** 2026-05-07

**Session scope:**

- Full independent critical review (CR) of the entire pipeline. 25 findings produced (5 critical, 9 major, 8 minor, 3 note); overall assessment: needs_revision; publication release blocked pending closure of the critical findings.
- CR-remediation implementation: 18 discrete changes (reviewed and approved by the researcher) were applied across the pipeline codebase to address all 25 CR findings. Key methodological changes include:
  - *GII decision-theoretic framing*: the GII formula is documented as a geometric mean of two utility components: M (magnitude) and V (variability). A feature must score positively on BOTH dimensions for the geometric mean to be nonzero, preventing high-magnitude features with no dose-response variation from reaching significance. The V component is anchored conceptually to Hill (1910) dose-response theory and visualized via individual conditional expectation curves (Goldstein et al., 2015). README.md and INPUT_SPECIFICATION.md updated; `shap_utils.py` docstrings updated.
  - *V-component sample standard deviation*: all six V-computation sites in `shap_utils.py` converted from population SD (ddof=0) to sample SD (ddof=1; Fisher, 1925) for unbiased variance estimation. Len < 2 guard added at each site (returns NaN for degenerate resamples).
  - *Shadow model leakage closure*: Phase 2 shadow model training removed `eval_set` and `early_stopping_rounds` (Kursa & Rudnicki, 2010). Prior implementation allowed shadow-model iteration-count selection to be influenced by validation outcomes, biasing the noise calibration baseline. Fixed iteration count `tuned_iters * 2` is retained without early stopping.
  - *Three independent BH-FDR calls*: exceedance p-values for M, V, and GII now each receive an independent Benjamini-Hochberg FDR correction (Benjamini & Hochberg, 1995). Prior implementation pooled all three into a single FDR call, inflating false discovery rate when the three families are differentially powered.
  - *Degenerate bootstrap CI fallback*: `compute_bootstrap_ci` returns `(base_score, NaN, NaN)` with a `RuntimeWarning` when `n_boot_effective = 0` (all bootstrap iterations dropped). Prior implementation returned `(base_score, base_score, base_score)`, which could be mistaken for a valid zero-width CI.
  - *Two-tier nominal unseen validation*: `_validate_nominal_unseen` added to `utils.py`. Tier 1 raises `ValueError` when > 50% of unique unseen values are absent from the nominal codebook (systematic naming mismatch). Tier 2 emits `UserWarning` when > 10% of observations encounter unseen levels (data quality issue).
- New test file `tests/test_build_20260507.py` (30 tests across 8 classes) covering seven previously unexercised code paths from the remediation cycle.
- Post-remediation test suite: 625 tests across 17 test files; 624 passing (1 skipped, psutil not yet declared as a hard dependency at this point), 0 failing.

**LLM tools used:**

- Claude Opus 4.7 (Anthropic): used for critical review, brainstorm sessions, implementation plan structuring support, disposition adjudication.
- Claude Sonnet 4.6 (Anthropic): used for code scaffolding under direction, test implementation support, documentation updates, file management.

LLM tool use carries no claim to scientific credit under project policy. All algorithmic decisions, scope determinations, and plan approvals were made by the researcher prior to any code-generation step.

**Key algorithmic decisions (researcher-approved):**

- *GII as a geometric-mean composite*: the geometric-mean structure requires both M and V to be meaningfully positive. This decision-theoretic framing was approved by the researcher as the canonical formulation and is now reflected in README.md, INPUT_SPECIFICATION.md Section 3, and `shap_utils.py` docstrings.
- *BH-FDR with three independent families*: pooled FDR was identified as a critical finding. The researcher approved separation into three independent BH calls (one per component family) to prevent cross-family FDR inflation.
- *Shadow leakage closure*: the researcher approved removing `eval_set` from the Phase 2 shadow model. The doubled-ceiling fixed-iteration approach is retained as the iteration budget.
- *Backwards-compatibility shim retention*: a `nominal_codebooks`-absent fallback in `predict.py` and `infer.py` (for models trained before this release) was explicitly reviewed and approved by the researcher. The shim is a narrow compatibility layer, not a behavioral default.

**Test metrics:**

- Pre-remediation: 461 tests across 16 test files (458 passing; 3 failing on missing `nanoparquet` R package, environment-only).
- Post-remediation: 625 tests across 17 test files; 624 passing, 1 skipped (psutil not yet declared as a hard dependency at this point), 0 failing.

**Audit trail references (.aid/reports/):**

- Critical review: `boost-shap-gii_cr_20260424_192405.md` (25 findings with severity classifications and researcher triage decisions)
- Implementation plan: `boost-shap-gii_implement_plan_20260507_104713.md` (18-change technical specification)
- Implementation build report: `boost-shap-gii_implement_build_20260507_124302.md`
- Test reports: `boost-shap-gii_test_20260507_130304.md`, `boost-shap-gii_test_20260508_090928.md`

---

### Session 2026-08-12 -- Post-release maintenance and aggregate SHAP feature

**Date:** 2026-08-12

**Session scope:**

- Full independent critical review (CR) of the pipeline (blank-slate, not based on prior session memory). Produced 3 minor findings; overall assessment: defensible; 0 critical, 0 major.
- Brainstorm session to validate proposed fixes and check for interactions with the existing pipeline. All three solutions confirmed sound with zero interaction risk.
- Implementation of aggregate SHAP (group-level GII) feature and 3 minor fixes:
  - *Aggregate SHAP feature*: new `aggregate_shap` config block enabling post-hoc group-level SHAP analysis. Sums member SHAP values within user-defined feature groups to compute group-level M, V, and GII. Produces singleton aggregates, within-group interaction aggregates, between-group interaction aggregates, and group-by-ungrouped interaction aggregates. Shadow noise calibration uses block-permutation (Au et al. 2022, S1) to preserve within-group correlation. Implemented across `utils.py` (`_block_permute_shadow`), `train.py` (`_validate_aggregate_shap`, block-permute integration), `shap_utils.py` (`_aggregate_effects`, `_is_aggregate_effect`, `is_aggregate` output flag, X_micro fallback for aggregate columns), and `infer.py` (`train_dir` context key separation). Config template updated in `example_config_advanced.yaml`. CatBoost refit HP strategy changed from allowlist to blocklist in `indiv_reports.py` (`_extract_user_level_params`).
  - *Docstring correction for `_nan_safe_fdr`*: the docstring incorrectly claimed NaN p-values were "excluded from the BH denominator." The code replaces them with 1.0 conservative placeholders that remain in the denominator (slightly more conservative than documented). Docstring corrected to describe the actual behavior. No code logic changed.
  - *Pre-flight CatBoost refit probe*: new `_probe_and_strip_refit_params` helper in `indiv_reports.py`. Trial-constructs a CatBoost model once before the bootstrap-of-CV loop to discover any internal-only parameters not covered by the static blocklist. Discovered params are stripped from all fold HP dicts and a `RuntimeWarning` is emitted. Defends against future CatBoost version upgrades that may add new internal-only keys to `get_all_params()`.
  - *Inference-mode X_stacked shadow feature correction*: replaced fold-0 tiling (`pd.concat([chunks_X[0]] * n_folds)`) with full concatenation (`pd.concat(chunks_X, ignore_index=True)`) in `shap_utils.py`. Preserves each fold's independent shadow permutation for correct shadow V computation in the exceedance test. Prior behavior used fold 0's shadow features for all K copies, creating a minor anti-conservative bias in the shadow noise distribution.
- Documentation: README.md updated with Aggregate SHAP section; INPUT_SPECIFICATION.md updated with config reference, aggregation algorithm, block-permutation details, `is_aggregate` column, and edge cases.
- Test suite: 5 new tests for the pre-flight probe helper; 1 re-expressed test (strengthened postcondition) for the X_stacked fix. Post-session: 658 tests across 17 test files; 658 passing, 0 failing.

**LLM tools used:**

- Claude Opus 4.6 (Anthropic): used for critical review, brainstorm, implementation plan structuring support.
- Claude Sonnet 4.6 (Anthropic): used for code scaffolding under direction, test execution support.
- Claude Sonnet 5 (Anthropic): used for test suite execution support and test run coordination.

LLM tool use carries no claim to scientific credit under project policy. All decisions were made by the researcher prior to any code-generation step.

**Key decisions (researcher-approved):**

- *Aggregate SHAP design*: the researcher designed the aggregate_shap config block, the block-permutation strategy (Au et al. 2022, S1), the four aggregate effect types (singleton, within-group, between-group, group-by-ungrouped), validation rules (disjoint membership, no nominal features, name-collision prohibition), and the `is_aggregate` output flag.
- *Allowlist-to-blocklist refit strategy*: the researcher approved replacing the CatBoost HP allowlist with a blocklist approach (all params pass except known internal-only keys), combined with the pre-flight probe fallback for unknown rejected keys.
- *Docstring-only fix for `_nan_safe_fdr`*: the researcher chose to correct the documentation rather than implement true BH denominator exclusion, preserving the slightly conservative behavior against which the simulation study baseline was run.
- *Pre-flight probe design*: the researcher approved Option B (single trial construction before the bootstrap loop, capped at 5 retries) over alternatives (try/except inside the hot loop, or CatBoost version pinning).
- *X_stacked tiling fix*: the researcher approved the one-line change after a brainstorm session verified zero interaction risk across all seven downstream consumers.

**Test metrics:**

- Pre-session: 653 tests across 17 test files (653 passing, 0 failing).
- Post-session: 658 tests across 17 test files (658 passing, 0 failing).

**Audit trail references (.aid/reports/):**

- Critical review: `boost-shap-gii_cr_20260812_142729.md`
- Brainstorm: `boost-shap-gii_brainstorm_20260812_144151.md`
- Implementation plan: `boost-shap-gii_implement_plan_20260812_144410.md`
- Implementation build: `boost-shap-gii_implement_build_20260812_144717.md`
- Test report: `boost-shap-gii_test_20260812_150327.md`

---

### Session 2026-08-13 -- CV strategy feature and fold-assignment artifact

**Date:** 2026-08-13

**Session scope:**

- Brainstorm session (4 topics) to design the CV strategy feature: literal `cv_strategy` splitter selector (T1: "uniform", "stratified", "group"), inner CV repeats via `n_inner_repeats` (T2), group column exclusion and validation (T3), and fold-assignment persistence as `fold_assignments.json` artifact (T5, replacing the prior splitter-reconstruction approach).
- Implementation plan and build (7 changes, C1-C7): added `get_cv_splitter()` refactor with three-strategy dispatch, `_StratifiedRegressionKFold` (quantile binning with `pd.qcut`/`pd.cut` fallback), `_GroupKFoldWrapper`, `_RepeatedGroupKFold`, `validate_cv_config()`, and `fill_config_defaults()` extensions in `utils.py` (C1); `train.py` group column exclusion, `fold_assignments.json` persistence, unbalanced-fold and cost warnings (C2); `predict.py` fold-assignment artifact loading replacing splitter reconstruction (C3); `shap_utils.py` fold-assignment artifact loading (C4); `indiv_reports.py` fold-assignment artifact loading and group-strategy bootstrap fallback to plain KFold (C5); `example_config_advanced.yaml` new keys (C6); config validation integration (C7).
- Test design and execution: 9 pre-design failures all dispositioned as obsolete-test (zero product bugs); 42 new tests in `test_cv_strategy.py`; 9 re-expressed tests across `test_implementation_changes.py`, `test_indiv_reports_unit.py`, and `test_train.py`. Post-design suite: 705/705 passing.
- Full codebase and documentation clean review (7 findings: 0 critical, 3 major, 2 minor, 2 style/note); all actionable findings are documentation drift from the CV strategy feature (F1-F5), routed to `/document`.
- Documentation updates: INPUT_SPECIFICATION.md (Stage 0 dependency list corrected, Stage 4 CV description rewritten, Stage 5 fold-assignment artifact loading, config table rows added, Section 10 fold reconstruction paragraph replaced), README.md (new Cross-Validation Strategy section with Group CV and Inner CV Repeats subsections), `indiv_reports.py` module docstring updated.

**LLM tools used:**

- Claude Opus 4.6 (Anthropic): used for brainstorm, implementation plan, clean review, documentation coordination.
- Claude Sonnet 4.6 (Anthropic): used for code scaffolding under direction, documentation edits, file management.
- Claude Sonnet 5 (Anthropic): used for test design and execution.

LLM tool use carries no claim to scientific credit under project policy. All decisions were made by the researcher prior to any code-generation step.

**Key decisions (researcher-approved):**

- *Literal cv_strategy selector*: the researcher directed "We don't want this option to do anything automatically." Each value is a literal splitter selector regardless of task type, with zero cardinality-dependent or task-dependent branching. This is an explicit, approved backward-compatibility break (prior behavior auto-stratified classification when y had < 20 unique values).
- *Fold-assignment persistence*: the researcher approved persisting `fold_assignments.json` (integer fold-index array) as an artifact from `train.py`, replacing the prior approach of reconstructing fold assignments at predict-time via `get_cv_splitter()`. This eliminates dependence on data identity, sklearn version determinism, and stratification replication.
- *Group-strategy bootstrap fallback*: when `cv_strategy="group"`, bootstrap resampling breaks group structure (resampled indices no longer correspond to original group labels). The researcher approved falling back to plain `KFold` for bootstrap inner splits in `indiv_reports.py`, with a logged warning.
- *_RepeatedGroupKFold design*: no sklearn equivalent exists. The researcher approved a custom implementation that permutes group-to-fold mapping per repeat with a seeded RNG while preserving group integrity (all observations sharing a group label remain in the same fold).
- *Quantile-binned regression stratification*: the researcher approved `_StratifiedRegressionKFold`, which bins continuous y via `pd.qcut` (with `pd.cut` fallback for tied values) and delegates to `StratifiedKFold`.

**Test metrics:**

- Pre-session: 658 tests across 17 test files (658 passing, 0 failing).
- Post-session: 705 tests across 19 test files (705 passing, 0 failing).

**Audit trail references (.aid/reports/):**

- Brainstorm: `boost-shap-gii_brainstorm_20260813_171500.md`
- Implementation plan: `boost-shap-gii_implement_plan_20260813_183000.md`, `boost-shap-gii_implement_plan_20260813_220028.md`
- Implementation build: `boost-shap-gii_implement_build_20260813_223500.md`
- Test report: `boost-shap-gii_test_20260813_231500.md`
- Clean report: `boost-shap-gii_clean_20260814_014500.md`

---

### Session 2026-08-14 -- CV strategy hardening: cluster bootstrap, fdr_method, and bug fix

**Date:** 2026-08-14

**Session scope:**

- Critical review (CR) of the CV strategy feature from Session 2026-08-13 produced 3 findings routed to brainstorm: population-level SHAP bootstrap should use cluster-aware resampling when group CV is active (cluster bootstrap; Cameron et al. 2008), the hardcoded BH-FDR should be configurable (fdr_method key), and several minor hardening items (Graham 1966 scheduling, cardinality validation, diagnostic warnings).
- Brainstorm session locked all design decisions and produced 7 action items for implementation.
- Implementation plan and build (7 changes, C1-C7): cluster bootstrap for group CV strategy with i.i.d. fallback at n < 20 (Ukoumunne et al. 2003) and variable-length list-of-arrays resampling for unequal group sizes (C1); Graham (1966) greedy list scheduling in `_RepeatedGroupKFold` replacing round-robin (C2); `validate_cv_config` cardinality check for n_unique_groups vs cv_folds/inner_cv_folds (C3); redundant splitter recreation removal in train.py (C4); configurable `fdr_method` key with "bh"/"by" values (C5); inner-groups diagnostic warning in `run_optuna_tuning` (C6); `stratify_labels_for_regression` bin-count warning (C7).
- Test cycle (first pass): 19 new tests in `test_build_20260814.py`, 1 re-expressed test in `test_train.py`. 9 failures identified as product bug: the i.i.d. fallback guard's reassignment of `cluster_ids` to `None` leaked into the microdata deduplication branch, causing a length-mismatch crash when fallback fired.
- Bug fix implementation: introduced `original_cluster_ids` variable in `_run_bootstrap_pipeline` (captured before the fallback guard), used for the microdata deduplication conditional. Re-expressed the obsolete `test_unequal_cluster_sizes_raises` test (now `test_unequal_cluster_sizes_completes_with_fallback`).
- Test cycle (verification pass): 1 new test (`TestClusterBootstrapMicrodataNoFallback`, covering the non-fallback branch at N >= 20). Final suite: 725/725 passing across 19 test files.
- Documentation updates: README.md (Group CV section, cluster bootstrap, fdr_method), INPUT_SPECIFICATION.md (fdr_method config key, cluster bootstrap section, Graham 1966 scheduling, cardinality validation, FDR references).

**LLM tools used:**

- Claude Opus 4.6 (Anthropic): used for critical review, brainstorm, implementation plan structuring.
- Claude Sonnet 4.6 (Anthropic): used for code scaffolding under direction, test execution.
- Claude Sonnet 5 (Anthropic): used for test design and coordination, code edits under direction, documentation coordination.

LLM tool use carries no claim to scientific credit under project policy. All decisions were made by the researcher prior to any code-generation step.

**Key decisions (researcher-approved):**

- *Cluster bootstrap for group CV*: the researcher approved cluster-aware resampling at the population level (resample entire groups, expand to member rows) when `cv_strategy="group"` is active in non-inference mode. This preserves within-group correlation for correctly calibrated significance tests (Cameron, Gelbach, & Miller, 2008).
- *i.i.d. fallback threshold at n < 20*: the researcher approved falling back to i.i.d. bootstrap with a RuntimeWarning when the number of unique groups is below 20, following Ukoumunne et al. (2003). Microdata deduplication uses the original cluster structure regardless of fallback status (the K-replication structure is independent of the bootstrap method).
- *Graham (1966) list scheduling*: the researcher approved replacing round-robin group-to-fold assignment with greedy-in-random-order scheduling to minimize fold-size imbalance under unequal group sizes.
- *Configurable fdr_method*: the researcher approved exposing the FDR correction method as a config key (`"bh"` for Benjamini-Hochberg, `"by"` for Benjamini-Yekutieli), defaulting to "bh" to preserve backward compatibility.
- *Variable-length cluster resampling*: the researcher approved storing bootstrap indices as a list of arrays (rather than a fixed 2-D ndarray) to support unequal group sizes without padding or truncation.

**Test metrics:**

- Pre-session: 705 tests across 19 test files (705 passing, 0 failing).
- Post-session: 725 tests across 19 test files (725 passing, 0 failing).

**Audit trail references (.aid/reports/):**

- Critical review: `boost-shap-gii_cr_20260814_133000.md`
- Brainstorm: `boost-shap-gii_brainstorm_20260814_135928.md`
- Implementation plan: `boost-shap-gii_implement_plan_20260814_141500.md`, `boost-shap-gii_implement_plan_20260814_155000.md`
- Implementation build: `boost-shap-gii_implement_build_20260814_142500.md`, `boost-shap-gii_implement_build_20260814_155500.md`
- Test reports: `boost-shap-gii_test_20260814_121500.md`, `boost-shap-gii_test_20260814_150500.md`, `boost-shap-gii_test_20260814_160500.md`

---

### Session 2026-08-16 -- psutil dependency fix and spline diagnostic verbosity

**Session scope:**

- A production pipeline run on an institutional HPC cluster revealed two issues: (1) `psutil`, required by the `indiv_reports.py` memory guard (introduced in v1.2.0), was never declared as a dependency in `pyproject.toml`, `environment.yaml`, or `check_env.py`, causing an `ImportError` at the individual reports stage; (2) the per-iteration spline degree downgrade warnings produced thousands of identical `[SHAP] Spline degree downgraded...` messages for low-cardinality features, obscuring earlier pipeline output.

**Changes implemented:**

- *psutil hard dependency*: `psutil` added to `pyproject.toml` dependencies, `environment.yaml` dependencies, and `check_env.py` `PYTHON_DEPS` list. The `boost-shap-gii check-env` preflight now validates psutil availability.
- *Spline downgrade diagnostic*: the per-call print in `_get_adaptive_knots_and_degree` was removed. A new function `_diagnose_spline_downgrades` inspects each non-nominal feature's unique interior knot count in the full dataset and emits a single summary at the start of the SHAP pipeline, listing all features whose spline degree will be downgraded and noting that their interactions are also affected.

**LLM tools used:**

- Claude Opus 4.6 (Anthropic): used for implementation planning support and documentation updates.
- Claude Sonnet 4.6 (Anthropic): used for code scaffolding under direction and test implementation support.

LLM tool use carries no claim to scientific credit under project policy. All scope decisions and remediation strategy were determined by the researcher.

**Test metrics:**

- Pre-design: 725/725 passing.
- Post-design: 733/733 passing (+8 new tests covering `_diagnose_spline_downgrades`, psutil dependency pinning in `pyproject.toml`, `environment.yaml`, and `check_env.py`).

**Audit trail references (.aid/reports/):**

- Implementation plan: `boost-shap-gii_implement_plan_20260816_120000.md`
- Implementation build: `boost-shap-gii_implement_build_20260816_121500.md`
- Test report: `boost-shap-gii_test_20260816_124800.md`
- Document report: `boost-shap-gii_document_20260816_130000.md`

---

## 8. Version and Release Notes

### Post-1.4.0 maintenance -- 2026-08-16

Maintenance commit (no version bump). Adds `psutil` as a declared hard dependency (was used since v1.2.0 but never added to `pyproject.toml`, `environment.yaml`, or `check_env.py`). Replaces per-iteration spline degree downgrade warnings with a single upfront diagnostic block at the start of the SHAP pipeline.

---

### Version 1.4.0 -- 2026-08-14 (CV strategy hardening and cluster bootstrap)

This release adds group-aware cross-validation infrastructure, cluster bootstrap for SHAP significance testing, configurable FDR correction, and a product bug fix.

- **Group CV with Graham (1966) scheduling**: new `_RepeatedGroupKFold` splitter assigns groups to folds via greedy list scheduling (Graham, 1966), minimizing fold-size imbalance under unequal group sizes. A warning is emitted when the max/min fold-size ratio exceeds 2.0. Group-cardinality validation enforces that the number of unique groups is at least `cv_folds` (and at least `inner_cv_folds` when tuning is configured).
- **Cluster bootstrap for group CV SHAP significance**: when the group CV strategy is active, population-level bootstrap significance testing uses cluster-aware resampling (resample entire groups with replacement, then expand to member rows) to preserve within-group correlation (Cameron, Gelbach, & Miller, 2008). Variable-length bootstrap indices support unequal group sizes.
- **i.i.d. fallback guard**: when the number of unique groups is below 20, cluster bootstrap falls back to i.i.d. resampling with a `RuntimeWarning` (Ukoumunne, Gulliford, Chinn, Sterne, & Burney, 2003).
- **Configurable FDR correction method**: new `shap.bootstrapping.fdr_method` config key supports `"bh"` (Benjamini-Hochberg, 1995; default) and `"by"` (Benjamini-Yekutieli, 2001) for multiple-comparison correction.
- **`original_cluster_ids` bug fix**: the i.i.d. fallback guard previously overwrote `cluster_ids` to `None`, which caused the microdata deduplication branch to skip group-averaging on K-replicated inference data. The fix captures `original_cluster_ids` before the fallback guard; microdata deduplication uses the original value since K-replication structure is independent of bootstrap method.
- **Documentation**: README.md and INPUT_SPECIFICATION.md updated with cluster bootstrap, fdr_method, Graham scheduling, and cardinality validation documentation.
- **Tests**: 67 new tests across 4 test files; 1 re-expressed test (strengthened postcondition). Post-session: 725/725 passing.

---

### Version 1.3.0 -- 2026-08-12 (aggregate SHAP feature and post-release maintenance)

This release adds user-configurable group-level SHAP analysis and resolves three minor post-release findings from independent critical review.

- **Aggregate SHAP (group-level GII)**: new `aggregate_shap` config block enables post-hoc group-level importance analysis. Users define feature groups; the pipeline computes group-level M, V, and GII via SHAP additivity. Shadow noise calibration uses block-permutation (Au et al. 2022, S1) to preserve within-group correlation. Four aggregate effect types produced: singleton, within-group interaction, between-group interaction, and group-by-ungrouped interaction. Results flagged with `is_aggregate = True` in `shap_stats_global.csv`.
- **Allowlist-to-blocklist CatBoost refit strategy**: `indiv_reports.py` HP handling changed from an allowlist (explicit list of known-good params) to a blocklist (all params pass except known internal-only keys), with a pre-flight probe (`_probe_and_strip_refit_params`) as a safety net for undiscovered internal params.
- **`_nan_safe_fdr` docstring correction**: docstring updated to describe actual NaN-to-1.0 conservative-placeholder behavior (code unchanged).
- **Inference-mode X_stacked shadow feature correction**: fold-0 tiling replaced with full concatenation to preserve per-fold independent shadow permutations for correct shadow V computation.
- **Documentation**: README.md and INPUT_SPECIFICATION.md updated with aggregate SHAP config reference, algorithmic details, and edge cases.
- **Tests**: 5 new tests (pre-flight probe helper); 1 re-expressed test (X_stacked, strengthened postcondition). Post-session: 658/658 passing.

---

### Version 1.2.0 -- 2026-05-08 (CR-remediation release)

This release encompasses the critical review (CR) remediation cycle completed 2026-05-07. All 25 CR findings are resolved in code.

- **GII decision-theoretic framing**: GII documented as a geometric mean of M and V. README.md, INPUT_SPECIFICATION.md, and `shap_utils.py` docstrings updated.
- **V-component sample SD (ddof=1)**: all six V-computation sites in `shap_utils.py` corrected from population SD to sample SD; len < 2 NaN guard added.
- **Shadow model leakage closure**: `eval_set` and `early_stopping_rounds` removed from Phase 2 shadow model training.
- **Three independent BH-FDR calls**: one per component family (M, V, GII) for exceedance p-value correction.
- **Degenerate CI fallback**: `compute_bootstrap_ci` returns `(base_score, NaN, NaN)` with `RuntimeWarning` when no bootstrap iterations are valid.
- **Two-tier nominal unseen validation**: Tier 1 `ValueError` (> 50% unique-value mismatch) and Tier 2 `UserWarning` (> 10% observation-level mismatch).
- **Adaptive-knot LSQ spline parity**: `plot.R` V-spline computation aligned with Python `LSQUnivariateSpline` (splines::splineDesign + qr.solve).
- **Nominal top-5 selection by V-contribution**: `plot.R` selects top nominal levels by V-contribution rank (count_k * (mean_SHAP_k - grand_mean)^2) rather than frequency.
- **Documentation**: INPUT_SPECIFICATION.md Sections 3, 4, and 8 updated; README.md CLI interface corrected; per-individual reports relocated to a dedicated subsection.
- **Breaking changes**: none beyond those introduced in v1.1.0 (see v1.1.0 entry).

---

### Version 1.1.0 -- 2026-04-24 (bundled Option B release)

This release bundles all uncommitted work from Session 2026-04-23 onward into a single tagged release:

- **A1 patch (pandas-3.0 Categorical fillna compatibility):** `train.py`, `predict.py`, `infer.py` patched; 19 regression tests added.
- **indiv_reports feature:** new module `src/boost_shap_gii/indiv_reports.py` providing per-individual SHAP reports with coupled-bootstrap CIs; `predict.py` and `infer.py` updated as consumers; `plot.R` extended with per-individual plots.
- **Config schema extensions:** six `shap.*` keys, six `plot.*` keys; `train_outcome_stats.json` new artifact.
- **Environment preflight hardening:** `run_preflight()` called at every CLI entry point.
- **Documentation:** `INPUT_SPECIFICATION.md` Section 10 added; `README.md` updated; `AID_LOG.md` updated (this entry).
- **Breaking changes:** `infer.py` default SHAP behavior change; `plot.R` CLI surface simplified; all configs require new required keys.

---

## 9. References

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

- Bridgeford, E. W., et al. (2025). Ten simple rules for AI-assisted coding in science. *arXiv preprint*, arXiv:2510.22254.

- Carpenter, J., & Bithell, J. (2000). Bootstrap confidence intervals: when, which, what? A practical guide for medical statisticians. *Statistics in Medicine*, 19(9), 1141-1164.

- Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and Their Application*. Cambridge University Press.

- Efron, B. (1983). Estimating the error rate of a prediction rule: improvement on cross-validation. *Journal of the American Statistical Association*, 78(382), 316-331.

- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.

- Fisher, R. A. (1925). *Statistical Methods for Research Workers*. Oliver and Boyd.

- Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E. (2015). Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation. *Journal of Computational and Graphical Statistics*, 24(1), 44-65.

- Hill, A. V. (1910). The possible effects of the aggregation of the molecules of haemoglobin on its dissociation curves. *Journal of Physiology*, 40, iv-vii.

- Jamieson, A. J., et al. (2024). Protecting scientific integrity in an age of generative AI. *Proceedings of the National Academy of Sciences*, 121(41), e2407886121.

- Kursa, M. B., & Rudnicki, W. R. (2010). Feature selection with the Boruta package. *Journal of Statistical Software*, 36(11), 1-13.

- Nussberger, A.-M., et al. (2024). Ten simple rules for using large language models in science. *PLOS Computational Biology*, 20(7), e1012291.

- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, 31, 6638-6648.

- Weaver, J. B. (2025). The AI Disclosure (AID) Framework. *arXiv preprint*, arXiv:2408.01904v2.
