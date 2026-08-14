<clean_report>
  <meta project="boost-shap-gii" mode="clean" timestamp="2026-08-14T01:45:00Z" />
  <scope>Full codebase (11 source files, 7628 LOC under src/boost_shap_gii/) and all GitHub-committed documentation (README.md, INPUT_SPECIFICATION.md, example_config_advanced.yaml, example_config_minimal.yaml, pyproject.toml, environment.yaml, AID_LOG.md, LICENSE; 1964 LOC total). Review focused on post-CV-strategy-feature state following implement build 20260813_223500 and test cycle 20260813_231500 (705/705 passing).</scope>
  <research_conducted>None required. All findings are based on direct codebase and documentation comparison; no external claims or framework-specific optimizations needed verification.</research_conducted>
  <metrics>
    <loc>9592 (7628 source + 1964 documentation/config)</loc>
    <files>19 (11 source + 8 documentation/config)</files>
    <avg_complexity>Estimated CC ~25-35 for monolithic main() functions; helper functions and utility modules are low complexity (CC ~5-10).</avg_complexity>
  </metrics>
  <findings>
    <finding id="F1" severity="major" category="correctness">
      <location file="INPUT_SPECIFICATION.md" lines="112-114" />
      <description>Stage 4 CV description states that StratifiedKFold is used "when task_type is a classification variant and the label set has fewer than 20 unique values; KFold otherwise." This auto-stratification behavior was deliberately removed in the current session's brainstorm decision T1 (approach A1: cv_strategy as a literal splitter selector with zero task-dependent branching). The documented behavior directly contradicts the implementation.</description>
      <current>Documentation describes auto-stratification based on task_type and y.nunique() cardinality.</current>
      <proposed>Replace with text describing the config-driven cv_strategy selector: "uniform" (KFold always), "stratified" (StratifiedKFold for classification, quantile-binned StratifiedKFold for regression), "group" (GroupKFold with group_column). Note the default is "uniform" and the selector is task-type-independent.</proposed>
      <literature>N/A (internal documentation drift).</literature>
      <impact>Prevents user confusion about CV fold construction; aligns spec with the approved backward-compatibility break documented in the brainstorm report.</impact>
    </finding>
    <finding id="F2" severity="major" category="correctness">
      <location file="INPUT_SPECIFICATION.md" lines="828-831" />
      <description>Section 10 states "Fold assignments for training individuals are reconstructed deterministically at predict-time from the saved config_resolved.yaml (which contains random_seed and cv_folds) without persisting a new artifact, using the same get_cv_splitter() call that predict.py uses internally." The implementation now persists fold_assignments.json from train.py and loads it in predict.py, shap_utils.py, and indiv_reports.py. The get_cv_splitter() import was removed from predict.py entirely.</description>
      <current>Documentation describes splitter reconstruction from config without a persisted artifact.</current>
      <proposed>Replace with text describing the fold_assignments.json artifact: written by train.py after the outer CV loop; loaded by predict.py, shap_utils.py, and indiv_reports.py; contains a JSON array of integer fold indices (one per training sample). Note this eliminates dependence on data identity, sklearn version determinism, and stratification replication.</proposed>
      <literature>N/A (internal documentation drift).</literature>
      <impact>Prevents a reviewer from expecting splitter-reconstruction architecture when the actual architecture is artifact-based; accurately documents a new output file (fold_assignments.json) that appears in the training output directory.</impact>
    </finding>
    <finding id="F3" severity="minor" category="correctness">
      <location file="INPUT_SPECIFICATION.md" lines="65" />
      <description>Stage 0 pre-flight description lists "shap" among the Python dependencies checked by check_env.py. The shap package was removed as a direct dependency in Session 4 (the pipeline uses CatBoost's native SHAP implementation). Confirmed: grep for "shap" in check_env.py returns only the project-name mention in the module docstring, not an import check. The package is absent from pyproject.toml dependencies.</description>
      <current>"Imports catboost, optuna, shap, pyarrow, sklearn, scipy, pandas, yaml, joblib, statsmodels."</current>
      <proposed>Remove "shap" from the import list. Verify the remaining list against the actual check_env.py imports.</proposed>
      <literature>N/A (stale dependency reference).</literature>
      <impact>Prevents users from unnecessarily installing the shap package; aligns spec with the actual dependency set.</impact>
    </finding>
    <finding id="F4" severity="major" category="correctness">
      <location file="README.md" lines="1-194" />
      <location file="INPUT_SPECIFICATION.md" lines="1-1138" />
      <description>The three new user-configurable keys introduced this session (cv_strategy, group_column, n_inner_repeats) have zero documentation in README.md or INPUT_SPECIFICATION.md. All three are present in example_config_advanced.yaml (L61-62, L71), implemented in utils.py (get_cv_splitter, validate_cv_config, fill_config_defaults), and exercised in train.py. They constitute the entire user-facing surface of the CV strategy feature.</description>
      <current>No mention of cv_strategy, group_column, or n_inner_repeats in either documentation file.</current>
      <proposed>Add all three keys to: (a) README.md config reference section with brief descriptions and valid values; (b) INPUT_SPECIFICATION.md Section 2 config table with full schema details (types, defaults, constraints, interactions); (c) INPUT_SPECIFICATION.md Section 3 modeling behavior narrative describing the cv_strategy semantics, group_column exclusion from features, group-strategy bootstrap fallback, and n_inner_repeats effect on inner CV during Optuna tuning.</proposed>
      <literature>N/A (missing documentation for new feature).</literature>
      <impact>Users cannot discover or correctly configure the CV strategy feature without reading source code or the example config's inline comments. Reviewers cannot verify that the implementation matches any documented specification.</impact>
    </finding>
    <finding id="F5" severity="minor" category="correctness">
      <location file="src/boost_shap_gii/indiv_reports.py" lines="11" />
      <description>Module docstring states inference-mode bootstrap generates "a fresh K-fold split on s_b (KFold/StratifiedKFold, seed = random_seed + b + 1)." The actual implementation now calls get_cv_splitter() which is config-driven (cv_strategy selector), and includes a third path: group-strategy fallback to plain KFold with a logged warning (because bootstrap resampling breaks group structure). The parenthetical "KFold/StratifiedKFold" implies only two possible splitter types and an automatic choice, neither of which is accurate.</description>
      <current>"generate a fresh K-fold split on s_b (KFold/StratifiedKFold, seed = random_seed + b + 1)"</current>
      <proposed>Update to reference the config-driven get_cv_splitter() call, note the group-strategy KFold fallback, and remove the implication of automatic splitter selection.</proposed>
      <literature>N/A (internal docstring drift).</literature>
      <impact>Prevents developer confusion when reading the module header about the inference-mode bootstrap fold-construction logic.</impact>
    </finding>
    <finding id="F6" severity="style" category="redundancy">
      <location file="src/boost_shap_gii/train.py" lines="892-900" />
      <description>The CV splitter is constructed at L892, consumed at L895 for unbalanced-fold size checking (group strategy only), then reconstructed identically at L900 before entering the outer CV loop. This double construction is functionally correct and was documented in the implement plan as an intentional pattern: the balance check needs to consume the generator before the training loop does. The second get_cv_splitter() call is negligible relative to model training cost.</description>
      <current>Two sequential get_cv_splitter() calls with identical arguments; the first consumed for fold-size analysis, the second for the actual CV loop.</current>
      <proposed>No action required. If a future refactor touches this area, consider materializing list(splitter.split(X, y_for_split)) once and iterating over the materialized list.</proposed>
      <literature>N/A.</literature>
      <impact>Negligible performance impact; informational only.</impact>
    </finding>
    <finding id="F7" severity="style" category="maintainability">
      <location file="src/boost_shap_gii/train.py" lines="626-1109" />
      <location file="src/boost_shap_gii/predict.py" lines="44-505" />
      <location file="src/boost_shap_gii/infer.py" lines="46-602" />
      <description>All three pipeline entry-point modules have monolithic main() functions spanning 460-560 lines each. These are pre-existing (not introduced by this session's work) and have been stable across multiple sessions. The monolithic structure is the primary reason integration-level testing was explicitly scoped out of the test design phase (test report P2 action item). Cyclomatic complexity is moderate (estimated CC ~25-35), driven by task-type branching. The functions are linear (sequential pipeline stages), not deeply nested.</description>
      <current>Single main() functions containing all pipeline orchestration logic per module.</current>
      <proposed>No immediate action. A future dedicated refactor cycle could decompose into stage-specific helpers (data loading, CV setup, training loop, output persistence), which would unlock integration-level test coverage for fold-assignment round-tripping and group-column exclusion.</proposed>
      <literature>N/A.</literature>
      <impact>Informational. Decomposition would improve testability and readability but carries non-trivial refactor risk for a pipeline passing 705 tests.</impact>
    </finding>
  </findings>
  <summary>
    <critical_count>0</critical_count>
    <major_count>3</major_count>
    <total_findings>7</total_findings>
    <overall_assessment>needs_minor_work</overall_assessment>
  </summary>
  <action_items>
    <item priority="P1" target_mode="document" finding_ref="F1,F2,F3,F4,F5" description="Update INPUT_SPECIFICATION.md and README.md for CV strategy feature: fix Stage 4 CV description (F1), fix Section 10 fold reconstruction description (F2), remove shap from Stage 0 dependency list (F3), add cv_strategy/group_column/n_inner_repeats to config tables and modeling narrative (F4), and update indiv_reports.py module docstring (F5). F5 is a code docstring edit and may require /implement rather than /document depending on skill write-scope constraints." />
    <item priority="P2" target_mode="implement" finding_ref="F7" description="Decompose monolithic main() functions in train.py, predict.py, and infer.py into stage-specific helpers to improve testability. This would enable integration-level test coverage for fold_assignments.json round-tripping and group_column exclusion (test report P2 action item). Defer to a dedicated refactor cycle." />
  </action_items>
</clean_report>
