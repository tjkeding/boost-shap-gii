<clean_report>
  <meta project="boost-shap-gii" mode="clean" timestamp="2026-08-25T13:55:59Z" />
  <scope>Full codebase: 9 Python modules (train.py, predict.py, infer.py, shap_utils.py, indiv_reports.py, utils.py, cli.py, check_env.py, __init__.py), 1 R script (plot.R), 1 shell script (run_boost-shap-gii.sh). All production code under src/boost_shap_gii/.</scope>
  <research_conducted>None required. All findings are based on AST analysis, grep cross-referencing, and manual code review of production call sites.</research_conducted>
  <metrics>
    <loc>7049 Python, 1045 R, 86 shell</loc>
    <files>11 production files</files>
    <avg_complexity>manual estimate: moderate (deep nesting in train.py main(); otherwise well-factored)</avg_complexity>
  </metrics>
  <findings>
    <finding id="F1" severity="minor" category="redundancy">
      <location file="src/boost_shap_gii/predict.py, src/boost_shap_gii/infer.py, src/boost_shap_gii/shap_utils.py, src/boost_shap_gii/utils.py, src/boost_shap_gii/indiv_reports.py" lines="various" />
      <description>14 unused imports across 5 production modules. Verified dead by AST parse and grep cross-referencing. No __all__ re-exports, no dynamic references, no isinstance() usage of CatBoost base class. predict.py: is_classification, save_json_atomic, Any, Dict, List. infer.py: is_classification, yaml. shap_utils.py: sys, yaml, detect_task, stats (scipy), bare joblib module, Union. indiv_reports.py: Any, Callable, CatBoost. utils.py: copy.</description>
      <current>14 imports consume module load time and create false dependency signals for readers. scipy.stats is the heaviest (full scipy.stats namespace).</current>
      <proposed>Remove all 14 unused imports.</proposed>
      <literature>n/a</literature>
      <impact>Marginal reduction in import time; improved readability and accurate dependency signaling.</impact>
    </finding>
    <finding id="F2" severity="minor" category="correctness">
      <location file="src/boost_shap_gii/utils.py" lines="673-680, 758-830" />
      <description>validate_bootstrap_config and validate_plot_config are implemented but never called from production code. validate_plot_config's own docstring states "Called only from cmd_plot", but cmd_plot (cli.py:64-93) dispatches directly to Rscript without calling it. validate_bootstrap_config validates shap.bootstrapping.fdr_method but is not wired into any entry point. Both have test coverage (tests call them directly) confirming the validation logic works.</description>
      <current>End users who pass invalid plot.* config keys receive cryptic R errors instead of clear Python-side ValueError messages. Invalid fdr_method values fall through to statsmodels.multipletests, which raises its own less informative error.</current>
      <proposed>Wire validate_plot_config into cmd_plot (after run_preflight(), reading the YAML config first). Wire validate_bootstrap_config into train.py's early validation sequence.</proposed>
      <literature>n/a</literature>
      <impact>Restores intended user-facing validation that was written but never connected. Improves error messages for misconfigured plot and bootstrap settings.</impact>
    </finding>
    <finding id="F3" severity="style" category="redundancy">
      <location file="src/boost_shap_gii/indiv_reports.py" lines="696-700" />
      <description>indiv_reports.py's orchestrate_bootstrap_cache inlines the CatBoost task-branched model instantiation pattern (CatBoostRegressor/CatBoostClassifier + load_model) despite the same file defining _load_one_model (line 142) that encapsulates exactly this pattern. The cross-module duplication (predict.py, infer.py, shap_utils.py) was assessed and skipped: the 4-line pattern has zero divergence risk, and extracting it to utils.py would introduce cross-module coupling that does not currently exist.</description>
      <current>orchestrate_bootstrap_cache inlines 5 lines that _load_one_model already provides within the same file.</current>
      <proposed>Replace the inline block at lines 696-700 with a call to _load_one_model(mpath, task). Internal consistency fix only; no cross-module refactor.</proposed>
      <literature>n/a</literature>
      <impact>Internal consistency within indiv_reports.py. Eliminates the only intra-file instance where the helper is not used.</impact>
    </finding>
    <finding id="F4" severity="minor" category="redundancy">
      <location file="src/boost_shap_gii/train.py, src/boost_shap_gii/predict.py, src/boost_shap_gii/infer.py" lines="train.py:650-664, predict.py:77-89, infer.py:95-107" />
      <description>Near-identical 8-10 line data-loading block (CSV/Parquet extension dispatch, CSV fallback re-parse with sep=None, whitespace-to-NaN cleanup) duplicated across all three main modules. Only difference is the source of data_path (config key vs. args.data) and log message wording. All three modules already import from utils.py.</description>
      <current>Three independent copies of the same I/O logic. A change to the loading pattern (e.g., adding TSV support or changing the NaN regex) must be applied in three places.</current>
      <proposed>Extract to utils.py as load_dataframe(data_path: str) -> pd.DataFrame. Each caller retains its own log message and path resolution. Reduces each call site from 8-10 lines to 1.</proposed>
      <literature>n/a</literature>
      <impact>Eliminates 3-way duplication of I/O logic. Centralizes the extension-dispatch and sanitization contract.</impact>
    </finding>
    <finding id="F5" severity="style" category="redundancy">
      <location file="src/boost_shap_gii/indiv_reports.py" lines="878-881" />
      <description>if/else branches for shared_indices reconstruction are identical. Both branches execute [shared_indices_arr[b] for b in range(B)] regardless of whether dtype is object (ragged) or rectangular. numpy array indexing with arr[b] produces the same result for both dtypes.</description>
      <current>Dead conditional suggests the two branches were meant to differ, potentially confusing future readers.</current>
      <proposed>Collapse to the single list comprehension. Retain the existing comment ("handles both ragged and rectangular") to document why one expression suffices for both cases.</proposed>
      <literature>n/a</literature>
      <impact>Removes misleading dead conditional. No behavioral change.</impact>
    </finding>
    <finding id="F6" severity="minor" category="redundancy">
      <location file="src/boost_shap_gii/shap_utils.py" lines="94-114" />
      <description>_to_numeric_matrix has three branches (category, string, object dtype) that share identical NaN sentinel logic (max_code + 1 replacement, 3 lines each). The dtype dispatch itself is intentional: the category branch correctly avoids a redundant .astype('category') call. But the sentinel logic (codes.where, max_code + 1, assignment) is copy-pasted verbatim three times.</description>
      <current>20 lines with 9 lines of pure duplication in the sentinel block.</current>
      <proposed>Restructure so each branch produces codes, then a single shared block applies the sentinel and assignment. Reduces to ~12 lines.</proposed>
      <literature>n/a</literature>
      <impact>Eliminates 3-way sentinel logic duplication. Preserves the intentional dtype-specific code-extraction difference.</impact>
    </finding>
    <finding id="F7" severity="minor" category="redundancy">
      <location file="src/boost_shap_gii/train.py, src/boost_shap_gii/predict.py, src/boost_shap_gii/infer.py" lines="train.py:907-941, predict.py:181-216, infer.py:175-207" />
      <description>~30-line ordinal coercion block (quote normalization, two-tier validation with 50% hard-error and 10% warning thresholds, CategoricalDtype construction, .where() out-of-category guard, .cat.codes coercion, NaN restoration) duplicated across all three main modules. predict.py and infer.py are nearly identical (both use df_raw and feature_meta). train.py differs in data source (X directly, selector.feature_metadata) but the validation logic and coercion pattern are structurally identical.</description>
      <current>~90 total lines, ~60 of which are pure duplication. The two-tier validation thresholds and logic must be maintained in three places independently.</current>
      <proposed>Extract coerce_ordinal_column(series: pd.Series, levels: list, column_name: str) -> pd.Series to utils.py. Handles quote normalization, two-tier validation, CategoricalDtype coercion, and NaN restoration. Each call site reduces to one function call plus DataFrame assignment.</proposed>
      <literature>n/a</literature>
      <impact>Centralizes the ordinal coercion contract. Eliminates divergence risk for validation thresholds.</impact>
    </finding>
  </findings>
  <summary>
    <critical_count>0</critical_count>
    <major_count>0</major_count>
    <total_findings>7</total_findings>
    <overall_assessment>needs_minor_work</overall_assessment>
  </summary>
  <action_items>
    <item priority="P1" target_mode="implement" finding_ref="F1" description="Remove 14 verified unused imports across predict.py, infer.py, shap_utils.py, utils.py, indiv_reports.py." />
    <item priority="P1" target_mode="implement" finding_ref="F2" description="Wire validate_plot_config into cmd_plot (cli.py) and validate_bootstrap_config into train.py early validation." />
    <item priority="P2" target_mode="implement" finding_ref="F3" description="Replace inline model-loading block in indiv_reports.py orchestrate_bootstrap_cache (lines 696-700) with _load_one_model call." />
    <item priority="P1" target_mode="implement" finding_ref="F4" description="Extract load_dataframe(data_path) to utils.py; replace 3 duplicated data-loading blocks in train.py, predict.py, infer.py." />
    <item priority="P2" target_mode="implement" finding_ref="F5" description="Collapse identical if/else branches in indiv_reports.py shared_indices reconstruction (lines 878-881) to single comprehension." />
    <item priority="P2" target_mode="implement" finding_ref="F6" description="Consolidate _to_numeric_matrix sentinel logic in shap_utils.py: restructure so dtype dispatch produces codes, shared block applies sentinel." />
    <item priority="P1" target_mode="implement" finding_ref="F7" description="Extract coerce_ordinal_column() to utils.py; replace 3 duplicated ordinal coercion blocks in train.py, predict.py, infer.py." />
  </action_items>
</clean_report>
