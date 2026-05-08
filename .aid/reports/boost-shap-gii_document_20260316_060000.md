<document_report>
 <meta project="boost-shap-gii" mode="document" timestamp="2026-03-16T16:11:10Z" />
 <files_updated>

 <file path="<project_root>/boost-shap-gii/README.md"
 changes="Added four new sections: Missing Value Handling (nominal __NA__ sentinel, ordinal, continuous), GII Interpretation (GII formula, non-summability, singleton vs interaction scale convention, Boruta adaptive noise baseline). Existing sections unchanged.">
 <type>readme</type>
 </file>

 <file path="<project_root>/boost-shap-gii/INPUT_SPECIFICATION.md"
 changes="Full rewrite to match current codebase. Added: two-tier ordinal validation; nominal __NA__ sentinel with informativeness note; shadow model early stopping with 2x ceiling; predict.py model-count assertion; depth floor and one_hot_max_size fixed range in search space table; joblib and statsmodels in Stage 0; exceedance p-value +1 correction; NaN preservation instead of 0.0 replacement; inner CV seed offset; stacked-spline routing in V-component method selection; Boruta noise baseline model-adaptive note; bootstrap drop-and-warn with n_boot_effective; permutation test while-loop with 2*n_perm cap; full artifact directory structure; shap_stats_global.csv column reference; interaction scale convention; GII non-additivity; high-M low-V insensitivity.">
 <type>input_spec</type>
 </file>

 <file path="<project_root>/boost-shap-gii/train.py"
 changes="Updated FeatureSelector class docstring (added deterministic ordering note). Updated _match, fit, get_feature_lists docstrings. Updated report_missingness docstring. Full NumPy-style docstring for run_optuna_tuning (added fold_idx parameter description, inner seed offset rationale). Added inline comment on nominal __NA__ sentinel in Type Enforcement block.">
 <type>docstring</type>
 </file>

 <file path="<project_root>/boost-shap-gii/predict.py"
 changes="Added inline comment on nominal __NA__ sentinel in Cast Nominals block.">
 <type>inline_comment</type>
 </file>

 <file path="<project_root>/boost-shap-gii/infer.py"
 changes="Added inline comment on nominal __NA__ sentinel in Cast Nominals block.">
 <type>inline_comment</type>
 </file>

 <file path="<project_root>/boost-shap-gii/utils.py"
 changes="Full NumPy-style docstring for compute_permutation_test (added while-loop rationale, +1 correction, contrast with bootstrap behavior). Full NumPy-style docstring for compute_bootstrap_ci (added n_boot_effective explanation, contrast with permutation retry). Updated _default_search_space docstring to note depth floor and one_hot_max_size fixed range.">
 <type>docstring</type>
 </file>

 <file path="<project_root>/boost-shap-gii/shap_utils.py"
 changes="Module docstring expanded: added exceedance +1 correction, interaction scale convention, Boruta adaptive noise baseline note, NaN preservation rationale. Updated _flatten_interaction_matrix docstring to explain singleton vs. interaction scale convention with full NumPy-style parameters/returns. Added run_shap_pipeline docstring with full ctx parameter specification.">
 <type>docstring</type>
 </file>

 </files_updated>

 <coverage>
 <public_functions_documented>18/18</public_functions_documented>
 <classes_documented>1/1</classes_documented>
 <modules_with_docstrings>5/5</modules_with_docstrings>
 </coverage>

 <summary>
 Documentation updated to reflect all 15 code changes from the implement sessions
 (changes 1-15) and the 2 documentation-only items from the brainstorm. README.md
 received four new sections covering the most interpretation-critical topics: nominal
 missingness handling, GII non-summability, interaction SHAP scale convention, and
 Boruta adaptive noise baseline. INPUT_SPECIFICATION.md was substantially expanded from
 a 55-line stub to a comprehensive 7-section reference covering all config parameters,
 pipeline stages, statistical methodology, artifact structure, and known limitations.
 Source file docstrings were updated to reflect implementation changes (inner CV seed
 offset, two-tier ordinal validation, shadow model early stopping, permutation while-loop,
 bootstrap drop tracking, NaN preservation, stacked-spline routing) and to clarify
 interpretation-sensitive design decisions. All 6 modified Python files pass py_compile.
 </summary>
</document_report>
