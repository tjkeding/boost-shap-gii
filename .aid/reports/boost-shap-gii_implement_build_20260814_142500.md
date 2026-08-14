<implement_report>
  <meta project="boost-shap-gii" mode="implement" submodule="build" timestamp="2026-08-14T14:25:00Z" />
  <spec_ref>boost-shap-gii_implement_plan_20260814_141500.md</spec_ref>
  <changes_applied>
    <change id="C4" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/train.py" lines_changed="1" />
      </files_modified>
      <notes>Removed redundant splitter recreation line. The splitter created at line 892 is now the sole instance used by both the fold-size diagnostic and the CV loop.</notes>
    </change>
    <change id="C2" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/utils.py" lines_changed="28" />
      </files_modified>
      <notes>Replaced round-robin group assignment in _RepeatedGroupKFold.split() with greedy-in-random-order (Graham 1966 list scheduling). Groups are shuffled per repetition, then each group is assigned to the fold with the fewest total samples, minimizing fold-size imbalance under unequal group sizes.</notes>
    </change>
    <change id="C3" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/utils.py" lines_changed="16" />
      </files_modified>
      <notes>Added validation in validate_cv_config that n_unique_groups >= cv_folds and n_unique_groups >= inner_cv_folds, raising ValueError with a diagnostic message when violated.</notes>
    </change>
    <change id="C7" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/utils.py" lines_changed="12" />
      </files_modified>
      <notes>stratify_labels_for_regression now captures the binned result, checks the actual number of unique bins, and emits a warning when actual_bins &lt; n_bins (fewer unique quantile edges than requested).</notes>
    </change>
    <change id="C6" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/train.py" lines_changed="7" />
      </files_modified>
      <notes>Added inner-groups diagnostic in run_optuna_tuning that warns when the number of unique groups in the inner CV dataset is less than 2 * inner_cv_folds, indicating potentially unreliable tuning estimates.</notes>
    </change>
    <change id="C5" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/shap_utils.py" lines_changed="8" />
        <file path="src/boost_shap_gii/utils.py" lines_changed="20" />
        <file path="example_config_advanced.yaml" lines_changed="1" />
      </files_modified>
      <notes>Exposed fdr_method config key with "bh" default (BH-FDR) and "by" option (BY-FDR). fill_config_defaults sets the default; validate_bootstrap_config validates the value; _run_bootstrap_pipeline extracts the key and maps to scipy method string; _nan_safe_fdr uses the parameterized method instead of hardcoded fdr_bh. Config template updated with inline documentation.</notes>
    </change>
    <change id="C1" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/predict.py" lines_changed="5" />
        <file path="src/boost_shap_gii/shap_utils.py" lines_changed="65" />
      </files_modified>
      <notes>Implemented cluster bootstrap for group CV strategy. predict.py threads group_column values and cv_strategy into shap_ctx. run_shap_pipeline extracts and passes groups/cv_strategy to _run_shap_for_slice, which sets cluster_ids when cv_strategy=="group" in non-inference mode. _run_bootstrap_pipeline implements cluster resampling with i.i.d. fallback when n_clusters &lt; 20 (RuntimeWarning per Ukoumunne et al. 2003), variable-length list-of-arrays for unequal cluster sizes, and isinstance-based branching in indices_split. _bootstrap_worker_chunk handles both list and ndarray index formats via isinstance check on n_iter derivation.</notes>
    </change>
  </changes_applied>
  <summary>
    <total_changes>7</total_changes>
    <completed>7</completed>
    <skipped>0</skipped>
    <blocked>0</blocked>
  </summary>
  <next_steps>Recommended: run /test to validate all changes.</next_steps>
</implement_report>
