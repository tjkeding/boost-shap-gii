<implement_report>
  <meta project="boost-shap-gii" mode="implement" submodule="build" timestamp="2026-08-11T22:36:51Z" />
  <spec_ref>boost-shap-gii_implement_plan_20260811_215631.md</spec_ref>
  <changes_applied>

    <change id="C1" status="done" user_decision="n/a">
      <files_modified>
        <file path="tests/test_build_20260507.py" lines_changed="25" />
      </files_modified>
      <notes>Renamed TestCobbDouglasAnchorPresence to TestGIIFramingPresence. Re-expressed three tests (geometric-mean framing, input specification section 3, README GII interpretation) with file-specific assertion sets. Quarantine test unchanged; no Cobb-Douglas added to forbidden list. All prior Cobb-Douglas assertions removed.</notes>
    </change>

    <change id="C2" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/train.py" lines_changed="68" />
      </files_modified>
      <notes>Added _validate_aggregate_shap function (lines 401-467) enforcing six invariants: name collision, empty/invalid list, single-member warning, constituent existence, nominal prohibition, disjoint membership. Call site inserted at line 709 after the all-missing column drop block. No-op when aggregate_shap is absent.</notes>
    </change>

    <change id="C3" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/train.py" lines_changed="42" />
      </files_modified>
      <notes>Replaced independent per-column permutation loops (lines 993-1033) with block-aware permutation. Grouped features share a single rng.permutation index per group per split (train/val). Ungrouped features retain independent permutation. When aggregate_shap is absent, grouped_features is empty and all features fall through to the ungrouped loop, preserving prior behavior exactly.</notes>
    </change>

    <change id="C4" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/shap_utils.py" lines_changed="24" />
      </files_modified>
      <notes>Replaced per-column permutation in _process_boruta_fold closure (lines 1332-1354) with block-aware permutation matching the train.py pattern. Shared permutation index preserves intra-group correlation in shadow generation for Boruta fold SHAP computation.</notes>
    </change>

    <change id="C5" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/shap_utils.py" lines_changed="205" />
      </files_modified>
      <notes>Added _aggregate_effects function (lines 503-706) computing four aggregate SHAP effect types: singleton aggregate (sum of member singletons), within-group interaction (sum of member x member pairs via combinations), between-group interaction (sum of cross-group pairs), and group x ungrouped interaction (sum of member x ungrouped pairs). Shadow equivalents computed in parallel using shadow_ prefix convention. Group-total X columns registered as continuous. Uses _inter_col helper for bidirectional column name matching and _sum_cols for NaN-safe summation. No-op when aggregate_shap absent.</notes>
    </change>

    <change id="C6" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/shap_utils.py" lines_changed="5" />
      </files_modified>
      <notes>Inserted _aggregate_effects call site at lines 1434-1439, after fold-merge (both inference and OOF paths) and before nan_mask computation (line 1441). Augments df_shap_real, df_shap_shadow, X_stacked, meta_real, meta_shadow_all, and all_feature_types in-place so downstream nan_mask and metadata extraction include aggregate columns.</notes>
    </change>

    <change id="C7" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/shap_utils.py" lines_changed="22" />
      </files_modified>
      <notes>Added _is_aggregate_effect helper (lines 122-139) near _get_effect_stratum. Added is_aggregate column to df_res in _run_bootstrap_pipeline (line 1248). Returns True for singleton aggregate names, within-group interaction names, between-group interaction names, and group x ungrouped interaction names. Returns False for all native effects and when aggregate_shap is absent.</notes>
    </change>

    <change id="C8" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/indiv_reports.py" lines_changed="18" />
      </files_modified>
      <notes>Replaced _CATBOOST_USER_PARAM_ALLOWLIST (19-entry inclusion set) with _CATBOOST_REFIT_BLOCKLIST (5-entry exclusion set: cat_features, text_features, embedding_features, task_type, devices). Inverted _extract_user_level_params filter logic from allowlist inclusion to blocklist exclusion. All user-facing hyperparameters pass through by default.</notes>
    </change>

    <change id="C9" status="done" user_decision="n/a">
      <files_modified>
        <file path="example_config_advanced.yaml" lines_changed="14" />
      </files_modified>
      <notes>Inserted fully commented aggregate_shap section (lines 176-189) between the shap: section and plot: section. All lines prefixed with #. Includes descriptive header, constraint notes (disjoint membership, no nominal features, block-permuted Boruta), and example group definitions.</notes>
    </change>

  </changes_applied>
  <summary>
    <total_changes>9</total_changes>
    <completed>9</completed>
    <skipped>0</skipped>
    <blocked>0</blocked>
  </summary>
  <next_steps>Recommended: run /test to validate all changes, covering re-expressed test assertions, aggregate SHAP validation rules, block-permutation behavior, aggregate M/V/GII computation correctness, and blocklist refactor.</next_steps>
</implement_report>
