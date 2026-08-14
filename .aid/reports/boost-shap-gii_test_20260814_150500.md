<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-14T15:05:00Z" />
  <pre_design_run>
    <total>705</total>
    <passed>695</passed>
    <failed>10</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="TestClusterBootstrapEqualSizeAssertion::test_unequal_cluster_sizes_raises" file="tests/test_inference_shap.py" line="205">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:1315 in _run_bootstrap_pipeline -&gt; shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
      </failure>
      <failure test="TestClusterBootstrapSeedReproducibility::test_same_seed_same_results" file="tests/test_inference_shap.py" line="117">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:1315 in _run_bootstrap_pipeline -&gt; shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
      </failure>
      <failure test="TestClusterBootstrapTinyN::test_tiny_n_no_crash" file="tests/test_inference_shap.py" line="117">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:1315 in _run_bootstrap_pipeline -&gt; shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
      </failure>
      <failure test="TestInferenceShapParquetHasNRows::test_microdata_has_n_rows" file="tests/test_inference_shap.py" line="117">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:1315 in _run_bootstrap_pipeline -&gt; shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
      </failure>
      <failure test="TestInferenceMicrodataHasNRows::test_microdata_x_values_align" file="tests/test_inference_shap.py" line="117">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:1315 in _run_bootstrap_pipeline -&gt; shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
      </failure>
      <failure test="TestInferenceShadowShapSameStructure::test_shadow_uses_cluster_bootstrap_path" file="tests/test_inference_shap.py" line="117">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:1315 in _run_bootstrap_pipeline -&gt; shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
      </failure>
      <failure test="TestAllZeroShapValues::test_zero_shap_produces_zero_gii" file="tests/test_inference_shap.py" line="117">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:1315 in _run_bootstrap_pipeline -&gt; shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
      </failure>
      <failure test="TestZeroVarianceFeatureInInference::test_constant_feature_produces_zero_v" file="tests/test_inference_shap.py" line="681">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:1315 in _run_bootstrap_pipeline -&gt; shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
      </failure>
      <failure test="TestInferenceSingleObservation::test_n1_k3_no_crash" file="tests/test_inference_shap.py" line="117">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:1315 in _run_bootstrap_pipeline -&gt; shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
      </failure>
      <failure test="TestFillConfigDefaults::test_user_values_never_overwritten" file="tests/test_train.py" line="688">
        <error_type>Failed</error_type>
        <message>Default applied to user-provided field: shap.bootstrapping.fdr_method</message>
        <traceback>tests/test_train.py:688 in test_user_values_never_overwritten -&gt; pytest.fail(f'Default applied to user-provided field: {f_path}')</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
    <disposition test="TestClusterBootstrapEqualSizeAssertion::test_unequal_cluster_sizes_raises" file="tests/test_inference_shap.py" classification="obsolete-test">
      <intended_contract>Historically, cluster bootstrap required equal cluster sizes and raised AssertionError otherwise. The implement build (change C1) generalized cluster bootstrap to support unequal cluster sizes via variable-length list-of-arrays resampling, so this contract no longer holds; the underlying assertion this test checks for was intentionally removed.</intended_contract>
      <current_test_claim>pytest.raises(AssertionError, match="equal cluster sizes") when cluster_ids has unequal group sizes ([3, 2]).</current_test_claim>
      <evidence>src/boost_shap_gii/shap_utils.py:1012-1036 (list-of-arrays cluster resampling path, no equal-size assertion); implement build report change C1.</evidence>
      <action>Not re-expressed in this cycle. The test's premise (unequal sizes should error) directly contradicts the new intended contract (unequal sizes are now supported); re-expressing it requires the microdata product-bug fix below to land first, since even a corrected assertion-free version of this test would still hit the same crash under fallback conditions. Left failing and covered by the product-bug routing below.</action>
    </disposition>
    <disposition test="TestClusterBootstrapSeedReproducibility::test_same_seed_same_results" file="tests/test_inference_shap.py" classification="product-bug">
      <intended_contract>Two identical calls to _run_bootstrap_pipeline with the same seed and cluster_ids must produce identical microdata output (determinism).</intended_contract>
      <current_test_claim>Calls _run_bootstrap_pipeline twice with N=10 clusters, K=3 replicates, same seed, and asserts identical results.</current_test_claim>
      <evidence>N=10 triggers the C1 i.i.d. fallback guard (n_clusters=10 &lt; 20 threshold) at shap_utils.py:1000-1010, which reassigns cluster_ids=None. The microdata deduplication branch at shap_utils.py:1297 reads the same now-None cluster_ids variable and takes the else branch, attempting pd.DataFrame({"id": ids (N=10 elements), "shap_value": phi (K*N=30 elements)}) at shap_utils.py:900, raising ValueError.</evidence>
      <action>Routed to implement as a P0 action item (see action_items below). No assertion edit.</action>
    </disposition>
    <disposition test="TestClusterBootstrapTinyN::test_tiny_n_no_crash" file="tests/test_inference_shap.py" classification="product-bug">
      <intended_contract>Cluster bootstrap must not crash on tiny N (edge case robustness).</intended_contract>
      <current_test_claim>Asserts _run_bootstrap_pipeline completes without raising for small N.</current_test_claim>
      <evidence>Same root cause as above: small N triggers the i.i.d. fallback, and the fallback reassignment of cluster_ids leaks into the unrelated microdata deduplication branch.</evidence>
      <action>Routed to implement (same fix as above). No assertion edit.</action>
    </disposition>
    <disposition test="TestInferenceShapParquetHasNRows::test_microdata_has_n_rows" file="tests/test_inference_shap.py" classification="product-bug">
      <intended_contract>Inference-mode microdata output must have exactly N rows (one per original observation), deduplicated from K*N SHAP rows.</intended_contract>
      <current_test_claim>Asserts microdata_M.parquet has N rows after _run_bootstrap_pipeline with cluster_ids present.</current_test_claim>
      <evidence>Same root cause: fallback triggers before the microdata deduplication branch is reached.</evidence>
      <action>Routed to implement (same fix as above). No assertion edit.</action>
    </disposition>
    <disposition test="TestInferenceMicrodataHasNRows::test_microdata_x_values_align" file="tests/test_inference_shap.py" classification="product-bug">
      <intended_contract>Microdata X values must align correctly with the deduplicated N-row structure in inference mode.</intended_contract>
      <current_test_claim>Asserts X values in microdata match expected per-observation values.</current_test_claim>
      <evidence>Same root cause.</evidence>
      <action>Routed to implement (same fix as above). No assertion edit.</action>
    </disposition>
    <disposition test="TestInferenceShadowShapSameStructure::test_shadow_uses_cluster_bootstrap_path" file="tests/test_inference_shap.py" classification="product-bug">
      <intended_contract>Shadow SHAP microdata must follow the same cluster-bootstrap-aware deduplication path as real SHAP microdata.</intended_contract>
      <current_test_claim>Asserts shadow microdata output structure matches real microdata structure under cluster_ids.</current_test_claim>
      <evidence>Same root cause.</evidence>
      <action>Routed to implement (same fix as above). No assertion edit.</action>
    </disposition>
    <disposition test="TestAllZeroShapValues::test_zero_shap_produces_zero_gii" file="tests/test_inference_shap.py" classification="product-bug">
      <intended_contract>All-zero SHAP values must produce GII=0 without crashing the pipeline.</intended_contract>
      <current_test_claim>Asserts zero GII output for all-zero SHAP input under cluster_ids.</current_test_claim>
      <evidence>Same root cause; small N in this synthetic fixture triggers fallback.</evidence>
      <action>Routed to implement (same fix as above). No assertion edit.</action>
    </disposition>
    <disposition test="TestZeroVarianceFeatureInInference::test_constant_feature_produces_zero_v" file="tests/test_inference_shap.py" classification="product-bug">
      <intended_contract>A constant feature must produce V=0 without crashing the pipeline.</intended_contract>
      <current_test_claim>Asserts zero V output for a constant feature under cluster_ids.</current_test_claim>
      <evidence>Same root cause.</evidence>
      <action>Routed to implement (same fix as above). No assertion edit.</action>
    </disposition>
    <disposition test="TestInferenceSingleObservation::test_n1_k3_no_crash" file="tests/test_inference_shap.py" classification="product-bug">
      <intended_contract>N=1 observation with K=3 folds must not crash the inference SHAP pipeline (extreme edge case).</intended_contract>
      <current_test_claim>Asserts no crash for N=1, K=3.</current_test_claim>
      <evidence>N=1 triggers the i.i.d. fallback guard (n_clusters=1 &lt; 20), same downstream crash.</evidence>
      <action>Routed to implement (same fix as above). No assertion edit.</action>
    </disposition>
    <disposition test="TestFillConfigDefaults::test_user_values_never_overwritten" file="tests/test_train.py" classification="obsolete-test">
      <intended_contract>fill_config_defaults must never overwrite a user-provided config value; only fields absent from the user's config may receive a default.</intended_contract>
      <current_test_claim>allowed_fills allowlist (fields legitimately auto-filled on a config fixture that predates recent schema additions) omitted "shap.bootstrapping.fdr_method", so its presence in the filled list caused pytest.fail.</current_test_claim>
      <evidence>src/boost_shap_gii/utils.py:485 adds _set(["shap","bootstrapping","fdr_method"], "bh") as a new default (implement build change C5). The sample_config fixture does not specify fdr_method, so the default was correctly applied via setdefault semantics; the test's allowlist was simply not updated for the new key.</evidence>
      <action>Re-expressed: added "shap.bootstrapping.fdr_method" to the allowed_fills set at tests/test_train.py:685 (plus a docstring line documenting the new allowed fill). Postcondition preserved and strengthened: the test still fails if ANY unlisted field is auto-filled; it now additionally documents that fdr_method is a known, intentional new default rather than silently tolerating it.</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>19</tests_created>
    <tests_modified>1</tests_modified>
    <files_created>
      <file path="tests/test_build_20260814.py" test_count="19" coverage_target="Coverage for implement build changes C2 (Graham 1966 greedy list scheduling), C3 (n_unique_groups cardinality validation), C5 (fdr_method config key: validate_bootstrap_config + fill_config_defaults), C6 (inner-groups diagnostic, source-level), C7 (stratify_labels_for_regression bin-count warning)" />
    </files_created>
    <design_rationale>
      Each new test class targets one build change (C2, C3, C5, C6, C7) not previously exercised by the suite. C1 (cluster bootstrap) is already extensively covered by the pre-existing tests/test_inference_shap.py, whose failures are routed as a product bug rather than duplicated here. C4 (redundant splitter recreation removal) has no independently observable behavioral contract distinguishable from the existing splitter tests, which continue to pass unchanged. C6's diagnostic is verified at the source level (gating condition, scope, and message text) rather than via full execution, following the existing codebase precedent (test_package_structure.py, test_implementation_changes.py) of not invoking run_optuna_tuning's full Optuna/CatBoost machinery in unit-test scope. Every numeric assertion in the new file (Graham-scheduling fold-size ratios, cardinality validation error messages, fdr_method default/preservation values, bin-count warning thresholds) was independently executed against the live codebase before being committed to the test file, per Design by Contract discipline; one initial assertion (a guessed 1.15 fold-size-ratio threshold) was caught by this verification step and corrected to a direct greedy-vs-round-robin comparison before being finalized.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>724</total>
    <passed>715</passed>
    <failed>9</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="TestClusterBootstrapEqualSizeAssertion::test_unequal_cluster_sizes_raises" file="tests/test_inference_shap.py" line="205">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
        <likely_cause>Same product bug as pre-design run; unaffected by this cycle's test edits (no product code was modified).</likely_cause>
      </failure>
      <failure test="TestClusterBootstrapSeedReproducibility::test_same_seed_same_results" file="tests/test_inference_shap.py" line="275">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
        <likely_cause>Product bug, unresolved (see disposition above).</likely_cause>
      </failure>
      <failure test="TestClusterBootstrapTinyN::test_tiny_n_no_crash" file="tests/test_inference_shap.py" line="298">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
        <likely_cause>Product bug, unresolved (see disposition above).</likely_cause>
      </failure>
      <failure test="TestInferenceShapParquetHasNRows::test_microdata_has_n_rows" file="tests/test_inference_shap.py" line="350">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
        <likely_cause>Product bug, unresolved (see disposition above).</likely_cause>
      </failure>
      <failure test="TestInferenceMicrodataHasNRows::test_microdata_x_values_align" file="tests/test_inference_shap.py" line="390">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
        <likely_cause>Product bug, unresolved (see disposition above).</likely_cause>
      </failure>
      <failure test="TestInferenceShadowShapSameStructure::test_shadow_uses_cluster_bootstrap_path" file="tests/test_inference_shap.py" line="411">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
        <likely_cause>Product bug, unresolved (see disposition above).</likely_cause>
      </failure>
      <failure test="TestAllZeroShapValues::test_zero_shap_produces_zero_gii" file="tests/test_inference_shap.py" line="620">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
        <likely_cause>Product bug, unresolved (see disposition above).</likely_cause>
      </failure>
      <failure test="TestZeroVarianceFeatureInInference::test_constant_feature_produces_zero_v" file="tests/test_inference_shap.py" line="681">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
        <likely_cause>Product bug, unresolved (see disposition above).</likely_cause>
      </failure>
      <failure test="TestInferenceSingleObservation::test_n1_k3_no_crash" file="tests/test_inference_shap.py" line="716">
        <error_type>ValueError</error_type>
        <message>All arrays must be of the same length</message>
        <traceback>shap_utils.py:900 in _process_and_save_microdata -&gt; block = pd.DataFrame({...})</traceback>
        <likely_cause>Product bug, unresolved (see disposition above).</likely_cause>
      </failure>
    </failures>
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>1</bugs_routed_to_implement>
    <recommendation>implement_fixes</recommendation>
  </summary>
  <action_items>
    <item priority="P0" target_mode="implement" description="Fix cluster-bootstrap i.i.d.-fallback / microdata-deduplication interaction in _run_bootstrap_pipeline (src/boost_shap_gii/shap_utils.py). The i.i.d. fallback guard (around line 1010, triggered when n_clusters &lt; 20 per Ukoumunne et al. 2003) reassigns the cluster_ids parameter to None so the bootstrap resampling logic switches to i.i.d. sampling. However, the microdata deduplication branch later in the same function (around line 1297) reads the same now-overwritten cluster_ids variable to decide whether inference-mode microdata should be group-averaged from K*N rows down to N rows. When fallback fires, this branch incorrectly takes the non-clustered path and attempts to build a pandas DataFrame from an N-length ids array and a K*N-length SHAP values array, raising ValueError('All arrays must be of the same length'). Fix: introduce a separate variable (e.g., original_cluster_ids) captured before the fallback reassignment, and use it (not the post-fallback cluster_ids) to gate the microdata deduplication branch, since the K-replication structure used for microdata averaging is independent of which bootstrap resampling method was ultimately selected. This affects 9 tests in tests/test_inference_shap.py, all currently failing with the identical traceback." />
  </action_items>
</test_report>
