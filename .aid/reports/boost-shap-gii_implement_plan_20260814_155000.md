<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-14T15:50:00Z" />
  <input_reports>
    <report path="boost-shap-gii_test_20260814_150500.md" mode="test" key_items="1" />
  </input_reports>
  <changes>
    <change id="C1" priority="P0" source_item="action_items[0]: Fix cluster-bootstrap i.i.d.-fallback / microdata-deduplication interaction">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>The i.i.d. fallback guard (line 1010) reassigns the cluster_ids parameter to None when n_clusters is below the minimum threshold (20, per Ukoumunne et al. 2003). This correctly switches bootstrap resampling to i.i.d. mode. However, the microdata deduplication branch at line 1297 reads the same now-None variable to decide whether to group-average inference-mode microdata from K*N rows to N rows. When fallback fires on K-replicated data, the else branch produces a DataFrame from an N-length ids array and a K*N-length SHAP column, raising ValueError. Fix: capture the original cluster_ids before the fallback guard and use it for the microdata deduplication decision, since K-replication structure is independent of the bootstrap method selected.</description>
      <spec>
In _run_bootstrap_pipeline (line 943+):

1. Before line 994 (`if cluster_ids is not None:`), insert:
   `original_cluster_ids = cluster_ids`

2. At line 1297, change the microdata deduplication guard from:
   `if cluster_ids is not None:`
   to:
   `if original_cluster_ids is not None:`

3. At lines 1299-1309, change all remaining references to cluster_ids within this microdata block to original_cluster_ids:
   - Line 1299: `df_shap_micro.index = cluster_ids` -> `df_shap_micro.index = original_cluster_ids`

No other references to cluster_ids in the microdata block need changing (the variable is not used elsewhere in lines 1300-1320).
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Single-variable introduction with a narrow, well-understood blast radius. The change only affects the microdata deduplication decision path; all bootstrap resampling logic (lines 994-1037) continues to use the post-fallback cluster_ids as before.</risk>
      <rollback>Revert the three edits: remove the original_cluster_ids assignment and restore the two references from original_cluster_ids back to cluster_ids.</rollback>
    </change>
    <change id="C2" priority="P0" source_item="failing_test_dispositions[0]: obsolete-test for test_unequal_cluster_sizes_raises, deferred pending C1">
      <file path="tests/test_inference_shap.py" action="modify" />
      <description>Re-express the obsolete test test_unequal_cluster_sizes_raises. The test's prior contract (unequal cluster sizes raise AssertionError) was invalidated by the C1 build cycle's introduction of variable-length list-of-arrays cluster resampling. The new contract is that unequal cluster sizes are supported (with i.i.d. fallback when n_clusters is below the threshold). The test report deferred this re-expression because the microdata product bug (C1 above) would have caused a crash even in a corrected test. With C1 now resolved, the re-expression can proceed.</description>
      <spec>
Replace the existing test_unequal_cluster_sizes_raises method body. The current body uses pytest.raises(AssertionError, match="equal cluster sizes"), which asserts a contract that no longer exists.

New test body:
1. Keep the same fixture setup (cluster_ids with unequal sizes [3, 2], same synthetic data construction).
2. Remove the pytest.raises wrapper.
3. Call _run_bootstrap_pipeline directly (no exception expected).
4. Assert: the function completes without error (implicit by reaching the assertions below).
5. Assert: a RuntimeWarning is emitted containing "Falling back to i.i.d. bootstrap" (the Ukoumunne et al. 2003 fallback diagnostic, since n_clusters=2 is below the threshold of 20).
6. Assert: microdata output files exist in the temp directory (microdata_M.parquet, microdata_V.parquet, microdata_GII.parquet), confirming the microdata deduplication path executed correctly.

Use warnings.catch_warnings() + warnings.simplefilter("always") + a manual check, or pytest.warns(RuntimeWarning, match="Falling back to i.i.d. bootstrap").

Update the test docstring to reflect the new contract: "Unequal cluster sizes should complete successfully with i.i.d. fallback warning."

Update the class docstring from "test_cluster_bootstrap_equal_size_assertion" to reflect the new intent.

Postcondition analysis: the old assertion was "unequal sizes raise." The new assertion is "unequal sizes complete without error AND emit the correct fallback diagnostic AND produce valid microdata output." This is a contract change (obsolete-test re-expression), not a weakening; the new postcondition exercises the actual current contract more thoroughly than the old one exercised the old contract.
      </spec>
      <dependencies>C1 (the microdata bug fix must land first, otherwise the re-expressed test would still crash with ValueError)</dependencies>
      <risk>low - Confined to one test method in tests/test_inference_shap.py. The test exercises the same code path as 8 other tests in the same file that are currently failing; once C1 lands, all 8 of those will pass without modification, and this re-expressed test will additionally verify the fallback warning pathway.</risk>
      <rollback>Restore the original pytest.raises(AssertionError, match="equal cluster sizes") wrapper and original docstrings.</rollback>
    </change>
  </changes>
  <execution_order>C1, C2</execution_order>
</implement_plan>
