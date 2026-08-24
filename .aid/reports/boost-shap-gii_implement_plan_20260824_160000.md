<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-24T16:00:00Z" />
  <input_reports>
    <report path="boost-shap-gii_test_20260824_152923.md" mode="test" key_items="1" />
  </input_reports>
  <changes>
    <change id="C1" priority="P0" source_item="action_items[0] (P0: fix microdata-saving length mismatch for cv_strategy=group)">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>The microdata-saving block in _run_bootstrap_pipeline (~lines 1332-1359) conflates two distinct semantics of original_cluster_ids. In infer mode, cluster_ids holds per-observation original indices used to collapse K fold-duplicate SHAP rows back to N rows via groupby(level=0).mean(); ids stays length-N and matches correctly. In predict mode with cv_strategy="group", cluster_ids holds group labels (e.g., 20 unique clusters repeated across 80 observations). The same groupby collapses 80 rows to 20 (one per cluster), but ids_micro = ids remains length 80. _process_and_save_microdata (line 924) then builds pd.DataFrame({"id": id_vals (len 80), "shap_value": phi (len 20)}), raising ValueError. The fix gates the groupby block on inference_mode, since the K-duplicate collapse is exclusively an inference-mode operation. For predict-mode group CV, each observation appears exactly once in OOF, so no groupby is needed; the standard (non-grouped) microdata path is correct.</description>
      <spec>
Three edits in src/boost_shap_gii/shap_utils.py:

1. FUNCTION SIGNATURE (line 967-986): Add `inference_mode: bool = False` parameter.

   Current:
   ```python
   def _run_bootstrap_pipeline(
       ...
       cluster_ids: Optional[np.ndarray] = None,
       X_display: Optional[pd.DataFrame] = None,
       shap_scale_factor: float = 1.0,
   ) -> pd.DataFrame:
   ```

   Change to:
   ```python
   def _run_bootstrap_pipeline(
       ...
       cluster_ids: Optional[np.ndarray] = None,
       X_display: Optional[pd.DataFrame] = None,
       shap_scale_factor: float = 1.0,
       inference_mode: bool = False,
   ) -> pd.DataFrame:
   ```

2. MICRODATA GROUPBY GUARD (line 1336): Change the condition from checking only original_cluster_ids to also requiring inference_mode.

   Current:
   ```python
   if original_cluster_ids is not None:
   ```

   Change to:
   ```python
   if original_cluster_ids is not None and inference_mode:
   ```

   Rationale: The groupby(level=0).mean() operation is specifically designed to collapse K-fold duplicates in inference mode (where every fold produces predictions for all N observations). In predict mode (OOF), each observation appears in exactly one fold's test set, so there are no duplicates to collapse. When cv_strategy="group", cluster_ids are used upstream for the cluster bootstrap resampling (lines 1046-1059), which is correct and remains unchanged. But applying the same groupby to OOF microdata incorrectly averages distinct observations within the same cluster, producing fewer rows than the ids array expects.

3. CALL SITE (lines 1509-1523 in _run_shap_for_slice): Pass inference_mode through.

   Current:
   ```python
   _run_bootstrap_pipeline(
       df_shap_real, df_shap_shadow,
       X_stacked,
       eff_idx_real, eff_idx_shadow,
       eff_names_real, eff_names_shadow,
       eff_type_real,
       config, shap_dir,
       all_feature_names,
       all_feature_types,
       nan_mask,
       X_raw, ids,
       cluster_ids=cluster_ids,
       X_display=chunks_X[0] if inference_mode else None,
       shap_scale_factor=shap_scale_factor,
   )
   ```

   Change to:
   ```python
   _run_bootstrap_pipeline(
       df_shap_real, df_shap_shadow,
       X_stacked,
       eff_idx_real, eff_idx_shadow,
       eff_names_real, eff_names_shadow,
       eff_type_real,
       config, shap_dir,
       all_feature_names,
       all_feature_types,
       nan_mask,
       X_raw, ids,
       cluster_ids=cluster_ids,
       X_display=chunks_X[0] if inference_mode else None,
       shap_scale_factor=shap_scale_factor,
       inference_mode=inference_mode,
   )
   ```

   The `inference_mode` variable is already a parameter of `_run_shap_for_slice` (line 1375), so it is in scope at the call site.

WHAT REMAINS UNCHANGED:
- The cluster bootstrap resampling logic (lines 1028-1059) is not affected; it correctly uses cluster_ids for group-level resampling regardless of mode.
- The original_cluster_ids assignment (line 1026) is unchanged; it still saves the pre-fallback value for use in infer-mode microdata.
- _process_and_save_microdata itself is unchanged; the fix ensures it only receives correctly-shaped inputs.
- The i.i.d. fallback (lines 1034-1044) is unchanged.

BEHAVIORAL CONSEQUENCES:
- Predict-mode with cv_strategy="group": microdata now flows through the `else` branch (lines 1350-1353), producing per-observation (not per-cluster) microdata rows. ids and SHAP values are both length-N. No crash.
- Infer-mode: behavior is identical to before (inference_mode=True gates the groupby on).
- Predict-mode without groups: behavior is identical to before (original_cluster_ids is None, else branch taken regardless).
- Predict-mode with groups but fewer than 20 clusters: cluster_ids is set to None by the i.i.d. fallback (line 1044), but original_cluster_ids retains the original value. Previously, this would have triggered the buggy groupby in microdata. With the fix, inference_mode=False prevents the groupby, which is correct: OOF SHAP values have no K-duplicate semantics regardless of cluster count.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Three-line change; only the conditional gate and parameter threading. The bootstrap resampling path and all other code paths are unaffected. The fix narrows the condition from "cluster_ids present" to "cluster_ids present AND inference mode", which is the semantically correct intent.</risk>
      <rollback>Revert the three edits: remove inference_mode parameter from signature, restore the original `if original_cluster_ids is not None:` condition, remove inference_mode=inference_mode from the call site.</rollback>
    </change>
  </changes>
  <execution_order>C1</execution_order>
</implement_plan>
