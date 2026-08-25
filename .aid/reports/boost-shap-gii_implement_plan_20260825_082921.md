<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-25T12:29:21Z" />
  <input_reports>
    <report path="boost-shap-gii_cr_20260825_082540.md" mode="cr" key_items="3" />
  </input_reports>
  <changes>
    <change id="C1" priority="P1" source_item="F1">
      <file path="src/boost_shap_gii/indiv_reports.py" action="modify" />
      <description>Replace the undocumented config["data"]["cluster_id_col"] lookup with config["modeling"]["group_column"] at both cluster bootstrap sites (training mode and inference mode), aligning individual-level SHAP CI bootstrap with the pipeline's established convention for obtaining group structure.</description>
      <spec>
        Site 1 (training mode, lines 703-706):
        Replace:
          cluster_ids = None
          cluster_col = config.get("data", {}).get("cluster_id_col")
          if cluster_col and cluster_col in X_train.columns:
              cluster_ids = X_train[cluster_col].values
        With:
          cluster_ids = None
          cv_strategy = config.get("modeling", {}).get("cv_strategy", "uniform")
          group_column = config.get("modeling", {}).get("group_column")
          if cv_strategy == "group" and group_column is not None and group_column in X_train.columns:
              cluster_ids = X_train[group_column].values

        Site 2 (inference mode, lines 1032-1036):
        Replace:
          # Resolve cluster_ids from config and X_train column (mirrors orchestrate_bootstrap_cache).
          infer_cluster_ids: Optional[np.ndarray] = None
          cluster_col = config.get("data", {}).get("cluster_id_col")
          if cluster_col and cluster_col in X_train.columns:
              infer_cluster_ids = X_train[cluster_col].values
        With:
          infer_cluster_ids: Optional[np.ndarray] = None
          cv_strategy = config.get("modeling", {}).get("cv_strategy", "uniform")
          group_column = config.get("modeling", {}).get("group_column")
          if cv_strategy == "group" and group_column is not None and group_column in X_train.columns:
              infer_cluster_ids = X_train[group_column].values

        The pattern (cv_strategy == "group" and group_column is not None and group_column in columns) mirrors predict.py:515-519 exactly.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - localized config key replacement; no logic changes to the bootstrap procedure itself</risk>
      <rollback>Revert the two blocks to the original config["data"]["cluster_id_col"] pattern.</rollback>
    </change>

    <change id="C2" priority="P2" source_item="F2">
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <description>Replace the input_transform re-call for fold_meta reconstruction with loading from the persisted fold_transform_metadata.json artifact, mirroring infer.py's approach and closing the single-write-point contract gap.</description>
      <spec>
        Step 1: After the transform_module initialization block (after line 294), load fold_transform_metadata.json:

        After line 294 ("print(f"[INFO] Transformations active...")"), add:
            ftm_path = os.path.join(run_dir, "fold_transform_metadata.json")
            with open(ftm_path) as f:
                _fold_transform_meta = json.load(f)

        Step 2: In the per-fold back-transformation block (lines 318-328), replace the input_transform re-call with an index into the loaded metadata:

        Replace:
            if transform_module is not None:
                outcome_col = outcome_cols[0] if len(outcome_cols) == 1 else outcome_cols
                train_idx_k = np.where(fold_assignments != fold_idx)[0]
                _, _, fold_meta = transform_module.input_transform(
                    df_raw, train_idx_k, val_idx, outcome_col, tx_info.get("params", {})
                )
                preds_bt = transform_module.output_transform(
                    np.asarray(preds, dtype=float), fold_meta, tx_info.get("params", {}),
                    df_raw=df_raw, row_indices=val_idx
                )
                oof_preds[val_idx] = preds_bt
        With:
            if transform_module is not None:
                preds_bt = transform_module.output_transform(
                    np.asarray(preds, dtype=float), _fold_transform_meta[fold_idx],
                    tx_info.get("params", {}),
                    df_raw=df_raw, row_indices=val_idx
                )
                oof_preds[val_idx] = preds_bt

        This removes the outcome_col and train_idx_k computation (no longer needed) and the input_transform re-call, sourcing fold_meta from the persisted artifact instead. The pattern matches infer.py:316-317.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - predict.py operates on the same data as train.py, so the persisted metadata is guaranteed to match; the artifact is always written by train.py before predict.py runs</risk>
      <rollback>Revert to the input_transform re-call pattern and remove the fold_transform_metadata.json load.</rollback>
    </change>

    <change id="C3" priority="P2" source_item="F3">
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>Apply the same permutation test resolution floor (minimum 1000) that predict.py uses, harmonizing permutation test behavior between the two modules.</description>
      <spec>
        At line 500, replace:
          n_perm = n_boot
        With:
          n_perm = max(n_boot, 1000)

        This matches predict.py:441 exactly.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - single expression change; only affects behavior when n_boot &lt; 1000, which is an atypical configuration</risk>
      <rollback>Revert to n_perm = n_boot.</rollback>
    </change>
  </changes>
  <execution_order>C1, C2, C3</execution_order>
</implement_plan>
