<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-26T14:46:18Z" />
  <input_reports>
    <report path="boost-shap-gii_cr_20260826_143601.md" mode="cr" key_items="5" />
  </input_reports>
  <changes>
    <change id="C1" priority="P1" source_item="F1">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>Add upfront hard halt when transformations are configured with a non-regression task. The input_transform/output_transform API contract assumes a continuous, invertible outcome transformation; classification tasks produce discrete labels (training) and probabilities (prediction), neither of which is compatible.</description>
      <spec>
Three insertion sites, each raising ValueError if transform_module is not None and not is_regression(task):

1. train.py, after line 899 (task = detect_task(config)):
   Insert immediately after `task = detect_task(config)` and before the classification class-count block (line 902):
   ```python
   if transform_module is not None and not is_regression(task):
       raise ValueError(
           f"Outcome transformations are only supported for regression tasks "
           f"(regression, multi_regression). Detected task: '{task}'. "
           f"The input_transform/output_transform API contract assumes a "
           f"continuous, invertible outcome transformation."
       )
   ```
   Note: transform_module is loaded at line 767, task is detected at line 899. The guard bridges these two.

2. predict.py, after line 248 (shap_scale_factor assignment, inside active=True branch):
   Insert after line 252 (_fold_transform_meta loaded) before the fold loop at line 254:
   ```python
   if not is_regression(task):
       raise ValueError(
           f"Outcome transformations are only supported for regression tasks "
           f"(regression, multi_regression). Detected task: '{task}'. "
           f"The input_transform/output_transform API contract assumes a "
           f"continuous, invertible outcome transformation."
       )
   ```
   Requires adding `is_regression` to the imports from utils.py if not already present.

3. infer.py, after line 241 (shap_scale_factor assignment, inside active=True branch):
   Insert after `shap_scale_factor = tx_info.get(...)` and before the fold_transform_metadata loading at line 244:
   ```python
   if not is_regression(task):
       raise ValueError(
           f"Outcome transformations are only supported for regression tasks "
           f"(regression, multi_regression). Detected task: '{task}'. "
           f"The input_transform/output_transform API contract assumes a "
           f"continuous, invertible outcome transformation."
       )
   ```
   Requires adding `is_regression` to the imports from utils.py if not already present.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - additive guard with no side effects on existing logic; triggers only on currently-undefined (and semantically invalid) configurations</risk>
      <rollback>Remove the three if-raise blocks</rollback>
    </change>

    <change id="C2" priority="P1" source_item="F2">
      <file path="src/boost_shap_gii/indiv_reports.py" action="modify" />
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>Thread shap_scale_factor through generate_indiv_reports and apply it to all individual SHAP point estimates and CI bounds before output assembly. Ensures global and individual SHAP reports are on the same scale when back_transform_shap is active.</description>
      <spec>
1. indiv_reports.py - generate_indiv_reports signature (line 635):
   Add `shap_scale_factor: float = 1.0` as a keyword-only parameter after the existing `df_raw` parameter:
   ```python
   def generate_indiv_reports(
       ...,
       df_raw: Optional[pd.DataFrame] = None,
       shap_scale_factor: float = 1.0,
   ) -> None:
   ```

2. indiv_reports.py - scaling block (insert between lines 1049 and 1054):
   After class_labels resolution (line 1048 `ids_list = list(ids_target)`) and before the output loop (line 1054 `for i in range(N_target):`), insert a scaling block:
   ```python
   if shap_scale_factor != 1.0:
       point_shap = (point_shap.astype(np.float64) * shap_scale_factor).astype(np.float32)
       if point_shap_int is not None:
           point_shap_int = (point_shap_int.astype(np.float64) * shap_scale_factor).astype(np.float32)
       if mode == "inference":
           main_ci_lo_inf *= shap_scale_factor
           main_ci_hi_inf *= shap_scale_factor
           if int_ci_lo_inf is not None:
               int_ci_lo_inf *= shap_scale_factor
               int_ci_hi_inf *= shap_scale_factor
       else:
           shap_ci_buf *= shap_scale_factor
           if int_ci_buf is not None:
               int_ci_buf *= shap_scale_factor
       print(f"[INFO] Scaled individual SHAP values by shap_scale_factor={shap_scale_factor:.6f}")
   ```
   Key implementation notes:
   - point_shap and point_shap_int are reassigned (not in-place) because they are float32 arrays and need float64 intermediate precision for the multiplication.
   - main_ci_lo_inf, main_ci_hi_inf are scaled IN-PLACE (`*=`) because _compute_ci_inf_main (line 928) is a closure that captures these arrays by reference; reassignment would break the closure.
   - shap_ci_buf and int_ci_buf are scaled in-place; _compute_ci (line 1027) receives slices of these buffers in the output loop, so in-place scaling propagates correctly.
   - pred_ci_buf is NOT scaled: predictions are already back-transformed via output_transform at the call site; shap_scale_factor applies only to SHAP values.

3. predict.py - call site (lines 528-547):
   Add `shap_scale_factor=shap_scale_factor` to the generate_indiv_reports call:
   ```python
   generate_indiv_reports(
       ...,
       df_raw=df_raw,
       shap_scale_factor=shap_scale_factor,
   )
   ```

4. infer.py - call site (lines 589-611):
   Add `shap_scale_factor=shap_scale_factor` to the generate_indiv_reports call:
   ```python
   generate_indiv_reports(
       ...,
       df_raw=df_raw,
       shap_scale_factor=shap_scale_factor,
   )
   ```
      </spec>
      <dependencies>none</dependencies>
      <risk>low - for additive residualization (current use case), shap_scale_factor=1.0, so the block is a no-op; for multiplicative transforms (future use case), the block is exercised</risk>
      <rollback>Remove the scaling block and the shap_scale_factor parameter from the signature and call sites</rollback>
    </change>

    <change id="C3" priority="P1" source_item="F3">
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>Assert that load_transform_module returns non-None when transform_config.json indicates active transforms. Prevents silent failure if the config YAML is modified between training and prediction/inference.</description>
      <spec>
1. predict.py, after line 239 (transform_module = load_transform_module(config)):
   Insert immediately after the load_transform_module call:
   ```python
   if transform_module is None:
       raise ValueError(
           "transform_config.json indicates active transforms but "
           "load_transform_module returned None. Verify the config "
           "YAML contains a valid transformations block matching "
           "the training config."
       )
   ```

2. infer.py, after line 232 (transform_module = load_transform_module(config)):
   Insert immediately after the load_transform_module call:
   ```python
   if transform_module is None:
       raise ValueError(
           "transform_config.json indicates active transforms but "
           "load_transform_module returned None. Verify the config "
           "YAML contains a valid transformations block matching "
           "the training config."
       )
   ```
      </spec>
      <dependencies>none</dependencies>
      <risk>low - additive assertion; fires only when config is inconsistent between training and prediction/inference</risk>
      <rollback>Remove the two if-raise blocks</rollback>
    </change>

    <change id="C4" priority="P1" source_item="F4">
      <file path="src/boost_shap_gii/utils.py" action="modify" />
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <file path="src/boost_shap_gii/indiv_reports.py" action="modify" />
      <description>Implement per-bootstrap-iteration transforms in the bootstrap cache workers so each bootstrap model is trained on a freshly-fitted transform (matching the re-estimation structure of CatBoost's internal parameters). Additionally, compute per-fold alpha during training and store it in fold_transform_metadata.json entries; validate cross-fold alpha consistency. transform_config.json stays global.</description>
      <spec>
This change has four sub-parts:

### C4a: resolve_transform_path helper (utils.py)

Add a new helper function after load_transform_module (after line 622):
```python
def resolve_transform_path(config: dict) -> Optional[str]:
    """Resolve the absolute file path to the user's transform module.

    Returns None if no transformations block is present in config.
    """
    if "transformations" not in config:
        return None
    raw_path = config["transformations"]["file"]
    if not os.path.isabs(raw_path):
        base = config.get("data", {}).get("data_dir", os.getcwd())
        return os.path.join(base, raw_path)
    return raw_path
```
This is factored from load_transform_module's existing path-resolution logic (lines 596-603) so orchestrate_bootstrap_cache can pass the resolved path to workers without loading the module object (modules are not picklable).

### C4b: Per-fold alpha computation (train.py)

Modify train.py's alpha computation (lines 1022-1044) to run on EVERY fold instead of only fold 0:

1. Remove the `fold_idx == 0` guard from line 1022 so the alpha computation runs on every fold.

2. Store the computed alpha alongside each fold's user metadata. Inject a `"_pipeline_alpha"` field into each fold_meta dict BEFORE appending to all_fold_transform_meta (line 1019):
   ```python
   if transform_module is not None:
       y_train, y_val, fold_meta = transform_module.input_transform(...)
       ...
       all_fold_transform_meta.append(fold_meta)
       save_json_atomic(all_fold_transform_meta, ...)
   ```
   becomes:
   ```python
   if transform_module is not None:
       y_train, y_val, fold_meta = transform_module.input_transform(...)
       ...
       # Compute per-fold alpha
       if tx_cfg.get("back_transform_shap", False) and is_affine:
           probe_0 = np.zeros(len(val_idx))
           probe_1 = np.ones(len(val_idx))
           ot_0 = np.asarray(transform_module.output_transform(
               probe_0, fold_meta, tx_cfg.get("params", {}),
               df_raw=df_raw, row_indices=val_idx
           ), dtype=float)
           ot_1 = np.asarray(transform_module.output_transform(
               probe_1, fold_meta, tx_cfg.get("params", {}),
               df_raw=df_raw, row_indices=val_idx
           ), dtype=float)
           alpha_vec = ot_1 - ot_0
           if not np.allclose(alpha_vec, alpha_vec[0], rtol=1e-6):
               raise ValueError(
                   f"back_transform_shap=true but output_transform has "
                   f"non-constant slope across samples in fold {fold_idx}. "
                   f"This indicates a sample-dependent scale factor."
               )
           fold_alpha = float(alpha_vec[0])
       else:
           fold_alpha = 1.0
       fold_meta["_pipeline_alpha"] = fold_alpha
       all_fold_transform_meta.append(fold_meta)
       save_json_atomic(all_fold_transform_meta, ...)
   ```

3. Remove the old fold_idx == 0 alpha block entirely (lines 1022-1044). Replace it with a post-loop cross-fold validation AFTER the fold loop completes, BEFORE the global artifacts block (insert before line 1198):
   ```python
   if transform_module is not None and len(all_fold_transform_meta) > 0:
       fold_alphas = [fm.get("_pipeline_alpha", 1.0) for fm in all_fold_transform_meta]
       if not np.allclose(fold_alphas, fold_alphas[0], rtol=1e-6):
           raise ValueError(
               f"Cross-fold alpha inconsistency detected: {fold_alphas}. "
               f"The SHAP scale factor must be constant across folds for "
               f"the global shap_scale_factor to be well-defined."
           )
       shap_scale_factor = fold_alphas[0]
       print(f"[INFO] Cross-fold alpha validation passed: "
             f"shap_scale_factor={shap_scale_factor:.6f} (K={len(fold_alphas)} folds)")
   ```
   Note: shap_scale_factor must remain defined at this scope so it is available for the transform_config.json artifact (line 1210).

4. The `_pipeline_alpha` field is injected into the user's fold_meta dict. output_transform implementations that only read their own keys (e.g., "intercept", "slope") will ignore `_pipeline_alpha`. This is a reasonable assumption for user modules that follow the documented API contract.

### C4c: Per-bootstrap transforms in workers (indiv_reports.py)

1. _fit_and_save_refit signature (line 264): Add optional transform parameters:
   ```python
   def _fit_and_save_refit(
       b: int,
       k: int,
       sample_indices: np.ndarray,
       params_k: dict,
       X_train_parquet_path: str,
       y_train_path: str,
       nom_feats: list,
       task: str,
       out_path: str,
       df_raw_parquet_path: str = None,
       transform_module_path: str = None,
       tx_params: dict = None,
       outcome_col = None,
   ) -> None:
   ```

2. _fit_and_save_refit body (lines 279-284): Add conditional transform logic:
   ```python
   X_train = pd.read_parquet(X_train_parquet_path)

   if transform_module_path is not None:
       # Per-bootstrap transform: fit on the bootstrap resample
       df_raw = pd.read_parquet(df_raw_parquet_path)
       spec = importlib.util.spec_from_file_location("_transforms", transform_module_path)
       mod = importlib.util.module_from_spec(spec)
       spec.loader.exec_module(mod)
       y_boot, _, _ = mod.input_transform(
           df_raw, sample_indices, sample_indices, outcome_col, tx_params or {}
       )
       if not isinstance(y_boot, np.ndarray):
           y_boot = np.asarray(y_boot, dtype=float)
   else:
       y_arr = np.load(y_train_path, allow_pickle=True)
       if y_arr.ndim == 1:
           y_boot = y_arr[sample_indices]
       else:
           y_boot = y_arr[sample_indices, :]

   X_boot = X_train.iloc[sample_indices]
   ```
   Requires adding `import importlib.util` at the top of indiv_reports.py if not already present.

3. orchestrate_bootstrap_cache signature (line 480): Add optional transform parameters:
   ```python
   def orchestrate_bootstrap_cache(
       run_dir: str,
       X_train: pd.DataFrame,
       y_train: Union[pd.Series, pd.DataFrame],
       task: str,
       outcome_cols: list,
       nom_feats: list,
       config: dict,
       n_jobs: int,
       random_seed: int,
       *,
       cluster_ids: Optional[np.ndarray] = None,
       transform_module_path: Optional[str] = None,
       tx_params: Optional[dict] = None,
       df_raw: Optional[pd.DataFrame] = None,
       outcome_col = None,
   ) -> dict:
   ```

4. orchestrate_bootstrap_cache body: Serialize df_raw if transforms active (insert after line 566, before step 4):
   ```python
   df_raw_tmp = None
   if transform_module_path is not None and df_raw is not None:
       df_raw_tmp = os.path.join(cache_dir, "_df_raw_tmp.parquet")
       df_raw.to_parquet(df_raw_tmp)
       print(f"[INFO] Serialized df_raw for per-bootstrap transforms.")
   ```

5. orchestrate_bootstrap_cache _make_tasks (line 571): Thread transform args:
   ```python
   def _make_tasks():
       for b in range(B):
           s = shared_indices_list[b]
           for k in range(K):
               out_path = os.path.join(cache_dir, f"iter_{b:05d}", f"fold_{k}.cbm")
               yield (b, k, s, params[k], x_tmp, y_tmp, nom_feats, task, out_path,
                      df_raw_tmp, transform_module_path, tx_params, outcome_col)
   ```

6. orchestrate_bootstrap_cache executor.submit (line 585): Thread additional args:
   ```python
   futures = {
       executor.submit(
           _fit_and_save_refit, b, k, s, p, x_tmp, y_tmp, nom_feats, task, out,
           df_raw_p, tx_mod_p, tx_par, oc
       ): (b, k)
       for b, k, s, p, _, _, _, _, out, df_raw_p, tx_mod_p, tx_par, oc in tasks
   }
   ```

7. orchestrate_bootstrap_cache cleanup (line 604): Add df_raw_tmp to cleanup:
   ```python
   for tmp in (x_tmp, y_tmp, df_raw_tmp):
       if tmp is not None and os.path.exists(tmp):
           os.remove(tmp)
   ```

### C4d: Caller-side changes (predict.py)

1. predict.py lines 493-505: Remove the global input_transform call for bootstrap y conditioning. Replace with passing raw y and transform parameters to orchestrate_bootstrap_cache:
   ```python
   # (Delete lines 493-505: the global transform block)

   # Resolve transform module path for per-bootstrap transforms
   tx_module_path = None
   _outcome_col_boot = None
   if transform_module is not None:
       tx_module_path = resolve_transform_path(config)
       _outcome_col_boot = outcome_cols[0] if len(outcome_cols) == 1 else outcome_cols

   cache_summary = orchestrate_bootstrap_cache(
       run_dir=run_dir,
       X_train=X,
       y_train=y,  # raw y, not transformed
       task=task,
       outcome_cols=outcome_cols,
       nom_feats=nom_feats,
       config=config,
       n_jobs=n_jobs,
       random_seed=config["execution"]["random_seed"],
       cluster_ids=cluster_ids_indiv,
       transform_module_path=tx_module_path,
       tx_params=tx_info.get("params", {}) if tx_info is not None else None,
       df_raw=df_raw if transform_module is not None else None,
       outcome_col=_outcome_col_boot,
   )
   ```
   Requires adding `resolve_transform_path` to the imports from utils.py.
      </spec>
      <dependencies>none (C4a is internal to C4)</dependencies>
      <risk>medium - modifies the bootstrap training estimand and the worker serialization path; requires end-to-end dry-run testing to validate; however, for the current use case (additive residualization), the per-bootstrap transform produces the same result as the global transform at large N, so the change is statistically transparent for existing pipelines</risk>
      <rollback>Revert the four sub-parts: remove resolve_transform_path from utils.py; restore fold_idx == 0 alpha guard in train.py; restore the global input_transform call in predict.py; remove the transform parameters from _fit_and_save_refit and orchestrate_bootstrap_cache in indiv_reports.py</rollback>
    </change>

    <change id="C5" priority="P2" source_item="F5">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <description>Add [transformed scale] qualifier to fold-level metric print statements in train.py when outcome transforms are active, to prevent users from misinterpreting transformed-scale metrics as raw-scale performance.</description>
      <spec>
Modify the four print statements in the fold metric block (lines 1102, 1112, 1120/1122, 1126/1130):

1. Determine the label suffix once per fold, before the metric computation (insert before line 1092):
   ```python
   scale_label = " [transformed scale]" if transform_module is not None else ""
   ```

2. Update each print statement to append scale_label:
   - Line 1102: `print(f"  > Scores{scale_label}: RMSE={rmse:.3f}, R2={r2:.3f}")`
   - Line 1112: `print(f"  > Scores{scale_label}: Mean RMSE={metrics['rmse_mean']:.3f}")`
   - Line 1120: `print(f"  > Scores{scale_label}: Balanced Acc={acc:.3f}, AUC-OVR={auc:.3f}")`
   - Line 1122: `print(f"  > Scores{scale_label}: Balanced Acc={acc:.3f} (AUC-OVR skipped)")`
   - Line 1126: `print(f"  > Scores{scale_label}: AUC={auc:.3f}")`
   - Line 1130: `print(f"  > Scores{scale_label}: ACC={acc:.3f} (AUC Failed)")`

Note: For classification tasks (lines 1120-1130), the label will never fire because C1 halts before this point when transforms + classification are combined. Including the label in all print statements is defensive (applies uniformly regardless of task) and costs nothing.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - cosmetic change to print output only; no effect on computation or artifacts</risk>
      <rollback>Remove scale_label variable and revert print statements to original format</rollback>
    </change>
  </changes>
  <execution_order>C1, C3, C5, C2, C4</execution_order>
</implement_plan>
