<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-24T08:50:52-04:00" />
  <input_reports>
    <report path="boost-shap-gii_brainstorm_20260824_010000.md" mode="brainstorm" key_items="4" />
  </input_reports>

  <assumptions>
    <assumption id="A1">The `_get_effect_stratum` function (shap_utils.py:142) already handles unknown type strings by including them verbatim in the stratum label (e.g., `singleton_aggregate`). No modification to `_get_effect_stratum` is required for C1. Verified: line 151 does `feature_types.get(effect_name, feature_types.get(base_name, "continuous"))`, so if a feature's type is `"aggregate"`, the returned stratum is `singleton_aggregate`.</assumption>
    <assumption id="A2">The upfront smoke test uses a 20-row subset sampled from `df_raw` with a deterministic seed (the pipeline's `random_seed`). This is consistent with the brainstorm's locked design (T4) specifying "20-row subset of df_raw." The subset is drawn without replacement; if `len(df_raw) < 20`, the full dataframe is used.</assumption>
    <assumption id="A3">The affinity test for determining whether a transform is affine checks whether `output_transform(alpha * x + beta, metadata) ≈ alpha * output_transform(x, metadata) + (alpha - 1) * output_transform(0, metadata) + beta_offset` for multiple probe values. A simpler and more robust approach: compute `output_transform` on three linearly independent probe vectors and check whether the results lie on a plane (i.e., the function is affine). Operationalized: for probe vectors p1, p2, p3 in R^n, compute o1, o2, o3 = output_transform(p_i, metadata), then check `max|o3 - (o1 + (o2 - o1) * t)| < tol` where t is the linear interpolation coefficient. More simply: `alpha = (o2 - o1) / (p2 - p1)` element-wise, check all alphas are equal (constant slope), and `beta_i = o1 - alpha * p1`. This gives the affine decomposition `output_transform(x) = alpha * x + beta_i`. If the transform is affine, `alpha` is the SHAP scale factor.</assumption>
    <assumption id="A4">The `back_transform_shap` config key lives under the `transformations` block: `transformations.back_transform_shap`. This is consistent with T4 locking it as a property of the transformation, not a global SHAP option.</assumption>
    <assumption id="A5">`transform_config.json` is written to the same `run_dir` as other train-time artifacts (`feature_names.json`, `fold_assignments.json`, etc.). predict.py and infer.py load it from the same directory.</assumption>
    <assumption id="A6">The `shap_scale_factor` is a scalar float (not per-feature). This follows from the transform API contract: `input_transform` and `output_transform` operate on the outcome variable as a whole. For multi_regression, the transform API does not support per-target scaling (the user's transform script handles multi-target logic internally). The scale factor is computed from the first fold's metadata and applied uniformly. If a user needs per-target scaling in multi_regression, they would need separate transform scripts per target, which is outside the current API scope.</assumption>
    <assumption id="A7">The smoke test's `output_transform` call signature includes the optional `df_raw` and `row_indices` kwargs (as documented in the brainstorm's transform API contract). The smoke test passes the 20-row subset's raw data and indices to validate the full call signature.</assumption>
  </assumptions>

  <changes>
    <change id="C1" priority="P0" source_item="T1 (brainstorm), P0 action item 1">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Aggregate SHAP noise stratum split: change aggregate feature type from "continuous" to "aggregate" (line 604), creating a dedicated `singleton_aggregate` stratum in the Stratified Max Boruta Exceedance Test. Add a small-stratum warning when any stratum has fewer than 3 shadow features (after line 1156). No changes to `_get_effect_stratum` required (it already handles arbitrary type strings).</description>
      <spec>
**Site 1 (line 604):** Change `feature_types[group_name] = "continuous"` to `feature_types[group_name] = "aggregate"`.

**Site 2 (after line 1156, inside the stratum-count log block):** After the existing `print(f"[SHAP]   Stratum '{stratum}': {n_real_in} real, {n_shadow_in} shadow")` line, add:
```python
if n_shadow_in > 0 and n_shadow_in < 3:
    print(f"[SHAP]   WARNING: Stratum '{stratum}' has only {n_shadow_in} shadow "
          f"effect(s). Statistical power for significance detection is limited "
          f"(null exceedance rate = 1/{n_shadow_in + 1} = "
          f"{1/(n_shadow_in + 1):.0%}).")
```
The warning fires only for nonzero strata (zero-shadow strata already have the global-max fallback warning at line 1144). The `1/(k+1)` formula is the exact null exceedance probability from order-statistic theory.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - single type-string change plus informational warning; no behavioral change to the exceedance test logic</risk>
      <rollback>Revert line 604 to `"continuous"` and remove the warning block.</rollback>
    </change>

    <change id="C2" priority="P0" source_item="T4 (brainstorm), P0 action item 1">
      <file path="src/boost_shap_gii/utils.py" action="modify" />
      <description>Add transformations config defaults to `fill_config_defaults` and two new utility functions: `load_transform_module` (importlib-based module loader with callable validation) and `validate_transform_config` (required_cols column-existence check).</description>
      <spec>
**Site 1: `fill_config_defaults` (after line 446, after the `n_inner_repeats` default).** Add transformations block normalization, gated on presence of the `transformations` key:
```python
if "transformations" in config:
    tx = config["transformations"]
    if "file" not in tx:
        raise ValueError("transformations.file is required when the transformations block is present")
    _set(["transformations", "params"], {})
    _set(["transformations", "required_cols"], [])
    _set(["transformations", "back_transform_shap"], False)
```
The `file` key is required (not defaulted); its absence is a hard error. `params`, `required_cols`, and `back_transform_shap` are defaulted via `_set` (only if absent).

**Site 2: New function `load_transform_module` (after `fill_config_defaults`, before `_VALID_INDIV_SCALING_MODES` at line 513).** Signature:
```python
def load_transform_module(config: dict) -> Optional[ModuleType]:
```
Logic:
1. If `"transformations"` not in config, return None.
2. Resolve `config["transformations"]["file"]` to an absolute path (relative paths resolved against `config["data"]["data_dir"]` or CWD).
3. Validate the file exists; raise `FileNotFoundError` with a clear message if not.
4. Use `importlib.util.spec_from_file_location` + `importlib.util.module_from_spec` + `spec.loader.exec_module` to load the module.
5. Validate `input_transform` and `output_transform` are callable attributes; raise `AttributeError` if either is missing or not callable.
6. Return the loaded module.

Requires adding `import importlib.util` and `from types import ModuleType` to utils.py imports.

**Site 3: New function `validate_transform_config` (immediately after `load_transform_module`).** Signature:
```python
def validate_transform_config(required_cols: list, df: pd.DataFrame, stage: str) -> None:
```
Logic:
1. Compute `missing = [c for c in required_cols if c not in df.columns]`.
2. If missing, raise `ValueError(f"[{stage}] transformations.required_cols missing from dataframe: {missing}")`.

The `stage` parameter produces contextual error messages ("train", "predict", "infer").
      </spec>
      <dependencies>none</dependencies>
      <risk>low - additive functions with no modification to existing behavior; hard error on missing `file` key prevents silent misconfiguration</risk>
      <rollback>Remove the three additions (config block in fill_config_defaults, two new functions) and the two new imports.</rollback>
    </change>

    <change id="C7" priority="P0" source_item="T4 (brainstorm), P0 action item 1">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Add `shap_scale_factor` parameter to `_run_bootstrap_pipeline` and scale both real and shadow SHAP matrices before M/V/GII bootstrap computation. Thread the parameter from `run_shap_pipeline` through `_run_shap_for_slice`.</description>
      <spec>
**Site 1: `_run_bootstrap_pipeline` signature (line 967).** Add `shap_scale_factor: float = 1.0` as the last parameter (after `X_display`):
```python
def _run_bootstrap_pipeline(
    ...,
    X_display: Optional[pd.DataFrame] = None,
    shap_scale_factor: float = 1.0,
) -> pd.DataFrame:
```

**Site 2: Inside `_run_bootstrap_pipeline`, after line 991** (`SHAP_vals_shadow = ...`). Add scaling block:
```python
if shap_scale_factor != 1.0:
    SHAP_vals = SHAP_vals * shap_scale_factor
    if SHAP_vals_shadow is not None:
        SHAP_vals_shadow = SHAP_vals_shadow * shap_scale_factor
    print(f"[SHAP] Scaled SHAP values by alpha={shap_scale_factor:.6f} "
          f"(back_transform_shap=true)")
```
This scales ALL SHAP values (real + shadow) by the same constant alpha before any M/V/GII computation. The exceedance test is invariant: P(alpha * real > alpha * max_shadow) = P(real > max_shadow). All downstream artifacts (M, V, GII, microdata, spline fits, noise distributions) automatically reflect original-scale units.

**Site 3: `_run_shap_for_slice` (line 1354).** Read `shap_scale_factor` from ctx:
After line 1379 (`n_jobs = config["execution"]["n_jobs"]`), add:
```python
shap_scale_factor = ctx.get("shap_scale_factor", 1.0)
```

**Site 4: `_run_bootstrap_pipeline` call site (line 1495).** Pass the new parameter:
After `X_display=chunks_X[0] if inference_mode else None` (line 1507), add:
```python
shap_scale_factor=shap_scale_factor,
```
      </spec>
      <dependencies>none</dependencies>
      <risk>low - default value 1.0 is a no-op; scaling only activates when transform pipeline explicitly sets a non-unity factor; exceedance test invariance proven algebraically</risk>
      <rollback>Remove the new parameter from `_run_bootstrap_pipeline` signature, remove the scaling block, remove the ctx read in `_run_shap_for_slice`, and remove the kwarg from the call site.</rollback>
    </change>

    <change id="C3" priority="P0" source_item="T2, T4 (brainstorm), P0 action items 1 and 2">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <description>Full transformations API integration in train.py: imports, transform module loading, upfront smoke test with affinity check, scaler guard, in-fold transform application, first-fold alpha computation, and transform_config.json artifact write.</description>
      <spec>
**Site 1: Imports (line 27-39).** Add `load_transform_module` and `validate_transform_config` to the utils import block:
```python
from .utils import (
    ...,
    load_transform_module,
    validate_transform_config,
)
```

**Site 2: Transform module loading (after line 764, after outcome distribution diagnostics).** Insert between the outcome diagnostics block and the nominal type enforcement block (line 766):
```python
transform_module = load_transform_module(config)
if transform_module is not None:
    tx_cfg = config["transformations"]
    validate_transform_config(tx_cfg.get("required_cols", []), df_raw, "train")
    print(f"[INFO] Transformations module loaded: {tx_cfg['file']}")
```

**Site 3: Upfront smoke test (immediately after Site 2, before line 766).** When `transform_module is not None`:
```python
if transform_module is not None:
    seed = config["execution"]["random_seed"]
    n_smoke = min(20, len(df_raw))
    rng = np.random.RandomState(seed)
    smoke_idx = rng.choice(len(df_raw), size=n_smoke, replace=False)
    smoke_train = smoke_idx[:n_smoke // 2]
    smoke_val = smoke_idx[n_smoke // 2:]
    tx_params = tx_cfg.get("params", {})
    outcome_col = outcome_cols[0] if len(outcome_cols) == 1 else outcome_cols

    # 3a. Execute input_transform
    try:
        y_sm_train, y_sm_val, sm_meta = transform_module.input_transform(
            df_raw, smoke_train, smoke_val, outcome_col, tx_params
        )
    except Exception as e:
        raise RuntimeError(
            f"Smoke test: input_transform failed on {n_smoke}-row subset: {e}"
        ) from e

    # 3b. Shape validation
    if len(y_sm_train) != len(smoke_train):
        raise ValueError(
            f"Smoke test: input_transform returned y_train with length "
            f"{len(y_sm_train)}, expected {len(smoke_train)}"
        )
    if len(y_sm_val) != len(smoke_val):
        raise ValueError(
            f"Smoke test: input_transform returned y_val with length "
            f"{len(y_sm_val)}, expected {len(smoke_val)}"
        )

    # 3c. Finiteness check
    y_sm_all = np.concatenate([np.asarray(y_sm_train), np.asarray(y_sm_val)])
    if not np.all(np.isfinite(y_sm_all)):
        n_nonfinite = int(np.sum(~np.isfinite(y_sm_all)))
        raise ValueError(
            f"Smoke test: input_transform produced {n_nonfinite} non-finite "
            f"value(s) (NaN or Inf)"
        )

    # 3d. Metadata JSON-serializability
    try:
        json.dumps(sm_meta)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"Smoke test: input_transform metadata is not JSON-serializable: {e}"
        ) from e

    # 3e. output_transform round-trip
    try:
        y_sm_rt = transform_module.output_transform(
            np.asarray(y_sm_val, dtype=float), sm_meta, tx_params,
            df_raw=df_raw, row_indices=smoke_val
        )
    except Exception as e:
        raise RuntimeError(
            f"Smoke test: output_transform failed: {e}"
        ) from e
    if len(y_sm_rt) != len(smoke_val):
        raise ValueError(
            f"Smoke test: output_transform returned length {len(y_sm_rt)}, "
            f"expected {len(smoke_val)}"
        )

    # 3f. Affinity test (is the transform affine?)
    # Probe with three linearly independent vectors to detect affinity.
    # If output_transform(x) = alpha * x + beta(metadata), then
    # (o2 - o1) / (p2 - p1) should be constant across all elements.
    n_probe = len(smoke_val)
    p1 = np.zeros(n_probe)
    p2 = np.ones(n_probe)
    p3 = np.full(n_probe, 2.0)
    o1 = np.asarray(transform_module.output_transform(
        p1, sm_meta, tx_params, df_raw=df_raw, row_indices=smoke_val
    ), dtype=float)
    o2 = np.asarray(transform_module.output_transform(
        p2, sm_meta, tx_params, df_raw=df_raw, row_indices=smoke_val
    ), dtype=float)
    o3 = np.asarray(transform_module.output_transform(
        p3, sm_meta, tx_params, df_raw=df_raw, row_indices=smoke_val
    ), dtype=float)

    # Affine check: o3 should equal 2*o2 - o1 (linearity of slope)
    expected_o3 = 2.0 * o2 - o1
    atol = 1e-6 * (np.abs(o2 - o1).max() + 1e-10)
    is_affine = np.allclose(o3, expected_o3, atol=atol, rtol=1e-6)

    if tx_cfg.get("back_transform_shap", False) and not is_affine:
        raise ValueError(
            "back_transform_shap=true but the output_transform is not affine. "
            "SHAP back-transformation requires output_transform(x) = alpha * x + beta "
            "for a constant alpha. Non-affine transforms break SHAP additivity."
        )

    print(f"[INFO] Smoke test passed ({n_smoke} rows). "
          f"Transform is {'affine' if is_affine else 'non-affine'}.")
```

**Site 4: Scaler guard (line 846).** Change:
```python
if task == "multi_regression":
```
to:
```python
if task == "multi_regression" and transform_module is None:
```
This prevents the StandardScaler from fitting/persisting when transforms are active. The transform takes full ownership of the outcome space.

**Site 5: In-fold transform application (after line 936).** After `y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]`, insert:
```python
fold_meta = None
if transform_module is not None:
    y_train, y_val, fold_meta = transform_module.input_transform(
        df_raw, train_idx, val_idx, outcome_col, tx_cfg.get("params", {})
    )
    y_train = pd.Series(y_train, index=y.iloc[train_idx].index)
    y_val = pd.Series(y_val, index=y.iloc[val_idx].index)
```

**Site 6: First-fold alpha computation (immediately after Site 5, inside fold_idx == 0 guard).** When `back_transform_shap` is true and the transform is affine, compute the precise scale factor from real fold metadata:
```python
if transform_module is not None and fold_idx == 0:
    if tx_cfg.get("back_transform_shap", False) and is_affine:
        # Compute alpha from real fold: slope of output_transform
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
                "back_transform_shap=true but output_transform has "
                "non-constant slope across samples in fold 0. "
                "This indicates a sample-dependent scale factor."
            )
        shap_scale_factor = float(alpha_vec[0])
        print(f"[INFO] SHAP scale factor (alpha) = {shap_scale_factor:.6f}")
    else:
        shap_scale_factor = 1.0
```

**Site 7: transform_config.json write (after the fold loop, near the metadata save block around line 860-886).** After all folds complete, write the transform configuration artifact:
```python
if transform_module is not None:
    tx_artifact = {
        "active": True,
        "file": tx_cfg["file"],
        "params": tx_cfg.get("params", {}),
        "required_cols": tx_cfg.get("required_cols", []),
        "is_affine": is_affine,
        "back_transform_shap": tx_cfg.get("back_transform_shap", False),
        "shap_scale_factor": shap_scale_factor,
    }
    save_json_atomic(tx_artifact, os.path.join(run_dir, "transform_config.json"))
    print(f"[INFO] Saved transform_config.json")
```
Place this after the fold loop and after `fold_assignments.json` is written, in the post-loop metadata block.
      </spec>
      <dependencies>C2 (load_transform_module, validate_transform_config must exist in utils.py)</dependencies>
      <risk>medium - largest change set; multiple integration points in the training loop; scaler guard modifies existing behavior for transform users; smoke test introduces a pre-training validation gate that could halt the pipeline on legitimate but unusual transforms</risk>
      <rollback>Remove all 7 site additions. Revert the scaler guard condition to `if task == "multi_regression":`. Remove the `load_transform_module` and `validate_transform_config` imports.</rollback>
    </change>

    <change id="C4" priority="P0" source_item="T3, T4 (brainstorm), P0 action item 1">
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <description>Transformations API integration in predict.py: import `load_transform_module` from utils, load transform_config.json artifact, validate required_cols against df_raw, apply per-fold back-transformation in the OOF prediction loop, guard the scaler inverse-transform, and pass `shap_scale_factor` to the SHAP pipeline context.</description>
      <spec>
**Site 1: Imports (line 17-31).** Add `load_transform_module` to the utils import block.

**Site 2: Transform detection (after loading run_dir metadata, before the OOF prediction loop at line 249).** Load transform_config.json and validate required_cols:
```python
tx_config_path = os.path.join(run_dir, "transform_config.json")
transform_module = None
tx_info = None
shap_scale_factor = 1.0
if os.path.exists(tx_config_path):
    with open(tx_config_path) as f:
        tx_info = json.load(f)
    if tx_info.get("active", False):
        transform_module = load_transform_module(config)
        required_cols = tx_info.get("required_cols", [])
        if required_cols:
            missing = [c for c in required_cols if c not in df_raw.columns]
            if missing:
                raise ValueError(
                    f"[predict] transformations.required_cols missing from "
                    f"dataframe: {missing}"
                )
        shap_scale_factor = tx_info.get("shap_scale_factor", 1.0)
        print(f"[INFO] Transformations active (file: {tx_info['file']})")
```

**Site 3: Per-fold back-transformation (inside the OOF loop, after line 270).** After `oof_preds[val_idx] = preds`, apply output_transform when the transform module is active:
```python
if transform_module is not None:
    _, _, fold_meta = transform_module.input_transform(
        df_raw, np.where(fold_assignments != fold_idx)[0],
        val_idx, outcome_col, tx_info.get("params", {})
    )
    preds_bt = transform_module.output_transform(
        np.asarray(preds, dtype=float), fold_meta, tx_info.get("params", {}),
        df_raw=df_raw, row_indices=val_idx
    )
    oof_preds[val_idx] = preds_bt
```
Note: `input_transform` is called to regenerate the per-fold metadata needed by `output_transform`. The `y_train` and `y_val` returns are discarded. The back-transformed predictions overwrite the transformed-space predictions in `oof_preds`. This follows Jensen's inequality: per-fold back-transformation before averaging avoids bias.

The `outcome_col` variable must be defined earlier in predict.py; verify it exists or derive it from config.

**Site 4: Scaler guard (line 284).** Change:
```python
if task == "multi_regression" and os.path.exists(scaler_path):
```
to:
```python
if task == "multi_regression" and os.path.exists(scaler_path) and transform_module is None:
```

**Site 5: SHAP context (line 441-455).** After the shap_ctx dict construction, add the shap_scale_factor:
```python
if shap_scale_factor != 1.0:
    shap_ctx["shap_scale_factor"] = shap_scale_factor
```
      </spec>
      <dependencies>C2 (load_transform_module), C7 (shap_scale_factor parameter in run_shap_pipeline)</dependencies>
      <risk>low-medium - per-fold metadata regeneration adds compute but is necessary for correctness; scaler guard modifies existing behavior for transform users only; `outcome_col` variable availability needs verification during build</risk>
      <rollback>Remove all 5 site additions. Revert the scaler guard condition. Remove the `load_transform_module` import.</rollback>
    </change>

    <change id="C5" priority="P0" source_item="T3, T4 (brainstorm), P0 action item 1">
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>Transformations API integration in infer.py: import `load_transform_module` from utils, load transform_config.json artifact, validate required_cols against df_raw, apply per-model back-transformation in the ensemble prediction loop (before accumulation), guard both scaler inverse-transform sites, and pass `shap_scale_factor` to the SHAP pipeline context.</description>
      <spec>
**Site 1: Imports (line 20-34).** Add `load_transform_module` to the utils import block.

**Site 2: Transform detection (after loading train_dir metadata, before the ensemble loop at line 265).** Same pattern as C4 Site 2:
```python
tx_config_path = os.path.join(train_dir, "transform_config.json")
transform_module = None
tx_info = None
shap_scale_factor = 1.0
if os.path.exists(tx_config_path):
    with open(tx_config_path) as f:
        tx_info = json.load(f)
    if tx_info.get("active", False):
        transform_module = load_transform_module(config)
        required_cols = tx_info.get("required_cols", [])
        if required_cols:
            missing = [c for c in required_cols if c not in df_raw.columns]
            if missing:
                raise ValueError(
                    f"[infer] transformations.required_cols missing from "
                    f"dataframe: {missing}"
                )
        shap_scale_factor = tx_info.get("shap_scale_factor", 1.0)
        print(f"[INFO] Transformations active (file: {tx_info['file']})")
```

**Site 3: Per-model back-transformation (inside the ensemble loop, after line 277).** After `preds = model.predict(...)` / `model.predict_proba(...)`, back-transform each model's predictions BEFORE accumulation:
```python
if transform_module is not None:
    fold_idx_for_meta = k
    fold_assignments_path = os.path.join(train_dir, "fold_assignments.json")
    with open(fold_assignments_path) as f:
        fa = np.array(json.load(f))
    train_idx_k = np.where(fa != fold_idx_for_meta)[0]
    val_idx_k = np.where(fa == fold_idx_for_meta)[0]
    # Regenerate fold metadata from the training split
    # Use full inference dataset indices for output_transform
    _, _, fold_meta_k = transform_module.input_transform(
        df_raw, train_idx_k, val_idx_k,
        outcome_col, tx_info.get("params", {})
    )
    preds = transform_module.output_transform(
        np.asarray(preds, dtype=float), fold_meta_k, tx_info.get("params", {}),
        df_raw=df_raw, row_indices=np.arange(len(df_raw))
    )
```
Note: `fold_assignments.json` should be read once before the loop (not per-model). The implementation should hoist the read. Per-model back-transformation before accumulation ensures Jensen's inequality is respected for nonlinear (but invertible) transforms. For affine transforms, the result is identical to back-transforming the ensemble average.

**Site 4: Scaler guard, per-model (line 284).** Change:
```python
if task == "multi_regression" and _scaler_info is not None:
```
to:
```python
if task == "multi_regression" and _scaler_info is not None and transform_module is None:
```

**Site 5: Scaler guard, ensemble (line 328).** Change:
```python
if task == "multi_regression" and _scaler_info is not None:
```
to:
```python
if task == "multi_regression" and _scaler_info is not None and transform_module is None:
```

**Site 6: SHAP context (line 512-528).** After the shap_ctx dict construction, add:
```python
if shap_scale_factor != 1.0:
    shap_ctx["shap_scale_factor"] = shap_scale_factor
```
      </spec>
      <dependencies>C2 (load_transform_module), C7 (shap_scale_factor parameter in run_shap_pipeline)</dependencies>
      <risk>low-medium - per-model metadata regeneration adds compute; fold_assignments.json read should be hoisted outside the loop for efficiency; two scaler guard sites need independent verification; `outcome_col` variable availability needs verification during build</risk>
      <rollback>Remove all 6 site additions. Revert both scaler guard conditions. Remove the `load_transform_module` import.</rollback>
    </change>

    <change id="C6" priority="P1" source_item="P0 action items 1 and 2 (documentation)">
      <file path="INPUT_SPECIFICATION.md" action="modify" />
      <description>Document the transformations config block, aggregate noise stratum, back_transform_shap option, smoke test behavior, and transform_config.json artifact in the technical reference. Also update example_config_advanced.yaml with the transformations block template.</description>
      <spec>
**INPUT_SPECIFICATION.md updates:**
1. Add a new subsection under the config documentation for the `transformations` block:
   - `file` (string, required): path to the Python transform script
   - `params` (dict, optional, default {}): arbitrary parameters passed to transform functions
   - `required_cols` (list, optional, default []): column names required in df_raw for output_transform
   - `back_transform_shap` (bool, optional, default false): when true, SHAP values are scaled by alpha to report in original-scale units; requires an affine output_transform

2. Document the transform function API contract:
   - `input_transform(df_raw, train_idx, val_idx, outcome_col, params) -> (y_train, y_val, metadata)`
   - `output_transform(predictions, metadata, params, *, df_raw=None, row_indices=None) -> predictions`
   - Metadata must be JSON-serializable
   - Per-fold metadata enables sample-dependent transforms (e.g., residualization)

3. Document the upfront smoke test: runs once before the fold loop on a 20-row deterministic subset; checks execution, shapes, finiteness, metadata serializability, output_transform round-trip, and affinity.

4. Document the aggregate noise stratum: aggregate SHAP features are assigned to a dedicated `singleton_aggregate` stratum in the Stratified Max Boruta Exceedance Test. Note the small-stratum warning when fewer than 3 shadow features exist in any stratum.

5. Document transform_config.json inter-stage artifact fields: `active`, `file`, `params`, `required_cols`, `is_affine`, `back_transform_shap`, `shap_scale_factor`.

**example_config_advanced.yaml update:** Add a commented-out `transformations` block template with all keys documented.
      </spec>
      <dependencies>C1, C2, C3, C4, C5, C7 (documents all implemented features)</dependencies>
      <risk>low - documentation only</risk>
      <rollback>Revert the documentation additions.</rollback>
    </change>
  </changes>

  <execution_order>
    <!-- Phase 1: Independent foundations (can be parallelized) -->
    C1 (aggregate stratum, shap_utils.py line 604 + warning)
    C2 (utils.py: config defaults + load_transform_module + validate_transform_config)
    C7 (shap_utils.py: shap_scale_factor in _run_bootstrap_pipeline)
    <!-- Phase 2: Depends on C2 -->
    C3 (train.py: full transformations integration)
    <!-- Phase 3: Depends on C2 and C7 -->
    C4 (predict.py: transformations integration)
    C5 (infer.py: transformations integration)
    <!-- Phase 4: Depends on all above -->
    C6 (documentation)
  </execution_order>
</implement_plan>
