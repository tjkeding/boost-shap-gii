<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-26T18:22:59Z" revised="2026-08-26T19:45:00Z" />
  <input_reports>
    <report path="boost-shap-gii_brainstorm_20260826_181550.md" mode="brainstorm" key_items="1" />
  </input_reports>
  <wiring_audit>
Complete grep audit of all `shap_scale_factor`, `fold_alpha`, `_pipeline_alpha`, `back_transform_shap`, `is_affine`, and `bootstrap_alpha` references across src/ and tests/. Every production reference accounted for in C1-C6 below. Test references (test_dry_run_transformations.py, test_indiv_reports_unit.py, test_transformations_wiring.py, test_transformations_api.py, test_shap_utils.py) will need updates in a subsequent /test cycle; they are NOT modified by this plan.

No-op guarantees:
- Transforms absent: transform_config.json does not exist; tx_info=None; fold_shap_scale_factors=None throughout. All new code guarded by `if fold_shap_scale_factors is not None` or `if shap_scale_factors is not None`. Zero behavioral change.
- Transforms present, back_transform_shap=False: C4/C5 gate fold_shap_scale_factors loading behind `if tx_info.get("back_transform_shap", False)`, so fold_shap_scale_factors stays None. Same no-op as above.
- Transforms present, back_transform_shap=True, is_affine=False: impossible (train.py:852-854 raises ValueError).
  </wiring_audit>
  <changes>
    <change id="C1" priority="P0" source_item="brainstorm T1/T2 action item">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <description>Replace the post-loop cross-fold alpha aggregation (hard halt + single scalar) with per-fold alpha array construction and an informational CV diagnostic. Change transform_config.json to store fold_shap_scale_factors (list of floats, one per fold) instead of shap_scale_factor (single float).</description>
      <spec>
Target: train.py lines 1209-1232.

1. Remove the rtol=1e-6 consistency check and hard halt (lines 1211-1216).
2. Remove `shap_scale_factor = fold_alphas[0]` (line 1217).
3. Remove the "Cross-fold alpha validation passed" print (lines 1218-1219).
4. Replace with informational diagnostic:

```python
if transform_module is not None and len(all_fold_transform_meta) > 0:
    fold_alphas = [fm.get("_pipeline_alpha", 1.0) for fm in all_fold_transform_meta]
    if len(fold_alphas) > 1:
        fa_arr = np.array(fold_alphas)
        cv_pct = float(np.std(fa_arr, ddof=1) / np.mean(fa_arr) * 100)
        print(f"[INFO] Per-fold shap_scale_factors: "
              f"{[round(a, 6) for a in fold_alphas]}, CV={cv_pct:.2f}%")
    else:
        print(f"[INFO] shap_scale_factor={fold_alphas[0]:.6f} (single fold)")
```

5. In the transform_config.json artifact (lines 1221-1232), replace `"shap_scale_factor": shap_scale_factor` with `"fold_shap_scale_factors": fold_alphas`. Keep all other fields unchanged.

The resulting transform_config.json schema:
```json
{
  "active": true,
  "file": "...",
  "params": {...},
  "required_cols": [...],
  "is_affine": true/false,
  "back_transform_shap": true/false,
  "fold_shap_scale_factors": [alpha_0, alpha_1, ...]
}
```

No-op when transforms absent: this entire block is inside `if transform_module is not None`, which is False when transforms are absent. No transform_config.json is emitted in that case.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - confined to post-loop metadata; no change to model fitting or SHAP computation logic</risk>
      <rollback>Restore the old rtol=1e-6 block and single shap_scale_factor field in transform_config.json</rollback>
    </change>

    <change id="C2" priority="P0" source_item="brainstorm T1 action item">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Change the GII-level SHAP scaling from a single scalar multiplication to per-row fold-specific scaling. The per-row alpha vector is constructed in _run_shap_for_slice from fold_shap_scale_factors (per-fold array from ctx) and the fold structure, then passed to _run_bootstrap_pipeline.</description>
      <spec>
Three functions are modified:

**A. _run_bootstrap_pipeline (line 961):**
Change parameter `shap_scale_factor: float = 1.0` to `shap_scale_factors: Optional[np.ndarray] = None`.
Add `from typing import Optional` if not already imported.

Replace the scaling block (lines 971-976):
```python
# OLD:
if shap_scale_factor != 1.0:
    SHAP_vals = SHAP_vals * shap_scale_factor
    if SHAP_vals_shadow is not None:
        SHAP_vals_shadow = SHAP_vals_shadow * shap_scale_factor
    print(...)

# NEW:
if shap_scale_factors is not None:
    SHAP_vals = SHAP_vals * shap_scale_factors[:, np.newaxis]
    if SHAP_vals_shadow is not None:
        SHAP_vals_shadow = SHAP_vals_shadow * shap_scale_factors[:, np.newaxis]
    n_unique = len(set(float(x) for x in shap_scale_factors))
    if n_unique == 1:
        print(f"[SHAP] Scaled SHAP values by alpha={shap_scale_factors[0]:.6f} "
              f"(back_transform_shap=true, uniform across folds)")
    else:
        print(f"[SHAP] Scaled SHAP values by per-fold alpha "
              f"(back_transform_shap=true, {len(shap_scale_factors)} rows, "
              f"{n_unique} distinct values)")
```

No-op: when `fold_shap_scale_factors` is absent from ctx (transforms not active), `shap_scale_factors` is None and the block is skipped entirely.

**B. _run_shap_for_slice (line 1349):**
Replace reading of `shap_scale_factor` (line 1375) with `fold_shap_scale_factors`:
```python
# OLD:
shap_scale_factor = ctx.get("shap_scale_factor", 1.0)

# NEW:
fold_shap_scale_factors = ctx.get("fold_shap_scale_factors", None)
```

After the fold merge (after lines 1462-1468 for predict mode, 1439-1461 for inference mode), construct the per-row alpha vector. Insert before the `_run_bootstrap_pipeline` call (before line 1491):

```python
shap_scale_factors = None
if fold_shap_scale_factors is not None:
    fsf = np.array(fold_shap_scale_factors, dtype=float)
    if inference_mode:
        N_obs = len(X_aligned)
        shap_scale_factors = np.repeat(fsf, N_obs)
    else:
        fold_assignments_arr = ctx.get("_fold_assignments")
        if fold_assignments_arr is not None:
            shap_scale_factors = fsf[fold_assignments_arr]
```

Note: `_fold_assignments` is stashed in ctx by `run_shap_pipeline` (see C2-C below). In inference mode, the pooled matrix has K*N rows in fold order (fold 0 first, fold 1 second, etc.), so `np.repeat(fsf, N_obs)` produces the correct per-row alpha vector.

Pass to `_run_bootstrap_pipeline`:
```python
# OLD:
shap_scale_factor=shap_scale_factor,

# NEW:
shap_scale_factors=shap_scale_factors,
```

**C. run_shap_pipeline (line 1509):**
After loading fold_assignments (lines 1562-1564), stash in ctx for downstream use by _run_shap_for_slice:

```python
# After: fold_assignments = np.array(json.load(f))
# Add:
ctx["_fold_assignments"] = fold_assignments
```

This addition is inside the `elif y is not None:` branch (predict mode, line 1561). Inference mode does not load fold_assignments here (inference splits are synthetic at line 1560), so `_fold_assignments` will be absent from ctx in inference mode. This is handled by the `if fold_assignments_arr is not None` guard in part B, and by the `if inference_mode:` branch using `np.repeat` instead.
      </spec>
      <dependencies>none (API-compatible with C4/C5 which supply the new ctx key)</dependencies>
      <risk>medium - touches the core GII SHAP scaling path; must preserve behavior when fold_shap_scale_factors is None (no-scaling default)</risk>
      <rollback>Restore scalar shap_scale_factor parameter and line 972 scalar multiplication</rollback>
    </change>

    <change id="C6" priority="P0" source_item="brainstorm T1 action item (exact per-bootstrap alpha)">
      <file path="src/boost_shap_gii/indiv_reports.py" action="modify" />
      <description>Compute and persist exact per-bootstrap-refit alpha values during cache construction. Modifies _fit_and_save_refit (compute and return per-refit alpha) and orchestrate_bootstrap_cache (thread flags, collect alphas, save bootstrap_alphas.npy). This change produces the artifact that C3's CI sections consume.</description>
      <spec>
**A. _fit_and_save_refit signature (lines 265-279):**
Add two new parameters after `outcome_col`:

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
    outcome_col=None,
    back_transform_shap: bool = False,
    is_affine: bool = False,
) -> float:
```

Return type changes from `None` to `float` (the bootstrap alpha, or 1.0 if no alpha computation needed).

**B. _fit_and_save_refit body (after line 297):**
Currently the third return of input_transform is discarded (`y_boot, _, _ = ...`). Capture it:

```python
# OLD (line 295-297):
y_boot, _, _ = mod.input_transform(
    df_raw, sample_indices, sample_indices, outcome_col, tx_params or {}
)

# NEW:
y_boot, _, boot_meta = mod.input_transform(
    df_raw, sample_indices, sample_indices, outcome_col, tx_params or {}
)
```

After the model is saved (after line 323 `m.save_model(out_path)`), compute and return alpha:

```python
    m.fit(pool)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    m.save_model(out_path)

    if back_transform_shap and is_affine and transform_module_path is not None:
        probe_0 = np.zeros(len(sample_indices))
        probe_1 = np.ones(len(sample_indices))
        ot_0 = np.asarray(mod.output_transform(
            probe_0, boot_meta, tx_params or {},
            df_raw=df_raw, row_indices=sample_indices
        ), dtype=float)
        ot_1 = np.asarray(mod.output_transform(
            probe_1, boot_meta, tx_params or {},
            df_raw=df_raw, row_indices=sample_indices
        ), dtype=float)
        alpha_vec = ot_1 - ot_0
        if not np.allclose(alpha_vec, alpha_vec[0], rtol=1e-6):
            raise ValueError(
                f"Bootstrap refit (b={b}, k={k}): output_transform has "
                f"non-constant slope across samples. "
                f"Range: [{alpha_vec.min():.8f}, {alpha_vec.max():.8f}]."
            )
        return float(alpha_vec[0])
    return 1.0
```

When `transform_module_path is None` (no transforms), the function never enters the input_transform branch (line 290 check), so `boot_meta` is never referenced, and the alpha block is skipped: returns 1.0.

When `back_transform_shap=False` or `is_affine=False`: alpha block is skipped, returns 1.0.

The within-sample constancy check (rtol=1e-6) is appropriate here: it verifies the transform is affine (constant slope) for THIS bootstrap resample, matching the train.py fold-level check at line 1039. This is a safety net, not an inter-fold comparison.

**C. orchestrate_bootstrap_cache signature (lines 501-517):**
Add two new keyword parameters:

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
    outcome_col=None,
    back_transform_shap: bool = False,
    is_affine: bool = False,
) -> dict:
```

**D. orchestrate_bootstrap_cache body:**

D1. Pre-allocate alpha collection array (after line 541 `K = len(model_files)`):
```python
boot_alphas = np.ones((B, K), dtype=np.float64)
```

D2. Thread new params to workers in task generator (lines 602-608). Add `back_transform_shap` and `is_affine` to the yield tuple:

```python
def _make_tasks():
    for b in range(B):
        s = shared_indices_list[b]
        for k in range(K):
            out_path = os.path.join(cache_dir, f"iter_{b:05d}", f"fold_{k}.cbm")
            yield (b, k, s, params[k], x_tmp, y_tmp, nom_feats, task, out_path,
                   df_raw_tmp, transform_module_path, tx_params, outcome_col,
                   back_transform_shap, is_affine)
```

D3. Update the executor.submit call (lines 616-621) to pass the two new args:

```python
futures = {
    executor.submit(
        _fit_and_save_refit, b, k, s, p, x_tmp, y_tmp, nom_feats, task, out,
        df_raw_p, tx_mod_p, tx_par, oc, bts, ia
    ): (b, k)
    for b, k, s, p, _, _, _, _, out, df_raw_p, tx_mod_p, tx_par, oc, bts, ia in tasks
}
```

D4. Collect returned alpha values (lines 623-631). Replace `fut.result()` (currently discarded) with alpha collection:

```python
for fut in concurrent.futures.as_completed(futures):
    b_idx, k_idx = futures[fut]
    try:
        boot_alpha = fut.result()
    except Exception as exc:
        raise RuntimeError(
            f"Bootstrap refit failed at iteration b={b_idx}, fold k={k_idx}: {exc}"
        ) from exc
    boot_alphas[b_idx, k_idx] = boot_alpha
    completed += 1
    if completed % max(1, total // 10) == 0:
        print(f"[INFO] Bootstrap refits: {completed}/{total} complete.")
```

D5. Save bootstrap_alphas.npy after all futures complete (after line 635 "All refits complete" print), gated by back_transform_shap:

```python
if back_transform_shap and is_affine:
    alphas_path = os.path.join(cache_dir, "bootstrap_alphas.npy")
    np.save(alphas_path, boot_alphas)
    n_unique = len(set(boot_alphas.ravel()))
    print(f"[INFO] Saved per-bootstrap alphas ({B}x{K} matrix, "
          f"{n_unique} distinct values) to {alphas_path}.")
```

When `back_transform_shap=False`: all workers return 1.0, no file is saved. No-op.

D6. Add `bootstrap_alphas_saved` to bootstrap_metadata.json (lines 643-655):

```python
meta = {
    ...existing fields...,
    "bootstrap_alphas_saved": back_transform_shap and is_affine,
}
```

This documents whether bootstrap_alphas.npy was computed, enabling future diagnostic queries without probing the filesystem.
      </spec>
      <dependencies>none (internal to indiv_reports.py; does not depend on C3's signature change since it modifies different functions)</dependencies>
      <risk>medium - modifies the bootstrap worker and cache orchestrator; must ensure the return value propagates correctly through ProcessPoolExecutor futures. Within-sample constancy check provides a safety net for pathological transforms.</risk>
      <rollback>Remove the two new parameters from both functions, revert _fit_and_save_refit to return None, remove alpha collection and bootstrap_alphas.npy emission from orchestrate_bootstrap_cache</rollback>
    </change>

    <change id="C3" priority="P0" source_item="brainstorm T1 action item">
      <file path="src/boost_shap_gii/indiv_reports.py" action="modify" />
      <description>Replace the post-hoc single-scalar shap_scale_factor scaling block in generate_indiv_reports with: (a) inline per-fold scaling at each point-estimate computation site using fold_shap_scale_factors[k], and (b) exact per-bootstrap alpha scaling at each CI accumulation site using bootstrap_alphas.npy (produced by C6). Point estimates use the original fold's alpha (exact for the original fold model). Bootstrap CI uses each refit's own alpha (exact for that bootstrap refit's transform parameters).</description>
      <spec>
**A. Signature change (line 688):**
```python
# OLD:
shap_scale_factor: float = 1.0,

# NEW:
fold_shap_scale_factors: Optional[List[float]] = None,
```

Verify `from typing import Optional, List` is present (check existing imports; List may need to be added).

**B. Load bootstrap_alphas.npy (after line 717 `cache_dir = ...`):**
Insert after cache_dir assignment, gated by fold_shap_scale_factors:

```python
boot_alphas = None
if fold_shap_scale_factors is not None:
    alphas_path = os.path.join(cache_dir, "bootstrap_alphas.npy")
    if not os.path.exists(alphas_path):
        raise FileNotFoundError(
            f"bootstrap_alphas.npy not found at {alphas_path}. "
            "The bootstrap cache was built without per-refit alpha "
            "computation. Re-run predict.py with the current version "
            "to rebuild the cache with per-bootstrap scale factors."
        )
    boot_alphas = np.load(alphas_path)
    if boot_alphas.shape != (B, K):
        raise ValueError(
            f"bootstrap_alphas.npy shape {boot_alphas.shape} does not match "
            f"expected (B={B}, K={K}). Re-run predict.py to rebuild."
        )
```

When `fold_shap_scale_factors is None` (no transforms, or back_transform_shap=False): `boot_alphas` stays None. No file access, no error. No-op.

**C. Training-mode point estimates (lines 812-828):**
After `sv = _shap_single(orig_models[k], pool_fold, task, X_fold.shape[0])` (line 818), apply per-fold scaling before assignment. Insert between sv computation and point_shap assignment:

```python
sv = _shap_single(orig_models[k], pool_fold, task, X_fold.shape[0])
pv = _predict_single(orig_models[k], pool_fold, task)
if fold_shap_scale_factors is not None:
    sv = (sv.astype(np.float64) * fold_shap_scale_factors[k]).astype(np.float32)
# ... (existing output_transform block at lines 820-826 unchanged)
point_shap[fold_mask] = sv
point_y_pred[fold_mask] = pv
```

SHAP scaling and prediction back-transformation are independent operations: SHAP scaling converts SHAP values from transformed-outcome units to original-outcome units; prediction back-transformation converts predictions from transformed scale to original scale. Both use fold k's transform parameters, but via different mechanisms (alpha multiplication for SHAP, output_transform function call for predictions).

**D. Inference-mode point estimates (lines 843-854):**
Scale each fold's SHAP before accumulating:

```python
for k in range(K):
    sv = _shap_single(orig_models[k], pool_all, task, N_target)
    pv = _predict_single(orig_models[k], pool_all, task)
    # ... (existing output_transform block at lines 846-852 unchanged)
    if fold_shap_scale_factors is not None:
        shap_accum += sv.astype(np.float64) * fold_shap_scale_factors[k]
    else:
        shap_accum += sv
    pred_accum += pv
```

Note: predictions are NOT scaled by alpha; they go through output_transform for back-transformation. Only SHAP values are scaled.

**E. Training-mode interaction point estimates (lines 870-877):**
After `sv_int = _shap_interaction_single(...)` (line 876), apply per-fold scaling:

```python
sv_int = _shap_interaction_single(orig_models[k], pool_fold)
if fold_shap_scale_factors is not None:
    sv_int = (sv_int.astype(np.float64) * fold_shap_scale_factors[k]).astype(np.float32)
point_shap_int[fold_mask] = sv_int
```

**F. Inference-mode interaction point estimates (lines 878-884):**
Scale each fold's interaction SHAP before accumulating:

```python
for k in range(K):
    sv_int = _shap_interaction_single(orig_models[k], pool_all)
    if fold_shap_scale_factors is not None:
        int_accum += sv_int.astype(np.float64) * fold_shap_scale_factors[k]
    else:
        int_accum += sv_int
```

**G. Inference-mode CI main SHAP (lines 914-919):**
Scale each fold's bootstrap SHAP by that refit's exact alpha (from boot_alphas[b, k]) before averaging:

```python
for k in range(K):
    sv_k = _shap_single(boot_models[k], pool_tgt, task, N_target)
    if boot_alphas is not None:
        shap_iter_folds[k] = (sv_k.astype(np.float64) * boot_alphas[b, k]).astype(np.float32)
    else:
        shap_iter_folds[k] = sv_k
```

Note: `boot_alphas[b, k]` is the alpha computed from bootstrap iteration b's refit of fold k's model, using that bootstrap resample's own fitted transform parameters. This is the exact alpha, not an approximation.

**H. Inference-mode CI interaction SHAP (lines 925-927):**
Same treatment:

```python
for k in range(K):
    sv_int_k = _shap_interaction_single(boot_models[k], pool_tgt)
    if boot_alphas is not None:
        int_iter_folds[k] = (sv_int_k.astype(np.float64) * boot_alphas[b, k]).astype(np.float32)
    else:
        int_iter_folds[k] = sv_int_k
```

**I. Training-mode CI main SHAP (lines 1022-1026):**
Scale per-fold bootstrap SHAP by that refit's exact alpha before storing:

```python
pool_tgt = Pool(X_target, cat_features=cat_feats)
for k in range(K):
    sv_k = _shap_single(boot_models[k], pool_tgt, task, N_target)
    pv_k = _predict_single(boot_models[k], pool_tgt, task).astype(np.float32)
    if boot_alphas is not None:
        shap_iter_folds[k] = (sv_k.astype(np.float64) * boot_alphas[b, k]).astype(np.float32)
    else:
        shap_iter_folds[k] = sv_k
    pred_iter_folds[k] = pv_k
```

Note: predictions are NOT scaled by alpha (they go through output_transform separately).

**J. Training-mode CI interaction SHAP (lines 1028-1035):**
```python
if compute_interactions:
    int_iter_folds = np.zeros(...)
    for k in range(K):
        sv_int_k = _shap_interaction_single(boot_models[k], pool_tgt)
        if boot_alphas is not None:
            int_iter_folds[k] = (sv_int_k.astype(np.float64) * boot_alphas[b, k]).astype(np.float32)
        else:
            int_iter_folds[k] = sv_int_k
```

**K. Remove post-hoc scaling block (lines 1085-1099):**
Delete the entire `if shap_scale_factor != 1.0:` block (15 lines). Per-fold/per-bootstrap scaling is now applied inline at each computation point (C through J above).

**L. Add diagnostic print (at the location of the removed block):**
```python
if fold_shap_scale_factors is not None:
    n_unique_fold = len(set(fold_shap_scale_factors))
    if n_unique_fold == 1:
        print(f"[INFO] Scaled individual SHAP values by "
              f"shap_scale_factor={fold_shap_scale_factors[0]:.6f} (uniform)")
    else:
        print(f"[INFO] Scaled individual SHAP values by per-fold "
              f"shap_scale_factors ({n_unique_fold} distinct values)")
    if boot_alphas is not None:
        n_unique_boot = len(set(boot_alphas.ravel()))
        print(f"[INFO] Scaled bootstrap CI SHAP values by per-refit "
              f"alphas ({boot_alphas.shape[0]}x{boot_alphas.shape[1]} matrix, "
              f"{n_unique_boot} distinct values)")
```

**Summary of scaling authority by computation site:**

| Site | Alpha source | Rationale |
|---|---|---|
| Point estimates (C,D,E,F) | fold_shap_scale_factors[k] | Original fold model uses original fold transform params |
| Inference CI (G,H) | boot_alphas[b, k] | Bootstrap refit uses bootstrap resample's transform params |
| Training CI (I,J) | boot_alphas[b, k] | Bootstrap refit uses bootstrap resample's transform params |

No-op: when fold_shap_scale_factors is None, boot_alphas is None, and every `if` guard evaluates to False. Zero behavioral change from the pre-fix code path (which never reaches line 1085 with shap_scale_factor != 1.0 when transforms are absent).
      </spec>
      <dependencies>C6 (bootstrap_alphas.npy must exist when fold_shap_scale_factors is not None)</dependencies>
      <risk>medium - multiple insertion points across training and inference code paths for both point estimates and CI; each must apply the correct alpha source (fold vs. bootstrap) at the correct scope (per-fold model, not after accumulation). The separation table above is the invariant to verify.</risk>
      <rollback>Restore single-scalar shap_scale_factor parameter and post-hoc scaling block at lines 1085-1099</rollback>
    </change>

    <change id="C4" priority="P0" source_item="brainstorm T1 action item">
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <description>Load fold_shap_scale_factors from transform_config.json (replacing single shap_scale_factor), pass to shap_ctx, generate_indiv_reports, and orchestrate_bootstrap_cache (with back_transform_shap + is_affine flags).</description>
      <spec>
**A. Load new artifact format (lines 235-256):**
Replace `shap_scale_factor = 1.0` declaration (line 235) and `shap_scale_factor = tx_info.get("shap_scale_factor", 1.0)` (line 256) with:

```python
# Replace line 235:
fold_shap_scale_factors = None

# Replace line 256:
if tx_info.get("back_transform_shap", False):
    if "fold_shap_scale_factors" not in tx_info:
        raise ValueError(
            "transform_config.json uses the legacy single-scalar "
            "shap_scale_factor format. Re-run train.py with the current "
            "version to regenerate per-fold scale factors."
        )
    fold_shap_scale_factors = tx_info["fold_shap_scale_factors"]
```

No-op: when `back_transform_shap` is False (or tx_info is None because transforms are absent), `fold_shap_scale_factors` stays None.

**B. Pass to shap_ctx (lines 490-491):**
Replace:
```python
# OLD:
if shap_scale_factor != 1.0:
    shap_ctx["shap_scale_factor"] = shap_scale_factor

# NEW:
if fold_shap_scale_factors is not None:
    shap_ctx["fold_shap_scale_factors"] = fold_shap_scale_factors
```

**C. Pass back_transform_shap and is_affine to orchestrate_bootstrap_cache (lines 516-531):**
Add two keyword arguments to the existing call:

```python
cache_summary = orchestrate_bootstrap_cache(
    ...existing args...,
    back_transform_shap=tx_info.get("back_transform_shap", False) if tx_info else False,
    is_affine=tx_info.get("is_affine", False) if tx_info else False,
)
```

No-op: when tx_info is None, both flags are False, and orchestrate_bootstrap_cache skips alpha computation entirely.

**D. Pass to generate_indiv_reports (line 559):**
Replace:
```python
# OLD:
shap_scale_factor=shap_scale_factor,

# NEW:
fold_shap_scale_factors=fold_shap_scale_factors,
```
      </spec>
      <dependencies>C1 (artifact format), C2 (shap_utils API), C3 (indiv_reports generate_indiv_reports API), C6 (orchestrate_bootstrap_cache API)</dependencies>
      <risk>low - wiring changes only; no algorithmic logic</risk>
      <rollback>Restore single shap_scale_factor loading and passthrough, remove back_transform_shap/is_affine args from orchestrate_bootstrap_cache call</rollback>
    </change>

    <change id="C5" priority="P0" source_item="brainstorm T1 action item">
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>Load fold_shap_scale_factors from transform_config.json (replacing single shap_scale_factor), pass to shap_ctx and generate_indiv_reports. infer.py does NOT call orchestrate_bootstrap_cache (confirmed by grep: only predict.py calls it), so no C6-related wiring is needed here.</description>
      <spec>
**A. Load new artifact format (lines 227-248):**
Replace `shap_scale_factor = 1.0` declaration (line 227) and `shap_scale_factor = tx_info.get("shap_scale_factor", 1.0)` (line 248) with:

```python
# Replace line 227:
fold_shap_scale_factors = None

# Replace line 248:
if tx_info.get("back_transform_shap", False):
    if "fold_shap_scale_factors" not in tx_info:
        raise ValueError(
            "transform_config.json uses the legacy single-scalar "
            "shap_scale_factor format. Re-run train.py with the current "
            "version to regenerate per-fold scale factors."
        )
    fold_shap_scale_factors = tx_info["fold_shap_scale_factors"]
```

No-op: when `back_transform_shap` is False (or tx_info is None because transforms are absent), `fold_shap_scale_factors` stays None.

**B. Pass to shap_ctx (lines 570-571):**
Replace:
```python
# OLD:
if shap_scale_factor != 1.0:
    shap_ctx["shap_scale_factor"] = shap_scale_factor

# NEW:
if fold_shap_scale_factors is not None:
    shap_ctx["fold_shap_scale_factors"] = fold_shap_scale_factors
```

**C. Pass to generate_indiv_reports (line 625):**
Replace:
```python
# OLD:
shap_scale_factor=shap_scale_factor,

# NEW:
fold_shap_scale_factors=fold_shap_scale_factors,
```

Inference-mode data boundary: infer.py accesses only pre-computed artifacts from train_dir (models, transform_config.json, fold_transform_metadata.json, bootstrap_refits/ cache including bootstrap_alphas.npy). No training data is accessed. bootstrap_alphas.npy was produced by orchestrate_bootstrap_cache in predict.py.
      </spec>
      <dependencies>C1 (artifact format), C2 (shap_utils API), C3 (indiv_reports generate_indiv_reports API)</dependencies>
      <risk>low - wiring changes only; no algorithmic logic</risk>
      <rollback>Restore single shap_scale_factor loading and passthrough</rollback>
    </change>
  </changes>
  <execution_order>Phase 1: C1, C2, C6 (independent: C1 modifies train.py, C2 modifies shap_utils.py, C6 modifies _fit_and_save_refit and orchestrate_bootstrap_cache in indiv_reports.py). Phase 2: C3 (modifies generate_indiv_reports in indiv_reports.py; depends on C6 for bootstrap_alphas.npy artifact). Phase 3: C4, C5 (independent: C4 modifies predict.py, C5 modifies infer.py; both depend on C1+C2+C3+C6 APIs).</execution_order>
  <test_impact_note>Existing tests referencing the old shap_scale_factor API (test_dry_run_transformations.py: 3 tests; test_indiv_reports_unit.py: 2 tests; test_transformations_wiring.py: 4 assertions; test_transformations_api.py: 1 assertion; test_shap_utils.py: 1 test class) will need re-expression in a subsequent /test cycle. These tests are NOT modified by this plan (per /implement write-scope constraints). The test_dry_run_transformations.py 32-error fixture failure is the P0 bug this plan resolves.</test_impact_note>
</implement_plan>
