<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-24T23:00:00-04:00" />
  <note>HANDOFF DOCUMENT. Per project policy, CFTSI-behavioral does not edit the boost-shap-gii
  pipeline repo directly. This plan specifies changes to be executed in the boost-shap-gii
  repository (https://github.com/tjkeding/boost-shap-gii). File paths below are relative to
  the boost-shap-gii repo root (local editable install).</note>
  <input_reports>
    <report path="conversation_context" mode="error_diagnosis" key_items="4" />
  </input_reports>
  <context>
    The v1.5.0 transformations API introduced `required_cols` as a config field that tells the
    pipeline which columns the transform module needs. `validate_transform_config()` (utils.py:575)
    validates column EXISTENCE in the dataframe but does not enforce per-row completeness. The
    pipeline drops rows with missing outcomes (train.py:680, predict.py:111) but does not drop
    rows where `required_cols` values are missing. This means `input_transform` receives rows
    with NaN in `required_cols` columns, which crashes OLS fitting (LAPACK SVD fails on
    non-finite input).

    The gap affects three modules:
    - train.py: crashes at input_transform -> _fit_ols (observed in production)
    - predict.py: re-runs input_transform, would crash the same way
    - infer.py: calls output_transform on all rows; NaN-baseline rows produce NaN predictions
      silently (no crash, but no warning either)
  </context>
  <changes>
    <change id="H1" priority="P0" source_item="production_crash_train">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <description>Add a required_cols row-drop step after the outcome-missing drop (line 680)
      and BEFORE X/y derivation (line 752-757). The required_cols are available from the config
      without needing the transform module to be loaded (which happens later at line 769). This
      ensures the transform module's `input_transform` never receives NaN values in columns it
      declared as required.</description>
      <spec>
After the outcome-missing drop block (lines 678-686), insert a required_cols drop block.
The block reads `required_cols` directly from `config["transformations"]["required_cols"]`
(no module needed). This must execute BEFORE feature selection (line 688), X/y derivation
(line 752-757), and transform module loading (line 769).

Insert after line 686 (`raise ValueError("No data left...")`):

```python
# Drop rows with missing values in transformation required_cols
tx_cfg = config.get("transformations", {})
tx_required_cols = tx_cfg.get("required_cols", [])
if tx_required_cols:
    # Validate columns exist before attempting drop
    tx_missing_cols = [c for c in tx_required_cols if c not in df_raw.columns]
    if tx_missing_cols:
        raise KeyError(
            f"transformations.required_cols references columns not in dataset: "
            f"{tx_missing_cols}"
        )
    pre_tx_len = len(df_raw)
    df_raw = df_raw.dropna(subset=tx_required_cols)
    tx_dropped = pre_tx_len - len(df_raw)
    if tx_dropped > 0:
        print(f"[INFO] Dropped {tx_dropped} rows with missing "
              f"transformations.required_cols value(s)")
    if len(df_raw) == 0:
        raise ValueError(
            "No data left after dropping rows with missing "
            "transformations.required_cols."
        )
```

Downstream effects: X, y, fold_assignments, oof_preds all derive from df_raw after this
point, so alignment is automatic. The missingness report (line 725) runs after this drop,
which is acceptable (it reports missingness in features, not in required_cols). The
`[INFO] Feature Matrix: N rows x M columns` message at line ~860 will correctly reflect
the reduced N.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - follows exact pattern of existing outcome-missing drop; no change to
      downstream logic; non-transform configs have empty required_cols so the block is a no-op</risk>
      <rollback>Remove the inserted block</rollback>
    </change>

    <change id="H2" priority="P0" source_item="production_crash_predict">
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <description>Mirror H1 in predict.py. predict.py already mirrors train.py's outcome-missing
      drop (line 111, with comment "Mirroring train.py"). Add the same required_cols drop
      immediately after the outcome-missing drop. predict.py re-runs `input_transform` at line 295,
      so it needs the same row filtering.</description>
      <spec>
After the outcome-missing drop block (lines 110-114), insert the same required_cols drop
pattern as H1:

```python
# Drop rows with missing values in transformation required_cols (Mirroring train.py)
tx_config_path_early = os.path.join(run_dir, "transform_config.json")
if os.path.exists(tx_config_path_early):
    with open(tx_config_path_early) as _f:
        _tx_info_early = json.load(_f)
    _tx_req = _tx_info_early.get("required_cols", [])
    if _tx_req:
        _tx_missing = [c for c in _tx_req if c not in df_raw.columns]
        if _tx_missing:
            raise KeyError(
                f"[predict] transformations.required_cols references columns "
                f"not in dataset: {_tx_missing}"
            )
        pre_tx_len = len(df_raw)
        df_raw = df_raw.dropna(subset=_tx_req)
        tx_dropped = pre_tx_len - len(df_raw)
        if tx_dropped > 0:
            print(f"[INFO] Dropped {tx_dropped} rows with missing "
                  f"transformations.required_cols value(s) "
                  f"(Mirroring train.py)")
```

predict.py reads transform_config.json (the artifact saved by train.py) rather than the
raw config's transformations block, because predict.py operates on a completed run
directory. The required_cols are stored in transform_config.json at the
"required_cols" key (train.py line 1224).

NOTE: predict.py loads transform_config.json later at line 250. This early read is a
separate, scoped read for the row-drop only. The later read at line 250 (which sets
transform_module, tx_info, shap_scale_factor) is unchanged.
      </spec>
      <dependencies>H1 (predict.py mirrors train.py's filtering logic)</dependencies>
      <risk>low - mirrors existing pattern; reads transform_config.json which is a stable
      artifact from train.py</risk>
      <rollback>Remove the inserted block</rollback>
    </change>

    <change id="H3" priority="P1" source_item="exhaustive_trace_infer_path">
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>Add a warning when `output_transform` will receive NaN values in
      `required_cols`. infer.py intentionally does NOT drop rows (inference runs on all samples),
      so NaN-baseline rows produce NaN predictions after back-transformation. This is the correct
      behavior (back-transformation is undefined for rows missing the baseline), but it should
      not be silent. Add a diagnostic warning after loading the transform config and before
      the prediction loop.</description>
      <spec>
After the transform module loading block (around line 284-289), and before the prediction
loop (line 291), insert a diagnostic check:

```python
if transform_module is not None:
    fold_meta_path = os.path.join(train_dir, "fold_transform_metadata.json")
    with open(fold_meta_path) as f:
        _fold_transform_meta = json.load(f)

    # Warn about NaN-baseline rows (back-transformation undefined)
    _tx_req = tx_info.get("required_cols", [])
    if _tx_req:
        _nan_counts = {c: int(df_raw[c].isna().sum()) for c in _tx_req
                       if c in df_raw.columns and df_raw[c].isna().any()}
        if _nan_counts:
            _total_nan = sum(_nan_counts.values())
            print(f"[WARNING] {_total_nan} row(s) have NaN in "
                  f"transformations.required_cols: {_nan_counts}. "
                  f"Back-transformed predictions for these rows "
                  f"will be NaN.")
```

No row-drop. No change to downstream logic. Warning only.
      </spec>
      <dependencies>none (independent of H1/H2)</dependencies>
      <risk>low - warning-only; no change to computation or output format</risk>
      <rollback>Remove the warning block</rollback>
    </change>

    <change id="H4" priority="P2" source_item="smoke_test_coverage_gap">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <description>Strengthen the smoke test to detect NaN in required_cols even when random
      sampling misses the NaN rows. After the smoke test block (line 862), add a deterministic
      NaN-in-required_cols check that does not depend on random row selection. This is a
      belt-and-suspenders measure; H1 already drops NaN rows before the smoke test, so this
      check should never fire. It serves as a regression guard if H1's drop is accidentally
      removed or bypassed.</description>
      <spec>
After the smoke test's final print statement (line 861-862) and before the nominal encoding
block (line 864), insert:

```python
    # Deterministic NaN check for required_cols (belt-and-suspenders with row-drop)
    _tx_req_cols = tx_cfg.get("required_cols", [])
    for rc in _tx_req_cols:
        _rc_nan = int(df_raw[rc].isna().sum())
        if _rc_nan > 0:
            raise ValueError(
                f"[INTERNAL] {_rc_nan} rows still have NaN in "
                f"transformations.required_cols column '{rc}' after "
                f"row-drop. This indicates a pipeline logic error."
            )
```

This is a fail-fast internal assertion, not a user-facing validation. If it fires, it
indicates a bug in the row-drop logic (H1), not a user config error.
      </spec>
      <dependencies>H1 (the assertion validates H1's row-drop completeness)</dependencies>
      <risk>low - assertion-only; fires only on internal pipeline bug</risk>
      <rollback>Remove the assertion block</rollback>
    </change>
  </changes>
  <execution_order>H1, H2, H3, H4</execution_order>
  <downstream_impact_summary>
    The required_cols row-drop (H1/H2) reduces the effective sample size by the number of rows
    with NaN in any required_cols column. For CFTSI-behavioral childchild cell: 311 -> 309
    (2 rows dropped). The per-cell Ns after complete-case filtering are:
    childchild=309, childparent=262, parentchild=271, parentparent=271.

    No impact on:
    - Non-transform configs (required_cols is empty; H1 is a no-op)
    - Feature selection logic (operates on columns, not rows)
    - aggregate_shap validation (operates on feature groups, not rows)
    - SHAP computation (shap_scale_factor path unchanged; SHAP operates on post-drop data)
    - Fold stratification (operates on post-drop outcome distribution)
    - back_transform_shap logic (unchanged; operates on fold metadata from post-drop training)
  </downstream_impact_summary>
</implement_plan>
