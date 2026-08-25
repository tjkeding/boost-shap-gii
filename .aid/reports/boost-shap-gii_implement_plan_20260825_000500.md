<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-25T00:05:00-04:00" />
  <input_reports>
    <report path="boost-shap-gii_brainstorm_20260824_235500.md" mode="brainstorm" key_items="1" />
    <report path="boost-shap-gii_implement_plan_20260824_230000.md" mode="implement (handoff)" key_items="4" />
  </input_reports>
  <changes>
    <change id="C1" priority="P0" source_item="brainstorm action_item P0 (H1 with refinement 1a)">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <description>Add a required_cols row-drop block after the outcome-missing drop (line 686)
      and before feature selection (line 688). Uses _tx_required_cols (inlined config access)
      to avoid shadowing the tx_cfg variable defined later at line 771. Ensures input_transform
      never receives NaN values in columns declared as required.</description>
      <spec>
Insert after line 686 (`raise ValueError("No data left after dropping rows with missing target.")`)
and before line 688 (`# 3. Feature Selection (THE NEW ENGINE)`).

The old_string for the edit anchor is:
```
    if len(df_raw) == 0:
        raise ValueError("No data left after dropping rows with missing target.")

    # 3. Feature Selection (THE NEW ENGINE)
```

Replace with:
```
    if len(df_raw) == 0:
        raise ValueError("No data left after dropping rows with missing target.")

    _tx_required_cols = config.get("transformations", {}).get("required_cols", [])
    if _tx_required_cols:
        _tx_missing_cols = [c for c in _tx_required_cols if c not in df_raw.columns]
        if _tx_missing_cols:
            raise KeyError(
                f"transformations.required_cols references columns not in dataset: "
                f"{_tx_missing_cols}"
            )
        pre_tx_len = len(df_raw)
        df_raw = df_raw.dropna(subset=_tx_required_cols)
        tx_dropped = pre_tx_len - len(df_raw)
        if tx_dropped > 0:
            print(f"[INFO] Dropped {tx_dropped} rows with missing "
                  f"transformations.required_cols value(s)")
        if len(df_raw) == 0:
            raise ValueError(
                "No data left after dropping rows with missing "
                "transformations.required_cols."
            )

    # 3. Feature Selection (THE NEW ENGINE)
```

Key refinement from brainstorm T1/1a: uses `_tx_required_cols` (inlined from
`config.get("transformations", {}).get("required_cols", [])`) instead of defining a
`tx_cfg` variable, which would shadow the `tx_cfg = config["transformations"]` at line 771.
The underscore prefix signals a scoped, transient variable.

For non-transform configs, `config.get("transformations", {})` returns `{}`, so
`_tx_required_cols` is `[]` and the entire block is a no-op.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - follows exact pattern of existing outcome-missing drop; no change to
      downstream logic; non-transform configs produce empty list so block is a no-op</risk>
      <rollback>Remove the inserted block (from _tx_required_cols through the empty-data guard)</rollback>
    </change>

    <change id="C2" priority="P0" source_item="brainstorm action_item P0 (H2 with refinements 1b, 1c)">
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <description>Mirror C1 in predict.py. Reads required_cols from transform_config.json
      (training artifact, architecturally correct source of truth for predict.py). Includes
      active flag check (brainstorm refinement 1b) and empty-data guard (refinement 1c) that
      the handoff plan omitted.</description>
      <spec>
Insert after line 114 (`print(f"[INFO] Dropped {dropped} rows with missing outcome(s) (Mirroring train.py).")`)
and before line 116 (`if len(outcome_cols) > 1:`).

The old_string for the edit anchor is:
```
        print(f"[INFO] Dropped {dropped} rows with missing outcome(s) (Mirroring train.py).")

    if len(outcome_cols) > 1:
```

Replace with:
```
        print(f"[INFO] Dropped {dropped} rows with missing outcome(s) (Mirroring train.py).")

    _tx_config_path_early = os.path.join(run_dir, "transform_config.json")
    if os.path.exists(_tx_config_path_early):
        with open(_tx_config_path_early) as _f:
            _tx_info_early = json.load(_f)
        if _tx_info_early.get("active", False):
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
                if len(df_raw) == 0:
                    raise ValueError(
                        "No data left after dropping rows with missing "
                        "transformations.required_cols (Mirroring train.py)."
                    )

    if len(outcome_cols) > 1:
```

Key refinements from brainstorm:
- 1b: `if _tx_info_early.get("active", False):` guard aligns with predict.py's existing
  active-check pattern at line 257. Prevents row-drop if a future change writes
  transform_config.json with active=False.
- 1c: `if len(df_raw) == 0: raise ValueError(...)` guard completes the train/predict
  mirroring contract (H1/C1 has this guard; H2 as-written did not).
- Source of truth: reads from transform_config.json (training artifact) rather than
  config["transformations"]["required_cols"]. This early read is scoped (underscore-prefixed
  variables); the later read at line 250 (which sets transform_module, tx_info,
  shap_scale_factor) is unchanged.
      </spec>
      <dependencies>C1 (predict.py mirrors train.py's filtering logic)</dependencies>
      <risk>low - mirrors existing pattern; reads transform_config.json which is a stable
      artifact from train.py; non-transform runs have no transform_config.json so the block
      is a no-op</risk>
      <rollback>Remove the inserted block (from _tx_config_path_early through the empty-data guard)</rollback>
    </change>

    <change id="C3" priority="P1" source_item="brainstorm action_item P0 (H3 unchanged)">
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>Add a NaN-baseline diagnostic warning in infer.py. No row drop (infer.py's
      contract is to predict on all samples). NaN in back-transformed predictions correctly
      signals that back-transformation is undefined for rows missing required_cols values.
      Warning gives the user full information to handle externally.</description>
      <spec>
Extend the existing `if transform_module is not None:` block at lines 286-289 by appending
the NaN warning after the fold_transform_metadata load.

The old_string for the edit anchor is:
```
    if transform_module is not None:
        fold_meta_path = os.path.join(train_dir, "fold_transform_metadata.json")
        with open(fold_meta_path) as f:
            _fold_transform_meta = json.load(f)

    for k, model_path in enumerate(model_files):
```

Replace with:
```
    if transform_module is not None:
        fold_meta_path = os.path.join(train_dir, "fold_transform_metadata.json")
        with open(fold_meta_path) as f:
            _fold_transform_meta = json.load(f)

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

    for k, model_path in enumerate(model_files):
```

The warning is inside the existing `if transform_module is not None:` block, so it only
executes when transforms are active. `tx_info` is defined at line 272 (inside the
transform_config.json load block at line 270-284), so it is in scope here.
      </spec>
      <dependencies>none (independent of C1/C2)</dependencies>
      <risk>low - warning-only; no change to computation or output format</risk>
      <rollback>Remove the warning block (from _tx_req through the print statement)</rollback>
    </change>

    <change id="C4" priority="P2" source_item="brainstorm action_item P0 (H4 unchanged)">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <description>Belt-and-suspenders internal assertion after the smoke test verifying no NaN
      remains in required_cols after C1's drop. Fires only on pipeline logic bugs (C1 removed
      or bypassed). Converts opaque LAPACK errors into clear pipeline diagnostics.</description>
      <spec>
Insert after line 862 (smoke test final print) and before line 864 (nominal encoding block).
This is inside the `if transform_module is not None:` block that starts at line 776.

The old_string for the edit anchor is:
```
        print(f"[INFO] Smoke test passed ({n_smoke} rows). "
              f"Transform is {'affine' if is_affine else 'non-affine'}.")

    # A. Force Nominal to String -> Category.
```

Replace with:
```
        print(f"[INFO] Smoke test passed ({n_smoke} rows). "
              f"Transform is {'affine' if is_affine else 'non-affine'}.")

        _tx_req_cols = tx_cfg.get("required_cols", [])
        for rc in _tx_req_cols:
            _rc_nan = int(df_raw[rc].isna().sum())
            if _rc_nan > 0:
                raise ValueError(
                    f"[INTERNAL] {_rc_nan} rows still have NaN in "
                    f"transformations.required_cols column '{rc}' after "
                    f"row-drop. This indicates a pipeline logic error."
                )

    # A. Force Nominal to String -> Category.
```

`tx_cfg` here is the variable defined at line 771 (`tx_cfg = config["transformations"]`),
which is in scope inside the `if transform_module is not None:` block. This is the correct
reference after C1's refinement 1a eliminated the shadowing variable.
      </spec>
      <dependencies>C1 (the assertion validates C1's row-drop completeness)</dependencies>
      <risk>low - assertion-only; fires only on internal pipeline bug; zero runtime cost</risk>
      <rollback>Remove the assertion block (from _tx_req_cols through the raise ValueError)</rollback>
    </change>
  </changes>
  <execution_order>C1, C4, C2, C3</execution_order>
  <dispatch_strategy>
    C1 and C4 modify train.py (non-overlapping regions: C1 at line 687, C4 at line 863).
    Dispatch as a single agent (train.py changes). C2 modifies predict.py. C3 modifies
    infer.py. Three agents total, all dispatchable in parallel (no cross-file dependencies).
    C1 must precede C4 within the train.py agent (C4 validates C1's drop).
  </dispatch_strategy>
</implement_plan>
