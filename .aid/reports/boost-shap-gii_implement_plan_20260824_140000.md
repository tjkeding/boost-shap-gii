<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-24T14:00:00-04:00" />
  <input_reports>
    <report path="(conversation)" mode="dry-run-finding" key_items="1" />
  </input_reports>

  <assumptions_and_decisions>
    None. The bug, root cause, and fix strategy are unambiguous from code inspection.
  </assumptions_and_decisions>

  <changes>
    <change id="C1" priority="P0" source_item="dry-run IndexError: infer.py passes inference df_raw to input_transform with training fold indices">
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>
        Bug: infer.py line 308 passes the inference-time df_raw (loaded from --data) to
        transform_module.input_transform() with fold indices from the training-time
        fold_assignments.json. When the inference dataset has fewer rows than the training
        set, training fold indices (e.g., index 41) exceed the inference dataframe's row
        count (e.g., 40 rows), causing IndexError.

        Root cause: predict.py's identical call works because predict.py loads df_raw from
        config["paths"]["input_data"] (the training data), so fold indices are valid.
        infer.py loads df_raw from args.data (the new independent dataset), but the
        per-model back-transformation block reuses the same variable name without loading
        the training data separately.

        Fix: Load the training data from config["paths"]["input_data"] into a separate
        variable (_df_train) within the existing `if transform_module is not None:` guard
        block (lines 286-290). Apply the same two preprocessing steps that train.py applies
        before the fold loop: (1) whitespace-only string replacement with NaN, and
        (2) dropna(subset=outcome_cols). Pass _df_train to input_transform; continue
        passing the inference df_raw to output_transform.
      </description>
      <spec>
        Location: infer.py, inside the `if transform_module is not None:` block that
        currently spans lines 286-290.

        After the existing fold_assignments and outcome_col setup (lines 286-290), add
        training-data loading:

        ```python
        _train_data_path = config["paths"]["input_data"]
        if _train_data_path.endswith('.csv'):
            try:
                _df_train = pd.read_csv(_train_data_path)
            except (pd.errors.ParserError, ValueError, Exception):
                _df_train = pd.read_csv(_train_data_path, sep=None, engine='python')
        else:
            _df_train = pd.read_parquet(_train_data_path)
        _df_train = _df_train.replace(r'^\s*$', pd.NA, regex=True)
        _df_train = _df_train.dropna(subset=outcome_cols)
        ```

        Then change line 308 from:
          `df_raw, train_idx_k, val_idx_k,`
        to:
          `_df_train, train_idx_k, val_idx_k,`

        The output_transform call (lines 312-314) keeps `df_raw=df_raw` (inference data)
        and `row_indices=np.arange(len(df_raw))` (inference row count) unchanged, because
        output_transform operates on inference predictions, not training data.

        Variable naming: _df_train uses underscore prefix consistent with the existing
        _fold_assign and _scaler_info variables in the same scope.

        Preprocessing parity: the two steps (whitespace→NaN, outcome-NaN-drop) match
        train.py lines 664 and 680, which are the only preprocessing steps applied to
        df_raw before fold_assignments are computed. No feature-type coercion or other
        column-level transforms are needed because input_transform only indexes into
        df_raw by row (via train_idx/val_idx) and accesses the outcome column.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Single variable substitution in one call site; predict.py's identical
        pattern (using training data for input_transform) already validates the approach.
        The training data file path is already present in the resolved config and was
        previously validated by train.py.</risk>
      <rollback>Revert the two edits: restore `df_raw` as the first arg to input_transform,
        and remove the _df_train loading block.</rollback>
    </change>
  </changes>
  <execution_order>C1</execution_order>
</implement_plan>
