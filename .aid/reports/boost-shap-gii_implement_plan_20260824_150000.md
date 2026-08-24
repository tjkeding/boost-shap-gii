<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-24T15:00:00-04:00" />
  <input_reports>
    <report path="(conversation)" mode="design-discussion" key_items="1" />
  </input_reports>

  <assumptions_and_decisions>
    None. The design was proposed with one recommendation (persist per-fold transform
    metadata at training time; infer.py loads it instead of re-reading training data)
    and the user approved proceeding by invoking this plan. No open options remain.
  </assumptions_and_decisions>

  <changes>
    <change id="C1" priority="P0" source_item="design-discussion: infer.py must not depend on training-data access">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <description>
        Accumulate each fold's transform metadata (already required to be
        JSON-serializable, validated by the existing upfront smoke test at line 815:
        `json.dumps(sm_meta)`) into a list across the fold loop, and persist it as a new
        artifact, fold_transform_metadata.json, alongside the existing
        transform_config.json. This is the missing piece that lets infer.py recover each
        model's fold-specific back-transformation metadata without needing to re-read the
        original training data file.
      </description>
      <spec>
        Insertion 1 (initialize accumulator before the fold loop): immediately after the
        existing line

          `fold_assignments = np.full(len(X), -1, dtype=int)`

        add:

          `all_fold_transform_meta = []`

        Insertion 2 (accumulate inside the fold loop): immediately after the existing two
        lines

          `y_train = pd.Series(y_train, index=y.iloc[train_idx].index)`
          `y_val = pd.Series(y_val, index=y.iloc[val_idx].index)`

        (these two lines are already inside the existing `if transform_module is not
        None:` block that also assigns `fold_meta`), add:

          `all_fold_transform_meta.append(fold_meta)`

        This must remain inside the same `if transform_module is not None:` block, at the
        same indentation level as the two lines above it, so exactly one entry is appended
        per fold, in fold order, only when transformations are active.

        Insertion 3 (persist after the fold loop): immediately after the existing two
        lines

          `save_json_atomic(tx_artifact, os.path.join(run_dir, "transform_config.json"))`
          `print(f"[INFO] Saved transform_config.json")`

        (these two lines are the last two lines of the existing `if transform_module is
        not None:` block that writes transform_config.json), add:

          `save_json_atomic(all_fold_transform_meta, os.path.join(run_dir, "fold_transform_metadata.json"))`
          `print(f"[INFO] Saved fold_transform_metadata.json ({len(all_fold_transform_meta)} folds)")`

        These two new lines must remain inside the same `if transform_module is not
        None:` block (the artifact is meaningless and must not be written when no
        transform is configured, matching the existing transform_config.json guard
        convention).

        No other lines in train.py change. `all_fold_transform_meta` is a plain Python
        list of dicts; `save_json_atomic` is already imported and used elsewhere in this
        file for structurally identical list/dict artifacts (`fold_assignments.tolist()`,
        `tx_artifact`), so no new imports are needed.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Purely additive: a new list, one append call inside an existing
        conditional block, and one new artifact write inside another existing conditional
        block. No existing computation, artifact content, or control flow is altered.
        Verified against the full set of existing train.py wiring tests
        (test_transformations_wiring.py::TestTrainPyWiring): none of their assertions
        reference lines this change touches or moves, so none break.</risk>
      <rollback>Remove the three insertions (accumulator initialization, append call,
        artifact write). train.py returns to not producing fold_transform_metadata.json.</rollback>
    </change>

    <change id="C2" priority="P0" source_item="design-discussion: remove infer.py's training-data dependency">
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>
        Replace the training-data-reload approach (added in the immediately prior build)
        with a direct load of the new fold_transform_metadata.json artifact. This removes
        infer.py's dependency on config["paths"]["input_data"] entirely: infer.py will no
        longer read the original training CSV/parquet file, matching its intended design
        as a stage that operates on a new, independent dataset using only train.py's
        persisted artifacts. This also removes the now-unnecessary fold_assignments.json
        read and the input_transform() call in the per-model loop, since the fold-specific
        metadata that call was reconstructing is now available directly.
      </description>
      <spec>
        Replace the current block (the `if transform_module is not None:` setup block
        immediately before the per-model loop, plus the per-model loop's own
        `if transform_module is not None:` block):

          ```python
          if transform_module is not None:
              fold_assignments_path = os.path.join(train_dir, "fold_assignments.json")
              with open(fold_assignments_path) as f:
                  _fold_assign = np.array(json.load(f))
              outcome_col = outcome_cols[0] if len(outcome_cols) == 1 else outcome_cols
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

          for k, model_path in enumerate(model_files):
              ...
              if transform_module is not None:
                  train_idx_k = np.where(_fold_assign != k)[0]
                  val_idx_k = np.where(_fold_assign == k)[0]
                  _, _, fold_meta_k = transform_module.input_transform(
                      _df_train, train_idx_k, val_idx_k,
                      outcome_col, tx_info.get("params", {})
                  )
                  preds = transform_module.output_transform(
                      np.asarray(preds, dtype=float), fold_meta_k, tx_info.get("params", {}),
                      df_raw=df_raw, row_indices=np.arange(len(df_raw))
                  )
          ```

        with:

          ```python
          if transform_module is not None:
              fold_meta_path = os.path.join(train_dir, "fold_transform_metadata.json")
              with open(fold_meta_path) as f:
                  _fold_transform_meta = json.load(f)

          for k, model_path in enumerate(model_files):
              ...
              if transform_module is not None:
                  preds = transform_module.output_transform(
                      np.asarray(preds, dtype=float), _fold_transform_meta[k], tx_info.get("params", {}),
                      df_raw=df_raw, row_indices=np.arange(len(df_raw))
                  )
          ```

        (the `...` represents the unchanged model-loading/prediction lines in between,
        which are not part of either block and must not be touched).

        Net effect: `_fold_assign`, `outcome_col`, `_train_data_path`, `_df_train`,
        `train_idx_k`, and `val_idx_k` are all removed from infer.py; none are referenced
        anywhere else in the file (verified: `outcome_col` — singular — appears nowhere
        outside this block; `_fold_assign`, `_df_train`, `train_idx_k`, `val_idx_k` were
        all introduced by the immediately prior build specifically for this block). The
        `input_transform` call is removed entirely from infer.py; only `output_transform`
        is called, now using `_fold_transform_meta[k]` (a plain list index lookup) in
        place of `fold_meta_k` (previously reconstructed via `input_transform`). The
        `output_transform` call itself is otherwise unchanged: it still receives the
        inference-time `df_raw` and `row_indices=np.arange(len(df_raw))`.

        Operational note: this means any run_dir produced by train.py before this change
        (i.e., lacking fold_transform_metadata.json) will cause infer.py to fail with a
        clear FileNotFoundError if transformations are active. This is expected: it forces
        a re-train under the corrected train.py rather than silently proceeding with
        incomplete artifacts.
      </spec>
      <dependencies>C1</dependencies>
      <risk>medium - Removes code from the immediately prior build and touches the exact
        call site responsible for the previously fixed IndexError. Mitigated by the fact
        that the new logic is strictly simpler (a JSON load and a list index, versus a
        second CSV/parquet parse and a second call into user-supplied transform code) and
        has fewer failure modes than what it replaces. Confirmed by direct grep that none
        of the removed identifiers (`_fold_assign`, `outcome_col`, `_df_train`,
        `train_idx_k`, `val_idx_k`) are referenced anywhere else in infer.py.

        Known test impact: this change will break
        test_transformations_wiring.py::TestInferPyWiring::test_fold_assignments_hoisted_outside_loop,
        which currently asserts that infer.py reads and hoists fold_assignments.json
        before the per-model loop. That assertion encoded the design this change replaces
        and will no longer be true. Per the established skill boundary, /implement does
        not edit files under tests/; this is a disposition for the next /test cycle
        (the intended contract changed: infer.py no longer needs fold_assignments.json at
        all, so the assertion should be re-expressed to test the new contract, e.g., that
        infer.py loads fold_transform_metadata.json and does not reference
        config["paths"]["input_data"], rather than removed outright).</risk>
      <rollback>Restore the training-data-reload block exactly as specified above in the
        "replace" source, i.e., revert to the state produced by the immediately prior
        build.</rollback>
    </change>
  </changes>
  <execution_order>C1, C2</execution_order>
</implement_plan>
