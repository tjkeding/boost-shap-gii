<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-12T11:16:58Z" />
  <input_reports>
    <report path="boost-shap-gii_test_20260811_234500.md" mode="test" key_items="2" />
  </input_reports>
  <changes>
    <change id="C1" priority="P0" source_item="test_report action_item[0]: NaN propagation in group-total X column">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Fix NaN propagation in the group-total X column computed by _aggregate_effects. The current call X_stacked[existing].sum(axis=1) uses pandas' default skipna=True, which silently drops NaN constituents instead of propagating NaN through the sum. This understates missingness in the group total and in the downstream nan_mask computed immediately afterward (line 1441). Adding min_count=len(existing) ensures that any NaN constituent yields a NaN group total, matching the implement_plan C5 spec from the prior session.</description>
      <spec>
        File: src/boost_shap_gii/shap_utils.py
        Location: _aggregate_effects function, line 575
        Current:  X_stacked[group_name] = X_stacked[existing].sum(axis=1)
        Replace:  X_stacked[group_name] = X_stacked[existing].sum(axis=1, min_count=len(existing))

        Semantics: pandas DataFrame.sum with min_count=N requires at least N non-NaN values to produce a non-NaN result. Since len(existing) equals the number of constituent columns being summed, this means ALL constituents must be non-NaN for the sum to produce a non-NaN value. Any single NaN constituent produces NaN in the group total.

        No other lines change. The downstream nan_mask (X_stacked.isnull().values at the C6 call site) will automatically pick up the corrected NaN propagation.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - single parameter addition to an existing pandas call; behavior matches the original C5 spec; already covered by a failing test (tests/test_aggregate_shap.py::TestGroupTotalNanPropagation::test_nan_in_one_constituent_yields_nan_group_total) that will pass once fixed</risk>
      <rollback>Remove the min_count=len(existing) parameter from the sum() call</rollback>
    </change>

    <change id="C2" priority="P2" source_item="test_report action_item[1]: Extract duplicated block-permutation logic into shared helper">
      <file path="src/boost_shap_gii/utils.py" action="modify" />
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Extract the near-identical block-permutation logic currently duplicated inline in train.py (lines 993-1033, inside main()) and shap_utils.py (lines 1332-1354, inside the _process_boruta_fold closure) into a single shared helper _block_permute_shadow in utils.py. Both call sites are replaced with calls to the shared function. This eliminates code duplication and makes the block-permutation algorithm independently importable and testable.

      RNG consumption order change: the current train.py interleaves group permutations across two DataFrames (train perm for group, val perm for group, next group). After refactoring, each DataFrame is processed in a single call (all groups for train, then all groups for val). This changes the rng consumption order, so for a given seed, shadow permutations will differ from v1.2.0. Statistical properties are preserved: grouped features still share identical row-permutation within each DataFrame; ungrouped features still receive independent permutations. This is acceptable because (a) shadow values are intermediate noise-calibration artifacts whose exact values are not user-facing, (b) CatBoost itself is non-deterministic across thread counts (documented in INPUT_SPECIFICATION.md), and (c) any version bump resets reproducibility expectations.

      Test impact: TestBlockPermutationSourceInvariant (tests/test_aggregate_shap.py, lines 347-369) asserts specific inline string patterns in train.py and shap_utils.py source. These assertions will fail after the refactor because the patterns move to utils.py. TestBlockPermutationProperty's _block_permute static method mirrors the algorithm and could be replaced with a direct import of _block_permute_shadow. Both updates are /test scope, not /implement scope. Recommend /test after build.</description>
      <spec>
        === Part A: New function in utils.py ===

        Add _block_permute_shadow after the existing _normalize_quotes function (before load_config). Add List to the typing import if not already present.

        ```python
        def _block_permute_shadow(
            df: pd.DataFrame,
            agg_groups: Dict[str, List[str]],
            rng: np.random.Generator,
        ) -> pd.DataFrame:
            """Apply block-permutation for shadow generation (Au et al. 2022, S1).

            Grouped features receive a shared row-permutation index per group,
            preserving within-group correlation. Ungrouped features are permuted
            independently. Category dtypes are preserved through permutation.

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame to permute IN PLACE (caller should pass a copy).
            agg_groups : dict
                Mapping {group_name: [member_col, ...]} from config["aggregate_shap"].
                Empty dict is valid (all columns permuted independently).
            rng : np.random.Generator
                Initialized random generator.

            Returns
            -------
            pd.DataFrame
                The same DataFrame, permuted in place.
            """
            grouped_features: set = set()
            for members in agg_groups.values():
                grouped_features.update(members)

            for _group_name, members in agg_groups.items():
                n = len(df)
                perm_idx = rng.permutation(n)
                for c in members:
                    if c not in df.columns:
                        continue
                    orig_dtype = df[c].dtype
                    df[c] = df[c].values[perm_idx]
                    if orig_dtype.name == 'category':
                        df[c] = df[c].astype(orig_dtype)

            for c in df.columns:
                if c in grouped_features:
                    continue
                orig_dtype = df[c].dtype
                df[c] = rng.permutation(df[c].values)
                if orig_dtype.name == 'category':
                    df[c] = df[c].astype(orig_dtype)

            return df
        ```

        === Part B: Replace inline code in train.py ===

        Add _block_permute_shadow to the existing `from .utils import (...)` block.

        Replace lines 993-1033 (the block starting at "# Extract aggregate group definitions" through the end of the ungrouped val loop, just before "# Rename columns") with:

        ```python
        agg_groups = config.get("aggregate_shap", {})
        _block_permute_shadow(X_train_shadow, agg_groups, rng)
        _block_permute_shadow(X_val_shadow, agg_groups, rng)
        ```

        Lines to remove: 993-1033 (the agg_groups extraction, grouped_features set construction, grouped-feature for-loop for train and val, ungrouped-feature for-loop for train and val).
        Lines to keep: 987-988 (X_train_shadow/X_val_shadow copy), 990-991 (rng initialization), 1035-1037 (column rename).

        === Part C: Replace inline code in shap_utils.py ===

        Add _block_permute_shadow to the existing `from .utils import (...)` line.

        Replace lines 1332-1354 (inside _process_boruta_fold, from "agg_groups = config.get..." through the ungrouped permutation loop, just before the column rename) with:

        ```python
        agg_groups = config.get("aggregate_shap", {})
        _block_permute_shadow(X_val_shadow, agg_groups, rng)
        ```

        Lines to remove: 1332-1354 (agg_groups extraction, grouped_features set, grouped-feature loop, ungrouped-feature loop).
        Lines to keep: 1329 (rng initialization), 1330 (X_val_shadow copy), 1355 (column rename).
      </spec>
      <dependencies>C1 (both touch shap_utils.py; C1 modifies _aggregate_effects which is above the _process_boruta_fold block that C2 modifies, so no line-number conflict, but sequential execution avoids any risk)</dependencies>
      <risk>medium - pure refactor across 3 files; no behavioral change in the block-permutation algorithm itself; RNG consumption order changes in train.py (documented above); 2 source-invariant tests will require /test re-expression</risk>
      <rollback>Revert the utils.py addition, restore inline block-permutation code in train.py and shap_utils.py, remove the import additions</rollback>
    </change>
  </changes>
  <execution_order>C1, C2 (sequential: C1 first because P0 priority and shared file overlap in shap_utils.py)</execution_order>
</implement_plan>
