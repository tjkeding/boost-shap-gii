<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-11T21:56:31Z" />
  <input_reports>
    <report path="boost-shap-gii_brainstorm_20260811_214642.md" mode="brainstorm" key_items="3" />
  </input_reports>
  <changes>

    <change id="C1" priority="P1" source_item="T1: C18 test re-expression">
      <file path="tests/test_build_20260507.py" action="modify" />
      <description>Re-express three failing C18 tests to assert current geometric-mean framing. Rename class. Leave quarantine test unchanged.</description>
      <spec>
In tests/test_build_20260507.py, modify lines 400-465:

1. Rename class `TestCobbDouglasAnchorPresence` to `TestGIIFramingPresence`. Update docstring to describe the geometric-mean framing of GII = sqrt(M * V) across three public-repo files, citing Hill (1910) and Goldstein et al. (2015) where present.

2. Rename `test_shap_utils_gii_function_docstring` to `test_shap_utils_geometric_mean_framing`. Assertions:
   - `assert "geometric mean" in src`
   - `assert "Hill (1910)" in src`
   - `assert "Goldstein" in src`
   Remove all Cobb-Douglas assertions.

3. Keep `test_input_specification_section3_framing` name. Assertions:
   - `assert "Decision-theoretic interpretation" in src`
   - `assert "geometric mean" in src`
   - `assert "Hill (1910)" in src`
   - `assert "Goldstein" in src`
   Remove the Cobb-Douglas assertion.

4. Rename `test_readme_cobb_douglas_subsection` to `test_readme_gii_interpretation_section`. Assertions:
   - `assert "GII Interpretation" in src`
   - `assert "geometric mean" in src`
   - `assert "sqrt(M" in src` (covers "sqrt(M * V)" or "sqrt(M × V)")
   Remove ALL prior assertions (Cobb-Douglas, Hill, Goldstein not present in README by design).

5. Leave `test_quarantine_no_calibration_or_in_prep_in_public_files` unchanged. Do NOT add "Cobb-Douglas" to the forbidden list.
      </spec>
      <dependencies>None</dependencies>
      <risk>low - aligned-classification re-expression; no behavioral changes to pipeline code</risk>
      <rollback>Restore original class and test method names/assertions from git</rollback>
    </change>

    <change id="C2" priority="P1" source_item="T2/T3: aggregate_shap config validation">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <description>Add config validation for the aggregate_shap section at pipeline startup, after feature selection resolves the final column set and feature types.</description>
      <spec>
In train.py, add a new function `_validate_aggregate_shap(config, final_cols, nom_feats)` and call it after feature selection (after line 637, where nom_feats is finalized).

Function signature:
```python
def _validate_aggregate_shap(config: dict, final_cols: list, nom_feats: list) -> None:
```

Logic:
1. Read `agg_cfg = config.get("aggregate_shap", {})`. If empty or absent, return (no-op).
2. `all_constituents = set()` for tracking disjoint membership.
3. For each `group_name, members` in `agg_cfg.items()`:
   a. If `group_name in final_cols`: raise ValueError (name collision with a resolved feature).
   b. If not isinstance(members, list) or len(members) == 0: raise ValueError (empty feature list for group `group_name`).
   c. If len(members) == 1: print a WARNING (single-feature group; no aggregation benefit).
   d. For each `feat` in `members`:
      - If `feat not in final_cols`: raise ValueError (constituent `feat` not in resolved feature set).
      - If `feat in nom_feats`: raise ValueError (nominal feature `feat` not permitted in aggregate groups).
      - If `feat in all_constituents`: raise ValueError (feature `feat` appears in multiple aggregate groups; disjoint membership required).
      - `all_constituents.add(feat)`.
4. Print summary: `[INFO] aggregate_shap: {len(agg_cfg)} group(s) validated, {len(all_constituents)} constituent features.`

Call site: Insert `_validate_aggregate_shap(config, final_cols, nom_feats)` immediately after the all-missing column drop block (after line 637).
      </spec>
      <dependencies>None</dependencies>
      <risk>low - additive validation function; no-op when aggregate_shap absent</risk>
      <rollback>Remove function and call site</rollback>
    </change>

    <change id="C3" priority="P1" source_item="T2/S1: block-permutation in train.py shadow generation">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <description>Modify the shadow feature permutation loop (lines 918-930) to apply block-permutation for features within defined aggregate groups. Ungrouped features retain independent permutation.</description>
      <spec>
In train.py, replace the per-column independent permutation loops (lines 921-930) with block-aware permutation:

```python
# Extract aggregate group definitions (empty dict if absent)
agg_groups = config.get("aggregate_shap", {})
# Build set of all grouped features for O(1) lookup
grouped_features = set()
for members in agg_groups.values():
    grouped_features.update(members)

# Block-permute grouped features; independent-permute ungrouped
for group_name, members in agg_groups.items():
    # Shared permutation index for all group members
    n_train = len(X_train_shadow)
    perm_idx_train = rng.permutation(n_train)
    for c in members:
        orig_dtype = X_train_shadow[c].dtype
        X_train_shadow[c] = X_train_shadow[c].values[perm_idx_train]
        if orig_dtype.name == 'category':
            X_train_shadow[c] = X_train_shadow[c].astype(orig_dtype)

    n_val = len(X_val_shadow)
    perm_idx_val = rng.permutation(n_val)
    for c in members:
        orig_dtype = X_val_shadow[c].dtype
        X_val_shadow[c] = X_val_shadow[c].values[perm_idx_val]
        if orig_dtype.name == 'category':
            X_val_shadow[c] = X_val_shadow[c].astype(orig_dtype)

# Independent permutation for ungrouped features
for c in X_train_shadow.columns:
    if c in grouped_features:
        continue
    orig_dtype = X_train_shadow[c].dtype
    X_train_shadow[c] = rng.permutation(X_train_shadow[c].values)
    if orig_dtype.name == 'category':
        X_train_shadow[c] = X_train_shadow[c].astype(orig_dtype)
for c in X_val_shadow.columns:
    if c in grouped_features:
        continue
    orig_dtype = X_val_shadow[c].dtype
    X_val_shadow[c] = rng.permutation(X_val_shadow[c].values)
    if orig_dtype.name == 'category':
        X_val_shadow[c] = X_val_shadow[c].astype(orig_dtype)
```

When `agg_groups` is empty, `grouped_features` is empty and all features hit the ungrouped loop, preserving current behavior exactly.
      </spec>
      <dependencies>C2 (validation must exist so config is trusted)</dependencies>
      <risk>medium - modifies shadow generation; incorrect permutation indices would corrupt the null distribution</risk>
      <rollback>Restore original per-column independent permutation loop</rollback>
    </change>

    <change id="C4" priority="P1" source_item="T2/S1: block-permutation in shap_utils.py Boruta fold processor">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Modify the shadow permutation in _process_boruta_fold (lines 1097-1103) to apply block-permutation for aggregate groups, matching the train.py change (C3).</description>
      <spec>
In shap_utils.py, within `_run_shap_for_slice`, modify the `_process_boruta_fold` closure (lines 1097-1103). The closure already captures `config` from the enclosing scope.

Replace the per-column permutation loop with the same block-aware pattern as C3:

```python
agg_groups = config.get("aggregate_shap", {})
grouped_features = set()
for members in agg_groups.values():
    grouped_features.update(members)

# Block-permute grouped features
for group_name, members in agg_groups.items():
    n_val = len(X_val_shadow)
    perm_idx = rng.permutation(n_val)
    for c in members:
        orig_dtype = X_val_shadow[c].dtype
        X_val_shadow[c] = X_val_shadow[c].values[perm_idx]
        if orig_dtype.name == 'category':
            X_val_shadow[c] = X_val_shadow[c].astype(orig_dtype)

# Independent permutation for ungrouped
for c in X_val_shadow.columns:
    if c in grouped_features:
        continue
    orig_dtype = X_val_shadow[c].dtype
    X_val_shadow[c] = rng.permutation(X_val_shadow[c].values)
    if orig_dtype.name == 'category':
        X_val_shadow[c] = X_val_shadow[c].astype(orig_dtype)
```
      </spec>
      <dependencies>None (reads config from closure scope)</dependencies>
      <risk>medium - same risk profile as C3; must be consistent with C3's block-permutation</risk>
      <rollback>Restore original per-column permutation loop</rollback>
    </change>

    <change id="C5" priority="P1" source_item="T2/T3: _aggregate_effects function">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Add new function _aggregate_effects() that computes aggregate SHAP columns (singleton, within-group, between-group, group x ungrouped) and augments the DataFrames and metadata.</description>
      <spec>
Add new function in shap_utils.py, placed after `_flatten_interaction_matrix` (after line 480) and before the bootstrap engine section:

```python
def _aggregate_effects(
    df_shap_real: pd.DataFrame,
    df_shap_shadow: pd.DataFrame,
    X_stacked: pd.DataFrame,
    config: Dict[str, Any],
    meta_real: Dict[str, Tuple[Tuple[int, int], str]],
    meta_shadow: Dict[str, Tuple[Tuple[int, int], str]],
    feature_types: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           Dict[str, Tuple[Tuple[int, int], str]],
           Dict[str, Tuple[Tuple[int, int], str]],
           Dict[str, str]]:
```

Returns: (df_shap_real, df_shap_shadow, X_stacked, meta_real, meta_shadow, feature_types) -- all augmented in-place or as new copies.

Logic:
1. `agg_cfg = config.get("aggregate_shap", {})`. If empty, return all inputs unchanged.

2. Identify ungrouped features: features in `df_shap_real.columns` that are Singleton effects (from meta_real) and whose name does not appear in any aggregate group's member list and is not a shadow feature.

3. Build group-total X columns: for each group G with members [f1, f2, ...], compute `X_stacked[group_name] = X_stacked[f1] + X_stacked[f2] + ...` (NaN propagation: if any constituent is NaN, group total is NaN). Register `feature_types[group_name] = "continuous"`. Record the column index of the group-total column in X_stacked.

4. For each group G, compute aggregate SHAP columns:

   a. **Singleton aggregate**: column name = `group_name`. Value = sum of `df_shap_real[fi]` for fi in G (only singletons, i.e., diagonal entries). Meta: `(group_total_idx, group_total_idx), "Singleton"`.

   b. **Within-group interaction**: column name = `{group_name} x {group_name}`. Value = sum of all `df_shap_real[fi x fj]` columns where both fi and fj are in G. If no such columns exist (no within-group interactions detected by TreeSHAP), skip. Meta: `(group_total_idx, group_total_idx), "Interaction"`.

   c. Shadow equivalents: same summation over `df_shap_shadow` columns using `shadow_{fi}` names and `shadow_{fi} x shadow_{fj}` patterns.

5. For each ordered pair of groups (G1, G2) where G1 < G2 alphabetically:

   a. **Between-group interaction**: column name = `{G1_name} x {G2_name}`. Value = sum of all `df_shap_real[fi x gj]` columns where fi in G1 and gj in G2 (in either order in the column name). Meta: `(G1_total_idx, G2_total_idx), "Interaction"`.

   b. Shadow equivalent: same with shadow columns.

6. For each group G and each ungrouped feature u:

   a. **Group x ungrouped interaction**: column name = `{group_name} x {u}`. Value = sum of all `df_shap_real[fi x u]` columns where fi in G (in either order in the column name). Meta: `(group_total_idx, u_col_idx), "Interaction"`.

   b. Shadow equivalent: same with shadow columns.

7. Return augmented (df_shap_real, df_shap_shadow, X_stacked, meta_real, meta_shadow, feature_types).

Column name matching: interaction columns in the flattened DataFrames follow the pattern `{name_i} x {name_j}` where i < j in column order. When searching for a pair (fi, fj), check both `f"{fi} x {fj}"` and `f"{fj} x {fi}"` as column names.
      </spec>
      <dependencies>None (standalone function)</dependencies>
      <risk>high - core aggregation logic; incorrect summation would produce wrong aggregate importance values</risk>
      <rollback>Remove function</rollback>
    </change>

    <change id="C6" priority="P1" source_item="T2/T3: _aggregate_effects call site in _run_shap_for_slice">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Insert _aggregate_effects() call in _run_shap_for_slice after fold-merge and before metadata extraction/bootstrap pipeline call.</description>
      <spec>
In _run_shap_for_slice, insert the call AFTER the fold-merge block (after both the inference_mode and OOF paths produce df_shap_real, df_shap_shadow, X_stacked) and BEFORE the metadata extraction block (before line 1185 `eff_names_real = ...`).

Insert at approximately line 1183 (after `nan_mask = X_stacked.isnull().values`), but BEFORE nan_mask is computed (since X_stacked may gain new columns). Correct insertion point: BEFORE line 1183, AFTER line 1181 (OOF path end) or line 1174 (inference path end):

```python
# --- Aggregate effects (post-hoc group-level SHAP summation) ---
(df_shap_real, df_shap_shadow, X_stacked,
 meta_real, meta_shadow_all, all_feature_types) = _aggregate_effects(
    df_shap_real, df_shap_shadow, X_stacked,
    config, meta_real, meta_shadow_all, all_feature_types
)
```

This must be placed BEFORE `nan_mask = X_stacked.isnull().values` (line 1183) so the nan_mask includes any new group-total columns. And BEFORE the metadata extraction lines (1185-1191) so the augmented metadata is used.

Note: `all_feature_types` is the variable name in the function signature (line 1068) for the feature_types parameter.
      </spec>
      <dependencies>C5 (function must exist)</dependencies>
      <risk>medium - insertion point is critical; wrong placement would miss aggregate columns or corrupt nan_mask</risk>
      <rollback>Remove the inserted call</rollback>
    </change>

    <change id="C7" priority="P1" source_item="T3: is_aggregate column in output">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Add boolean is_aggregate column to the shap_stats_global.csv output in _run_bootstrap_pipeline.</description>
      <spec>
In _run_bootstrap_pipeline, modify the df_res construction (lines 993-1017) to include an `is_aggregate` column.

The `is_aggregate` flag is True for any effect whose name matches an aggregate group name or an aggregate interaction pattern. To determine this, pass the aggregate_shap config to _run_bootstrap_pipeline (add `config` access, which is already available as a parameter at line 719).

Add to the df_res DataFrame (after line 1016 `"v_failure_rate": v_failure_rate,`):
```python
"is_aggregate": [_is_aggregate_effect(n, config) for n in effect_names],
```

Add helper function (near _get_effect_stratum):
```python
def _is_aggregate_effect(effect_name: str, config: Dict[str, Any]) -> bool:
    agg_groups = config.get("aggregate_shap", {})
    if not agg_groups:
        return False
    group_names = set(agg_groups.keys())
    # Singleton aggregate or within-group interaction
    if effect_name in group_names:
        return True
    # Interaction involving at least one aggregate group
    if " x " in effect_name:
        parts = effect_name.split(" x ")
        if parts[0] in group_names or parts[1] in group_names:
            return True
    return False
```

When no aggregate_shap is configured, all effects get `is_aggregate = False`.
      </spec>
      <dependencies>C5, C6 (aggregate effects must exist in the data)</dependencies>
      <risk>low - additive column; does not affect existing pipeline logic</risk>
      <rollback>Remove column and helper function</rollback>
    </change>

    <change id="C8" priority="P1" source_item="T4: blocklist refactor">
      <file path="src/boost_shap_gii/indiv_reports.py" action="modify" />
      <description>Replace _CATBOOST_USER_PARAM_ALLOWLIST (18-entry allowlist) with _CATBOOST_REFIT_BLOCKLIST (narrowly-scoped blocklist of dangerous internal keys). Invert _extract_user_level_params logic.</description>
      <spec>
In indiv_reports.py, replace lines 72-103:

1. Replace the constant:
```python
# Blocklist of CatBoost internal/runtime keys that conflict with Pool
# construction or are hardware-specific. All other keys pass through,
# giving users maximum flexibility over hyperparameter choices.
_CATBOOST_REFIT_BLOCKLIST = {
    "cat_features",
    "text_features",
    "embedding_features",
    "task_type",
    "devices",
}
```

2. Replace `_extract_user_level_params`:
```python
def _extract_user_level_params(all_params: dict) -> dict:
    """Return all params except blocklisted internal/runtime keys."""
    return {k: v for k, v in all_params.items() if k not in _CATBOOST_REFIT_BLOCKLIST}
```

The comment on the old line 72 ("Allowlist of user-facing CatBoost HP keys...") is replaced by the new blocklist comment.
      </spec>
      <dependencies>None</dependencies>
      <risk>low - inverted filter logic; all existing allowlisted params pass through (they are not in the blocklist); two new params (model_size_reg, max_ctr_complexity) also pass through automatically</risk>
      <rollback>Restore original allowlist and filter logic</rollback>
    </change>

    <change id="C9" priority="P1" source_item="T2/T3: example config update">
      <file path="example_config_advanced.yaml" action="modify" />
      <description>Add a commented aggregate_shap section to the advanced example config, placed as a sibling to the existing top-level keys.</description>
      <spec>
In example_config_advanced.yaml, add a new top-level section after the `shap:` section and before `plot:`. The section is fully commented out (all lines prefixed with `#`) to show the format without activating it:

```yaml
# aggregate_shap:
#   # Post-hoc group-level SHAP analysis. Sums individual SHAP values
#   # within user-defined groups to compute group-level M, V, and GII.
#   # Each feature may belong to at most one group. Nominal features
#   # are not permitted. Groups receive shadow-calibrated significance
#   # testing via block-permuted Boruta exceedance.
#   #
#   # subscale_A_total:
#   #   - "subscale_A_item1"
#   #   - "subscale_A_item2"
#   #   - "subscale_A_item3"
#   # subscale_B_total:
#   #   - "subscale_B_item1"
#   #   - "subscale_B_item2"
```
      </spec>
      <dependencies>None</dependencies>
      <risk>low - commented-out config; no behavioral change</risk>
      <rollback>Remove the added block</rollback>
    </change>

  </changes>
  <execution_order>
    Parallel group 1 (no cross-file dependencies): C1, C8, C9
    Sequential group 2 (train.py, internal dependency): C2 then C3
    Sequential group 3 (shap_utils.py, internal dependencies): C4, C5, C6, C7

    Group 2 and group 3 can execute in parallel with each other and with group 1.
    Within group 3: C4 and C5 are independent and can be done in either order; C6 depends on C5; C7 depends on C6.

    Agent dispatch grouping:
    - Agent A: C1 (tests/test_build_20260507.py)
    - Agent B: C2 + C3 (train.py)
    - Agent C: C4 + C5 + C6 + C7 (shap_utils.py)
    - Agent D: C8 (indiv_reports.py)
    - Agent E: C9 (example_config_advanced.yaml)
    All five agents can be dispatched in parallel.
  </execution_order>
</implement_plan>
