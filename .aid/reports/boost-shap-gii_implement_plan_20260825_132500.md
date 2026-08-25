<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-25T13:25:00Z" />
  <input_reports>
    <report path="boost-shap-gii_test_20260825_085706.md" mode="test" key_items="1" />
  </input_reports>
  <changes>
    <change id="C1" priority="P0" source_item="test_report/action_items/item[@finding_ref='F1']">
      <file path="src/boost_shap_gii/indiv_reports.py" action="modify" />
      <description>Add an explicit cluster_ids parameter to orchestrate_bootstrap_cache and generate_indiv_reports, replacing the internal X_train.columns-based resolution that is a no-op in production (group_column is unconditionally stripped from features in train.py:760-769, so it is never present in X_train).</description>
      <spec>
**orchestrate_bootstrap_cache (line 653)**: Add `cluster_ids: Optional[np.ndarray] = None` as a new keyword-only parameter after `random_seed`. Remove the internal 4-line resolution block at lines 703-707:
```python
cluster_ids = None
cv_strategy = config.get("modeling", {}).get("cv_strategy", "uniform")
group_column = config.get("modeling", {}).get("group_column")
if cv_strategy == "group" and group_column is not None and group_column in X_train.columns:
    cluster_ids = X_train[group_column].values
```
Replace with a single passthrough: the function now uses the `cluster_ids` parameter directly (it is already named `cluster_ids` in the existing code at line 712, so all downstream references to `cluster_ids` remain unchanged).

**generate_indiv_reports (line 815)**: Add `cluster_ids: Optional[np.ndarray] = None` as a new keyword-only parameter after `sig_GII_interaction`. Remove the internal 5-line resolution block at lines 1033-1037:
```python
infer_cluster_ids: Optional[np.ndarray] = None
cv_strategy = config.get("modeling", {}).get("cv_strategy", "uniform")
group_column = config.get("modeling", {}).get("group_column")
if cv_strategy == "group" and group_column is not None and group_column in X_train.columns:
    infer_cluster_ids = X_train[group_column].values
```
Replace with: `infer_cluster_ids = cluster_ids` (single assignment, preserving the existing local variable name used at line 1059).

No changes to the function's internal bootstrap logic, _bootstrap_sample_indices calls, or _bootstrap_of_cv_inference calls; those already accept cluster_ids/infer_cluster_ids correctly.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - signature addition with a default of None preserves backward compatibility; all internal consumers of cluster_ids are unchanged; only the source of the value changes from an always-False internal lookup to an explicit caller-provided parameter.</risk>
      <rollback>Revert the two signature additions and restore the two internal resolution blocks verbatim.</rollback>
    </change>
    <change id="C2" priority="P0" source_item="test_report/action_items/item[@finding_ref='F1']">
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <description>Thread cluster_ids from df_raw[group_column] into both the orchestrate_bootstrap_cache and generate_indiv_reports call sites. This mirrors the working shap_ctx["groups"] pattern already used correctly at predict.py:516-517.</description>
      <spec>
**Before the orchestrate_bootstrap_cache call (around line 528-535)**: Resolve cluster_ids from df_raw using the same pattern as shap_ctx["groups"] at lines 514-517:
```python
cluster_ids_indiv = None
if cv_strategy == "group" and group_column is not None and group_column in df_raw.columns:
    cluster_ids_indiv = df_raw[group_column].values
```
Note: `cv_strategy` and `group_column` are already in scope at this point (assigned at lines 514-515).

**orchestrate_bootstrap_cache call (~line 535)**: Add `cluster_ids=cluster_ids_indiv` as a keyword argument.

**generate_indiv_reports call (~line 554)**: Add `cluster_ids=cluster_ids_indiv` as a keyword argument.
      </spec>
      <dependencies>C1</dependencies>
      <risk>low - df_raw is in scope; cv_strategy and group_column are already resolved; the resolution logic is an exact copy of the proven shap_ctx pattern at predict.py:516-517.</risk>
      <rollback>Remove the 3-line cluster_ids_indiv resolution block and the two keyword arguments from the call sites.</rollback>
    </change>
    <change id="C3" priority="P0" source_item="test_report/action_items/item[@finding_ref='F1']">
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>Thread cluster_ids from df_raw[group_column] into the generate_indiv_reports call site. infer.py has df_raw in scope from its own data load.</description>
      <spec>
**Before the generate_indiv_reports call (around line 620-640)**: Resolve cluster_ids from df_raw:
```python
cluster_ids_indiv = None
cv_strategy_indiv = config["modeling"].get("cv_strategy", "uniform")
group_column_indiv = config["modeling"].get("group_column")
if cv_strategy_indiv == "group" and group_column_indiv is not None and group_column_indiv in df_raw.columns:
    cluster_ids_indiv = df_raw[group_column_indiv].values
```
Note: infer.py does not have cv_strategy/group_column pre-resolved at this scope (unlike predict.py where they are used for shap_ctx), so fresh local variables are needed. The `_indiv` suffix avoids any potential name collision.

**generate_indiv_reports call (~line 640)**: Add `cluster_ids=cluster_ids_indiv` as a keyword argument.
      </spec>
      <dependencies>C1</dependencies>
      <risk>low - df_raw is in scope; the resolution logic is identical to predict.py's pattern; no existing variables are shadowed.</risk>
      <rollback>Remove the 4-line cluster_ids_indiv resolution block and the keyword argument from the call site.</rollback>
    </change>
  </changes>
  <execution_order>C1, C2, C3 (C2 and C3 are independent of each other but both depend on C1)</execution_order>
</implement_plan>
