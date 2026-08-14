<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-13T18:30:00Z" />
  <input_reports>
    <report path="boost-shap-gii_brainstorm_20260813_171500.md" mode="brainstorm" key_items="7" />
  </input_reports>
  <changes>
    <change id="C1" priority="P0" source_item="P0: cv_strategy wrapper classes + get_cv_splitter refactor + validate_cv_config + fill_config_defaults">
      <file path="src/boost_shap_gii/utils.py" action="modify" />
      <description>Add wrapper classes, refactor get_cv_splitter(), add validation and defaults for cv_strategy/n_inner_repeats. This is the foundation change: every downstream module depends on the new get_cv_splitter interface and wrapper classes.</description>
      <spec>
**Imports (L15):** Add to the sklearn.model_selection import line: `GroupKFold, RepeatedKFold, RepeatedStratifiedKFold`.

**New helper function `stratify_labels_for_regression()` (insert after L175, before get_scoring_function):**
```python
def stratify_labels_for_regression(y: pd.Series, n_bins: int) -> pd.Series:
    try:
        return pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        return pd.cut(y, bins=n_bins, labels=False)
```
Signature: `(y: pd.Series, n_bins: int) -> pd.Series`. No docstring per project conventions.

**New class `_StratifiedRegressionKFold` (insert before get_cv_splitter):**
- Constructor: `__init__(self, n_splits, random_state, n_repeats=1)`. Stores n_splits. When n_repeats > 1, inner delegate is `RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)`. When n_repeats == 1, inner delegate is `StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)`.
- `split(self, X, y=None, groups=None)`: Calls `stratify_labels_for_regression(y, self.n_splits)` to bin continuous y, then `yield from self._inner.split(X, y_binned)`.
- `get_n_splits(self, X=None, y=None, groups=None)`: Delegates to `self._inner.get_n_splits()`.

**New class `_GroupKFoldWrapper` (insert before get_cv_splitter):**
- Constructor: `__init__(self, n_splits, groups)`. Stores groups and creates `GroupKFold(n_splits=n_splits)` as inner delegate.
- `split(self, X, y=None, groups=None)`: `yield from self._inner.split(X, y, groups=self.groups)`.
- `get_n_splits(self, X=None, y=None, groups=None)`: Returns `self.n_splits`.

**New class `_RepeatedGroupKFold` (insert before get_cv_splitter):**
- Constructor: `__init__(self, n_splits, n_repeats, groups, random_state)`. Stores all params.
- `split(self, X, y=None, groups=None)`: Uses `self.groups`. Gets unique groups. Creates `np.random.default_rng(self.random_state)`. For each repeat in range(n_repeats): permutes unique_groups via rng, assigns group_to_fold = {group: i % n_splits for i, group in enumerate(permuted_groups)}. For each fold_idx in range(n_splits): builds val_mask from group_to_fold, yields (train_idx, val_idx).
- `get_n_splits(self, X=None, y=None, groups=None)`: Returns `self.n_splits * self.n_repeats`.

**Refactor `get_cv_splitter()` (L163-175):**
New signature: `get_cv_splitter(config: Dict, y: pd.Series, seed_override: int = None, groups=None, n_repeats: int = 1, n_folds_override: int = None)`.

Logic:
- `n_folds = n_folds_override if n_folds_override is not None else int(config["modeling"]["cv_folds"])`
- `seed = seed_override if seed_override is not None else int(config["execution"]["random_seed"])`
- `task = detect_task(config)`
- `cv_strategy = config["modeling"].get("cv_strategy", "uniform")`
- "uniform": if n_repeats > 1, return RepeatedKFold; else return KFold. No auto-stratification logic (removed).
- "stratified": if is_regression(task), return _StratifiedRegressionKFold(n_splits, seed, n_repeats); else if n_repeats > 1, return RepeatedStratifiedKFold; else return StratifiedKFold.
- "group": if groups is None, raise ValueError. If n_repeats > 1, return _RepeatedGroupKFold; else return _GroupKFoldWrapper.
- else: raise ValueError for unknown cv_strategy.

This removes the existing auto-stratification for classification (the `y.nunique() < 20` branch). This is a deliberate backward-compatibility break approved in T1.

**New function `validate_cv_config()` (insert after validate_spline_config, before validate_indiv_reports_config):**
Signature: `validate_cv_config(config: dict, df: pd.DataFrame = None) -> None`. Validates:
1. `cv_strategy` must be in {"uniform", "stratified", "group"} (or absent, defaulting to "uniform").
2. When cv_strategy="group", `modeling.group_column` must be present in config and (if df provided) present in df.columns.
3. `n_inner_repeats` must be a positive integer (or absent, defaulting to 1).

**Update `fill_config_defaults()` (L326 area, after the execution defaults):**
Add two new defaults:
```python
_set(["modeling", "cv_strategy"], "uniform")
_set(["modeling", "tuning", "n_inner_repeats"], 1)
```
Insert these after the `_set(["execution", "random_seed"], 42)` line and before the task_type inference block.

**Export additions:** The new public symbols are: `stratify_labels_for_regression`, `validate_cv_config`, `_StratifiedRegressionKFold`, `_GroupKFoldWrapper`, `_RepeatedGroupKFold`. The wrapper classes use underscore-prefix (internal), but are importable for testing.
      </spec>
      <dependencies>none</dependencies>
      <risk>medium - Refactoring get_cv_splitter changes the contract for all callers. The wrapper classes must maintain sklearn's split(X, y) interface exactly. _RepeatedGroupKFold is custom code with no sklearn equivalent.</risk>
      <rollback>Revert utils.py to prior version via git checkout. All downstream changes depend on this, so reverting C1 requires reverting C2-C5.</rollback>
    </change>

    <change id="C2" priority="P0" source_item="P0: train.py outer/inner CV, fold_assignments.json, group_column, WARNINGs">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <description>Refactor both CV call sites in train.py: outer CV to handle groups + persist fold_assignments.json, inner CV (Optuna) to use get_cv_splitter with n_inner_repeats. Add validate_cv_config call, group_column exclusion, and computational cost WARNINGs. Remove direct KFold/StratifiedKFold imports.</description>
      <spec>
**Imports (L16, L28-39):**
- Remove `from sklearn.model_selection import KFold, StratifiedKFold` (L16). These are no longer used directly.
- Add `validate_cv_config` to the utils import (L28-39).

**Group column exclusion (insert after `_validate_aggregate_shap` call at L710, before type enforcement at L713):**
```python
cv_strategy = config["modeling"].get("cv_strategy", "uniform")
group_column = config["modeling"].get("group_column")
groups = None
if cv_strategy == "group":
    if group_column in final_cols:
        final_cols = [c for c in final_cols if c != group_column]
        con_feats = [c for c in con_feats if c != group_column]
        ord_feats = [c for c in ord_feats if c != group_column]
        nom_feats = [c for c in nom_feats if c != group_column]
        print(f"[INFO] group_column '{group_column}' excluded from feature candidates.")
    groups = df_raw[group_column].values
```

**Validation call (insert after fill_config_defaults at ~L679, before the resolved_config.yaml write):**
```python
validate_cv_config(config, df=df_raw)
```

**Outer CV (L865-886):**
Replace:
```python
splitter = get_cv_splitter(config, y_for_split)
```
With:
```python
splitter = get_cv_splitter(config, y_for_split, groups=groups)
```
The split call `splitter.split(X, y_for_split)` remains unchanged (wrappers handle groups internally).

After the outer CV loop completes (insert after L978 `fold_metrics.append(metrics)`, before phase 2 shadow training), add fold assignment recording:
```python
if fold_idx == 0:
    fold_assignments = np.full(len(X), -1, dtype=int)
fold_assignments[val_idx] = fold_idx
```
Actually, better to initialize fold_assignments before the loop and populate during:
- Before the loop (after `fold_metrics = []` at L882): `fold_assignments = np.full(len(X), -1, dtype=int)`
- Inside the loop (after `X_train, X_val = ...` at L889): `fold_assignments[val_idx] = fold_idx`

After the entire CV loop completes (at L1037, after `print("\n[INFO] CV Complete..."`):
```python
save_json_atomic(fold_assignments.tolist(), os.path.join(run_dir, "fold_assignments.json"))
```

**Unbalanced fold WARNING (insert after splitter creation, before the loop):**
When cv_strategy="group":
```python
if cv_strategy == "group":
    fold_sizes = [len(val) for _, val in splitter.split(X, y_for_split)]
    ratio = max(fold_sizes) / max(min(fold_sizes), 1)
    if ratio > 2.0:
        print(f"[WARNING] GroupKFold folds are unbalanced: sizes {fold_sizes}. "
              f"Max/min ratio = {ratio:.2f} (threshold: 2.0).")
    splitter = get_cv_splitter(config, y_for_split, groups=groups)
```
Note: since GroupKFold is deterministic (no shuffle), iterating split twice produces identical assignments. But to avoid consuming the iterator, reconstruct the splitter after the check.

**run_optuna_tuning signature (L475-483):**
Add `groups: np.ndarray = None` parameter.

**Inner CV refactor (L525-535):**
Replace the direct KFold/StratifiedKFold construction with:
```python
n_inner_repeats = int(config["modeling"]["tuning"].get("n_inner_repeats", 1))
inner_cv_folds = tuning_cfg["inner_cv_folds"]
inner_seed = seed + fold_idx + 1

y_for_stratify = y_train if isinstance(y_train, pd.Series) else y_train.iloc[:, 0]

inner_cv = get_cv_splitter(
    config, y_for_stratify, seed_override=inner_seed,
    groups=groups, n_repeats=n_inner_repeats,
    n_folds_override=inner_cv_folds,
)
```

The inner CV split call at L564 `for t_idx, v_idx in inner_cv.split(X_train, y_for_stratify):` remains unchanged.

**Outer fold run_optuna_tuning call (L897):**
Add `groups=inner_groups` where `inner_groups = groups[train_idx] if groups is not None else None`:
```python
inner_groups = groups[train_idx] if groups is not None else None
best_params, tuned_iters = run_optuna_tuning(
    X_train, y_train, nom_feats, task, config, n_jobs,
    fold_idx=fold_idx, groups=inner_groups,
)
```

**Computational cost WARNINGs (insert in run_optuna_tuning, after inner_cv construction):**
```python
if n_inner_repeats > 10:
    print("[WARNING] n_inner_repeats > 10: diminishing returns expected "
          "(Vanwinckelen & Blockeel 2012).")

total_inner_fits = inner_cv_folds * n_inner_repeats * n_trials
if total_inner_fits > 5000:
    print(f"[WARNING] Total inner fits per outer fold = {total_inner_fits} "
          f"({inner_cv_folds} folds x {n_inner_repeats} repeats x "
          f"{n_trials} trials). Consider reducing n_inner_repeats or n_iter.")
```
      </spec>
      <dependencies>C1</dependencies>
      <risk>medium - Inner CV loop restructuring touches the Optuna objective function. The n_inner_repeats integration via Repeated* splitters changes the number of iterations per trial. Must verify that the Optuna objective still returns the correct mean score.</risk>
      <rollback>Revert train.py to prior version. Downstream modules (predict.py, shap_utils.py, indiv_reports.py) will fall back to the old reconstruction approach until C3-C5 are also reverted.</rollback>
    </change>

    <change id="C3" priority="P0" source_item="P0: predict.py fold_assignments.json loading">
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <description>Replace CV splitter reconstruction with fold_assignments.json artifact loading. This eliminates the fragile dependency on data identity and sklearn determinism for fold replication.</description>
      <spec>
**Imports (L17-32):**
Remove `get_cv_splitter` from the utils import line.

**OOF Prediction Loop (L235-249):**
Replace the splitter reconstruction block:
```python
# Replicate Splitter from train.py
y_for_split = y if isinstance(y, pd.Series) else y.iloc[:, 0]
splitter = get_cv_splitter(config, y_for_split)
```
With fold_assignments.json loading:
```python
fold_assignments_path = os.path.join(run_dir, "fold_assignments.json")
with open(fold_assignments_path) as f:
    fold_assignments = np.array(json.load(f))
n_folds = int(fold_assignments.max()) + 1
```

Replace the model file count validation (L240-245):
```python
expected_folds = splitter.get_n_splits()
if len(model_files) != expected_folds:
```
With:
```python
if len(model_files) != n_folds:
    raise AssertionError(
        f"Found {len(model_files)} model file(s) in {run_dir} but fold_assignments.json "
        f"indicates {n_folds} fold(s). Re-run train.py or check output_dir."
    )
```

Replace the split iteration (L249):
```python
for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y_for_split)):
```
With:
```python
for fold_idx in range(n_folds):
    val_idx = np.where(fold_assignments == fold_idx)[0]
```
The remaining loop body (L250-270) stays identical, just using the val_idx from fold_assignments.

Remove the `print(f"[INFO] Generating OOF Predictions using {len(model_files)} folds...")` line's splitter reference if any, and update the count validation print to use n_folds.

Also remove the now-unused `y_for_split` variable and its associated `splitter` construction. The `y_for_split` variable may still be needed elsewhere in predict.py (check all usages). Actually, y_for_split is not used elsewhere in predict.py beyond the splitter, so it can be removed along with the splitter.

Wait: y_for_split IS still referenced at L236 for the splitter. After removing the splitter, check if y_for_split is referenced anywhere else. It is not used after the loop. Remove both lines.
      </spec>
      <dependencies>C1 (for API compatibility), C2 (fold_assignments.json must exist at runtime)</dependencies>
      <risk>low - Straightforward replacement of reconstruction with artifact loading. The artifact is authoritative (written by train.py). Risk is limited to file-not-found if run against an old training directory; the json.load call will raise a clear FileNotFoundError.</risk>
      <rollback>Revert predict.py to prior version. Old reconstruction approach will still work since get_cv_splitter's interface is backward-compatible (new params have defaults).</rollback>
    </change>

    <change id="C4" priority="P0" source_item="P0: shap_utils.py fold_assignments.json loading">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Replace CV splitter reconstruction in non-inference mode with fold_assignments.json artifact loading. Inference mode is unaffected (bypasses splitter entirely).</description>
      <spec>
**Imports (L61):**
Remove `get_cv_splitter` from the utils import line:
```python
from .utils import detect_task, is_regression, _block_permute_shadow
```

**run_shap_pipeline (L1505-1514):**
Replace the non-inference splitter reconstruction:
```python
elif y is not None:
    y_for_split = y if isinstance(y, pd.Series) else y.iloc[:, 0]
    splitter = get_cv_splitter(config, y_for_split)
    splits = list(splitter.split(X_aligned, y_for_split))
else:
    splits = [(None, np.arange(len(X_aligned)))] * len(shadow_paths)
```
With fold_assignments.json loading for non-inference mode:
```python
elif y is not None:
    fold_assignments_path = os.path.join(train_dir, "fold_assignments.json")
    with open(fold_assignments_path) as f:
        fold_assignments = np.array(json.load(f))
    n_folds = int(fold_assignments.max()) + 1
    splits = []
    for k in range(n_folds):
        val_idx = np.where(fold_assignments == k)[0]
        train_idx = np.where(fold_assignments != k)[0]
        splits.append((train_idx, val_idx))
else:
    splits = [(None, np.arange(len(X_aligned)))] * len(shadow_paths)
```

Note: `train_dir` is already defined at L1499: `train_dir = ctx.get("train_dir", run_dir)`. In predict.py (OOF mode), train_dir == run_dir, so fold_assignments.json is read from the correct location. Need to add `import json` at the top of shap_utils.py if not already present. Check: json is NOT imported in shap_utils.py currently. Add `import json` to the imports block (L7 area).
      </spec>
      <dependencies>C1, C2</dependencies>
      <risk>low - Same pattern as C3. Inference mode (infer.py) is completely unaffected since it bypasses the splitter with synthetic full-dataset splits.</risk>
      <rollback>Revert shap_utils.py to prior version.</rollback>
    </change>

    <change id="C5" priority="P0" source_item="P0: indiv_reports.py fold assignment + bootstrap-of-CV refactor">
      <file path="src/boost_shap_gii/indiv_reports.py" action="modify" />
      <description>Refactor _reconstruct_fold_assignments to load fold_assignments.json artifact. Refactor bootstrap-of-CV inner loop to use get_cv_splitter with GroupKFold-to-KFold fallback.</description>
      <spec>
**Imports (L62-64):**
- Remove `KFold, StratifiedKFold` from the sklearn.model_selection import (L62). These are no longer used directly.
- Keep `get_cv_splitter` in the utils import (L64), as it is still used in the bootstrap-of-CV refactored code.
- Add `import json` if not already present (it is not currently imported at module level; json is used via the utils module's save_json_atomic. Need explicit import for json.load).

Wait, checking indiv_reports.py imports: L53 has `import json`. Good, already imported.

So only remove KFold, StratifiedKFold from L62:
```python
# Before:
from sklearn.model_selection import KFold, StratifiedKFold
# After:
# (line removed entirely)
```

**_reconstruct_fold_assignments (L219-244):**
Replace the entire function body. New implementation loads fold_assignments.json:
```python
def _reconstruct_fold_assignments(
    config: dict,
    X_train: pd.DataFrame,
    y_train,
    train_dir: str = None,
) -> np.ndarray:
    run_dir = train_dir if train_dir is not None else config["paths"]["output_dir"]
    fold_assignments_path = os.path.join(run_dir, "fold_assignments.json")
    with open(fold_assignments_path) as f:
        fold_of = np.array(json.load(f), dtype=np.int32)
    N = len(X_train)
    assert len(fold_of) == N, (
        f"fold_assignments.json has {len(fold_of)} entries but X_train has {N} rows. "
        "Verify that the training data matches the training run."
    )
    assert (fold_of >= 0).all(), (
        "fold_assignments.json contains negative fold indices. "
        "This indicates a corrupted or incomplete training run."
    )
    return fold_of
```
The signature changes: adds `train_dir: str = None` parameter. Callers must be updated.

**Callers of _reconstruct_fold_assignments:**
In `generate_indiv_reports` (L937):
```python
# Before:
fold_of = _reconstruct_fold_assignments(config, X_train, y_target)
# After:
fold_of = _reconstruct_fold_assignments(config, X_train, y_target, train_dir=train_dir)
```

**Bootstrap-of-CV inner loop (L362-369):**
Replace the direct KFold/StratifiedKFold construction:
```python
fold_seed = random_seed + b + 1
if is_cls and y_b.ndim == 1:
    splitter = StratifiedKFold(n_splits=K, shuffle=True, random_state=fold_seed)
    fold_iter = list(splitter.split(X_b, y_b))
else:
    splitter = KFold(n_splits=K, shuffle=True, random_state=fold_seed)
    fold_iter = list(splitter.split(X_b))
```
With get_cv_splitter, respecting cv_strategy with GroupKFold fallback:
```python
fold_seed = random_seed + b + 1
cv_strategy = config.get("modeling", {}).get("cv_strategy", "uniform")
if cv_strategy == "group":
    from sklearn.model_selection import KFold as _KFold
    splitter = _KFold(n_splits=K, shuffle=True, random_state=fold_seed)
    fold_iter = list(splitter.split(X_b))
else:
    y_b_series = pd.Series(y_b) if y_b.ndim == 1 else pd.Series(y_b[:, 0])
    splitter = get_cv_splitter(
        config, y_b_series, seed_override=fold_seed, n_folds_override=K,
    )
    fold_iter = list(splitter.split(X_b, y_b_series))
```
Note: the `from sklearn.model_selection import KFold as _KFold` is a local import to avoid re-adding KFold to the module-level imports. This keeps the module-level import clean. Alternatively, since this is a single fallback case, a local import is acceptable.

Actually, a cleaner approach: since we removed KFold from the module-level import, we can either (a) keep KFold in the module-level import just for this fallback, or (b) use a local import. Option (b) is cleaner since it makes the fallback explicit. But the user's codebase convention uses module-level imports. Let me keep KFold in the module-level import, but only import KFold (not StratifiedKFold):
```python
from sklearn.model_selection import KFold
```
Then the fallback uses `KFold(n_splits=K, shuffle=True, random_state=fold_seed)` directly.
      </spec>
      <dependencies>C1, C2</dependencies>
      <risk>medium - Two interacting changes in the same file. _reconstruct_fold_assignments signature change requires updating all callers. The bootstrap-of-CV refactor must correctly handle all three cv_strategy values. The GroupKFold fallback is methodologically sound (bootstrap resampling breaks group structure) but adds a code path that diverges from the configured cv_strategy.</risk>
      <rollback>Revert indiv_reports.py to prior version.</rollback>
    </change>

    <change id="C6" priority="P1" source_item="P1: Simplify example configs + add new keys">
      <file path="example_config_advanced.yaml" action="modify" />
      <description>Add cv_strategy and n_inner_repeats config entries. Simplify all comments to one-line inline. Remove multi-paragraph explanations (theory moved to INPUT_SPECIFICATION.md). Remove fully commented-out blocks (uncomment aggregate_shap as a real example).</description>
      <spec>
**New entries in modeling section (after `cv_folds: 10`):**
```yaml
  cv_strategy: "uniform"      # CV splitter: "uniform" (KFold), "stratified" (StratifiedKFold), or "group" (GroupKFold)
  # group_column: "subject_id"  # Required when cv_strategy: "group"; names the grouping column
```

**New entry in modeling.tuning section (after `inner_cv_folds: 5`):**
```yaml
    n_inner_repeats: 1         # Averaged inner CV repeats per Optuna trial (1 = no repeats)
```

**Comment simplification across the entire file:**
Every multi-line comment block is condensed to a single inline comment on the same line as the key. Specific areas:

1. L63-89 (Huber loss / MAD explanation): Remove the entire multi-line comment block. Replace with a one-line inline comment on the `loss_function` key:
```yaml
  loss_function: "RMSE"        # Training loss: RMSE, MAE, Huber:delta=VALUE, MultiRMSE, Logloss, MultiClass
```

2. L93-98 (scoring explanation): Remove multi-line comment. The scoring key already has context from the loss_function comment.
```yaml
    scoring: "neg_rmse"        # Tuning metric: neg_rmse, neg_mae, r2, roc_auc, balanced_accuracy, f1_weighted
```

3. L155-161 (indiv_ci_nboot multi-line explanation): Condense to inline:
```yaml
  indiv_ci_nboot: 2500         # Coupled bootstrap iterations for per-individual SHAP CIs (0 disables)
```

4. L162-163 (indiv_scaling_mode multi-line): Condense to inline:
```yaml
  indiv_scaling_mode: "sd"     # Scaling: "raw", "sd" (regression only), or "custom_value"
```

5. L167-169 (indiv_scaling_value multi-line): Condense to inline:
```yaml
  indiv_scaling_value: 1.0     # Divisor when indiv_scaling_mode="custom_value"; must be > 0
```

6. L171-175 (compute_global_on_inference multi-line): Condense to inline:
```yaml
  compute_global_on_inference: false  # If true, infer.py emits population-level GII on inference data
```

7. L176-190 (fully commented-out aggregate_shap block): Uncomment as a real example with inline comments:
```yaml
# aggregate_shap:
#   subscale_A_total:            # Group name (must not collide with a feature name)
#     - "subscale_A_item1"       # Constituent features (disjoint, no nominals)
#     - "subscale_A_item2"
#     - "subscale_A_item3"
#   subscale_B_total:
#     - "subscale_B_item1"
#     - "subscale_B_item2"
```
Keep this as a commented-out example block (since it requires study-specific feature names), but strip the multi-paragraph explanation that preceded it.

8. L192-195 (outcome_max multi-line): Condense to inline:
```yaml
  outcome_max: 100.0           # Theoretical maximum of the outcome (scales GII plot magnitudes)
```

9. L196-199 (negate_shap multi-line): Condense to inline:
```yaml
  negate_shap: false           # If true, sign-flip SHAP y-axis values on plots
```

10. L200-204 (label strings): These are already effectively one-line, but remove the multi-line prose comment above them. Keep as:
```yaml
  gii_y_label: "Global Importance Index (GII)"
  gii_y_sublabel: "SHAP-based feature importance"
  indiv_y_label: "SHAP Value (scaled)"
  indiv_y_sublabel: "Per-individual contribution to predicted outcome"
```

11. The top-of-section comments for `execution:`, `paths:`, `features:`, `modeling:`, `shap:`, `plot:` sections: Remove or keep as simple one-line section headers. The file already has the header comment at L1.

12. Task type comment block (L62-64): Condense to inline on `task_type`:
```yaml
  # task_type: "regression"    # Optional: regression, binary_classification, multiclass_classification, multi_regression
```
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Config file formatting only. No behavioral change.</risk>
      <rollback>Revert example_config_advanced.yaml to prior version.</rollback>
    </change>

    <change id="C7" priority="P1" source_item="P1: Simplify example_config_minimal.yaml">
      <file path="example_config_minimal.yaml" action="modify" />
      <description>Ensure minimal config follows one-line-comment convention. New optional keys (cv_strategy, n_inner_repeats) are omitted since defaults apply. No structural changes needed, only comment cleanup if any multi-line comments exist.</description>
      <spec>
Review all comments in the file. The minimal config is already relatively clean (71 lines). Ensure:
1. No multi-line comment blocks.
2. All comments are inline (on the same line as the key they describe).
3. The header comment at L1-3 is kept as-is (brief and functional).

Specific changes:
- L57 `outcome: "outcome_total"` has a long inline comment. Keep but verify it's a single line.
- L60 `indiv_ci_nboot: 2500` has a one-line comment. Keep.
- L61 `indiv_scaling_mode: "sd"` has a one-line comment. Keep.
- L62 `# indiv_scaling_value: 1.0` is a commented-out entry with inline comment. Keep (it illustrates the optional key).

No new keys are added (defaults apply for cv_strategy and n_inner_repeats in minimal mode).
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Formatting only.</risk>
      <rollback>Revert example_config_minimal.yaml to prior version.</rollback>
    </change>
  </changes>
  <execution_order>C1, C2, C3, C4, C5, C6, C7</execution_order>
</implement_plan>
