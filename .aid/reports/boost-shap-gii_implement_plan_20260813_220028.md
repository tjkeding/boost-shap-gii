<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-13T22:00:28Z" />
  <input_reports>
    <report path="boost-shap-gii_implement_plan_20260813_183000.md" mode="implement" key_items="5 (unimplemented C3-C7 from the prior plan; C1-C2 verified as completed)" />
  </input_reports>
  <preamble>
    The prior implement plan (boost-shap-gii_implement_plan_20260813_183000.md) specified 7 changes (C1-C7).
    The interrupted build completed C1 (utils.py: wrapper classes, get_cv_splitter refactor, validate_cv_config,
    fill_config_defaults) and C2 (train.py: outer/inner CV refactor, fold_assignments.json persistence, group_column
    exclusion, cost WARNINGs) in full. Both diffs verified line-by-line against the spec.

    C3 through C7 have zero uncommitted changes: predict.py, shap_utils.py, indiv_reports.py,
    example_config_advanced.yaml, and example_config_minimal.yaml are all at the v1.3.0 baseline.

    This plan covers the 5 remaining changes, renumbered C1-C5, with line references verified
    against the current file state.
  </preamble>
  <changes>
    <change id="C1" priority="P0" source_item="Prior-plan C3: predict.py fold_assignments.json loading">
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <description>Replace CV splitter reconstruction with fold_assignments.json artifact loading. This eliminates the fragile dependency on data identity and sklearn determinism for fold replication, and is required because get_cv_splitter's new interface (cv_strategy-aware wrappers, optional groups parameter) cannot reproduce the original fold assignments without the same group array and strategy that train.py used.</description>
      <spec>
**Imports (L17-32):**
Remove `get_cv_splitter` from the utils import block. Current L27 reads:
```python
    get_cv_splitter,
```
Delete this line.

**Fold assignment loading (replace L235-237):**
Current code:
```python
    # Replicate Splitter from train.py
    y_for_split = y if isinstance(y, pd.Series) else y.iloc[:, 0]
    splitter = get_cv_splitter(config, y_for_split)
```
Replace with:
```python
    # Load authoritative fold assignments from train.py output
    fold_assignments_path = os.path.join(run_dir, "fold_assignments.json")
    with open(fold_assignments_path) as f:
        fold_assignments = np.array(json.load(f))
    n_folds = int(fold_assignments.max()) + 1
```
Note: `json` is already imported at L9 (`import json`); `np` is already imported at L12; `os` is already imported at L10.

**Fold count validation (replace L239-245):**
Current code:
```python
    # Validate model file count against expected CV fold count
    expected_folds = splitter.get_n_splits()
    if len(model_files) != expected_folds:
        raise AssertionError(
            f"Found {len(model_files)} model file(s) in {run_dir} but CV splitter "
            f"expects {expected_folds} fold(s). Re-run train.py or check output_dir."
        )
```
Replace with:
```python
    # Validate model file count against fold assignments
    if len(model_files) != n_folds:
        raise AssertionError(
            f"Found {len(model_files)} model file(s) in {run_dir} but fold_assignments.json "
            f"indicates {n_folds} fold(s). Re-run train.py or check output_dir."
        )
```

**OOF loop (replace L249):**
Current code:
```python
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y_for_split)):
```
Replace with:
```python
    for fold_idx in range(n_folds):
        val_idx = np.where(fold_assignments == fold_idx)[0]
```
The loop body (L250-270) remains identical; it only uses `fold_idx`, `val_idx`, and `X`. The `train_idx` variable was unused in the loop body.
      </spec>
      <dependencies>none (depends on C1-C2 from the prior plan, which are already completed)</dependencies>
      <risk>low - Straightforward replacement of reconstruction with artifact loading. The json.load call will raise a clear FileNotFoundError if run against a pre-v1.3.0 training directory that lacks fold_assignments.json.</risk>
      <rollback>Revert predict.py to prior version (git checkout src/boost_shap_gii/predict.py).</rollback>
    </change>

    <change id="C2" priority="P0" source_item="Prior-plan C4: shap_utils.py fold_assignments.json loading">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Replace CV splitter reconstruction in non-inference mode with fold_assignments.json artifact loading. Inference mode is unaffected (bypasses the splitter entirely with synthetic full-dataset splits).</description>
      <spec>
**Imports (L61):**
Current line:
```python
from .utils import get_cv_splitter, detect_task, is_regression, _block_permute_shadow
```
Remove `get_cv_splitter` from this import:
```python
from .utils import detect_task, is_regression, _block_permute_shadow
```
Note: `json` is already imported at L45 (`import json`).

**Non-inference fold reconstruction (replace L1509-1512):**
Current code:
```python
    elif y is not None:
        y_for_split = y if isinstance(y, pd.Series) else y.iloc[:, 0]
        splitter = get_cv_splitter(config, y_for_split)
        splits = list(splitter.split(X_aligned, y_for_split))
```
Replace with:
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
```
Note: `train_dir` is already defined at L1499 (`train_dir = ctx.get("train_dir", run_dir)`). In OOF mode (predict.py), train_dir == run_dir, so fold_assignments.json is read from the correct location.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Same pattern as C1. Inference mode is completely unaffected since it bypasses the splitter.</risk>
      <rollback>Revert shap_utils.py to prior version.</rollback>
    </change>

    <change id="C3" priority="P0" source_item="Prior-plan C5: indiv_reports.py fold assignment + bootstrap-of-CV refactor">
      <file path="src/boost_shap_gii/indiv_reports.py" action="modify" />
      <description>Two changes in one file. (A) Refactor _reconstruct_fold_assignments to load fold_assignments.json artifact instead of re-splitting, adding a train_dir parameter. (B) Refactor _bootstrap_of_cv_inference to use get_cv_splitter for non-group strategies, with KFold fallback for group CV (since bootstrap resampling breaks group structure).</description>
      <spec>
**Imports (L62):**
Current line:
```python
from sklearn.model_selection import KFold, StratifiedKFold
```
Replace with (keep KFold for group-CV bootstrap fallback, remove StratifiedKFold):
```python
from sklearn.model_selection import KFold
```

**Imports (L64):**
Current line:
```python
from .utils import get_cv_splitter, save_json_atomic
```
Keep unchanged. get_cv_splitter is still needed for the bootstrap-of-CV non-group path.

**_reconstruct_fold_assignments (L219-244):**
Replace the entire function with:
```python
def _reconstruct_fold_assignments(
    config: dict,
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, pd.DataFrame],
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
Signature change: adds `train_dir: str = None`. The existing parameters `config`, `X_train`, `y_train` are preserved for interface compatibility (X_train is used for length validation; y_train is unused but kept for signature stability; config provides fallback output_dir path).

**Caller of _reconstruct_fold_assignments (L937):**
Current code:
```python
        fold_of = _reconstruct_fold_assignments(config, X_train, y_target)
```
Replace with:
```python
        fold_of = _reconstruct_fold_assignments(config, X_train, y_target, train_dir=train_dir)
```

**_bootstrap_of_cv_inference signature (L272-283):**
Add `config: dict` parameter after `nom_feats`. Current signature ends:
```python
    nom_feats: List[str],
    point_shap_main: np.ndarray,
```
Insert after `nom_feats`:
```python
    nom_feats: List[str],
    config: dict,
    point_shap_main: np.ndarray,
```

**Bootstrap fold construction (replace L362-369):**
Current code:
```python
        # Fresh K-fold split on s_b; seed decoupled from s_b draw.
        fold_seed = random_seed + b + 1
        if is_cls and y_b.ndim == 1:
            splitter = StratifiedKFold(n_splits=K, shuffle=True, random_state=fold_seed)
            fold_iter = list(splitter.split(X_b, y_b))
        else:
            splitter = KFold(n_splits=K, shuffle=True, random_state=fold_seed)
            fold_iter = list(splitter.split(X_b))
```
Replace with:
```python
        # Fresh K-fold split on s_b; seed decoupled from s_b draw.
        fold_seed = random_seed + b + 1
        cv_strategy = config.get("modeling", {}).get("cv_strategy", "uniform")
        if cv_strategy == "group":
            splitter = KFold(n_splits=K, shuffle=True, random_state=fold_seed)
            fold_iter = list(splitter.split(X_b))
        else:
            y_b_series = pd.Series(y_b) if y_b.ndim == 1 else pd.Series(y_b[:, 0])
            splitter = get_cv_splitter(
                config, y_b_series, seed_override=fold_seed, n_folds_override=K,
            )
            fold_iter = list(splitter.split(X_b, y_b_series))
```
Rationale for group fallback to KFold: bootstrap resampling breaks group structure (individuals are drawn with replacement from a pooled sample; group membership is not preserved). Using KFold on the resampled data is methodologically sound. For uniform and stratified strategies, get_cv_splitter applies the configured strategy to the resampled data, which is correct.

**Docstring for _bootstrap_of_cv_inference (L291-292):**
Current text references KFold/StratifiedKFold. Replace:
```
         seed = random_seed + b + 1) using KFold for regression and StratifiedKFold for
         classification tasks.
```
With:
```
         seed = random_seed + b + 1) using get_cv_splitter for uniform/stratified strategies,
         or KFold fallback for group CV (bootstrap resampling breaks group structure).
```

**Caller of _bootstrap_of_cv_inference (L1049-1063):**
Add `config=config` to the call. Current call block:
```python
        main_ci_lo_inf, main_ci_hi_inf, int_ci_lo_inf, int_ci_hi_inf = (
            _bootstrap_of_cv_inference(
                X_train=X_train,
                y_train=y_train_arr,
                inference_pool=pool_infer_tgt,
                fold_hyperparameters=frozen_hps,
                B=B,
                K=K,
                random_seed=random_seed_cfg,
                cluster_ids=infer_cluster_ids,
                task=task,
                nom_feats=nom_feats,
                point_shap_main=point_shap,
                point_shap_int=point_shap_int if compute_interactions else None,
            )
        )
```
Insert `config=config,` after `nom_feats=nom_feats,`:
```python
        main_ci_lo_inf, main_ci_hi_inf, int_ci_lo_inf, int_ci_hi_inf = (
            _bootstrap_of_cv_inference(
                X_train=X_train,
                y_train=y_train_arr,
                inference_pool=pool_infer_tgt,
                fold_hyperparameters=frozen_hps,
                B=B,
                K=K,
                random_seed=random_seed_cfg,
                cluster_ids=infer_cluster_ids,
                task=task,
                nom_feats=nom_feats,
                config=config,
                point_shap_main=point_shap,
                point_shap_int=point_shap_int if compute_interactions else None,
            )
        )
```
      </spec>
      <dependencies>none</dependencies>
      <risk>medium - Two interacting changes in the same file. The _reconstruct_fold_assignments signature change adds a new parameter; the bootstrap refactor adds `config` to a different function signature. Both callers must be updated. The group-CV fallback to KFold is methodologically sound but adds a code path that diverges from the configured cv_strategy.</risk>
      <rollback>Revert indiv_reports.py to prior version.</rollback>
    </change>

    <change id="C4" priority="P1" source_item="Prior-plan C6: example_config_advanced.yaml new keys + comment cleanup">
      <file path="example_config_advanced.yaml" action="modify" />
      <description>Add cv_strategy and n_inner_repeats config entries. Simplify all multi-line comments to one-line inline comments. Remove multi-paragraph explanations (theory stays in INPUT_SPECIFICATION.md). Retain the commented-out aggregate_shap example block.</description>
      <spec>
**New entries in modeling section (insert after L60 `cv_folds: 10`):**
```yaml
  cv_strategy: "uniform"         # CV splitter: "uniform" (KFold), "stratified" (StratifiedKFold), or "group" (GroupKFold)
  # group_column: "subject_id"   # Required when cv_strategy: "group"; names the grouping column
```

**New entry in modeling.tuning section (insert after L100 `inner_cv_folds: 5`):**
```yaml
    n_inner_repeats: 1           # Averaged inner CV repeats per Optuna trial (1 = no repeats)
```

**Comment simplification throughout the file:**
1. L62-89 (task_type comment + Huber/MAD block): Remove entire multi-line block. Replace L62-64 task_type lines with:
```yaml
  # task_type: "regression"      # Optional: regression, binary_classification, multiclass_classification, multi_regression
```
Replace L66-90 loss_function block with:
```yaml
  loss_function: "RMSE"          # Training loss: RMSE, MAE, Huber:delta=VALUE, MultiRMSE, Logloss, MultiClass
```

2. L93-99 scoring block: Replace with:
```yaml
    scoring: "neg_rmse"          # Tuning metric: neg_rmse, neg_mae, r2, roc_auc, balanced_accuracy, f1_weighted
```

3. L155-161 indiv_ci_nboot multi-line: Replace with:
```yaml
  indiv_ci_nboot: 2500           # Coupled bootstrap iterations for per-individual SHAP CIs (0 disables)
```

4. L162-163 indiv_scaling_mode: Replace with:
```yaml
  indiv_scaling_mode: "sd"       # Scaling: "raw", "sd" (regression only), or "custom_value"
```

5. L167-169 indiv_scaling_value: Replace with:
```yaml
  indiv_scaling_value: 1.0       # Divisor when indiv_scaling_mode="custom_value"; must be > 0
```

6. L171-175 compute_global_on_inference: Replace with:
```yaml
  compute_global_on_inference: false  # If true, infer.py emits population-level GII on inference data
```

7. L176-190 aggregate_shap block: Strip the multi-paragraph explanation but keep the commented-out example:
```yaml
# aggregate_shap:
#   subscale_A_total:              # Group name (must not collide with a feature name)
#     - "subscale_A_item1"         # Constituent features (disjoint, no nominals)
#     - "subscale_A_item2"
#     - "subscale_A_item3"
#   subscale_B_total:
#     - "subscale_B_item1"
#     - "subscale_B_item2"
```

8. L192-199 outcome_max/negate_shap: Replace with:
```yaml
  outcome_max: 100.0             # Theoretical maximum of the outcome (scales GII plot magnitudes)
  negate_shap: false             # If true, sign-flip SHAP y-axis values on plots
```

9. L200-209 label strings: Keep as-is (already one-line, no comments to remove). Remove any multi-line comment blocks above them.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Config file formatting only. No behavioral change. The new keys (cv_strategy, n_inner_repeats) have matching defaults in fill_config_defaults(), so existing configs that omit them will behave identically.</risk>
      <rollback>Revert example_config_advanced.yaml to prior version.</rollback>
    </change>

    <change id="C5" priority="P1" source_item="Prior-plan C7: example_config_minimal.yaml comment cleanup">
      <file path="example_config_minimal.yaml" action="modify" />
      <description>Verify minimal config follows one-line-comment convention. No new keys are added (cv_strategy and n_inner_repeats defaults apply when omitted). The file is already 71 lines and relatively clean; changes are limited to any multi-line comments that exist.</description>
      <spec>
Review all comments in the file. The minimal config is already compliant:
- L1-3: header comment (brief, functional) — keep.
- L57: `outcome: "outcome_total"` with inline comment — keep.
- L60-62: indiv_ci_nboot, indiv_scaling_mode, indiv_scaling_value — all one-line inline — keep.
- L64-70: plot section entries — all one-line inline — keep.

No structural changes are needed. If any multi-line comment is discovered during build execution, condense it to a single inline comment. Otherwise, this change is a verified no-op.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Formatting verification only.</risk>
      <rollback>N/A (no-op if already compliant).</rollback>
    </change>
  </changes>
  <execution_order>C1, C2, C3, C4, C5 (C1-C3 are P0 and have no mutual dependencies; C4-C5 are P1 config formatting. C1 and C2 can be parallelized. C3 is independent. C4-C5 can be parallelized.)</execution_order>
</implement_plan>
