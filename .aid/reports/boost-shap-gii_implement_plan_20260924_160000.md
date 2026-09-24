<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-09-24T16:00:00-04:00" />
  <input_reports>
    <report path="boost-shap-gii_checkpoint_resume_spec_20260924.md" mode="brainstorm+cr (harmonized)" key_items="7" />
  </input_reports>
  <changes>

    <change id="C1" priority="P0" source_item="Spec T1, T2, T9, CR F2, CR F3">
      <file path="src/boost_shap_gii/utils.py" action="modify" />
      <description>Add checkpoint infrastructure to utils.py: config hash computation, atomic CSV writes, checkpoint load/save helpers, and a predecessor mtime helper. All downstream changes (C2-C6) depend on these primitives.</description>
      <spec>
**New imports** (top of file): `hashlib`.

**New constant** (after existing imports, before function definitions):

```python
CONFIG_HASH_SCOPES = {
    "train": ["execution", "paths", "features", "modeling", "aggregate_shap", "transformations"],
    "predict": ["execution", "paths", "features", "modeling", "aggregate_shap", "transformations", "shap"],
    "infer": ["execution", "paths", "features", "modeling", "aggregate_shap", "transformations", "shap"],
}
```

Predict and infer scopes are strict supersets of train scope (union `{shap}`). Any config change in train's scope automatically cascades to predict/infer hashes. `plot` is excluded from all scopes (consumed only by R subprocess). Verified against codebase: train.py has zero `config["shap"]` references; no stage accesses `config["plot"]`.

**New function: `compute_config_hash(config, stage)`**

```python
def compute_config_hash(config, stage):
    sections = CONFIG_HASH_SCOPES[stage]
    filtered = {s: config.get(s, {}) for s in sections}
    canonical = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
```

`config` must be the raw config from `load_config()`, BEFORE `fill_config_defaults()` is called (CR F3). The `default=str` serializer handles Path objects and other non-JSON-native types that may appear in config values. `sort_keys=True` ensures key-order independence.

**New function: `save_csv_atomic(df, path, **kwargs)`**

Mirrors existing `save_json_atomic` pattern (lines 121-126): write to `path + ".tmp"`, then `os.replace(tmp_path, path)`.

```python
def save_csv_atomic(df, path, **kwargs):
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, **kwargs)
    os.replace(tmp_path, path)
```

CR F2: prevents silent data corruption from truncated CSV writes during crashes.

**New function: `load_checkpoint(path)`**

Returns the parsed JSON dict, or `None` if the file does not exist.

```python
def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
```

**New function: `save_checkpoint(data, path)`**

Thin wrapper delegating to `save_json_atomic` for POSIX-atomic checkpoint writes.

```python
def save_checkpoint(data, path):
    save_json_atomic(data, path)
```

**New function: `get_predecessor_mtime(path)`**

Returns `os.path.getmtime(path)` or `None` if the file does not exist (CR F7). Used by predict.py and infer.py to store and compare predecessor checkpoint mtimes.

```python
def get_predecessor_mtime(path):
    try:
        return os.path.getmtime(path)
    except (FileNotFoundError, OSError):
        return None
```
      </spec>
      <dependencies>none</dependencies>
      <risk>low - pure additions to utils.py; no existing functions or signatures modified</risk>
      <rollback>Delete the six new definitions (CONFIG_HASH_SCOPES, compute_config_hash, save_csv_atomic, load_checkpoint, save_checkpoint, get_predecessor_mtime) and the hashlib import.</rollback>
    </change>

    <change id="C2" priority="P1" source_item="Spec T8">
      <file path="src/boost_shap_gii/cli.py" action="modify" />
      <description>Add --force-restart flag to train, predict, and infer CLI subcommands and wire through sys.argv dispatch.</description>
      <spec>
**Three subparser additions** (in `build_parser()`):

For the `train` subparser (currently has only `--config`):
```python
train_parser.add_argument("--force-restart", action="store_true",
                          help="Delete this stage's checkpoint and restart from scratch")
```

For the `predict` subparser (currently has only `--config`):
```python
predict_parser.add_argument("--force-restart", action="store_true",
                            help="Delete this stage's checkpoint and restart from scratch")
```

For the `infer` subparser (currently has `--config`, `--data`, `--output-subdir`):
```python
infer_parser.add_argument("--force-restart", action="store_true",
                          help="Delete this stage's checkpoint and restart from scratch")
```

**Three cmd_* dispatch modifications:**

In `cmd_train` (line 36): after the existing `sys.argv` assignment, add:
```python
if args.force_restart:
    sys.argv.append("--force-restart")
```

In `cmd_predict` (line 45): same pattern.

In `cmd_infer` (line 54-59): same pattern, appending to the existing list.

Not added to `plot` or `check-env` subcommands (no checkpoint semantics for those stages).
      </spec>
      <dependencies>none</dependencies>
      <risk>low - argparse additions and sys.argv list appends; no existing behavior modified</risk>
      <rollback>Remove the three add_argument calls and the three conditional appends.</rollback>
    </change>

    <change id="C3" priority="P1" source_item="Spec T1, T2, T3, T9, CR F2, CR F3, CR F6">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <description>Add per-fold checkpoint/resume infrastructure to train.py: config hash computation from raw config, checkpoint lifecycle, five-artifact fold completion predicate, per-fold artifact saves, accumulator reconstruction on resume, fold_transform_metadata truncation, and resume/restart UX logging.</description>
      <spec>
**New imports** (add to existing import block):
- `import datetime`
- From `.__init__` or `boost_shap_gii`: `from . import __version__` (or `from boost_shap_gii import __version__`)
- From utils: add `compute_config_hash`, `save_csv_atomic`, `load_checkpoint`, `save_checkpoint` to the existing `from .utils import ...` statement

**Argparse modification** (line 629-631):
Add `--force-restart` argument:
```python
parser.add_argument("--force-restart", action="store_true")
```

**Config hash computation** (immediately after `config = load_config(args.config)` at line 634, BEFORE any other config operations):
```python
config_hash = compute_config_hash(config, "train")
```
This captures the raw config hash before `fill_config_defaults()` modifies config in-place at line 702.

**Checkpoint handling block** (after `os.makedirs(run_dir, exist_ok=True)` at line 636, before data loading at line 638):
```python
checkpoint_path = os.path.join(run_dir, "_checkpoint_train.json")
force_restart = getattr(args, "force_restart", False)

if force_restart and os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print("[RESTART] Forced restart; checkpoint removed.")

existing_checkpoint = load_checkpoint(checkpoint_path)
if existing_checkpoint is not None:
    if existing_checkpoint["config_hash"] != config_hash:
        os.remove(checkpoint_path)
        existing_checkpoint = None
        print("[RESTART] Config changed; checkpoint invalidated.")
    elif existing_checkpoint["status"] == "complete":
        print("[RESUME] Training already complete; nothing to do.")
        return
    else:
        print("[RESUME] Partial checkpoint found; will resume after setup.")
```

**New helper function: `_fold_complete(run_dir, fold_idx)`**

Place before `main()`. Returns True if all five per-fold artifacts exist:
```python
def _fold_complete(run_dir, fold_idx):
    artifacts = [
        f"model_fold_{fold_idx}.cbm",
        f"shadow_model_fold_{fold_idx}.cbm",
        f"_oof_fold_{fold_idx}.csv",
        f"_metrics_fold_{fold_idx}.json",
        f"_params_fold_{fold_idx}.json",
    ]
    return all(os.path.exists(os.path.join(run_dir, a)) for a in artifacts)
```

**Accumulator reconstruction block** (after in-memory accumulator initialization at lines 990-1009, before the fold loop at line 1013):

```python
completed_folds = set()
if existing_checkpoint is not None and existing_checkpoint["status"] == "partial":
    n_folds_total = splitter.get_n_splits()
    for k in range(n_folds_total):
        if _fold_complete(run_dir, k):
            completed_folds.add(k)

    if completed_folds:
        for k in sorted(completed_folds):
            fold_df = pd.read_csv(os.path.join(run_dir, f"_oof_fold_{k}.csv"))
            row_idx = fold_df["row_idx"].values
            pred_cols = [c for c in fold_df.columns if c != "row_idx"]
            if task == "multiclass_classification":
                oof_preds.iloc[row_idx] = fold_df[pred_cols].values
            elif task == "multi_regression":
                oof_preds.iloc[row_idx] = fold_df[pred_cols].values
            else:
                oof_preds.iloc[row_idx] = fold_df["y_pred"].values
            fold_assignments[row_idx] = k

            with open(os.path.join(run_dir, f"_metrics_fold_{k}.json")) as f:
                fold_metrics.append(json.load(f))

        # CR F6: truncate fold_transform_metadata to completed fold count
        if transform_module is not None:
            ftm_path = os.path.join(run_dir, "fold_transform_metadata.json")
            if os.path.exists(ftm_path):
                with open(ftm_path) as f:
                    all_fold_transform_meta = json.load(f)[:len(completed_folds)]

        print(f"[RESUME] Reconstructed {len(completed_folds)}/{n_folds_total} folds; resuming.")

# Initialize checkpoint if new run
now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
if existing_checkpoint is None:
    existing_checkpoint = {
        "pipeline_version": __version__,
        "config_hash": config_hash,
        "stage": "train",
        "status": "partial",
        "started_at": now_utc,
        "updated_at": now_utc,
        "predecessor_checkpoint_mtime": None,
        "progress": {"completed_folds": sorted(completed_folds), "total_folds": splitter.get_n_splits()}
    }
    save_checkpoint(existing_checkpoint, checkpoint_path)
```

**Fold loop skip gate** (inside the fold loop, immediately after `for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y_for_split)):` at line 1013, before the existing fold body):

```python
if fold_idx in completed_folds:
    continue
```

The accumulator reconstruction above already filled oof_preds, fold_assignments, and fold_metrics for completed folds; the `continue` simply skips the training body.

**Per-fold artifact saves** (inside the fold loop, after `model_shadow.save_model(shadow_model_path)` at line 1203, before the loop's next iteration):

Save three new per-fold artifacts:

1. `_oof_fold_{fold_idx}.csv`: row_idx (val_idx positions) + prediction columns. Column names depend on task type:
   - regression/binary: `["row_idx", "y_pred"]`
   - multiclass: `["row_idx"] + class_labels` (the class label column names from oof_preds.columns)
   - multi_regression: `["row_idx"] + [f"y_pred_{col}" for col in outcome_cols]`

   Use `save_csv_atomic` for crash safety (CR F2). The predictions stored are post-back-transform (if applicable), matching the state of `oof_preds.iloc[val_idx]` at this point in the loop.

2. `_metrics_fold_{fold_idx}.json`: the `metrics` dict for this fold (already computed at this point).
   Use `save_json_atomic`.

3. `_params_fold_{fold_idx}.json`: `{"best_params": best_params, "tuned_iters": tuned_iters}`.
   Use `save_json_atomic`.

After saving artifacts, update checkpoint:
```python
existing_checkpoint["progress"]["completed_folds"].append(fold_idx)
existing_checkpoint["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
save_checkpoint(existing_checkpoint, checkpoint_path)
```

**Finalization checkpoint** (after the last statement in the finalization block, before `print(f"[SUCCESS] Training finished...")` at line 1265):

```python
existing_checkpoint["status"] = "complete"
existing_checkpoint["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
save_checkpoint(existing_checkpoint, checkpoint_path)
```

This must be the LAST write: only set status=complete when all finalization artifacts (fold_assignments.json, full_oof_predictions.csv, metrics_oof.csv/json, task_info.json, train_outcome_stats.json, transform_config.json) are already saved.
      </spec>
      <dependencies>C1 (utils.py checkpoint primitives)</dependencies>
      <risk>medium - modifies the training fold loop (core algorithm). The skip gate and accumulator reconstruction must not alter the content of finalization artifacts. Mitigation: all existing code paths are untouched; new code is additive (skip gate + artifact saves + reconstruction block).</risk>
      <rollback>Revert the file to its pre-change state. Per-fold artifact files (_oof_fold_*.csv, _metrics_fold_*.json, _params_fold_*.json) and _checkpoint_train.json in run_dir are inert (no existing code reads them) and can be left in place or manually removed.</rollback>
    </change>

    <change id="C4" priority="P1" source_item="Spec T6">
      <file path="src/boost_shap_gii/indiv_reports.py" action="modify" />
      <description>Add bootstrap refit cache skip pattern: per-refit alpha sidecar writes in _fit_and_save_refit, and existence-check skip with alpha reconstruction in orchestrate_bootstrap_cache.</description>
      <spec>
**Modification 1: `_fit_and_save_refit` (line ~265)**

After the `.cbm` model file is saved (line ~327: `model.save_model(...)`), save a `fold_{k}_alpha.json` sidecar file in the same directory containing the alpha float:

```python
alpha_sidecar_path = os.path.join(os.path.dirname(model_path), f"fold_{k}_alpha.json")
save_json_atomic({"alpha": alpha}, alpha_sidecar_path)
```

This must use the existing `save_json_atomic` import. The alpha value is already computed and returned by `_fit_and_save_refit`; the sidecar makes it recoverable per-refit without re-fitting the model. When `back_transform_shap` is inactive, alpha defaults to 1.0; the sidecar still records it.

**Modification 2: `orchestrate_bootstrap_cache` (line ~526)**

Before task generation (the `for b in range(B): for k in range(K)` loop at line ~630), add an existence-check scan:

```python
# Scan for completed (b, k) pairs
completed_pairs = set()
for b in range(B):
    iter_dir = os.path.join(cache_dir, f"iter_{b:05d}")
    for k in range(K):
        cbm_path = os.path.join(iter_dir, f"fold_{k}.cbm")
        if os.path.exists(cbm_path):
            completed_pairs.add((b, k))

if completed_pairs:
    print(f"[RESUME] Bootstrap cache: {len(completed_pairs)}/{B * K} refits found; "
          f"{B * K - len(completed_pairs)} remaining.")
```

Filter the task list to exclude completed pairs:
```python
# In the existing task generation loop, add guard:
if (b, k) in completed_pairs:
    continue
```

After all futures resolve (line ~667-672), reconstruct `boot_alphas` from sidecar files for completed pairs that were skipped:

For each `(b, k)` in `completed_pairs`, read `iter_{b:05d}/fold_{k}_alpha.json` to get the alpha value. Fill the corresponding cell in the `boot_alphas` array. The existing code fills cells for newly-completed pairs from future results; the reconstruction fills cells for pre-existing pairs from sidecars.

```python
for (b, k) in completed_pairs:
    sidecar = os.path.join(cache_dir, f"iter_{b:05d}", f"fold_{k}_alpha.json")
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            boot_alphas[b, k] = json.load(f)["alpha"]
    else:
        boot_alphas[b, k] = 1.0  # absence implies alpha = 1.0 (spec T6)
```

`shared_indices.npz`: deterministic from seed. Overwrite unconditionally (or skip if exists; either is correct since content is identical).

`bootstrap_metadata.json`: completion marker, written only after ALL futures resolve (existing behavior). No change to this write.
      </spec>
      <dependencies>C1 (save_json_atomic already imported; no new utils dependency)</dependencies>
      <risk>low-medium - modifies bootstrap dispatch logic. The skip is additive (filter completed pairs from task list). Alpha reconstruction from sidecars is a new read path. Risk: incorrect alpha reconstruction if sidecar is missing (mitigated by the 1.0 default per T6 spec).</risk>
      <rollback>Revert the file. Sidecar files (fold_{k}_alpha.json) are inert; no existing code reads them.</rollback>
    </change>

    <change id="C5" priority="P1" source_item="Spec T1, T4, CR F1, CR F4, CR F7">
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <description>Add phase-gate checkpoint/resume to predict.py: config hash from raw config, predecessor mtime guard against train's checkpoint, five-phase dispatcher with terminal artifact checks, predictions_oof.csv reordering (CR F4), and resume UX logging.</description>
      <spec>
**New imports** (add to existing import block):
- `import datetime`
- `from . import __version__`
- From utils: add `compute_config_hash`, `load_checkpoint`, `save_checkpoint`, `get_predecessor_mtime` to the existing `from .utils import ...` statement

**Argparse modification** (line 44-47):
Add `--force-restart`:
```python
parser.add_argument("--force-restart", action="store_true")
```

**Config hash computation** (immediately after `config = load_config(args.config)` at line 50, BEFORE any config modification):
```python
config_hash = compute_config_hash(config, "predict")
```

**Checkpoint handling block** (after `run_dir = config["paths"]["output_dir"]` at line 51, before the metadata loading at line 56):

```python
checkpoint_path = os.path.join(run_dir, "_checkpoint_predict.json")
force_restart = getattr(args, "force_restart", False)
train_checkpoint_path = os.path.join(run_dir, "_checkpoint_train.json")

if force_restart and os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print("[RESTART] Forced restart; checkpoint removed.")

existing_checkpoint = load_checkpoint(checkpoint_path)
completed_phases = set()

if existing_checkpoint is not None:
    if existing_checkpoint["config_hash"] != config_hash:
        os.remove(checkpoint_path)
        existing_checkpoint = None
        print("[RESTART] Config changed; checkpoint invalidated.")
    else:
        # CR F7: predecessor mtime guard
        stored_mtime = existing_checkpoint.get("predecessor_checkpoint_mtime")
        current_mtime = get_predecessor_mtime(train_checkpoint_path)
        if stored_mtime != current_mtime:
            os.remove(checkpoint_path)
            existing_checkpoint = None
            print("[RESTART] Train checkpoint changed; predict checkpoint invalidated.")
        elif existing_checkpoint["status"] == "complete":
            print("[RESUME] Predict already complete; nothing to do.")
            return
        else:
            completed_phases = set(existing_checkpoint.get("progress", {}).get("completed_phases", []))
            print(f"[RESUME] Partial checkpoint found; completed phases: {sorted(completed_phases)}.")
```

**Initialize checkpoint if new run** (after the block above):
```python
now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
if existing_checkpoint is None:
    existing_checkpoint = {
        "pipeline_version": __version__,
        "config_hash": config_hash,
        "stage": "predict",
        "status": "partial",
        "started_at": now_utc,
        "updated_at": now_utc,
        "predecessor_checkpoint_mtime": get_predecessor_mtime(train_checkpoint_path),
        "progress": {"completed_phases": [], "skipped_phases": []}
    }
    save_checkpoint(existing_checkpoint, checkpoint_path)
```

**Helper to update checkpoint after each phase:**
```python
def _update_predict_checkpoint(checkpoint, checkpoint_path, phase_id):
    if phase_id not in checkpoint["progress"]["completed_phases"]:
        checkpoint["progress"]["completed_phases"].append(phase_id)
    checkpoint["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_checkpoint(checkpoint, checkpoint_path)
```
(This can be a local closure or a module-level helper.)

**Setup code** (lines 56-196): runs unconditionally on resume (fast, required by all downstream phases). No changes.

**CR F4: Move predictions_oof.csv write** from its current location (lines 480-492, after metrics) to immediately after the OOF loop completes (after line 314, before the metrics section at line 326).

The moved block is the pred_df construction and `pred_df.to_csv(...)` call. The metrics section (line 326+) reads from in-memory `oof_preds` and `y`, not from the CSV. No semantic change.

**Phase-gate dispatcher**: wrap each existing phase block in a completion check:

**P1 gate** (before OOF prediction loop at line 198):
```python
if "P1" not in completed_phases:
    # ... existing OOF loop (lines 198-314) ...
    # ... moved predictions_oof.csv write (from lines 480-492) ...
    _update_predict_checkpoint(existing_checkpoint, checkpoint_path, "P1")
else:
    # Reload oof_preds from predictions_oof.csv for downstream phases
    pred_df = pd.read_csv(os.path.join(run_dir, "predictions_oof.csv"))
    if task == "multiclass_classification":
        prob_cols = [f"prob_{cl}" for cl in class_labels]
        oof_preds = pred_df[prob_cols].values
    elif task == "multi_regression":
        pred_cols = [f"y_pred_{col}" for col in outcome_cols]
        oof_preds = pred_df[pred_cols].values
    else:
        oof_preds = pred_df["y_pred"].values
    counts = np.ones(len(oof_preds))  # all rows accounted for
    print("[RESUME] P1 (OOF predictions): skipped (complete).")
```

**P2 gate** (before metrics section at line 326):
```python
if "P2" not in completed_phases:
    # ... existing metrics, bootstrap, permutation code (lines 326-477) ...
    _update_predict_checkpoint(existing_checkpoint, checkpoint_path, "P2")
else:
    print("[RESUME] P2 (metrics & bootstrap): skipped (complete).")
```

**P3 gate** (before SHAP pipeline at line 494):
```python
if "P3" not in completed_phases:
    # ... existing SHAP pipeline (lines 494-522) ...
    _update_predict_checkpoint(existing_checkpoint, checkpoint_path, "P3")
else:
    print("[RESUME] P3 (SHAP pipeline): skipped (complete).")
```

**P4/P5 gate** (at the `if nboot_indiv > 0:` block, line 528):

P4 and P5 are conditional on `nboot_indiv > 0`. If `nboot_indiv == 0`, both are skipped and recorded in `skipped_phases`.

```python
if nboot_indiv > 0:
    if "P4" not in completed_phases:
        # ... existing orchestrate_bootstrap_cache call (lines 544-567) ...
        _update_predict_checkpoint(existing_checkpoint, checkpoint_path, "P4")
    else:
        print("[RESUME] P4 (bootstrap cache): skipped (complete).")

    if "P5" not in completed_phases:
        # ... existing generate_indiv_reports call (lines 569-592) ...
        _update_predict_checkpoint(existing_checkpoint, checkpoint_path, "P5")
    else:
        print("[RESUME] P5 (individual reports): skipped (complete).")
else:
    # Record both as skipped
    for p in ["P4", "P5"]:
        if p not in existing_checkpoint["progress"].get("skipped_phases", []):
            existing_checkpoint["progress"].setdefault("skipped_phases", []).append(p)
    print("[INFO] shap.indiv_ci_nboot=0; skipping P4/P5.")
```

**Terminal artifact checks** for phase completion detection (used by the phase-gate dispatcher when loading an existing checkpoint to verify claimed completions):

| Phase | Terminal artifact path (relative to run_dir) |
|---|---|
| P1 | `predictions_oof.csv` |
| P2 | `performance_final.csv` |
| P3 | `shap_analysis/shap_stats_global.csv` |
| P4 | `bootstrap_refits/bootstrap_metadata.json` |
| P5 | `indiv_reports/indiv_reports_metadata.json` (CR F1 corrected) |

The `completed_phases` set from the checkpoint is used directly (populated by `_update_predict_checkpoint` during the run). No terminal-artifact re-scan on resume; the checkpoint is the authoritative record, and each phase's code produces the artifact before the checkpoint is updated.

**Finalization** (after the last phase block, before `if __name__`):
```python
existing_checkpoint["status"] = "complete"
existing_checkpoint["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
save_checkpoint(existing_checkpoint, checkpoint_path)
```
      </spec>
      <dependencies>C1 (utils.py primitives), C2 (--force-restart in cli.py), C3 (train checkpoint must exist for predecessor mtime)</dependencies>
      <risk>medium - modifies predict.py flow. The predictions_oof.csv reordering (CR F4) is a code move with no semantic change (verified: metrics section reads in-memory arrays, not the CSV). Phase gates are additive wrapping of existing blocks. Risk: oof_preds reload from CSV on P1 skip must produce the same ndarray shape/dtype as the original computation; column name mismatch would cause silent data corruption. Mitigation: column names are derived from class_labels/outcome_cols which are loaded from training artifacts in the setup code.</risk>
      <rollback>Revert the file. _checkpoint_predict.json in run_dir is inert.</rollback>
    </change>

    <change id="C6" priority="P2" source_item="Spec T1, T5, CR F1, CR F5, CR F7">
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>Add phase-gate checkpoint/resume to infer.py: config hash from raw config, data_path guard, P4 as pass-through prerequisite check (CR F5), predecessor mtime guard against predict's checkpoint, and resume UX logging.</description>
      <spec>
**New imports** (add to existing import block):
- `import datetime`
- `from . import __version__`
- From utils: add `compute_config_hash`, `load_checkpoint`, `save_checkpoint`, `get_predecessor_mtime` to the existing `from .utils import ...` statement

**Argparse modification** (line 46-55):
Add `--force-restart`:
```python
parser.add_argument("--force-restart", action="store_true")
```

**Config hash computation** (immediately after `config = load_config(args.config)` at line 58):
```python
config_hash = compute_config_hash(config, "infer")
```

**Checkpoint handling block** (after `os.makedirs(infer_dir, exist_ok=True)` at line 61):

Checkpoint location: `{infer_dir}/_checkpoint_infer.json` (per-subdir, not per-run_dir, since infer runs multiple independent datasets into separate subdirs).

Predecessor: predict's checkpoint at `{run_dir}/_checkpoint_predict.json` (train_dir == run_dir for infer).

Additional guards beyond config hash:
- `data_path`: the `args.data` value. If the checkpoint's stored data_path differs from the current invocation's `args.data`, invalidate (user is running against a different dataset).

```python
checkpoint_path = os.path.join(infer_dir, "_checkpoint_infer.json")
force_restart = getattr(args, "force_restart", False)
predict_checkpoint_path = os.path.join(train_dir, "_checkpoint_predict.json")

if force_restart and os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print("[RESTART] Forced restart; checkpoint removed.")

existing_checkpoint = load_checkpoint(checkpoint_path)
completed_phases = set()

if existing_checkpoint is not None:
    if existing_checkpoint["config_hash"] != config_hash:
        os.remove(checkpoint_path)
        existing_checkpoint = None
        print("[RESTART] Config changed; checkpoint invalidated.")
    elif existing_checkpoint.get("data_path") != args.data:
        os.remove(checkpoint_path)
        existing_checkpoint = None
        print("[RESTART] Data path changed; checkpoint invalidated.")
    else:
        stored_mtime = existing_checkpoint.get("predecessor_checkpoint_mtime")
        current_mtime = get_predecessor_mtime(predict_checkpoint_path)
        if stored_mtime != current_mtime:
            os.remove(checkpoint_path)
            existing_checkpoint = None
            print("[RESTART] Predict checkpoint changed; infer checkpoint invalidated.")
        elif existing_checkpoint["status"] == "complete":
            print("[RESUME] Inference already complete; nothing to do.")
            return
        else:
            completed_phases = set(existing_checkpoint.get("progress", {}).get("completed_phases", []))
            print(f"[RESUME] Partial checkpoint found; completed phases: {sorted(completed_phases)}.")
```

**Initialize checkpoint if new:**
```python
now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
if existing_checkpoint is None:
    existing_checkpoint = {
        "pipeline_version": __version__,
        "config_hash": config_hash,
        "stage": "infer",
        "status": "partial",
        "started_at": now_utc,
        "updated_at": now_utc,
        "predecessor_checkpoint_mtime": get_predecessor_mtime(predict_checkpoint_path),
        "data_path": args.data,
        "output_subdir": args.output_subdir,
        "progress": {"completed_phases": [], "skipped_phases": []}
    }
    save_checkpoint(existing_checkpoint, checkpoint_path)
```

**Phase-gate dispatcher**: same pattern as predict.py (C5), wrapping each existing phase block.

**P1-P3**: Standard phase gates with terminal artifact checks on the infer_dir file tree.

**P4**: CR F5 correction. Infer's P4 is a prerequisite check, NOT a compute phase. It only calls `_load_bootstrap_cache_or_fail(train_dir)`. The phase gate does NOT record P4 in completed_phases. Instead, it verifies the prerequisite exists before P5:

```python
if nboot_indiv > 0:
    # P4: prerequisite check (not a compute phase; CR F5)
    bootstrap_meta_path = os.path.join(train_dir, "bootstrap_refits", "bootstrap_metadata.json")
    if not os.path.exists(bootstrap_meta_path):
        raise FileNotFoundError(
            f"Bootstrap cache not found at {bootstrap_meta_path}. "
            f"Run predict.py first to build the bootstrap refit cache."
        )

    if "P5" not in completed_phases:
        # ... existing generate_indiv_reports call ...
        _update_infer_checkpoint(existing_checkpoint, checkpoint_path, "P5")
    else:
        print("[RESUME] P5 (individual reports): skipped (complete).")
```

**P5 terminal artifact**: `indiv_reports/indiv_reports_metadata.json` (CR F1 corrected).

**Finalization**: same pattern as predict.py: set status=complete, save checkpoint.

**Phase terminal artifacts for infer** (all relative to infer_dir):

| Phase | Terminal artifact | Conditional |
|---|---|---|
| P1 | `predictions_ensemble.csv` | always |
| P2 | `performance_final.csv` | `outcomes_present` |
| P3 | `shap_analysis/shap_stats_global.csv` | `compute_global_on_inference` |
| P4 | (prerequisite check only; no terminal artifact) | `nboot_indiv > 0` |
| P5 | `indiv_reports/indiv_reports_metadata.json` | `nboot_indiv > 0` |
      </spec>
      <dependencies>C1 (utils.py primitives), C2 (--force-restart in cli.py), C5 (predict checkpoint must exist for predecessor mtime)</dependencies>
      <risk>low - mirrors the predict.py pattern (C5). P4 pass-through simplifies the logic relative to predict.py. infer.py is structurally simpler (no bootstrap cache build).</risk>
      <rollback>Revert the file. _checkpoint_infer.json in infer_dir is inert.</rollback>
    </change>

  </changes>

  <execution_order>C1 → C2 → C3 → C4 → C5 → C6</execution_order>

  <notes>
    **Execution order rationale**: C1 (utils.py) is foundational; all others import from it. C2 (cli.py) is independent but placed second for clarity. C3 (train.py) must precede C5 (predict.py) because predict.py references train's checkpoint for the predecessor mtime guard. C4 (indiv_reports.py) must precede C5 because predict.py's P4 phase calls orchestrate_bootstrap_cache. C5 (predict.py) must precede C6 (infer.py) for the same predecessor mtime reason.

    **Parallelization potential**: C1 and C2 touch different files and could be parallelized. After C1 completes, C3 and C4 touch different files and could be parallelized. C5 depends on both C3 and C4. C6 depends on C5.

    **Test coverage**: Not in scope for this plan (per /implement constraints). Recommend `/test` after build to validate: partial completions (crash mid-fold, crash mid-phase), config hash mismatch detection, --force-restart behavior, crash safety (partial artifact sets passing/failing the five-artifact predicate), predecessor mtime invalidation, fold_transform_metadata truncation, bootstrap cache skip with alpha sidecar reconstruction, and P4 pass-through behavior in infer.

    **Documentation**: Not in scope for this plan. Recommend `/document` after test to update README.md (--force-restart flag, checkpoint file locations, resume behavior), INPUT_SPECIFICATION.md (checkpoint schema, config hash scope per stage, known limitation: hash guards config-file changes, not data-content changes), and AID_LOG.md (session entry).

    **Known limitation (documented per CR F3)**: The config hash guards against config-file changes, not data-content changes. If a user replaces the data file at the same path without changing the config, the checkpoint will not detect the change. This is a standard limitation of path-based hashing systems and is documented in the harmonized spec.

    **KI-001 reminder**: The eventual /publish v1.7.0 must bump version in BOTH pyproject.toml and __init__.py.
  </notes>

</implement_plan>
