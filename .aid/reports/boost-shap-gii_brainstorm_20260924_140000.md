<brainstorm_report>
  <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-09-24T14:00:00-04:00" />
  <context_files>
    <file path="src/boost_shap_gii/train.py" relevance="Fold loop structure, per-fold artifact writes, in-memory accumulators, config key access patterns" />
    <file path="src/boost_shap_gii/predict.py" relevance="Sequential phase structure (OOF, metrics, SHAP, bootstrap refits, indiv_reports), config key access patterns" />
    <file path="src/boost_shap_gii/infer.py" relevance="Phase structure mirroring predict.py, bootstrap-of-CV CI, per-subdir output, CLI args (--data, --output-subdir)" />
    <file path="src/boost_shap_gii/indiv_reports.py" relevance="orchestrate_bootstrap_cache (B x K refit dispatch, per-(b,k) .cbm saves, bootstrap_metadata.json), generate_indiv_reports (CI accumulation loop)" />
    <file path="src/boost_shap_gii/shap_utils.py" relevance="run_shap_pipeline phase boundary, per-fold SHAP computation via joblib" />
    <file path="src/boost_shap_gii/utils.py" relevance="save_json_atomic utility, compute_bootstrap_ci, shared helpers" />
    <file path="src/boost_shap_gii/cli.py" relevance="CLI subcommand dispatch, argparse definitions, --force-restart flag target" />
    <file path="example_config_advanced.yaml" relevance="Full config structure showing all top-level sections (execution, paths, features, modeling, shap, aggregate_shap, transformations, plot)" />
  </context_files>
  <topics>
    <topic id="T1" title="Checkpoint state storage design">
      <summary>Per-stage checkpoint files in the run directory, co-located with each stage's output artifacts. train and predict write to run_dir; infer writes to {run_dir}/{output_subdir}/. Each file follows a common JSON schema with stage-specific progress fields.</summary>
      <approaches>
        <approach id="A1" label="Single unified file" feasibility="high" risk="medium">
          <description>One _checkpoint.json in run_dir with per-stage sections; infer keyed by output_subdir.</description>
          <pros>Single file to inspect/delete; single source of truth.</pros>
          <cons>Grows unbounded with many infer runs; concurrent infer runs could race on the file.</cons>
        </approach>
        <approach id="A2" label="Per-stage files" feasibility="high" risk="low">
          <description>_checkpoint_train.json and _checkpoint_predict.json in run_dir; _checkpoint_infer.json in {run_dir}/{output_subdir}/.</description>
          <pros>Natural separation; infer checkpoints co-locate with infer outputs; no concurrency risk.</pros>
          <cons>Multiple files to manage; --force-restart must find and delete the relevant file.</cons>
        </approach>
        <approach id="A3" label="Checkpoint subdirectory" feasibility="high" risk="low">
          <description>.checkpoint/ subdirectory inside run_dir with per-stage JSON files.</description>
          <pros>Clean separation from pipeline artifacts; rm -rf .checkpoint/ for manual reset.</pros>
          <cons>Extra directory; dot-prefix hides from casual ls.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A2">Per-stage files. Filenames: _checkpoint_train.json, _checkpoint_predict.json (in run_dir); _checkpoint_infer.json (in {run_dir}/{output_subdir}/). Common schema: pipeline_version, config_hash, stage, status (complete|partial), started_at, updated_at, progress (stage-specific). Underscore prefix sorts near top of directory listings without hiding.</decision>
    </topic>
    <topic id="T2" title="Config hash: scope and invalidation">
      <summary>Stage-scoped hashing with superset cascading. Each stage hashes only the config sections it reads. Because predict's scope is a strict superset of train's scope, any config change affecting train automatically invalidates predict. The plot section is excluded from all hashes (structurally consumed only by the R subprocess).</summary>
      <approaches>
        <approach id="A1" label="Full config hash" feasibility="high" risk="low">
          <description>Hash the entire YAML (minus plot). Any change invalidates all stages.</description>
          <pros>Maximally conservative; zero false negatives.</pros>
          <cons>False positives: changing shap.indiv_ci_nboot forces retrain; changing paths for infer invalidates training.</cons>
        </approach>
        <approach id="A2" label="Stage-scoped hash with superset cascading" feasibility="high" risk="low">
          <description>Each stage hashes its own config scope. Train scope: {execution, paths, features, modeling, aggregate_shap, transformations}. Predict/infer scope: train scope + {shap}. Upstream cascade is automatic via superset inclusion.</description>
          <pros>Zero false positives; zero false negatives; infer can run on new datasets without invalidating training.</pros>
          <cons>A new top-level config section requires adding it to the scope lists (one-line change, caught by testing).</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A2">Stage-scoped hashing. Train scope: {execution, paths, features, modeling, aggregate_shap, transformations}. Predict/infer scope: train scope union {shap}. Plot section excluded from all hashes (verified: train.py has zero references to config["shap"]; plot.* is consumed only by Rscript plot.R). Hash method: json.dumps(filtered_config, sort_keys=True, default=str) then SHA-256. On mismatch: discard checkpoint, restart stage from scratch, cascade forward to all successor stages. Each stage also verifies predecessor checkpoints are complete and hash-valid before resuming.</decision>
    </topic>
    <topic id="T3" title="train.py per-fold incremental artifacts">
      <summary>Save three additional per-fold files inside the fold loop (OOF predictions, metrics, tuned hyperparameters) to enable zero-reconstruction resume. Retained after completion for per-fold traceability.</summary>
      <approaches>
        <approach id="A1" label="Save per-fold artifacts" feasibility="high" risk="low">
          <description>Write _oof_fold_{k}.csv, _metrics_fold_{k}.json, _params_fold_{k}.json inside the fold loop alongside the existing .cbm files.</description>
          <pros>Zero reconstruction cost on resume; straightforward concatenation at end; adds per-fold traceability.</pros>
          <cons>2K additional small files in run_dir per training run (negligible).</cons>
        </approach>
        <approach id="A2" label="Reconstruct on resume" feasibility="high" risk="low">
          <description>On resume, reload saved .cbm models for completed folds, re-predict on val sets, recompute metrics.</description>
          <pros>No additional files.</pros>
          <cons>Adds a reconstruction pass (seconds, not minutes); requires re-loading data and re-splitting.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Save all three per-fold artifacts. Files: _oof_fold_{k}.csv (indices + predictions), _metrics_fold_{k}.json (fold metrics dict), _params_fold_{k}.json (tuned hyperparameters + tuned_iters). Underscore-prefixed to distinguish from terminal artifacts. Fold completion predicate: all five artifacts exist (model_fold_{k}.cbm, shadow_model_fold_{k}.cbm, _oof_fold_{k}.csv, _metrics_fold_{k}.json, _params_fold_{k}.json). Missing any one means incomplete fold, re-run. Retained after loop completion (not cleaned up).</decision>
    </topic>
    <topic id="T4" title="predict.py phase-gate boundaries">
      <summary>Five sequential phases with distinct terminal artifact sets. Phase-gate dispatcher at top of main() finds the first incomplete phase and jumps to it, re-executing setup code unconditionally.</summary>
      <approaches>
        <approach id="A1" label="Five-phase decomposition" feasibility="high" risk="low">
          <description>P1: OOF predictions (terminal: predictions_oof.csv). P2: Metrics and bootstrap (terminal: performance_final.csv, bootstrap_distributions_perf.parquet, permutation_results_*.csv). P3: SHAP pipeline (terminal: shap_analysis/shap_stats_global.csv). P4: Bootstrap refit cache (terminal: bootstrap_refits/bootstrap_metadata.json). P5: Individual reports (terminal: indiv_reports/indiv_metadata.json). P4 and P5 conditional on shap.indiv_ci_nboot > 0.</description>
          <pros>Clean boundaries at natural phase transitions; each phase's terminal artifacts are already written by the pipeline.</pros>
          <cons>None identified.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Five-phase decomposition as described. Checkpoint progress schema: completed_phases (list of phase name strings), skipped_phases (phases skipped by config, e.g., indiv_ci_nboot=0), bootstrap_refits sub-object for P4 fine-grained state (completed_pairs count, total_pairs). Data loading and setup code (lines 44-196) re-executed unconditionally on resume (fast, required by all phases).</decision>
    </topic>
    <topic id="T5" title="infer.py phase-gate boundaries">
      <summary>Same five-phase decomposition as predict.py, with structural differences: P2 conditional on outcomes present in inference data, P3 conditional on compute_global_on_inference config, P4 shared at run_dir level (not per-subdir), checkpoint includes data_path guard.</summary>
      <approaches>
        <approach id="A1" label="Five-phase decomposition mirroring predict.py" feasibility="high" risk="low">
          <description>Same phase boundaries, adapted for infer's structural differences. Checkpoint file at {run_dir}/{output_subdir}/_checkpoint_infer.json. Additional fields: data_path (from --data CLI arg), output_subdir, outcomes_present boolean.</description>
          <pros>Shared infrastructure with predict.py; P4 reuse across infer runs.</pros>
          <cons>None identified.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Five-phase decomposition mirroring predict.py. P2 conditional on outcomes_present. P3 conditional on compute_global_on_inference. P4 checks {run_dir}/bootstrap_refits/bootstrap_metadata.json (shared with predict); if predict already built the cache, infer skips P4 entirely. Checkpoint at {run_dir}/{output_subdir}/_checkpoint_infer.json with data_path guard: mismatch between stored data_path and current --data arg invalidates the checkpoint.</decision>
    </topic>
    <topic id="T6" title="Bootstrap refit cache skip pattern">
      <summary>Filter the task list in orchestrate_bootstrap_cache to exclude (b, k) pairs whose .cbm file already exists. Per-refit alpha values saved as tiny JSON sidecar files for zero-cost recovery on resume.</summary>
      <approaches>
        <approach id="A1" label="Re-probe alpha from existing .cbm files" feasibility="high" risk="low">
          <description>Load each completed refit model, re-run the alpha probe.</description>
          <pros>No additional files.</pros>
          <cons>Requires transform module and training data available; nonzero (though cheap) recomputation.</cons>
        </approach>
        <approach id="A2" label="Per-refit alpha sidecar file" feasibility="high" risk="low">
          <description>Each worker writes iter_{b:05d}/fold_{k}_alpha.json (one float) alongside the .cbm. Absence means alpha = 1.0.</description>
          <pros>Zero reconstruction cost; decoupled from transform module availability.</pros>
          <cons>One additional tiny file per (b, k) pair when transforms are active.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A2">Per-refit alpha sidecar files. Pattern: iter_{b:05d}/fold_{k}_alpha.json. Written by the worker process alongside the .cbm save when back_transform_shap is active. Absence implies alpha = 1.0 (no transform). On resume, completed pairs are detected by .cbm existence; alpha values for completed pairs are loaded from their sidecar files. shared_indices.npz is deterministic from seed (overwrite or skip if exists; no issue). bootstrap_metadata.json remains the completion marker (written only after all futures resolve). Resume logging: single line reporting n_existing/total found, n_remaining dispatched.</decision>
    </topic>
    <topic id="T7" title="CI accumulation checkpointing">
      <summary>Deferred from v1.7.0. The CI accumulation loop in generate_indiv_reports (B iterations, each loading K models and computing SHAP) is the post-refit wall-clock bottleneck, but checkpointing the accumulation arrays (especially interaction SHAP: N x F x F x B float32) produces potentially multi-GB snapshots. Defer and assess whether T3-T6 checkpointing is sufficient for the user's server time constraints.</summary>
      <approaches>
        <approach id="A1" label="Include in v1.7.0" feasibility="medium" risk="medium">
          <description>Save main_replicates and interaction_replicates arrays every N iterations to _ci_checkpoint.npz.</description>
          <pros>Prevents rework if timeout occurs during CI accumulation.</pros>
          <cons>Interaction tensor (N x F x F x B, float32) can be multi-GB; periodic saves add I/O overhead.</cons>
        </approach>
        <approach id="A2" label="Checkpoint main effects only" feasibility="high" risk="low">
          <description>Save only main_replicates (manageable sizes); recompute interactions on resume.</description>
          <pros>Manageable file sizes.</pros>
          <cons>Interactions still recomputed; partial solution.</cons>
        </approach>
        <approach id="A3" label="Defer entirely" feasibility="high" risk="low">
          <description>No CI accumulation checkpointing in v1.7.0. Reassess after evaluating whether T3-T6 checkpointing resolves the user's timeout issues.</description>
          <pros>Zero complexity; refit cache skip (T6) covers the most expensive individual computation units.</pros>
          <cons>Timeout during CI accumulation loop restarts the loop from iteration 0.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A3">Deferred from v1.7.0. Reassess after deployment if T3-T6 checkpointing proves insufficient. If needed, the preferred approach at that time would be informed by empirical interaction tensor sizes from real runs.</decision>
    </topic>
    <topic id="T8" title="CLI surface and resume UX">
      <summary>--force-restart flag on train, predict, infer subcommands. Single-line [RESUME]/[RESTART] log messages. No cascade delete behavior.</summary>
      <approaches>
        <approach id="A1" label="--force-restart flag with single-line messaging" feasibility="high" risk="low">
          <description>Add --force-restart to train, predict, infer argparse definitions. On pass, delete the stage's own checkpoint file and run from scratch. Single-line [RESUME] message when resuming (e.g., "[RESUME] Skipping folds 0-6, resuming at fold 7 (7/10 complete)"). Single-line [RESTART] on hash mismatch or --force-restart.</description>
          <pros>Minimal, informative output; consistent with existing [INFO]/[WARNING] prefix conventions.</pros>
          <cons>None identified.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">--force-restart flag (hyphenated). Added to train, predict, infer subcommands only (not plot or check-env). No cascade delete: --force-restart on predict does not delete train's checkpoint. Single-line [RESUME] and [RESTART] log prefixes. User correction applied: no multi-line verbose output, one line per resume/restart event.</decision>
    </topic>
    <topic id="T9" title="Crash safety and atomicity">
      <summary>Use the existing save_json_atomic pattern (write to .tmp, then os.rename) for all checkpoint file writes. Per-fold completion predicate (all five artifacts present) handles mid-fold crashes without special logic.</summary>
      <approaches>
        <approach id="A1" label="Write-tmp-then-rename (existing pattern)" feasibility="high" risk="low">
          <description>Reuse save_json_atomic from train.py for all checkpoint writes. POSIX os.rename is atomic within the same filesystem.</description>
          <pros>Already implemented; proven in the codebase; handles crash mid-write.</pros>
          <cons>None identified.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Use save_json_atomic (write-tmp-then-rename) for all checkpoint file writes. Per-fold completion predicate (all five artifacts present) handles mid-fold crashes: a fold with fewer than five artifacts is treated as incomplete and re-run. No additional crash-safety infrastructure needed.</decision>
    </topic>
  </topics>
  <action_items>
    <item priority="P0" target_mode="implement" description="Checkpoint infrastructure: config hash utility (stage-scoped, SHA-256, json.dumps canonical form), save/load checkpoint helpers using save_json_atomic, checkpoint common schema (pipeline_version, config_hash, stage, status, timestamps, progress). Target: utils.py." />
    <item priority="P1" target_mode="implement" description="train.py per-fold checkpointing: save _oof_fold_{k}.csv, _metrics_fold_{k}.json, _params_fold_{k}.json inside the fold loop; add resume gate at top of fold loop (check five-artifact completion predicate + config hash); reconstruct in-memory accumulators from per-fold artifacts for skipped folds; update _checkpoint_train.json after each fold completion." />
    <item priority="P1" target_mode="implement" description="predict.py phase-gate resume: add phase-gate dispatcher at top of main() checking five-phase terminal artifacts; update _checkpoint_predict.json after each phase completion; handle conditional phases (indiv_ci_nboot=0)." />
    <item priority="P1" target_mode="implement" description="indiv_reports.py bootstrap refit cache skip: add existence check for iter_{b:05d}/fold_{k}.cbm before dispatch; add fold_{k}_alpha.json sidecar writes in _fit_and_save_refit; reconstruct partial bootstrap_alphas array from sidecar files on resume." />
    <item priority="P2" target_mode="implement" description="infer.py phase-gate resume: mirror predict.py's phase-gate dispatcher; add data_path guard to checkpoint; handle shared P4 (bootstrap refit cache at run_dir level)." />
    <item priority="P1" target_mode="implement" description="CLI --force-restart flag: add to train, predict, infer subcommands in cli.py; wire through to each module's main() function; on pass, delete stage's checkpoint file before proceeding." />
    <item priority="P1" target_mode="implement" description="Resume UX: add [RESUME] and [RESTART] single-line log messages at checkpoint detection, hash verification, fold/phase skip, and hash mismatch events." />
    <item priority="P1" target_mode="test" description="Checkpoint/resume test suite: simulate partial completions (mid-fold, mid-phase, mid-bootstrap-refit), config hash mismatch detection, --force-restart behavior, crash safety (partial artifact sets), predecessor validation in predict/infer." />
    <item priority="P1" target_mode="document" description="Update README.md and INPUT_SPECIFICATION.md with checkpoint/resume documentation: --force-restart flag, checkpoint file locations, resume behavior, config hash scope per stage." />
  </action_items>
  <next_steps>Route to /implement for plan + build. The P0 item (checkpoint infrastructure in utils.py) must be built first as all other items depend on it. Followed by /test, /document, then /publish v1.7.0.</next_steps>
</brainstorm_report>
