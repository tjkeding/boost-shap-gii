<implement_report>
  <meta project="boost-shap-gii" mode="implement" submodule="build" timestamp="2026-09-24T17:00:00-04:00" />
  <spec_ref>boost-shap-gii_implement_plan_20260924_160000.md</spec_ref>
  <changes_applied>

    <change id="C1" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/utils.py" lines_changed="~65" />
      </files_modified>
      <notes>Applied exactly as specified: hashlib import, CONFIG_HASH_SCOPES constant, and five functions (compute_config_hash, save_csv_atomic, load_checkpoint, save_checkpoint, get_predecessor_mtime) in a contiguous block. No existing functions modified. Verified importable with a working hash computation via direct interpreter test.</notes>
    </change>

    <change id="C2" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/cli.py" lines_changed="~9" />
      </files_modified>
      <notes>Applied exactly as specified: --force-restart added to train, predict, infer subparsers only; each corresponding cmd_* function conditionally appends the flag to sys.argv. plot and check-env subcommands unchanged.</notes>
    </change>

    <change id="C3" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/train.py" lines_changed="~110" />
      </files_modified>
      <notes>Applied exactly as specified: new imports (datetime, __version__, four utils functions), --force-restart flag, config hash computed from raw config before fill_config_defaults, checkpoint load/validate/restart block, module-level _fold_complete helper, accumulator-reconstruction block before the fold loop, skip-and-continue gate as the first statement in the fold loop, three new per-fold artifact saves (_oof_fold_k.csv via save_csv_atomic, _metrics_fold_k.json, _params_fold_k.json) plus a checkpoint update after the shadow model save, and a finalization checkpoint write (status=complete) as the last statement before the success print. No existing computation logic altered.</notes>
    </change>

    <change id="C4" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/indiv_reports.py" lines_changed="~35" />
      </files_modified>
      <notes>Applied exactly as specified: _fit_and_save_refit now writes a fold_{k}_alpha.json sidecar alongside each .cbm file. orchestrate_bootstrap_cache scans for existing (b,k) pairs before dispatch, skips completed pairs in the task-generation loop, and reconstructs alpha values for skipped pairs from sidecar files (defaulting to 1.0 when a sidecar is absent) before saving bootstrap_alphas.npy. bootstrap_metadata.json write is unchanged (still written only after all futures resolve). No existing computation logic altered.</notes>
    </change>

    <change id="C5" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/predict.py" lines_changed="~140" />
      </files_modified>
      <notes>Applied exactly as specified, including the CR F4 reorder (predictions_oof.csv write relocated to immediately after the OOF loop, before the metrics section). All five phases gated on completed_phases membership with reload logic in the P1 else-branch (oof_preds reconstructed from predictions_oof.csv). P4/P5 remain nested inside the existing nboot_indiv&gt;0 conditional; the nboot_indiv==0 branch now records both as skipped_phases. Manually re-read and verified the full file post-build; no deviations from spec found.</notes>
    </change>

    <change id="C6" status="done" user_decision="modify">
      <files_modified>
        <file path="src/boost_shap_gii/infer.py" lines_changed="~230" />
      </files_modified>
      <notes>
        During implementation, two issues were discovered that the tech spec's "standard phase gates" description did not anticipate, both surfaced to the user for an explicit decision before proceeding:

        1. infer.py already imports `from datetime import datetime` (class-level import), unlike train.py/predict.py which use `import datetime` (module-level). The spec's `datetime.datetime.now(datetime.timezone.utc)` pattern would have collided with this existing import (either breaking the existing `datetime.now()` call at the metadata-write site, or breaking the new checkpoint code, depending on import order). Resolved without a user decision (mechanical fix, not a design choice): extended the existing import to `from datetime import datetime, timezone` and used `datetime.now(timezone.utc)` throughout the new checkpoint code instead.

        2. infer.py's ensemble prediction loop computes `per_model_metrics_rows` inline (consumed later by performance_per_model.csv), and the P1 terminal artifact (predictions_ensemble.csv) was written after the P2 terminal artifact (performance_final.csv) in the original code -- the same structural issue CR F4 fixed in predict.py, but complicated by this additional data dependency. Surfaced to the user with two options; the user selected: reorder predictions_ensemble.csv to immediately follow ensemble_preds computation (mirroring CR F4) and move performance_per_model.csv's construction into the same P1 block (since it is a byproduct of the prediction loop, not of metric computation), leaving P2 responsible only for performance_final.csv and the permutation test.

        Implementing this reorder surfaced a further mechanical consequence (not requiring a separate decision): several variables computed inside the original section-8 preamble (model_files/n_models validation, cat_features_indices, transform-module state: transform_module/tx_info/fold_shap_scale_factors/_fold_transform_meta, and id_col/ids) are consumed by P3 (SHAP) and P4/P5 (indiv_reports) regardless of whether P1 itself runs or is skipped on resume. These were hoisted to run unconditionally before the P1 gate, mirroring predict.py's existing unconditional-setup pattern. This is a direct, mechanical application of the plan's own stated principle (setup code that downstream phases need must run unconditionally) to code paths the plan's cursory "P1-P3 standard phase gates" language did not enumerate individually -- not a new design axis.

        Given the scale of restructuring required, this change was implemented directly (Read/Edit) rather than dispatched to a fresh execution agent, to reduce the risk of misinterpretation on a genuinely intricate multi-location relocation. The full file was re-read post-edit and traced end-to-end to confirm every variable used in P2/P3/P4/P5 is defined on both the P1-runs and P1-skipped code paths.

        One additional deviation caught and reverted during review: an initial edit had also changed the pre-existing, out-of-scope `inference_metadata.json` timestamp field from `datetime.now().isoformat()` to a timezone-aware form. This was outside the requested change set and was reverted to the original bare form.
      </notes>
    </change>

  </changes_applied>
  <summary>
    <total_changes>6</total_changes>
    <completed>6</completed>
    <skipped>0</skipped>
    <blocked>0</blocked>
  </summary>
  <environment_note>
    A `python3 -c "import boost_shap_gii.train"` sanity check failed with `ModuleNotFoundError: No module named 'catboost'` in both the ambient Python and the project's own `boost_shap_gii` conda environment. This is a pre-existing environment gap unrelated to this build (catboost is a core pipeline dependency per pyproject.toml/environment.yaml but is absent from the currently installed `boost_shap_gii` env) and was not remediated here, since package installation requires explicit per-package user approval under the Environment Pre-Flight procedure and is out of scope for a checkpoint/resume feature. All six modified files were verified via `python3 -m py_compile` (syntax-clean, individually and together) and via `boost_shap_gii.utils`/`boost_shap_gii.cli` runtime imports (catboost-independent, confirmed working). The three highest-risk changes (train.py, predict.py, infer.py) were additionally verified by a full manual re-read tracing every new/moved variable across all phase boundaries.
  </environment_note>
  <next_steps>Recommended: run /test to validate all changes, including partial-completion resume, config hash mismatch detection, --force-restart behavior, crash safety (partial artifact sets), predecessor mtime invalidation, fold_transform_metadata truncation, bootstrap cache skip with alpha sidecar reconstruction, and the infer.py P1/P2 boundary (predictions_ensemble.csv + performance_per_model.csv under P1, performance_final.csv + permutation test under P2). The catboost environment gap noted above should also be resolved (with explicit user approval) before any end-to-end pipeline run.</next_steps>
</implement_report>
