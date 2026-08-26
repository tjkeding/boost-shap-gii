<implement_report>
  <meta project="boost-shap-gii" mode="implement" submodule="build" timestamp="2026-08-26T19:15:50Z" />
  <spec_ref>boost-shap-gii_implement_plan_20260826_182259.md</spec_ref>
  <changes_applied>
    <change id="C1" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/train.py" lines_changed="~24" />
      </files_modified>
      <notes>Removed the rtol=1e-6 cross-fold consistency hard halt and the single-scalar shap_scale_factor assignment. Replaced with an informational diagnostic printing per-fold alphas and their coefficient of variation. transform_config.json now stores fold_shap_scale_factors (a list, one value per fold) in place of the scalar shap_scale_factor field. Applied exactly as specified.</notes>
    </change>
    <change id="C2" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/shap_utils.py" lines_changed="~40" />
      </files_modified>
      <notes>_run_bootstrap_pipeline now accepts shap_scale_factors as an Optional[np.ndarray] and applies per-row broadcasting instead of scalar multiplication. _run_shap_for_slice constructs the per-row alpha vector from fold_shap_scale_factors: np.repeat against the pooled fold-ordered matrix in inference mode, and indexing by stashed fold assignments in predict mode. run_shap_pipeline stashes fold_assignments into ctx for this purpose. Applied exactly as specified.</notes>
    </change>
    <change id="C6" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/indiv_reports.py" lines_changed="~55" />
      </files_modified>
      <notes>_fit_and_save_refit now captures boot_meta from input_transform (previously discarded) and, when back_transform_shap and is_affine are both true, probes output_transform with zero/one vectors to compute the exact per-bootstrap-refit alpha, returning it (1.0 otherwise). orchestrate_bootstrap_cache threads the two new flags to every worker task, collects the returned alphas into a (B, K) array, and persists it as bootstrap_alphas.npy when back_transform_shap and is_affine are both true. bootstrap_alphas_saved recorded in bootstrap_metadata.json. generate_indiv_reports was not touched by this change (confirmed by direct file read). Applied exactly as specified.</notes>
    </change>
    <change id="C3" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/indiv_reports.py" lines_changed="~90" />
      </files_modified>
      <notes>generate_indiv_reports signature changed from shap_scale_factor (float) to fold_shap_scale_factors (Optional[List[float]]). bootstrap_alphas.npy is loaded and shape-validated against (B, K) whenever fold_shap_scale_factors is not None. All eight point-estimate and CI-accumulation sites (training-mode and inference-mode, main and interaction SHAP) apply per-fold scaling via fold_shap_scale_factors[k] for point estimates and per-bootstrap scaling via boot_alphas[b, k] for CI accumulation, per the scaling-authority separation locked in the tech spec. The prior post-hoc single-scalar scaling block was removed and replaced with a diagnostic print. Predictions are never scaled by alpha in any of the eight sites (verified: only output_transform back-transforms predictions). Applied exactly as specified.</notes>
    </change>
    <change id="C4" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/predict.py" lines_changed="~20" />
      </files_modified>
      <notes>Loads fold_shap_scale_factors from transform_config.json (raising ValueError on the legacy scalar format when back_transform_shap is active), passes it to shap_ctx and to generate_indiv_reports, and threads back_transform_shap/is_affine flags into the orchestrate_bootstrap_cache call. Applied exactly as specified.</notes>
    </change>
    <change id="C5" status="done" user_decision="n/a">
      <files_modified>
        <file path="src/boost_shap_gii/infer.py" lines_changed="~15" />
      </files_modified>
      <notes>Loads fold_shap_scale_factors from transform_config.json (same legacy-format guard as C4), passes it to shap_ctx and to generate_indiv_reports. No orchestrate_bootstrap_cache call was added, preserving the inference-mode data boundary (infer.py reads only pre-computed train_dir artifacts, including the already-persisted bootstrap_alphas.npy via generate_indiv_reports). Applied exactly as specified.</notes>
    </change>
  </changes_applied>
  <verification_notes>
Every change was read back directly from the modified files (not inferred from agent self-reports) and cross-checked line-by-line against the tech spec. A whole-codebase grep confirmed zero stray references to the legacy scalar shap_scale_factor parameter outside of the intentional single-fold diagnostic print label in train.py. All five modified files (train.py, shap_utils.py, indiv_reports.py, predict.py, infer.py) were byte-compiled successfully with py_compile, confirming no syntax errors.

No-op guarantees confirmed by direct inspection:
- Transforms absent: transform_config.json is not emitted (train.py:1219 gate unchanged); fold_shap_scale_factors stays None throughout predict.py/infer.py/shap_utils.py/indiv_reports.py.
- Transforms present, back_transform_shap=False: predict.py:256 and infer.py:248 gate fold_shap_scale_factors loading behind back_transform_shap; stays None.
- Transforms present, back_transform_shap=True, is_affine=False: unreachable per train.py's existing upstream validation (unchanged by this build).
  </verification_notes>
  <summary>
    <total_changes>6</total_changes>
    <completed>6</completed>
    <skipped>0</skipped>
    <blocked>0</blocked>
  </summary>
  <next_steps>Recommended: run /test to validate all changes. The 32 dry-run tests in test_dry_run_transformations.py that were blocked by the pre-fix hard halt are the primary regression target; existing tests referencing the legacy shap_scale_factor API (test_dry_run_transformations.py, test_indiv_reports_unit.py, test_transformations_wiring.py, test_transformations_api.py, test_shap_utils.py) will need re-expression per the test design discipline (obsolete-test disposition, not weakening).</next_steps>
</implement_report>
