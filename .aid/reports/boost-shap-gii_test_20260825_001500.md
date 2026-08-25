<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-25T00:15:00-04:00" />
  <pre_design_run>
    <total>863</total>
    <passed>863</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct />
    <failures />
  </pre_design_run>
  <failing_test_dispositions>
    <!-- No pre-design failures; nothing to disposition at baseline. -->
  </failing_test_dispositions>
  <design_phase>
    <tests_created>8</tests_created>
    <tests_modified>0</tests_modified>
    <files_created>
      <file path="tests/test_required_cols_nan_handling.py" test_count="8" coverage_target="required_cols NaN handling across train.py (row-drop), predict.py (mirrored row-drop), infer.py (warn-and-produce-NaN), and the belt-and-suspenders internal assertion" />
    </files_created>
    <design_rationale>
      Covers all five brainstorm action items for the required_cols NaN-handling
      feature: (1) train.py drop-and-proceed, tested via two fast config-validation
      guards (missing column -&gt; KeyError; all-NaN -&gt; ValueError) plus a shared
      end-to-end fixture asserting the exact row-count reduction; (2) predict.py
      drop-and-proceed, tested via the same fixture's OOF row count mirroring
      train's reduction; (3) infer.py warn-and-produce-NaN, tested via the fixture's
      full (undropped) prediction count and per-row NaN-position verification;
      (4) no-transform no-op, NOT duplicated here (test_dry_run_no_transform_group_cv.py
      already covers this path; its continued pass in this same suite run is the
      regression evidence); (5) the internal belt-and-suspenders assertion, tested
      via a monkeypatch that bypasses the row-drop specifically for the
      required_cols subset, to verify the assertion fires when its precondition is
      defeated.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>871</total>
    <passed>865</passed>
    <failed>1</failed>
    <errors>5</errors>
    <coverage_pct />
    <failures>
      <failure test="test_c4_internal_assertion_fires_when_drop_bypassed"
               file="tests/test_required_cols_nan_handling.py" line="255">
        <error_type>AssertionError (wrong exception content)</error_type>
        <message>Expected ValueError matching "[INTERNAL].*still have NaN"; actual: ValueError "Smoke test: input_transform produced 2 non-finite value(s) (NaN or Inf)" raised at train.py:829.</message>
        <traceback>train.py:829 raises a pre-existing smoke-test finite-value check on input_transform's y_sm_train/y_sm_val outputs, BEFORE execution reaches the newly-added belt-and-suspenders assertion later in the same block.</traceback>
        <likely_cause>
          Test construction flaw, not a product defect. The monkeypatch bypasses
          the required_cols row-drop, leaving NaN in baseline_col at two row
          indices chosen without regard to the smoke test's random 20-of-30 row
          sample (seed=42). Because the transform module's input_transform
          computes y_train/y_val as (outcome - baseline), any smoke-sampled row
          with a NaN baseline produces a NaN in the smoke test's own
          finite-value check (train.py:829), which fires first and masks the
          belt-and-suspenders assertion under test. This is a pre-existing guard
          I was not tracking when designing the test; it does not indicate the
          belt-and-suspenders assertion is unreachable in general, only that this
          specific test's injected NaN positions collide with the smoke sample.
          Fix (not applied this pass, per Single-Pass Discipline): precompute the
          smoke test's exact sample indices (RandomState(seed=42).choice(30, 20,
          replace=False), mirroring train.py's own derivation) and place the
          injected NaN rows outside that sampled set, so execution reaches the
          belt-and-suspenders assertion instead of tripping the earlier guard.
        </likely_cause>
      </failure>
      <failure test="TestTrainRowDrop::test_train_row_count_reduced_by_nan_count, TestTrainRowDrop::test_transform_config_records_required_cols, TestPredictRowDrop::test_predict_oof_row_count_mirrors_train_drop, TestInferWarnNoNaN::test_infer_predicts_all_rows_no_drop, TestInferWarnNoNaN::test_infer_nan_baseline_rows_produce_nan_predictions"
               file="tests/test_required_cols_nan_handling.py" line="0">
        <error_type>ERROR at module-scoped fixture setup (required_cols_dry_run)</error_type>
        <message>ValueError: Input contains NaN, raised from an sklearn scoring function call inside infer.py's per-model performance-metrics computation.</message>
        <traceback>infer.py:361 (raw = fn(y_true_sup, fold_sup)) calls a scoring function (e.g. neg_rmse, r2) on fold_sup, which contains NaN at the two infer-dataset rows with NaN baseline_col (indices 5 and 10). sklearn's metric functions reject NaN input and raise ValueError, aborting infer.main() before the fixture can yield.</traceback>
        <likely_cause>
          Genuine product defect, not a test flaw. infer.py:132 defines
          supervised_mask purely from outcome-column non-nullity
          (df_raw[outcome_cols].notna().all(axis=1)), independent of whether the
          corresponding PREDICTION is finite. C3 (the new NaN-baseline warning)
          correctly lets required_cols-NaN rows produce NaN predictions rather
          than crashing at the prediction step itself, but the pre-existing
          per-model metrics block at infer.py:323-365 was not updated to exclude
          NaN predictions before calling sklearn scoring functions. Any infer run
          that supplies both (a) required_cols NaN values and (b) ground-truth
          outcomes for evaluation will crash at the metrics stage, which defeats
          the intended "warn and proceed" contract locked in brainstorm topic T2
          ("warning gives user full information to handle externally" implies the
          pipeline completes; it does not currently).
        </likely_cause>
      </failure>
    </failures>
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>1</bugs_routed_to_implement>
    <recommendation>implement_fixes</recommendation>
  </summary>
  <action_items>
    <item priority="P0" target_mode="implement" description="infer.py's per-model performance-metrics block (lines 323-365, specifically the sklearn scoring calls at line 361 and the multi_regression/multiclass branches at lines 335 and 345) crashes with 'Input contains NaN' whenever a required_cols-NaN row also has a ground-truth outcome, because supervised_mask (line 132) is defined purely from outcome non-nullity and does not exclude rows whose back-transformed prediction is NaN. Fix should intersect supervised_mask (or an equivalent per-metric mask) with an isfinite check on the relevant prediction column(s) before calling sklearn scoring functions, so metrics are computed only over rows with both a ground-truth outcome and a finite prediction. Two candidate approaches exist: (a) a single additional finite-prediction mask applied uniformly before all three metric branches, or (b) per-branch finite masks scoped to the specific column being scored (relevant for multi_regression, where a row could be NaN in one outcome column's prediction but not another's). Recommend a brainstorm or implement-plan pass to choose between these before building." />
    <item priority="P1" target_mode="test" description="Repair test_c4_internal_assertion_fires_when_drop_bypassed in tests/test_required_cols_nan_handling.py: the injected NaN row indices (currently [0, 1]) must be chosen outside the smoke test's precomputed random sample (RandomState(seed=42).choice(30, 20, replace=False)) so the test reaches the belt-and-suspenders assertion instead of tripping the pre-existing smoke-test finite-value check at train.py:829. Requires a second /test design pass; not performed in this invocation per Single-Pass Discipline." />
  </action_items>
</test_report>
