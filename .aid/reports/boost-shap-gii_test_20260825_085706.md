<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-25T12:57:06Z" />
  <pre_design_run>
    <total>875</total>
    <passed>875</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct />
    <failures />
  </pre_design_run>
  <failing_test_dispositions>
    <!-- No pre-design failures; nothing to disposition. The three changes from
         the preceding /implement build (boost-shap-gii_implement_build_20260825_083421.md,
         addressing CR findings F1-F3 from boost-shap-gii_cr_20260825_082540.md)
         were already applied before this /test cycle began. -->
  </failing_test_dispositions>
  <design_phase>
    <tests_created>13</tests_created>
    <tests_modified>0</tests_modified>
    <files_created>
      <file path="tests/test_indiv_reports_unit.py" test_count="5 (added to 60+ pre-existing)" coverage_target="orchestrate_bootstrap_cache's cluster_ids config resolution wiring (the F1 fix site) and source-parity guards against reintroducing the removed cluster_id_col key" />
      <file path="tests/test_transformations_wiring.py" test_count="4 (added to 25 pre-existing)" coverage_target="predict.py's fold_transform_metadata.json loading and per-fold indexing (the F2 fix site), mirroring the pre-existing infer.py wiring tests that predict.py previously lacked" />
      <file path="tests/test_predict.py" test_count="2 (added to existing TestPermutationCountFloor)" coverage_target="source-inspection of predict.py's and infer.py's actual n_perm floor lines (the F3 fix site), replacing reliance on a pre-existing tautological test that reimplemented the formula inline rather than reading either file's source" />
      <file path="tests/test_dry_run_transformations.py" test_count="2 (added to existing TestPredictArtifacts)" coverage_target="predict.py's actual OOF artifact (predictions_oof.csv), previously never checked by any test in this file (the pre-existing test_oof_predictions_in_original_scale reads full_oof_predictions.csv, a train.py artifact produced by a different code path); includes an independent oracle test that reloads each fold's model directly and recomputes the expected back-transformed value from the persisted fold_transform_metadata.json" />
    </files_created>
    <design_rationale>
      The pre-design run showed zero failures, so there was nothing to
      disposition per the Test Design Discipline routing rules. Design-phase
      analysis instead targeted the coverage gaps left by the preceding
      implement build's three changes (F1, F2, F3), grounded in tracing each
      change's actual call sites rather than trusting the existing test
      suite's pass count as evidence of correctness.

      For F2 (predict.py fold_transform_metadata.json) and F3 (n_perm floor),
      this traced cleanly: both fixes are correctly implemented and now have
      direct coverage. Notably, tracing F2's coverage surfaced that the
      pre-existing "OOF predictions in original scale" dry-run test was
      silently checking the wrong artifact (train.py's full_oof_predictions.csv
      instead of predict.py's own predictions_oof.csv), meaning predict.py's
      actual OOF output had never been asserted on at all; the new oracle
      test closes this pre-existing gap in addition to covering F2's specific
      fix. Tracing F3's coverage surfaced that the only existing "test" for
      the permutation-count floor was a standalone reimplementation of
      max(n_boot, 1000) asserting a property of its own inline computation,
      never reading either predict.py's or infer.py's actual source; this is
      precisely why a missing floor in infer.py went undetected until the CR.
      The new tests read the real source lines directly.

      For F1 (indiv_reports.py cluster_ids resolution), tracing the fix's
      actual production call sites surfaced that the implemented fix does not
      functionally work: orchestrate_bootstrap_cache and generate_indiv_reports
      receive X_train as the model's feature-only matrix (predict.py:219 X =
      X[trained_features]; infer.py:629 loads train.py:1017's same
      post-exclusion matrix from train_matrix.parquet), and group_column is
      unconditionally stripped from that feature set in train.py:760-769.
      The fix's guard (group_column in X_train.columns) therefore evaluates
      False unconditionally in production, so cluster_ids resolves to None
      regardless of cv_strategy -- identical behavior to before the fix. This
      was verified directly: a naive test using a synthetic X_train that
      artificially retains the group column (the intuitive but unrealistic
      way to unit-test this) would have passed against still-broken
      production code. Instead, the design phase added a test using a
      realistic feature-only X_train that documents the current (still
      broken) behavior precisely, so a future correct fix will make this test
      fail and force its deliberate re-expression, plus an isolation test
      proving the resolution logic itself is correct in isolation (confirming
      the fault is entirely in what callers pass, not in indiv_reports.py's
      internal logic) and two source-parity guards. This is routed as a P0
      action item below rather than silently treated as resolved.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>888</total>
    <passed>888</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct />
    <failures />
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>1</bugs_routed_to_implement>
    <recommendation>implement_fixes</recommendation>
  </summary>
  <action_items>
    <item priority="P0" target_mode="implement" finding_ref="F1" description="
      The CR's F1 finding (individual-level SHAP CI bootstrap not respecting
      group CV cluster structure) remains functionally unresolved despite the
      preceding implement build. orchestrate_bootstrap_cache and
      generate_indiv_reports (src/boost_shap_gii/indiv_reports.py) need an
      explicit cluster-identity parameter (e.g. cluster_ids: Optional[np.ndarray]
      = None) added to both function signatures, populated by the CALLER from
      df_raw[group_column] when cv_strategy == 'group' -- mirroring the
      shap_ctx['groups'] pattern already used correctly at predict.py:516-517
      for the population-level GII bootstrap. Neither indiv_reports.py
      function has access to df_raw today; both receive only X_train/X_target,
      the model's feature-only matrix, from which group_column is
      unconditionally excluded (train.py:760-769). The internal
      X_train.columns-based resolution at indiv_reports.py:703-707 (training
      mode) and :1033-1037 (inference mode) should be replaced with direct use
      of the new parameter. Callers to update: predict.py's
      orchestrate_bootstrap_cache call (~line 535) and generate_indiv_reports
      call (~line 554), both of which already have df_raw in scope; and
      infer.py's generate_indiv_reports call (~line 640), which has df_raw in
      scope via its own inference-mode data load. Once implemented,
      tests/test_indiv_reports_unit.py::TestOrchestrateBootstrapCacheClusterIdsResolution::test_cluster_ids_remain_none_under_group_cv_with_realistic_feature_only_x_train
      will fail by design and must be re-expressed (obsolete-test disposition)
      in the next /test cycle to assert the corrected behavior instead.
    " />
  </action_items>
</test_report>
