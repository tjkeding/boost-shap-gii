<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-13T23:15:00Z" />
  <pre_design_run>
    <total>658</total>
    <passed>649</passed>
    <failed>9</failed>
    <errors>0</errors>
    <coverage_pct></coverage_pct>
    <failures>
      <failure test="TestC5ModelCountAssertion::test_predict_py_has_assertion" file="tests/test_implementation_changes.py" line="220">
        <error_type>AssertionError</error_type>
        <message>assert "expected_folds" in content</message>
        <traceback>tests/test_implementation_changes.py:220: in test_predict_py_has_assertion
    assert "expected_folds" in content
E   assert 'expected_folds' in '...'</traceback>
      </failure>
      <failure test="TestReconstructFoldAssignments::test_determinism_same_seed_same_folds" file="tests/test_indiv_reports_unit.py" line="351">
        <error_type>KeyError</error_type>
        <message>KeyError: 'paths'</message>
        <traceback>src/boost_shap_gii/indiv_reports.py:225: in _reconstruct_fold_assignments
    run_dir = train_dir if train_dir is not None else config["paths"]["output_dir"]
E   KeyError: 'paths'</traceback>
      </failure>
      <failure test="TestReconstructFoldAssignments::test_different_seed_different_folds" file="tests/test_indiv_reports_unit.py" line="357">
        <error_type>KeyError</error_type>
        <message>KeyError: 'paths'</message>
        <traceback>src/boost_shap_gii/indiv_reports.py:225: in _reconstruct_fold_assignments
    run_dir = train_dir if train_dir is not None else config["paths"]["output_dir"]
E   KeyError: 'paths'</traceback>
      </failure>
      <failure test="TestReconstructFoldAssignments::test_partition_covers_every_individual" file="tests/test_indiv_reports_unit.py" line="364">
        <error_type>KeyError</error_type>
        <message>KeyError: 'paths'</message>
        <traceback>src/boost_shap_gii/indiv_reports.py:225: in _reconstruct_fold_assignments
    run_dir = train_dir if train_dir is not None else config["paths"]["output_dir"]
E   KeyError: 'paths'</traceback>
      </failure>
      <failure test="TestReconstructFoldAssignments::test_partition_balanced_to_k_folds" file="tests/test_indiv_reports_unit.py" line="372">
        <error_type>KeyError</error_type>
        <message>KeyError: 'paths'</message>
        <traceback>src/boost_shap_gii/indiv_reports.py:225: in _reconstruct_fold_assignments
    run_dir = train_dir if train_dir is not None else config["paths"]["output_dir"]
E   KeyError: 'paths'</traceback>
      </failure>
      <failure test="TestReconstructFoldAssignments::test_y_as_dataframe_first_column" file="tests/test_indiv_reports_unit.py" line="383">
        <error_type>KeyError</error_type>
        <message>KeyError: 'paths'</message>
        <traceback>src/boost_shap_gii/indiv_reports.py:225: in _reconstruct_fold_assignments
    run_dir = train_dir if train_dir is not None else config["paths"]["output_dir"]
E   KeyError: 'paths'</traceback>
      </failure>
      <failure test="TestGetCvSplitter::test_classification_uses_stratified" file="tests/test_train.py" line="285">
        <error_type>AssertionError</error_type>
        <message>assert 'KFold' == 'StratifiedKFold'</message>
        <traceback>tests/test_train.py:285: in test_classification_uses_stratified
    assert type(splitter).__name__ == "StratifiedKFold"
E   AssertionError: assert 'KFold' == 'StratifiedKFold'</traceback>
      </failure>
      <failure test="TestGetCvSplitter::test_explicit_task_type" file="tests/test_train.py" line="323">
        <error_type>AssertionError</error_type>
        <message>assert 'KFold' == 'StratifiedKFold'</message>
        <traceback>tests/test_train.py:323: in test_explicit_task_type
    assert type(splitter).__name__ == "StratifiedKFold"
E   AssertionError: assert 'KFold' == 'StratifiedKFold'</traceback>
      </failure>
      <failure test="TestFillConfigDefaults::test_user_values_never_overwritten" file="tests/test_train.py" line="645">
        <error_type>Failed</error_type>
        <message>Default applied to user-provided field: modeling.cv_strategy</message>
        <traceback>tests/test_train.py:645: in test_user_values_never_overwritten
    pytest.fail(f"Default applied to user-provided field: {f_path}")
E   Failed: Default applied to user-provided field: modeling.cv_strategy</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <failing_test_dispositions>
    <disposition test="TestC5ModelCountAssertion::test_predict_py_has_assertion" file="tests/test_implementation_changes.py" classification="obsolete-test">
      <intended_contract>predict.py must raise if the model file count does not match the CV fold count, with an actionable message directing the user to re-run train.py.</intended_contract>
      <current_test_claim>Literal source-string check for the variable name "expected_folds" and the comparison "len(model_files) != expected_folds".</current_test_claim>
      <evidence>src/boost_shap_gii/predict.py:239-245 (post-implement C1 change): the variable was renamed from expected_folds (computed via splitter.get_n_splits()) to n_folds (computed from the authoritative fold_assignments.json artifact). The behavioral contract (raise on mismatch, actionable message) is unchanged and independently verified by the sibling test test_assertion_message_is_actionable, which still passes.</evidence>
      <action>re-express: update the two literal-string assertions to check for "n_folds" and "len(model_files) != n_folds" instead of the retired variable name.</action>
    </disposition>
    <disposition test="TestReconstructFoldAssignments (5 tests)" file="tests/test_indiv_reports_unit.py" classification="obsolete-test">
      <intended_contract>Prior contract: recompute per-individual fold assignments deterministically from config + data via get_cv_splitter. New contract (implement C3, tracing to brainstorm decision T5 "Persist fold assignments"): load the authoritative fold_assignments.json artifact written by train.py, eliminating dependence on data identity, sklearn determinism, and stratification replication.</intended_contract>
      <current_test_claim>Fixtures call _reconstruct_fold_assignments(config, X, y) with an in-memory config lacking config["paths"]["output_dir"] and no on-disk artifact, expecting the function to compute a KFold split internally.</current_test_claim>
      <evidence>src/boost_shap_gii/indiv_reports.py:219-232 (post-implement C3 change): the function body no longer calls get_cv_splitter; it loads json.load(fold_assignments_path) from train_dir or config["paths"]["output_dir"]. Brainstorm report boost-shap-gii_brainstorm_20260813_171500.md, topic T5, decision: "Persist fold_assignments.json... Robust; immune to data/version drift; eliminates alignment bug class."</evidence>
      <action>re-express: fixture writes a real fold_assignments.json (generated via get_cv_splitter to mirror train.py's actual output) to a pytest tmp_path before each test; all original assertions preserved verbatim except test_different_seed_different_folds, whose premise (a seed parameter that no longer exists on this function) no longer applies -- re-expressed as test_different_artifact_different_folds, preserving the identical "not equal" postcondition via two distinct on-disk artifacts instead of two seeds. Two new tests added (missing-artifact raises FileNotFoundError; length-mismatch raises AssertionError) to cover the new failure modes this architecture introduces.</action>
    </disposition>
    <disposition test="TestGetCvSplitter::test_classification_uses_stratified, test_explicit_task_type" file="tests/test_train.py" classification="obsolete-test">
      <intended_contract>Prior contract: get_cv_splitter auto-stratifies classification tasks when y has fewer than 20 unique values. New contract (brainstorm decision T1, approach A1): cv_strategy is a literal splitter selector with zero task-dependent or cardinality-dependent branching; "uniform" always returns KFold regardless of task type.</intended_contract>
      <current_test_claim>Classification y (or explicit classification task_type) with cv_strategy omitted (defaulting to "uniform") is expected to yield StratifiedKFold.</current_test_claim>
      <evidence>src/boost_shap_gii/utils.py:242-252: the "uniform" branch has no task-type or y.nunique() conditional; it unconditionally returns KFold (or RepeatedKFold). Brainstorm report, topic T1, decision text: "User directive: 'We don't want this option to do anything automatically.' Each value is a literal splitter selector regardless of task type." Documented as an explicit, approved backward-compatibility break in the same report.</evidence>
      <action>re-express: each original test split into two -- one confirming the new default (KFold even for classification), one confirming that explicit cv_strategy="stratified" still yields StratifiedKFold. This is a strict superset of the original assertion power (adds coverage for the explicit-opt-in path that was previously untested). A third, previously-passing-but-misleadingly-labeled test (test_classification_many_classes_uses_kfold, whose docstring claimed a nunique()>=20 fallback that no longer exists in the code) was also corrected and paired with a new companion test confirming "stratified" ignores cardinality entirely.</action>
    </disposition>
    <disposition test="TestFillConfigDefaults::test_user_values_never_overwritten" file="tests/test_train.py" classification="obsolete-test">
      <intended_contract>fill_config_defaults must never overwrite a user-supplied config value; it may only fill genuinely absent keys. The test's own allowlist mechanism (established in prior sessions for modeling.task_type and shap.compute_global_on_inference) exists specifically to accommodate schema growth: as new optional keys are added, the shared sample_config fixture legitimately predates them.</intended_contract>
      <current_test_claim>An allowlist of two entries (modeling.task_type, shap.compute_global_on_inference); any other filled path fails the test.</current_test_claim>
      <evidence>src/boost_shap_gii/utils.py:414-424: _set() uses _setdefault_nested (fill-if-absent only; verified not a product bug -- independently confirmed the helper never overwrites a present key). The sample_config fixture (tests/test_train.py:43-98) does not set modeling.cv_strategy or modeling.tuning.n_inner_repeats, both newly introduced this session (implement C1/C4).</evidence>
      <action>re-express: extend allowed_fills with modeling.cv_strategy and modeling.tuning.n_inner_repeats, following the exact precedent of the two existing entries; docstring updated to document the addition using the same convention.</action>
    </disposition>
  </failing_test_dispositions>
  <design_phase>
    <tests_created>47</tests_created>
    <tests_modified>9</tests_modified>
    <files_created>
      <file path="tests/test_cv_strategy.py" test_count="42" coverage_target="get_cv_splitter across all three cv_strategy values (uniform/stratified/group) including n_repeats and combinations thereof; the three wrapper classes (_StratifiedRegressionKFold, _GroupKFoldWrapper, _RepeatedGroupKFold) exercised directly with group-integrity, determinism, and reshuffling-across-repeats guarantees; validate_cv_config's enum/group_column/n_inner_repeats validation branches; stratify_labels_for_regression including a monkeypatched pd.qcut-failure fallback to pd.cut; fill_config_defaults' two new default entries" />
    </files_created>
    <design_rationale>
      All 9 pre-design failures were dispositioned obsolete-test: every one traces to a deliberate, plan-locked design decision from this session's implement cycle (fold_assignments.json persistence per brainstorm T5; literal cv_strategy selector semantics per brainstorm T1; schema growth in fill_config_defaults). Zero product bugs were found. Assertions were re-expressed to match the new contracts without weakening any postcondition; several re-expressions strictly strengthened coverage (e.g., the classification splitter tests now cover both the new default AND the explicit-opt-in stratified path, where only the auto-stratify path was tested before).

      Beyond fixing breakage, a coverage-gap survey found zero existing tests referencing any of the CV strategy feature's new symbols (cv_strategy, the three wrapper classes, validate_cv_config, n_inner_repeats, stratify_labels_for_regression) outside the incidentally-affected fixtures above. tests/test_cv_strategy.py closes this gap for the directly-importable, unit-testable surface in utils.py -- the layer carrying the session's genuine novel algorithmic risk (custom RNG-based group-repeat shuffling in _RepeatedGroupKFold, quantile-binning fallback in stratify_labels_for_regression). Several assertions were empirically verified against the actual implementation before being written (exact stratified class balance, quantile-fold spread vs. unstratified spread, cross-repeat reshuffling, determinism) rather than assumed, per the project's evidence-based verification standard.

      Explicitly not attempted: integration-level tests for the fold_assignments.json round-tripping in predict.py (C1) and shap_utils.py (C2), or the group_column exclusion and WARNING emissions inline in train.py's main(). All three live in monolithic, non-decomposed functions with zero existing integration-test precedent anywhere in this 705-test suite; building first-of-its-kind fixture infrastructure for them was surfaced to the user as an explicit scope question before the design phase began, and the user approved proceeding with the unit-level scope only.

      One self-inflicted defect was found and corrected during the post-design run: three new tests in test_cv_strategy.py's classification-stratified cluster shared the base_config fixture, which hardcodes modeling.task_type="regression"; the tests changed tuning.scoring without also overriding task_type, and detect_task() gives explicit task_type precedence over scoring-based inference (the identical precedence rule already verified for the pre-existing test_explicit_task_type case above) -- so all three silently exercised the regression-stratification wrapper instead of classification's StratifiedKFold. This was a test-authoring bug in code written this session, not a disposition-worthy failure from the pre-design baseline; it was corrected directly and reconfirmed via a second full-suite run.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>705</total>
    <passed>705</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct></coverage_pct>
    <failures></failures>
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <recommendation>proceed_to_document</recommendation>
  </summary>
  <action_items>
    <item priority="P2" target_mode="brainstorm" description="Integration-level test coverage for predict.py's and shap_utils.py's fold_assignments.json round-tripping, and train.py's group_column exclusion / WARNING emissions, was explicitly scoped out of this cycle (no existing integration-test precedent for these monolithic functions in the suite). Consider whether a lightweight integration harness is warranted for these paths in a future test-design session, or whether the existing unit-level coverage plus the manual verification already performed during implement build (parse checks, import checks, grep-based cross-file consistency checks) is sufficient given the low complexity of the actual round-trip logic (a single os.path.join + json.load + np.where per site)." />
  </action_items>
</test_report>
