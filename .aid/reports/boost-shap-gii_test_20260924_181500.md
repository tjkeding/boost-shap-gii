<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-09-24T18:15:00-04:00" />
  <pre_design_run>
    <total>1005</total>
    <passed>1005</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures/>
  </pre_design_run>
  <failing_test_dispositions>
    <!-- No pre-design failures; the checkpoint/resume changes (C1-C6) had zero prior test coverage, so this design pass adds new coverage rather than dispositioning existing failures. -->
  </failing_test_dispositions>
  <design_phase>
    <tests_created>63</tests_created>
    <tests_modified>0</tests_modified>
    <files_created>
      <file path="tests/test_checkpoint_resume.py" test_count="63" coverage_target="Checkpoint/resume infrastructure (C1-C6): config hash scoping, atomic writes, checkpoint load/save, predecessor mtime guard (utils.py); --force-restart wiring (cli.py); per-fold checkpoint/resume, five-artifact completion predicate, partial-crash resume, fold_transform_metadata truncation (train.py); five-phase checkpoint/resume, predecessor mtime guard, phase-skip reload (predict.py); phase-gate resume, data_path guard, P4 pass-through, combined P1 artifacts (infer.py); bootstrap refit cache alpha sidecar and resume skip (indiv_reports.py)." />
    </files_created>
    <design_rationale>
      The build report (boost-shap-gii_implement_build_20260924_170000.md) explicitly flagged this feature set as untested and recommended /test. Checkpoint/resume is P0/P1-risk infrastructure intended to protect long-running HPC pipeline jobs (per project context) against crash-induced data loss or silent corruption, so coverage was designed to exercise every locked behavior in the tech spec (boost-shap-gii_implement_plan_20260924_160000.md) rather than a representative subset: all five config-hash scope relationships, the five-artifact fold-completion predicate, --force-restart wiring on all three stages, a genuine simulated-crash-and-resume cycle for train.py's fold loop (with bit-identical reconstruction verification against pre-crash data), the CR F6 fold_transform_metadata orphaned-entry truncation, predict.py's and infer.py's phase-gate skip/reload paths, the CR F7 predecessor-mtime invalidation guard on both predict and infer, the CR F5 P4 pass-through-not-a-phase behavior in infer.py, and the C4 bootstrap-refit cache skip with alpha-sidecar reconstruction (including the missing-sidecar default-to-1.0 path).

      One test, TestInferDataPathGuard, was written ahead of a routed fix rather than against current behavior, per explicit user instruction (see disposition below).
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>1068</total>
    <passed>1067</passed>
    <failed>1</failed>
    <errors>0</errors>
    <coverage_pct>null</coverage_pct>
    <failures>
      <failure test="TestInferDataPathGuard::test_relative_vs_absolute_path_to_same_file_does_not_invalidate" file="tests/test_checkpoint_resume.py" line="965">
        <error_type>AssertionError</error_type>
        <message>assert "[RESTART] Data path changed; checkpoint invalidated." not in captured.out</message>
        <traceback>tests/test_checkpoint_resume.py:965: AssertionError</traceback>
        <likely_cause>Expected and intentional. infer.py's data_path invalidation guard (infer.py:89,117) compares existing_checkpoint.get("data_path") != args.data as a raw string, with no os.path.abspath() normalization on either side. This test encodes the CORRECTED contract (normalized comparison) per explicit user instruction during the design phase, so a single /implement pass closes both the fix and this test's passing state without a second /test design cycle. See action item below.</likely_cause>
      </failure>
    </failures>
  </post_design_run>
  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>1</bugs_routed_to_implement>
    <recommendation>implement_fixes</recommendation>
  </summary>
  <action_items>
    <item priority="P2" target_mode="implement" finding_ref="TestInferDataPathGuard" description="Normalize the data_path checkpoint guard in infer.py to compare absolute paths rather than raw strings. Two one-line changes: (1) infer.py:117, store &quot;data_path&quot;: os.path.abspath(args.data) instead of args.data verbatim; (2) infer.py:89, compare existing_checkpoint.get(&quot;data_path&quot;) != os.path.abspath(args.data) instead of the raw args.data. This closes tests/test_checkpoint_resume.py::TestInferDataPathGuard, which currently documents the corrected (not-yet-implemented) contract and fails against current code. Discovered during /test design review of the checkpoint/resume build, not present in the original harmonized spec or CR report." />
  </action_items>
</test_report>
