<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-08-24T09:24:37-04:00" />
  <pre_design_run>
    <total>733</total>
    <passed>732</passed>
    <failed>1</failed>
    <errors>0</errors>
    <coverage_pct></coverage_pct>
    <failures>
      <failure test="TestAggregateEffects::test_singleton_aggregate_is_member_sum" file="tests/test_aggregate_shap.py" line="187">
        <error_type>AssertionError</error_type>
        <message>assert 'aggregate' == 'continuous'</message>
        <traceback>- continuous
+ aggregate
tests/test_aggregate_shap.py:187: AssertionError: assert 'aggregate' == 'continuous'</traceback>
      </failure>
    </failures>
  </pre_design_run>

  <failing_test_dispositions>
    <disposition test="TestAggregateEffects::test_singleton_aggregate_is_member_sum" file="tests/test_aggregate_shap.py" classification="obsolete-test">
      <intended_contract>Aggregate SHAP features must be assigned feature_type "aggregate" (not "continuous"), so the Stratified Max Boruta Exceedance Test calibrates aggregate significance thresholds against aggregate shadow noise rather than pooling them with individual continuous-feature shadow noise. This is a locked, intentional design change (brainstorm T1; implement plan C1).</intended_contract>
      <current_test_claim>assert ft["A"] == "continuous"</current_test_claim>
      <evidence>src/boost_shap_gii/shap_utils.py:604 sets feature_types[group_name] = "aggregate" (changed from "continuous" as the explicit, stated purpose of the change). boost-shap-gii_implement_plan_20260824_125052.md change C1 spec. boost-shap-gii_brainstorm_20260824_010000.md topic T1 (decided, chosen A3).</evidence>
      <action>re-express: assertion updated to assert ft["A"] == "aggregate" — same specificity (exact string equality), asserting the new correct contract. Not weakened.</action>
    </disposition>
  </failing_test_dispositions>

  <design_phase>
    <tests_created>2</tests_created>
    <tests_modified>2</tests_modified>
    <files_created>
      <file path="tests/test_transformations_api.py" test_count="21" coverage_target="fill_config_defaults transformations block; load_transform_module (None/valid/relative-path/FileNotFoundError/AttributeError x2); validate_transform_config (pass/raise/stage-label); affinity-check algorithm (affine/identity/nonaffine detection); first-fold alpha computation (known slope, identity, non-constant-slope rejection)" />
      <file path="tests/test_transformations_wiring.py" test_count="20" coverage_target="Source-inspection verification that the transformations API is correctly wired into train.py (imports, module loading, six-check smoke test, ordering relative to feature-type coercion, scaler guard, in-fold transform ordering, first-fold alpha gating, transform_config.json field completeness and post-loop placement), predict.py (import, artifact detection, back-transform-before-counts-increment ordering, scaler guard, shap_scale_factor passthrough), and infer.py (import, artifact detection from train_dir, hoisted fold_assignments read, back-transform-before-accumulation ordering, both scaler guards, shap_scale_factor passthrough)" />
    </files_created>
    <files_modified>
      <file path="tests/test_aggregate_shap.py" test_count="1 re-expressed" coverage_target="obsolete-test re-expression: aggregate feature_type assertion updated from 'continuous' to 'aggregate'" />
      <file path="tests/test_shap_utils.py" test_count="9 added" coverage_target="_get_effect_stratum singleton_aggregate/shadow-aggregate/interaction-aggregate cases (3); small-stratum-warning wiring checks (3); shap_scale_factor exceedance-test invariance under positive scaling, including the significance-decision invariance and the alpha=0 boundary case documenting why the proof requires alpha&gt;0 (3)" />
    </files_modified>
    <design_rationale>
Coverage was scoped to the 7 implemented changes from boost-shap-gii_implement_build_20260824_131044.md. Two coverage strategies were used depending on testability:

(1) Standalone functions and algorithms (fill_config_defaults, load_transform_module, validate_transform_config, the affinity-check three-probe linearity test, the first-fold alpha computation, and the shap_scale_factor exceedance-invariance property) were tested via genuine execution with hand-written affine/non-affine transform scripts and property-based numpy assertions, following the exact algorithms specified in the implement plan.

(2) Logic embedded inside train.py's, predict.py's, and infer.py's monolithic main() functions (smoke test, in-fold/per-model transform application, scaler guards, artifact I/O ordering) was verified via source inspection rather than full end-to-end CatBoost pipeline execution. This matches the established convention already present in this suite for other fold-loop-embedded changes (test_implementation_changes.py::TestC4ShadowModelEarlyStopping, TestC5ModelCountAssertion) — the existing 788-test suite contains no full train.main() -&gt; predict.main() end-to-end integration test, and none of the three pipeline modules expose the fold-loop internals as independently callable functions. This scope tradeoff was presented to the user via the Pre-Write Approval Gate before any files were written; the user selected the source-inspection-plus-unit-test scope over adding a new full end-to-end integration test.

The aggregate-stratum small-count warning (T1) was covered by source-inspection wiring checks rather than execution, since exercising it requires a full _run_bootstrap_pipeline fixture (a heavy function with no existing full-execution test in this suite; TestNanSafeFdr only imports it to confirm the module loads). The exceedance-invariance property that motivates back_transform_shap's correctness (T4's core statistical claim) was instead verified directly and rigorously as a pure numpy property test, which is a stronger and faster check of the actual mathematical guarantee than a full bootstrap fixture would have provided.
    </design_rationale>
  </design_phase>

  <post_design_run>
    <total>788</total>
    <passed>788</passed>
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
</test_report>
