<?xml version="1.0" encoding="UTF-8"?>
<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-03-16T03:00:00Z" />
  <pre_design_run>
    <total>283</total>
    <passed>279</passed>
    <failed>4</failed>
    <errors>0</errors>
    <coverage_pct>N/A</coverage_pct>
    <failures>
      <failure test="test_check_env_failure_missing_deps" file="tests/test_hardening.py" line="17">
        <error_type>AssertionError</error_type>
        <message>check_env.py has a SyntaxError (backslash in f-string expression, Python 3.12+ incompatible) so it exits with returncode=1 via SyntaxError, not via the expected "[ERROR] Missing Python packages" message.</message>
        <traceback>check_env.py line 59: f-string expression part cannot include a backslash</traceback>
      </failure>
      <failure test="test_multiclass_probability_extraction" file="tests/test_multiclass_pipeline.py" line="27">
        <error_type>AssertionError</error_type>
        <message>CatBoost SHAP interaction shape assertion is wrong. Expected (10, 2, 2, 3) but got (10, 3, 2, 2). The installed CatBoost version returns (N, C, P+1, P+1) not (N, P+1, P+1, C).</message>
        <traceback>assert (10, 3, 2, 2) == (10, 2, 2, 3)</traceback>
      </failure>
      <failure test="test_2d_spline_too_few_knots_returns_zero" file="tests/test_shap_utils.py" line="564">
        <error_type>AssertionError</error_type>
        <message>Stale test: C6 changed calculate_v_spline_2d to route to stacked-spline/group-means instead of returning 0.0. Test expected 0.0, got 0.452.</message>
        <traceback>assert 0.4520814146474982 == 0.0 +/- 1.0e-12</traceback>
      </failure>
      <failure test="test_one_hot_max_size_scales_with_p + test_search_space_scales_with_n_and_p" file="tests/test_train.py" line="513,725">
        <error_type>AssertionError</error_type>
        <message>Stale tests: C10 changed one_hot_max_size to fixed [2, 25] range. Tests assumed p-dependent scaling.</message>
        <traceback>assert 25 == 5; assert 25 &lt; 25</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <design_phase>
    <tests_created>1</tests_created>
    <tests_modified>2</tests_modified>
    <files_created>
      <file path="tests/test_implementation_changes.py" test_count="54" coverage_target="All 15 implementation changes (C1-C15) from brainstorm/implement cycle" />
    </files_created>
    <files_modified>
      <file path="tests/test_train.py" changes="Fixed 2 stale tests (test_one_hot_max_size_scales_with_p -> test_one_hot_max_size_is_fixed; test_search_space_scales_with_n_and_p updated for fixed one_hot_max_size)" />
      <file path="tests/test_shap_utils.py" changes="Fixed 1 stale test (test_2d_spline_too_few_knots_returns_zero -> test_2d_spline_too_few_knots_routes_to_fallback)" />
    </files_modified>
    <design_rationale>
      The 15 implementation changes from the brainstorm/implement cycle introduced behavioral changes
      across utils.py, shap_utils.py, train.py, predict.py, infer.py, check_env.py, and plot.R.
      Three existing tests were stale (asserted pre-implementation behavior). New tests were designed
      to validate each change individually (C1-C15) plus cross-change integration tests. Test classes
      cover: syntax correctness (C1), function signature changes (C2), ordinal validation tiers (C3),
      shadow model early stopping (C4), model-count assertion (C5), stacked-spline routing (C6),
      NaN handling (C7), p-value correction (C8), search space bounds (C9, C10), dependency checks (C11),
      bootstrap tracking (C12), plot.R changes (C13, C15), and permutation test loop (C14).
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>337</total>
    <passed>335</passed>
    <failed>2</failed>
    <errors>0</errors>
    <coverage_pct>N/A</coverage_pct>
    <failures>
      <failure test="test_check_env_failure_missing_deps" file="tests/test_hardening.py" line="17">
        <error_type>SyntaxError (pre-existing)</error_type>
        <message>check_env.py line 59 has f-string backslash syntax incompatible with Python 3.12+. The script fails with SyntaxError before reaching the dependency check logic.</message>
        <traceback>f-string expression part cannot include a backslash</traceback>
        <likely_cause>check_env.py was written for Python 3.10 or earlier where backslashes in f-string expressions were permitted. Python 3.12 (PEP 701) changed this; the environment uses Python 3.11 which also rejects this syntax. Fix requires modifying check_env.py (outside tests/ scope).</likely_cause>
      </failure>
      <failure test="test_multiclass_probability_extraction" file="tests/test_multiclass_pipeline.py" line="27">
        <error_type>AssertionError (pre-existing)</error_type>
        <message>CatBoost ShapInteractionValues shape assertion assumes (N, P+1, P+1, C) but installed CatBoost returns (N, C, P+1, P+1).</message>
        <traceback>assert (10, 3, 2, 2) == (10, 2, 2, 3)</traceback>
        <likely_cause>CatBoost version in boost_shap_gii environment uses a different axis ordering for multiclass SHAP interaction values than the test assumed. The test assertion should be updated to (10, 3, 2, 2), and shap_utils.py's multiclass SHAP handling should be verified against this shape. Fix requires modifying test_multiclass_pipeline.py (can be done in tests/) but the root cause may also affect shap_utils.py.</likely_cause>
      </failure>
    </failures>
  </post_design_run>
  <summary>
    <all_passing>false</all_passing>
    <recommendation>implement_fixes</recommendation>
  </summary>
  <action_items>
    <item priority="P0" target_mode="implement" description="Fix check_env.py line 59: replace backslash-in-f-string with a pre-computed variable (e.g., quoted_missing = ', '.join(f'&quot;{m}&quot;' for m in missing); print(f'[HINT] ... install.packages(c({quoted_missing}))')). This is a SyntaxError that prevents check_env.py from running on Python 3.11+." />
    <item priority="P1" target_mode="implement" description="Fix test_multiclass_pipeline.py: CatBoost ShapInteractionValues shape is (N, C, P+1, P+1) not (N, P+1, P+1, C). Update assertion and verify shap_utils.py multiclass SHAP indexing matches the actual CatBoost output shape." />
  </action_items>
</test_report>
