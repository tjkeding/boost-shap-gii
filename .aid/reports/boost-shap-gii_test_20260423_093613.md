<test_report>
 <meta project="boost-shap-gii" mode="test" timestamp="2026-04-23T13:36:13Z" />
 <pre_design_run>
 <total>442</total>
 <passed>439</passed>
 <failed>3</failed>
 <errors>0</errors>
 <coverage_pct>null</coverage_pct>
 <failures>
 <failure test="test_check_env_success_in_correct_env" file="tests/test_hardening.py" line="28">
 <error_type>AssertionError</error_type>
 <message>check_env.py failed (rc=1): Missing R packages: nanoparquet</message>
 <traceback>check-env reports missing R package 'nanoparquet'; install via R: install.packages("nanoparquet")</traceback>
 </failure>
 <failure test="TestCLI.test_check_env_runs_successfully" file="tests/test_package_structure.py" line="303">
 <error_type>AssertionError</error_type>
 <message>check-env CLI subcommand returns rc=1 due to missing R package nanoparquet</message>
 <traceback>Same root cause as test_check_env_success_in_correct_env.</traceback>
 </failure>
 <failure test="test_plot_r_discovers_shap_dirs" file="tests/test_plot_smoke.py" line="44">
 <error_type>AssertionError</error_type>
 <message>plot.R smoke test halts when nanoparquet is unavailable; stdout empty.</message>
 <traceback>R script cannot execute without nanoparquet available; same dependency chain as the other two.</traceback>
 </failure>
 </failures>
 </pre_design_run>
 <design_phase>
 <tests_created>19</tests_created>
 <tests_modified>0</tests_modified>
 <files_created>
 <file path="tests/test_categorical_fillna_bugfix.py" test_count="19" coverage_target="Pandas 3.0 Categorical fillna bugfix: idiom-behavior coverage of the patched nominal-encoding path under adversarial input conditions, plus source-contract coverage verifying the patched idiom is present in train.py, predict.py, and infer.py." />
 </files_created>
 <design_rationale>
Coverage gap identified during analysis: existing test_dtype_bugfix.py (35KB, Session 5) covers _to_numeric_matrix under Categorical inputs with "__NA__" but does not exercise the main/encoding loop in train.py/predict.py/infer.py. The pre-existing test_infer.py::test_nominal_cast_to_category is a mirror test that reimplements the idiom in the WRONG order (astype(str) before fillna, with sentinel "MISSING"), so it never exercised the failing case.

The new file splits coverage into two layers:

(1) Idiom behavior (13 tests across 4 classes):
 - TestCategoricalWithNaN: the primary regression case — CategoricalDtype with NaN, categories exclude "__NA__". Four assertions: no TypeError, "__NA__" is added to cat.categories, NaN positions map to "__NA__", no NaN remains in output.
 - TestCategoricalWithPreExistingNALevel: edge case where "__NA__" is already an explicit string category; must be preserved, and NaN values must coexist correctly.
 - TestCategoricalDegenerate: all-NaN and no-NaN inputs.
 - TestNonCategoricalInputs: non-regression coverage for object-dtype, pyarrow-string, numeric-coded Categorical, and ordered Categorical inputs.

(2) Source contract (6 parametrized tests in TestSourceContract):
 - test_patched_idiom_present[train|predict|infer]: regex-matches the patched line '.astype(object).fillna("__NA__").astype(str).astype("category")' in each module's source via inspect.getsource. Fails loudly if the patch is reverted.
 - test_no_unpatched_idiom_remains[train|predict|infer]: line-by-line check that no source line contains the old unpatched idiom without a preceding.astype(object) cast. Catches partial reverts that leave one site unpatched.

Deterministic (no randomness used in constructed Series). Hermetic (no filesystem, no subprocess, no CatBoost). Fast (all 19 tests run in subseconds). All assertions target observable output invariants (result dtype, category index membership, element-wise values) rather than internal pandas structures.
 </design_rationale>
 </design_phase>
 <post_design_run>
 <total>461</total>
 <passed>458</passed>
 <failed>3</failed>
 <errors>0</errors>
 <coverage_pct>null</coverage_pct>
 <failures>
 <failure test="test_check_env_success_in_correct_env" file="tests/test_hardening.py" line="28">
 <error_type>AssertionError</error_type>
 <message>check_env.py failed (rc=1): Missing R packages: nanoparquet</message>
 <traceback>Unchanged from baseline; environment state, not code defect.</traceback>
 <likely_cause>The nanoparquet R package is not installed in the local R library. This test passes in any environment where install.packages("nanoparquet") has been run. Orthogonal to the patch; pre-dates this session.</likely_cause>
 </failure>
 <failure test="TestCLI.test_check_env_runs_successfully" file="tests/test_package_structure.py" line="303">
 <error_type>AssertionError</error_type>
 <message>check-env CLI subcommand returns rc=1 due to missing R package nanoparquet</message>
 <traceback>Same root cause as test_check_env_success_in_correct_env.</traceback>
 <likely_cause>Same as above.</likely_cause>
 </failure>
 <failure test="test_plot_r_discovers_shap_dirs" file="tests/test_plot_smoke.py" line="44">
 <error_type>AssertionError</error_type>
 <message>plot.R smoke test halts when nanoparquet is unavailable; stdout empty.</message>
 <traceback>Same dependency chain.</traceback>
 <likely_cause>Same as above.</likely_cause>
 </failure>
 </failures>
 </post_design_run>
 <summary>
 <all_passing>false</all_passing>
 <recommendation>proceed_to_document</recommendation>
 </summary>
 <action_items>
 <item priority="P2" target_mode="implement" description="Environment-setup concern (not a code defect): install the nanoparquet R package in the local R library to clear the three pre-existing failures (test_hardening::test_check_env_success_in_correct_env, test_package_structure::TestCLI.test_check_env_runs_successfully, test_plot_smoke::test_plot_r_discovers_shap_dirs). Command: `Rscript -e 'install.packages(\"nanoparquet\", repos=\"https://cloud.r-project.org\")'`. This is orthogonal to the patch and can be handled out-of-band; it does not block /publish of the current patch." />
 </action_items>
</test_report>
