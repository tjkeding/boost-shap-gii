<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-03-26T15:07:33Z" />
  <pre_design_run>
    <total>340 (expected) / 30 (collected)</total>
    <passed>14</passed>
    <failed>16</failed>
    <errors>8 (collection errors; ~310 tests never ran)</errors>
    <coverage_pct>N/A</coverage_pct>
    <failures>
      <failure test="test_adversarial.py" file="tests/test_adversarial.py" line="26">
        <error_type>ModuleNotFoundError</error_type>
        <message>No module named 'train'</message>
        <traceback>Collection error: sys.path.insert(0, "..") pointed to project root where train.py no longer exists</traceback>
      </failure>
      <failure test="test_implementation_changes.py" file="tests/test_implementation_changes.py" line="23">
        <error_type>ModuleNotFoundError</error_type>
        <message>No module named 'utils'</message>
        <traceback>Collection error: bare `from utils import` after flat-to-src-layout migration</traceback>
      </failure>
      <failure test="test_infer.py" file="tests/test_infer.py" line="15">
        <error_type>ModuleNotFoundError</error_type>
        <message>No module named 'utils'</message>
        <traceback>Collection error: bare `from utils import` after migration</traceback>
      </failure>
      <failure test="test_infer_robustness.py" file="tests/test_infer_robustness.py" line="6">
        <error_type>ModuleNotFoundError</error_type>
        <message>No module named 'train'</message>
        <traceback>Collection error: bare `from train import FeatureSelector`</traceback>
      </failure>
      <failure test="test_inference_shap.py" file="tests/test_inference_shap.py" line="18">
        <error_type>ModuleNotFoundError</error_type>
        <message>No module named 'shap_utils'</message>
        <traceback>Collection error: bare `from shap_utils import`</traceback>
      </failure>
      <failure test="test_predict.py" file="tests/test_predict.py" line="14">
        <error_type>ModuleNotFoundError</error_type>
        <message>No module named 'utils'</message>
        <traceback>Collection error: bare `from utils import`</traceback>
      </failure>
      <failure test="test_shap_utils.py" file="tests/test_shap_utils.py" line="14">
        <error_type>ModuleNotFoundError</error_type>
        <message>No module named 'shap_utils'</message>
        <traceback>Collection error: bare `from shap_utils import`</traceback>
      </failure>
      <failure test="test_train.py" file="tests/test_train.py" line="18">
        <error_type>ModuleNotFoundError</error_type>
        <message>No module named 'train'</message>
        <traceback>Collection error: bare `from train import`</traceback>
      </failure>
      <failure test="test_hardening.py (6 tests)" file="tests/test_hardening.py" line="various">
        <error_type>FileNotFoundError</error_type>
        <message>check_env.py, run_boost-shap-gii.sh no longer at PROJECT_ROOT</message>
        <traceback>Files moved to src/boost_shap_gii/ and src/boost_shap_gii/scripts/</traceback>
      </failure>
      <failure test="test_plot_smoke.py (2 tests)" file="tests/test_plot_smoke.py" line="various">
        <error_type>FileNotFoundError / AssertionError</error_type>
        <message>plot.R no longer at PROJECT_ROOT</message>
        <traceback>plot.R moved to src/boost_shap_gii/scripts/plot.R</traceback>
      </failure>
      <failure test="test_shell_and_config.py (8 tests)" file="tests/test_shell_and_config.py" line="various">
        <error_type>FileNotFoundError / AssertionError</error_type>
        <message>run_boost-shap-gii.sh and plot.R no longer at PROJECT_ROOT; shell script content changed</message>
        <traceback>Shell script now uses python3 -m boost_shap_gii.{train,predict,infer} instead of bare .py scripts</traceback>
      </failure>
    </failures>
  </pre_design_run>
  <design_phase>
    <tests_created>45</tests_created>
    <tests_modified>340 (all existing tests updated for new import paths)</tests_modified>
    <files_created>
      <file path="tests/conftest.py" test_count="0" coverage_target="sys.path setup for src-layout package imports" />
      <file path="tests/_paths.py" test_count="0" coverage_target="Shared path constants (PROJECT_ROOT, PACKAGE_DIR, SCRIPTS_DIR) for all test files" />
      <file path="tests/test_package_structure.py" test_count="45" coverage_target="pyproject.toml validation, __init__.py exports, module importability, package data discovery, CLI entry point parsing, relative import integrity" />
    </files_created>
    <design_rationale>
      Three categories of changes were required:

      1. IMPORT MIGRATION (8 files): Replaced all `sys.path.insert(0, "..")` + `from {module} import` patterns with `from boost_shap_gii.{module} import`. This was the primary breakage: the old flat layout (train.py, utils.py, shap_utils.py at project root) was replaced by the src layout (src/boost_shap_gii/train.py, etc.), so bare module imports no longer resolve. A conftest.py was created to add src/ to sys.path, making boost_shap_gii importable without pip install -e (which fails due to requires-python >= 3.12 vs. Python 3.11 in the conda env).

      2. PATH REFERENCE MIGRATION (4 files): Tests that opened files using PROJECT_DIR (../check_env.py, ../run_boost-shap-gii.sh, ../plot.R, ../utils.py, etc.) were updated to use PACKAGE_DIR (src/boost_shap_gii/) and SCRIPTS_DIR (src/boost_shap_gii/scripts/). Shell script content assertions were updated to match the new `python3 -m boost_shap_gii.*` invocation pattern. Shell guard tests were rewritten to bypass the check_env module call using a `true` substitution instead of the old file-replacement approach.

      3. NEW PACKAGE STRUCTURE TESTS (1 new file, 45 tests): test_package_structure.py validates the new src layout end-to-end: pyproject.toml parsing and metadata (build system, dependencies, entry points, package-data), __init__.py version export and consistency with pyproject.toml, importability of all 7 production modules, existence of bundled scripts, CLI argument parsing for all 5 subcommands (train, predict, infer, plot, check-env), and relative import integrity (no bare sibling imports in production code).

      Additionally, one pre-existing test (test_source_code_uses_while_loop in test_implementation_changes.py) was fixed by increasing the source-code grep window from 2000 to 4000 characters, since the while-loop pattern now appears deeper in the function body after the docstring was expanded during the hardening pass.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>385</total>
    <passed>385</passed>
    <failed>0</failed>
    <errors>0</errors>
    <coverage_pct>N/A</coverage_pct>
    <failures>
    </failures>
  </post_design_run>
  <summary>
    <all_passing>true</all_passing>
    <recommendation>proceed_to_document</recommendation>
  </summary>
  <action_items>
    <item priority="P1" target_mode="implement" description="pyproject.toml specifies requires-python >= 3.12 but the boost_shap_gii conda environment has Python 3.11.14. This prevents pip install -e from working. Either update the conda environment to Python >= 3.12 or relax the constraint to >= 3.11. Tests currently work around this via PYTHONPATH injection in conftest.py." />
    <item priority="P2" target_mode="implement" description="Three Pandas4Warnings about deprecated Categorical construction with values not in dtype categories (test_adversarial.py:480, test_infer.py:203). These reflect production-code patterns that will break in a future pandas release." />
  </action_items>
</test_report>
