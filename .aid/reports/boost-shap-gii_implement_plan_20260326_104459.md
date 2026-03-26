<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-03-26T14:44:55Z" />
  <input_reports>
    <report path="(inline brainstorm decisions)" mode="brainstorm" key_items="13" />
  </input_reports>

  ## Assumptions and Decisions Made Without Explicit Approval

  1. **`src/` layout inner directory uses underscores**: The Python package directory will be `src/boost_shap_gii/` (underscores), which is the Python convention for importable package names. The project canonical name remains `boost-shap-gii` (hyphens) for the repository, pyproject.toml metadata, and pip install name.
  2. **`scripts/` directory lives inside `src/boost_shap_gii/`**: The `plot.R` file and `run_boost-shap-gii.sh` will be placed under `src/boost_shap_gii/scripts/` so they are included as package data and discoverable at runtime via `importlib.resources`. This differs from placing them at `scripts/` in the project root.
  3. **Tests remain outside the package**: Tests are already gitignored and will NOT be moved into the `src/` layout. Their `sys.path.insert` hacks must be updated to point to the new package location (but that update is deferred to `/test` mode, since tests are not tracked in git). I will note the required test update in the report.
  4. **`check_env.py` moves into the package** as `src/boost_shap_gii/check_env.py` and becomes callable via `boost-shap-gii check-env` CLI subcommand. The `shap` import check is removed per decision (5).
  5. **Example configs stay in project root**: `example_config_advanced.yaml` and `example_config_minimal.yaml` are user-facing templates, not package data. They remain in the project root.
  6. **`environment.yaml` stays in project root**: It is a conda environment spec, not a Python package artifact.
  7. **Entry point console_scripts**: `boost-shap-gii` will be the CLI command name, dispatching to subcommands `train`, `predict`, `infer`, `plot`, and `check-env` via `argparse` subparsers.
  8. **`__pycache__` and `catboost_info` directories**: These are build/runtime artifacts already gitignored (by the ignore-all-then-whitelist pattern). No action needed.
  9. **Documentation files (README.md, INPUT_SPECIFICATION.md, CLAUDE.md, GEMINI.md)**: Remain in project root. No movement.
  10. **The `plot` subcommand wraps `Rscript plot.R`**: The CLI will locate `plot.R` via `importlib.resources`, construct the `Rscript` invocation, and call it via `subprocess`. If `Rscript` is not found on PATH, it prints a clear error and exits non-zero.

  <changes>
    <change id="C1" priority="P0" source_item="decision-3, decision-6">
      <file path="src/boost_shap_gii/__init__.py" action="create" />
      <description>Create the package __init__.py with version string and public API exports. Exports the key public functions from submodules so that `from boost_shap_gii import load_config` etc. work. Sets `__version__ = "1.0.0"`.</description>
      <spec>
        - `__version__ = "1.0.0"`
        - Re-export key public symbols from `.utils`, `.shap_utils`, `.train`, `.predict`, `.infer` for convenience (lazy imports are acceptable)
        - Minimal file; this is the namespace anchor
      </spec>
      <dependencies>none (created first)</dependencies>
      <risk>low - new file, no existing code affected</risk>
      <rollback>delete src/boost_shap_gii/__init__.py</rollback>
    </change>

    <change id="C2" priority="P0" source_item="decision-1, decision-2">
      <file path="src/boost_shap_gii/utils.py" action="create (move from root)" />
      <file path="src/boost_shap_gii/shap_utils.py" action="create (move from root)" />
      <file path="src/boost_shap_gii/train.py" action="create (move from root)" />
      <file path="src/boost_shap_gii/predict.py" action="create (move from root)" />
      <file path="src/boost_shap_gii/infer.py" action="create (move from root)" />
      <file path="src/boost_shap_gii/check_env.py" action="create (move from root)" />
      <description>Move all Python source modules from project root into `src/boost_shap_gii/`. Use `git mv` to preserve history.</description>
      <spec>
        git mv utils.py src/boost_shap_gii/utils.py
        git mv shap_utils.py src/boost_shap_gii/shap_utils.py
        git mv train.py src/boost_shap_gii/train.py
        git mv predict.py src/boost_shap_gii/predict.py
        git mv infer.py src/boost_shap_gii/infer.py
        git mv check_env.py src/boost_shap_gii/check_env.py
      </spec>
      <dependencies>C1</dependencies>
      <risk>medium - must be followed immediately by C3 (import rewrite) or imports break</risk>
      <rollback>git mv each file back to root</rollback>
    </change>

    <change id="C3" priority="P0" source_item="decision-10, decision-11">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Refactor all intra-package imports to relative imports. Remove the sys.path.append fallback hack from predict.py and infer.py.</description>
      <spec>
        train.py:
          - `from utils import (...)` -> `from .utils import (...)`

        predict.py:
          - `from utils import (...)` -> `from .utils import (...)`
          - Remove the entire try/except/sys.path.append block for shap_utils
          - Replace with: `from .shap_utils import run_shap_pipeline`

        infer.py:
          - `from utils import (...)` -> `from .utils import (...)`
          - Remove the entire try/except/sys.path.append block for shap_utils
          - Replace with: `from .shap_utils import run_shap_pipeline`

        shap_utils.py:
          - `from utils import get_cv_splitter, detect_task, is_regression` -> `from .utils import get_cv_splitter, detect_task, is_regression`
      </spec>
      <dependencies>C2</dependencies>
      <risk>high - if any import is missed, module will fail to load. Must be thorough.</risk>
      <rollback>revert relative imports to absolute</rollback>
    </change>

    <change id="C4" priority="P0" source_item="decision-9">
      <file path="src/boost_shap_gii/scripts/" action="create directory" />
      <file path="src/boost_shap_gii/scripts/plot.R" action="create (move from root)" />
      <file path="src/boost_shap_gii/scripts/run_boost-shap-gii.sh" action="create (move from root)" />
      <description>Move plot.R and run_boost-shap-gii.sh into the package scripts/ directory. Update run_boost-shap-gii.sh to invoke modules via `python3 -m boost_shap_gii.train` instead of `python3 train.py`, and locate plot.R relative to itself.</description>
      <spec>
        mkdir -p src/boost_shap_gii/scripts/
        git mv plot.R src/boost_shap_gii/scripts/plot.R
        git mv run_boost-shap-gii.sh src/boost_shap_gii/scripts/run_boost-shap-gii.sh

        In run_boost-shap-gii.sh:
          - Change `python3 check_env.py` -> `python3 -m boost_shap_gii.check_env`
          - Change `python3 train.py` -> `python3 -m boost_shap_gii.train`
          - Change `python3 predict.py` -> `python3 -m boost_shap_gii.predict`
          - Change `python3 infer.py` -> `python3 -m boost_shap_gii.infer`
          - Change `Rscript plot.R` -> `Rscript "$(dirname "$0")/plot.R"` (so it finds plot.R co-located)
          - Remove `cd "$(dirname "$0")"` since scripts no longer at project root (replace with appropriate logic)
      </spec>
      <dependencies>C2, C3</dependencies>
      <risk>medium - shell script path logic must be correct for both pip-installed and development use</risk>
      <rollback>git mv files back to root, revert shell script changes</rollback>
    </change>

    <change id="C5" priority="P0" source_item="decision-7">
      <file path="src/boost_shap_gii/cli.py" action="create" />
      <description>Create the CLI entry point module with argparse subparsers for: train, predict, infer, plot, check-env. Each subcommand accepts --config (required) and subcommand-specific flags. The `plot` subcommand also accepts positional args for outcome_range, negate_shap, y_axis_label. The CLI dispatches to the appropriate module's main() function.</description>
      <spec>
        def main():
            parser = argparse.ArgumentParser(prog="boost-shap-gii", description="...")
            subparsers = parser.add_subparsers(dest="command", required=True)

            # train subcommand
            p_train = subparsers.add_parser("train")
            p_train.add_argument("--config", required=True)

            # predict subcommand
            p_predict = subparsers.add_parser("predict")
            p_predict.add_argument("--config", required=True)

            # infer subcommand
            p_infer = subparsers.add_parser("infer")
            p_infer.add_argument("--config", required=True)
            p_infer.add_argument("--data", required=True)
            p_infer.add_argument("--output-subdir", required=True)

            # plot subcommand
            p_plot = subparsers.add_parser("plot")
            p_plot.add_argument("--config", required=True)
            p_plot.add_argument("--outcome-range", required=True)
            p_plot.add_argument("--negate-shap", required=True)
            p_plot.add_argument("--y-axis-label", required=True)
            p_plot.add_argument("--run-dir", required=False, help="Override run dir (for inference plots)")

            # check-env subcommand
            p_check = subparsers.add_parser("check-env")

            args = parser.parse_args()
            # Dispatch based on args.command
            # For plot: locate plot.R via importlib.resources, invoke Rscript
            # For check-env: call check_env.main()
            # For train/predict/infer: inject sys.argv and call respective main()

        Internal details:
          - plot subcommand: uses importlib.resources to find plot.R path
          - Graceful degradation: if Rscript not on PATH, print error, exit 1
          - All subcommands independently callable
      </spec>
      <dependencies>C1, C2, C3</dependencies>
      <risk>medium - argparse interface must match existing module interfaces exactly</risk>
      <rollback>delete cli.py, remove console_scripts from pyproject.toml</rollback>
    </change>

    <change id="C6" priority="P0" source_item="decision-1, decision-2, decision-4, decision-5, decision-6">
      <file path="pyproject.toml" action="create" />
      <description>Create pyproject.toml with all package metadata, dependencies, build system, and console_scripts entry point. No setup.py.</description>
      <spec>
        [build-system]
        requires = ["setuptools>=68.0", "wheel"]
        build-backend = "setuptools.backends._legacy:_Backend"
        -> Actually use the standard: build-backend = "setuptools.build_meta"

        [project]
        name = "boost-shap-gii"
        version = "1.0.0"
        description = "Config-driven gradient boosting with SHAP-based global importance indices (GII)."
        readme = "README.md"
        license = "MIT"
        requires-python = ">=3.12"
        authors = [{name = "TJ Keding"}]
        dependencies = [
            "numpy",
            "pandas",
            "pyyaml",
            "scikit-learn",
            "scipy",
            "statsmodels",
            "joblib",
            "pyarrow",
            "catboost",
            "optuna",
        ]
        # NOTE: shap is NOT included (confirmed unused by codebase audit)

        [project.scripts]
        boost-shap-gii = "boost_shap_gii.cli:main"

        [tool.setuptools.packages.find]
        where = ["src"]

        [tool.setuptools.package-data]
        boost_shap_gii = ["scripts/*.R", "scripts/*.sh"]
      </spec>
      <dependencies>C1</dependencies>
      <risk>low - new file, standard setuptools configuration</risk>
      <rollback>delete pyproject.toml</rollback>
    </change>

    <change id="C7" priority="P1" source_item="decision-3">
      <file path="LICENSE" action="create" />
      <description>Create MIT license file.</description>
      <spec>Standard MIT license text with "2026 TJ Keding" as copyright holder.</spec>
      <dependencies>none</dependencies>
      <risk>low - new file</risk>
      <rollback>delete LICENSE</rollback>
    </change>

    <change id="C8" priority="P1" source_item="decision-5, check_env.py audit">
      <file path="src/boost_shap_gii/check_env.py" action="modify" />
      <description>Remove "shap" from PYTHON_DEPS list in check_env.py (confirmed unused by codebase).</description>
      <spec>
        Remove "shap" from the PYTHON_DEPS list.
        Current: PYTHON_DEPS = ["catboost", "optuna", "shap", "pyarrow", "sklearn", ...]
        New: PYTHON_DEPS = ["catboost", "optuna", "pyarrow", "sklearn", ...]
      </spec>
      <dependencies>C2 (file must be in new location)</dependencies>
      <risk>low - removing a check for an unused dependency</risk>
      <rollback>re-add "shap" to PYTHON_DEPS</rollback>
    </change>

    <change id="C9" priority="P1" source_item="gitignore update for new layout">
      <file path=".gitignore" action="modify" />
      <description>Rewrite .gitignore for the new src layout. The current approach (ignore everything, whitelist specific files) must be updated to whitelist the new directory structure and new files (pyproject.toml, LICENSE, src/ directory tree).</description>
      <spec>
        Replace entire .gitignore content with updated whitelist:
        - Keep ignore-all pattern
        - Add !src/ and !src/** recursion
        - Add !pyproject.toml
        - Add !LICENSE
        - Update file whitelist (remove root-level .py files that moved)
        - Keep example configs, environment.yaml, README.md, INPUT_SPECIFICATION.md, CLAUDE.md
      </spec>
      <dependencies>C2, C4 (files must be in new locations)</dependencies>
      <risk>medium - incorrect gitignore could cause files to be untracked or sensitive files to leak</risk>
      <rollback>restore original .gitignore</rollback>
    </change>

    <change id="C10" priority="P1" source_item="decision-6, environment.yaml sync">
      <file path="environment.yaml" action="modify" />
      <description>Update environment.yaml: bump python requirement to >=3.12 to match pyproject.toml, remove shap if present (it is not currently listed, so this may be a no-op). Verify consistency.</description>
      <spec>
        Change `python>=3.8` to `python>=3.12`
        Verify no shap dependency exists (confirmed: it does not)
      </spec>
      <dependencies>none</dependencies>
      <risk>low - only changes python version floor</risk>
      <rollback>revert python version to >=3.8</rollback>
    </change>

    <change id="C11" priority="P2" source_item="module runnability">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <file path="src/boost_shap_gii/check_env.py" action="modify" />
      <description>Ensure each module remains directly runnable via `python -m boost_shap_gii.train` by keeping the `if __name__ == "__main__": main()` guard. No changes needed if already present (they are), but confirm the guard is intact after C3 refactoring.</description>
      <spec>Verification pass only. Each file already has `if __name__ == "__main__": main()`. Ensure C3 did not accidentally remove them.</spec>
      <dependencies>C3</dependencies>
      <risk>low - verification only</risk>
      <rollback>n/a</rollback>
    </change>
  </changes>

  <execution_order>C7, C10, C1, C6, C2, C3, C8, C4, C5, C9, C11</execution_order>

  ## Execution Order Rationale

  1. **C7** (LICENSE) and **C10** (environment.yaml) - Independent, no dependencies. Can be done first/in parallel.
  2. **C1** (__init__.py) and **C6** (pyproject.toml) - Create package skeleton. C1 requires the directory to exist.
  3. **C2** (git mv files) - Move all Python source into package directory.
  4. **C3** (relative imports) - Must immediately follow C2 to restore import correctness.
  5. **C8** (remove shap from check_env) - Small modification after file is in place.
  6. **C4** (move scripts) - Move plot.R and shell script, update shell script paths.
  7. **C5** (cli.py) - Create CLI entry point. Depends on all modules being in their final locations.
  8. **C9** (.gitignore) - Update after all files are in their final locations.
  9. **C11** (verification) - Final check that __main__ guards survived.

  ## Notes for User

  - **Tests**: The existing tests under `tests/` use `sys.path.insert(0, "..")` to import from the project root. After this restructuring, those tests will need their import paths updated to either: (a) `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))`, or (b) install the package in editable mode (`pip install -e .`) and remove the sys.path hacks entirely. Recommend running `/test` after this implementation to update and verify all tests.
  - **Editable install**: After implementation, developers should use `pip install -e .` from the project root. This makes the package importable without sys.path hacks.
  - **Backward compatibility**: The `run_boost-shap-gii.sh` script will still work but now invokes modules via `python3 -m boost_shap_gii.X` instead of `python3 X.py`.

</implement_plan>
