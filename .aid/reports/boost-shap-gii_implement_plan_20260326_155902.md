# boost-shap-gii Implementation Plan

```xml
<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-03-26T15:59:02Z" />
  <input_reports>
    <report path="user-provided-inline" mode="direct-specification" key_items="3" />
  </input_reports>

  <assumptions>
    <!-- Assumptions made without explicit user approval -->
    1. The conda environment name is "boost_shap_gii" (underscores), not "boost-shap-gii" (hyphens).
       The environment.yaml declares name: boost-shap-gii but conda created it as boost_shap_gii.
       environment.yaml currently already specifies python>=3.12 — the spec is correct but the
       installed version is 3.11.14. The environment needs an update, not a spec change.
    2. mamba is available at <conda_env_base>/bin/mamba (v2.5.0) and will be used
       for the environment update since the user's global preferences specify mamba.
    3. The run_boost-shap-gii.sh script has moved to src/boost_shap_gii/scripts/run_boost-shap-gii.sh
       as part of the prior package restructuring. Documentation references will point there.
    4. README.md documentation updates will preserve all existing content about the shell script
       workflow and add new sections for pip install / CLI usage.
    5. INPUT_SPECIFICATION.md updates will reflect the new src/ layout and CLI entry points
       while preserving all existing technical content.
    6. Python 3.12 is the target (not 3.13) since it is the current stable release with
       broadest dependency compatibility.
  </assumptions>

  <changes>
    <change id="C1" priority="P0" source_item="Item 1: Conda environment upgrade">
      <file path="(conda environment)" action="modify" />
      <description>
        Upgrade the boost_shap_gii conda environment from Python 3.11.14 to Python 3.12+
        using mamba. The environment.yaml already specifies >=3.12, so the spec file is correct.
        The actual environment simply needs updating. After the upgrade, verify all existing
        dependencies install correctly.
      </description>
      <spec>
        1. Run: mamba update -n boost_shap_gii python=3.12 --yes
           (or if that fails: mamba env update -n boost_shap_gii -f environment.yaml --prune)
        2. Verify: conda activate boost_shap_gii && python --version (should show 3.12.x)
        3. Verify key packages: python -c "import catboost, optuna, sklearn, scipy, pandas, numpy, statsmodels, joblib, pyarrow, yaml"
      </spec>
      <dependencies>none</dependencies>
      <risk>medium - Python minor version upgrades can break compiled extension compatibility (catboost, numpy, scipy). Mamba should rebuild affected packages.</risk>
      <rollback>mamba install -n boost_shap_gii python=3.11 --yes</rollback>
    </change>

    <change id="C2" priority="P0" source_item="Item 2: Validate pip install -e .">
      <file path="(pip editable install)" action="modify" />
      <description>
        Run pip install -e . from within the upgraded conda environment to install the
        boost-shap-gii package in editable mode. Verify CLI entry points, version import,
        and run the full test suite.
      </description>
      <spec>
        1. cd to project root; run: pip install -e "."
        2. Verify CLI: boost-shap-gii --help
        3. Verify subcommands: boost-shap-gii train --help, boost-shap-gii predict --help, etc.
        4. Verify version: python -c "from boost_shap_gii import __version__; print(__version__)" -> "1.0.0"
        5. Run test suite: python -m pytest tests/ -v (385 tests expected)
      </spec>
      <dependencies>C1</dependencies>
      <risk>low - the package structure and pyproject.toml are already validated from prior work. The main risk is Python 3.12 compatibility issues surfaced during testing.</risk>
      <rollback>pip uninstall boost-shap-gii</rollback>
    </change>

    <change id="C3" priority="P1" source_item="Item 3: Update README.md">
      <file path="README.md" action="modify" />
      <description>
        Update README.md to document:
        - New pip install methods (from GitHub, editable dev install)
        - Note that R is optional (needed only for plot subcommand)
        - New CLI interface (boost-shap-gii train/predict/infer/plot/check-env with --config)
        - Preserve existing shell script documentation as an alternative method
        Maintain professional, third-person, academic tone throughout.
      </description>
      <spec>
        In the Quickstart section:
        - Add a new "Installation" subsection before "Configuration" that documents:
          a) pip install from GitHub: pip install git+https://github.com/tjkeding/boost-shap-gii
          b) Development install: pip install -e .
          c) Note that R is optional (only needed for visualization via plot subcommand)
        - Update "Running the Pipeline" to show the CLI as the primary method:
          a) boost-shap-gii train --config config.yaml
          b) boost-shap-gii predict --config config.yaml
          c) boost-shap-gii infer --config resolved_config.yaml --data new_data.csv --output-subdir subdir
          d) boost-shap-gii plot --config config.yaml --outcome-range RANGE --negate-shap BOOL --y-axis-label LABEL
          e) boost-shap-gii check-env
        - Keep existing shell script usage as "Alternative: Shell Script" subsection
      </spec>
      <dependencies>C2 (verify CLI works before documenting it)</dependencies>
      <risk>low - documentation-only change</risk>
      <rollback>git checkout README.md</rollback>
    </change>

    <change id="C4" priority="P1" source_item="Item 3: Update INPUT_SPECIFICATION.md">
      <file path="INPUT_SPECIFICATION.md" action="modify" />
      <description>
        Update INPUT_SPECIFICATION.md to reflect:
        - The new src/ package structure (src/boost_shap_gii/ layout)
        - CLI entry points (boost-shap-gii command with subcommands)
        - Package installation methods
        Preserve all existing technical content.
      </description>
      <spec>
        - Update Stage 0 Pre-flight section to mention both CLI (boost-shap-gii check-env)
          and module invocation (python -m boost_shap_gii.check_env).
        - Add a new section or update Section 5 (Directory Structure) to document the
          src/ package layout:
          src/boost_shap_gii/
            __init__.py, cli.py, train.py, predict.py, infer.py, shap_utils.py, utils.py, check_env.py
            scripts/plot.R, scripts/run_boost-shap-gii.sh
        - Add a brief section on CLI invocation alongside module invocation.
      </spec>
      <dependencies>C2 (verify package structure before documenting it)</dependencies>
      <risk>low - documentation-only change</risk>
      <rollback>git checkout INPUT_SPECIFICATION.md</rollback>
    </change>
  </changes>

  <execution_order>C1, C2, C3, C4</execution_order>
</implement_plan>
```
