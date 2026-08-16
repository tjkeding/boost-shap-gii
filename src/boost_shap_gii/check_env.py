#!/usr/bin/env python3
"""Cross-language dependency verification for boost-shap-gii."""

import sys
import subprocess
import importlib

PYTHON_DEPS = [
    "catboost", "optuna", "pyarrow", "sklearn", "scipy",
    "pandas", "yaml", "joblib", "statsmodels", "psutil"
]

R_DEPS = [
    "ggplot2", "dplyr", "nanoparquet", "tidyr", "foreach", "doParallel",
    "gridExtra", "stringr", "yaml"
]

def check_python():
    print("[CHECK] Verifying Python dependencies...")
    missing = []
    for dep in PYTHON_DEPS:
        try:
            importlib.import_module(dep if dep != "yaml" else "yaml")
        except ImportError:
            # Handle sklearn/scikit-learn naming
            if dep == "sklearn":
                try:
                    importlib.import_module("sklearn")
                except ImportError:
                    missing.append("scikit-learn")
            else:
                missing.append(dep)
    
    if missing:
        print(f"[ERROR] Missing Python packages: {', '.join(missing)}")
        print(f"[HINT]  Install via: pip install {' '.join(missing)}")
        return False
    print("   - All Python dependencies found.")
    return True

def check_r():
    print("[CHECK] Verifying R dependencies...")
    missing = []
    for dep in R_DEPS:
        # Run a quick R command to check if library(dep) succeeds
        try:
            result = subprocess.run(
                ["Rscript", "-e", f"library({dep})"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                missing.append(dep)
        except FileNotFoundError:
            print("[ERROR] Rscript not found in PATH. Please install R.")
            return False

    if missing:
        print(f"[ERROR] Missing R packages: {', '.join(missing)}")
        quoted = ', '.join(f'"{m}"' for m in missing)
        print(f"[HINT]  Install in R: install.packages(c({quoted}))")
        return False
    print("   - All R dependencies found.")
    return True

def run_preflight() -> None:
    """Run all environment checks; exit with status 2 on failure.

    Intended to be called from within other CLI command handlers (cmd_train,
    cmd_predict, cmd_infer, cmd_plot) before any other work, so environment
    problems surface as a fast early exit with actionable guidance.

    On success: prints "[ENV] Environment preflight passed." and returns.
    On failure: check_python() / check_r() have already printed the concrete
    list of missing packages with install commands; this function then calls
    sys.exit(2). Exit code 2 is distinct from main()'s sys.exit(1) so that CI
    and log scrapers can distinguish a preflight-gate failure from a standalone
    check-env invocation failure.
    """
    py_ok = check_python()
    r_ok = check_r()
    if not (py_ok and r_ok):
        sys.exit(2)
    print("[ENV] Environment preflight passed.")


def main():
    """Run all environment checks and exit non-zero on failure."""
    py_ok = check_python()
    r_ok = check_r()

    if not (py_ok and r_ok):
        sys.exit(1)

    print("[SUCCESS] Environment verification complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
