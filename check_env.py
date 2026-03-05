#!/usr/bin/env python3
"""Cross-language dependency verification for boost-shap-gii."""

import sys
import subprocess
import importlib

PYTHON_DEPS = [
    "catboost", "optuna", "shap", "pyarrow", "sklearn", "scipy", "pandas", "yaml"
]

R_DEPS = [
    "ggplot2", "dplyr", "arrow", "tidyr", "foreach", "doParallel", 
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
        print(f"[HINT]  Install in R: install.packages(c({', '.join([f'\"{m}\"' for m in missing])}))")
        return False
    print("   - All R dependencies found.")
    return True

if __name__ == "__main__":
    py_ok = check_python()
    r_ok = check_r()
    
    if not (py_ok and r_ok):
        sys.exit(1)
    
    print("[SUCCESS] Environment verification complete.")
    sys.exit(0)
