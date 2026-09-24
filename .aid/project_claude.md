# CLAUDE.MD FOR THE boost-shap-gii PIPELINE

## Overview
A pre-existing GitHub repository (https://github.com/tjkeding/boost-shap-gii); a machine learning pipeline utilizing gradient boosting with mixed data types and SHAP analysis (generating feature/feature interaction global importance indices [GIIs]). This pipeline is user config-driven (pull everything from config.yaml when possible), minimizes assumptions, hardcoded values, and defaults (err on kill), and is flexible for many different types of feature sets and outcomes.

## Core Modules
- `train.py`: Data input formatting, model training/hyperparameter tuning with nested cross validation, per-fold checkpoint/resume support
- `predict.py`: OOF model evaluation and SHAP analysis orchestration with phase-level checkpoint/resume
- `infer.py`: Independent-dataset ensemble inference with phase-level checkpoint/resume
- `shap_utils.py`: SHAP-based feature importance: generates magnitude (M) and variability (V) components from SHAP to create the global importance index (GII = sqrt(M * V))
- `indiv_reports.py`: Per-individual SHAP reports with bootstrap confidence intervals
- `utils.py`: Shared utility functions (config hashing, checkpoint I/O, atomic writes, CV splitters, validation)
- `cli.py`: CLI entry point with subcommand dispatch (train, predict, infer, plot, check-env)
- `plot.R`: Visualization of statistically significant SHAP (GII) effects (R/ggplot2)
- `check_env.py`: Pre-flight environment validation (Python and R dependencies)

## Key Supplementary Files
- `example_config_advanced.yaml`: the global controller for boost-shap-gii with all details included (should be the only file that is user-visible/editable)
- `example_config_minimal.yaml`: the global controller for boost-shap-gii with minimal details included (should be the only file that is user-visible/editable)
- `run_boost-shap-gii.sh`: a bash-based pipeline orchestrator for boost-shap-gii (currently called by the user)
- `environment.yaml`: environment software requirements
- `README.md`: user-friendly, detailed documentation of the pipeline
- `INPUT_SPECIFICATION.md`: LLM-friendly, detailed documentation of the pipeline
