# CLAUDE.MD FOR THE boost-shap-gii PIPELINE

## Overview
A pre-existing GitHub repository (https://github.com/tjkeding/boost-shap-gii); a machine learning pipeline utilizing gradient boosting with mixed data types and shap analysis (generating feature/feature interaction global importance indices [GIIs]). This pipeline is/needs to be user config-driven (pull everything from config.yaml when possible), minimizes assumptions, hardcoded values, and defaults (err on kill), and is flexible for many different types of feature sets and outcomes.

## Core Modules
- `train.py`: Data input formatting and model training/hyperparameter tuning with cross validation
- `predict.py`: Model evaluation and call to shap_utils.py for SHAP analysis
- `infer.py`: Model predictions, evaluation, and call to shap_utils.py for SHAP analysis using a new, independent (non-training) dataset
- `shap_utils.py`: SHAP-based feature importance: generates magnitude (M) and variability (V) components from SHAP to create the global importance index (GII)
- `utils.py`: Shared utility functions for the pipeline, primarily train.py, predict.py, and infer.py.
- `plot.R`: Visualization of statistically significant SHAP (GII) effects (NOTE: this is in R, not Python; I prefer ggplot2 for plotting)

## Key Supplementary Files
- `example_config_advanced.yaml`: the global controller for boost-shap-gii with all details included (should be the only file that is user-visible/editable)
- `example_config_minimal.yaml`: the global controller for boost-shap-gii with minimal details included (should be the only file that is user-visible/editable)
- `run_boost-shap-gii.sh`: a bash-based pipeline orchestrator for boost-shap-gii (currently called by the user)
- `environment.yaml`: environment software requirements
- `README.md`: user-friendly, detailed documentation of the pipeline
- `INPUT_SPECIFICATION.md`: LLM-friendly, detailed documentation of the pipeline
