#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mode=${1:-"train"}

# 1. PRE-FLIGHT ENVIRONMENT CHECK
if ! python3 -m boost_shap_gii.check_env; then
    echo "[ABORT] Environment check failed. Fix missing dependencies above."
    exit 1
fi

if [ "$mode" == "train" ]; then
    if [ $# -ne 5 ]; then
        echo "Error: train mode requires exactly 5 arguments."
        echo "Usage: $0 train CONFIG OUTCOME_RANGE NEGATE_SHAP Y_AXIS_LABEL"
        echo "  CONFIG       - absolute file path to the config.yaml file"
        echo "  OUTCOME_RANGE - theoretical maximum of outcome measure (for plotting)"
        echo "  NEGATE_SHAP  - 'true' or 'false' to reverse SHAP plot y-axis direction"
        echo "  Y_AXIS_LABEL - y-axis label for SHAP plots"
        exit 1
    fi
    config_file=$2 # absolute file path to the config.yaml file
    plot_outcome_theo_max=$3 # plotting - theoretical maximum (not sample maximum) of outcome measure
    plot_negate_shap=$4 # plotting - binary 'true' or 'false' (or 0/1) indicating whether y-axis of SHAP plots should reverse direction
    plot_y_axis_lab=$5 # plotting - y-axis label for SHAP plots (specific to outcome)

    # 2. VALIDATE PATHS
    if [ ! -f "${config_file}" ]; then
        echo "[ABORT] Config file not found: ${config_file}"
        exit 1
    fi

    # Extract input_data from YAML to verify its existence
    input_data=$(python3 -c "import yaml; c=yaml.safe_load(open('${config_file}')); print(c['paths']['input_data'])" 2>/dev/null || echo "")
    if [ -z "${input_data}" ]; then
        echo "[ABORT] 'paths.input_data' not found in ${config_file}"
        exit 1
    elif [ ! -f "${input_data}" ]; then
        echo "[ABORT] Input data file not found: ${input_data}"
        exit 1
    fi

    # Tune and train a boosting model using tabular data with mixed data types
    python3 -m boost_shap_gii.train --config "${config_file}" 2>&1 | tee train_output.log

    # Evaluate boosting models against chance and generate global importance indices (GII; importance for each feature) from SHAP
    python3 -m boost_shap_gii.predict --config "${config_file}" 2>&1 | tee predict_shap_output.log

    # Plot features with statistically significant GII scores
    Rscript "${SCRIPT_DIR}/plot.R" "${config_file}" "${plot_outcome_theo_max}" "${plot_negate_shap}" "${plot_y_axis_lab}" 2>&1 | tee plot_output.log

elif [ "$mode" == "infer" ]; then
    if [ $# -ne 7 ]; then
        echo "Error: infer mode requires exactly 7 arguments."
        echo "Usage: $0 infer CONFIG DATA_PATH OUTPUT_SUBDIR OUTCOME_RANGE NEGATE_SHAP Y_AXIS_LABEL"
        echo "  CONFIG       - absolute file path to the resolved_config.yaml from a training run"
        echo "  DATA_PATH    - absolute file path to the new independent dataset (CSV or Parquet)"
        echo "  OUTPUT_SUBDIR - subdirectory name for inference outputs within the training output_dir"
        echo "  OUTCOME_RANGE - theoretical maximum of outcome measure (for plotting)"
        echo "  NEGATE_SHAP  - 'true' or 'false' to reverse SHAP plot y-axis direction"
        echo "  Y_AXIS_LABEL - y-axis label for SHAP plots"
        exit 1
    fi
    config_file=$2 # absolute file path to the resolved_config.yaml from a training run
    data_path=$3 # absolute file path to the new independent dataset (CSV or Parquet)
    output_subdir=$4 # subdirectory name for inference outputs within the training output_dir
    plot_outcome_theo_max=$5 # plotting - theoretical maximum of outcome measure
    plot_negate_shap=$6 # plotting - binary 'true' or 'false' for y-axis direction
    plot_y_axis_lab=$7 # plotting - y-axis label for SHAP plots

    # 2. VALIDATE PATHS
    if [ ! -f "${config_file}" ]; then
        echo "[ABORT] Config file not found: ${config_file}"
        exit 1
    fi
    if [ ! -f "${data_path}" ]; then
        echo "[ABORT] Inference data file not found: ${data_path}"
        exit 1
    fi

    # Apply trained models to independent dataset
    python3 -m boost_shap_gii.infer --config "${config_file}" --data "${data_path}" \
        --output-subdir "${output_subdir}" 2>&1 | tee infer_output.log

    # Plot inference SHAP results (pass inference dir as 5th arg to override RUN_DIR)
    infer_dir=$(python3 -c "import yaml; c=yaml.safe_load(open('${config_file}')); print(c['paths']['output_dir'] + '/${output_subdir}')")
    Rscript "${SCRIPT_DIR}/plot.R" "${config_file}" "${plot_outcome_theo_max}" "${plot_negate_shap}" "${plot_y_axis_lab}" "${infer_dir}" 2>&1 | tee plot_output.log

else
    echo "Usage: $0 {train|infer} [args...]"
    echo "  train: $0 train CONFIG OUTCOME_RANGE NEGATE_SHAP Y_AXIS_LABEL"
    echo "  infer: $0 infer CONFIG DATA_PATH OUTPUT_SUBDIR OUTCOME_RANGE NEGATE_SHAP Y_AXIS_LABEL"
    exit 1
fi
