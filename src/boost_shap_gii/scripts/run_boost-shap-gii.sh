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
    if [ $# -ne 2 ]; then
        echo "Error: train mode requires exactly 2 arguments."
        echo "Usage: $0 train CONFIG"
        echo "  CONFIG - absolute file path to the config.yaml file"
        echo "  All plot parameters (outcome_max, negate_shap, labels) are read from config.plot.*"
        exit 1
    fi
    config_file=$2 # absolute file path to the config.yaml file

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

    # Plot features with statistically significant GII scores (all plot params read from config.plot.*)
    Rscript "${SCRIPT_DIR}/plot.R" "${config_file}" 2>&1 | tee plot_output.log

elif [ "$mode" == "infer" ]; then
    if [ $# -ne 4 ]; then
        echo "Error: infer mode requires exactly 4 arguments."
        echo "Usage: $0 infer CONFIG DATA_PATH OUTPUT_SUBDIR"
        echo "  CONFIG        - absolute file path to the resolved_config.yaml from a training run"
        echo "  DATA_PATH     - absolute file path to the new independent dataset (CSV or Parquet)"
        echo "  OUTPUT_SUBDIR - subdirectory name for inference outputs within the training output_dir"
        echo "  All plot parameters (outcome_max, negate_shap, labels) are read from config.plot.*"
        exit 1
    fi
    config_file=$2 # absolute file path to the resolved_config.yaml from a training run
    data_path=$3 # absolute file path to the new independent dataset (CSV or Parquet)
    output_subdir=$4 # subdirectory name for inference outputs within the training output_dir

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

    # Plot inference SHAP results (pass inference dir as 2nd arg to override RUN_DIR)
    infer_dir=$(python3 -c "import yaml; c=yaml.safe_load(open('${config_file}')); print(c['paths']['output_dir'] + '/${output_subdir}')")
    Rscript "${SCRIPT_DIR}/plot.R" "${config_file}" "${infer_dir}" 2>&1 | tee plot_output.log

else
    echo "Usage: $0 {train|infer} [args...]"
    echo "  train: $0 train CONFIG"
    echo "  infer: $0 infer CONFIG DATA_PATH OUTPUT_SUBDIR"
    exit 1
fi
