#!/usr/bin/env python3
"""Independent Dataset Inference for boost-shap-gii"""

from __future__ import annotations

import argparse
import json
import os
import glob
import warnings
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml

from catboost import CatBoostRegressor, CatBoostClassifier, Pool

from .utils import (
    _normalize_quotes,
    load_config,
    save_json_atomic,
    detect_task,
    is_classification,
    is_regression,
    get_scoring_function,
    compute_bootstrap_ci,
    compute_permutation_test,
)

from .shap_utils import run_shap_pipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply trained boost-shap-gii models to an independent dataset"
    )
    parser.add_argument("--config", required=True,
                        help="Path to the training run's resolved_config.yaml")
    parser.add_argument("--data", required=True,
                        help="Path to the new independent dataset (CSV or Parquet)")
    parser.add_argument("--output-subdir", required=True,
                        help="Subdirectory name for inference outputs within output_dir")
    args = parser.parse_args()

    # 1. Load config and set up directories
    config = load_config(args.config)
    train_dir = config["paths"]["output_dir"]
    infer_dir = os.path.join(train_dir, args.output_subdir)
    os.makedirs(infer_dir, exist_ok=True)

    print(f"[INFO] Training directory: {train_dir}")
    print(f"[INFO] Inference output directory: {infer_dir}")

    # 2. Load training artifacts
    try:
        with open(os.path.join(train_dir, "feature_names.json"), "r") as f:
            trained_features = json.load(f)
        with open(os.path.join(train_dir, "feature_types.json"), "r") as f:
            feature_types = json.load(f)
        with open(os.path.join(train_dir, "feature_metadata.json"), "r") as f:
            feature_meta = json.load(f)

        shadow_features_path = os.path.join(train_dir, "feature_names_shadow.json")
        shadow_features = []
        if os.path.exists(shadow_features_path):
            with open(shadow_features_path, "r") as f:
                shadow_features = json.load(f)
            print(f"[INFO] Found {len(shadow_features)} shadow features for noise calibration.")

        task_info_path = os.path.join(train_dir, "task_info.json")
        if os.path.exists(task_info_path):
            with open(task_info_path, "r") as f:
                json.load(f)  # validate it exists and is parseable

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Missing training artifacts in {train_dir}. Run train.py first."
        ) from e

    # 3. Load new data
    data_path = args.data
    print(f"[INFO] Loading inference data from {data_path}")
    if data_path.endswith('.csv'):
        try:
            df_raw = pd.read_csv(data_path)
        except (pd.errors.ParserError, ValueError, Exception):
            print("[WARNING] Standard CSV parsing failed. Attempting auto-detection (sep=None, engine='python')...")
            df_raw = pd.read_csv(data_path, sep=None, engine='python')
    else:
        df_raw = pd.read_parquet(data_path)

    # Replace whitespace-only strings with NaN
    df_raw = df_raw.replace(r'^\s*$', pd.NA, regex=True)
    N = len(df_raw)
    print(f"[INFO] Loaded {N} rows from inference dataset.")

    # 4. Validate features
    missing = [f for f in trained_features if f not in df_raw.columns]
    if missing:
        raise KeyError(
            f"Inference data is missing {len(missing)} feature(s) used in training: {missing}"
        )

    extra = [c for c in df_raw.columns if c not in trained_features]
    # Extra columns are silently ignored (may include outcome, id, etc.)

    # 5. Outcome detection
    outcome_cfg = config["modeling"]["outcome"]
    if isinstance(outcome_cfg, list):
        outcome_cols = outcome_cfg
    else:
        outcome_cols = [outcome_cfg]

    has_outcomes = all(oc in df_raw.columns for oc in outcome_cols)

    if has_outcomes:
        # Identify rows with non-missing outcomes (supervised subset)
        supervised_mask = df_raw[outcome_cols].notna().all(axis=1).values
        n_supervised = int(supervised_mask.sum())
        print(f"[INFO] Outcomes found for {n_supervised}/{N} rows.")

        if n_supervised == 0:
            print("[WARNING] All outcome values are missing. "
                  "Predictions and SHAP will run, but performance metrics will be skipped.")
            has_outcomes = False
    else:
        supervised_mask = np.zeros(N, dtype=bool)
        n_supervised = 0
        print("[INFO] No outcome column(s) found in inference data. "
              "Predictions and SHAP will run; performance metrics will be skipped.")

    # 6. Type-cast features (identical to predict.py, using saved artifacts)
    # Capture raw data before encoding for SHAP microdata
    X_raw = df_raw[trained_features].copy()

    X = pd.DataFrame(index=df_raw.index)

    con_feats = [c for c, t in feature_types.items() if t == 'continuous']
    ord_feats = [c for c, t in feature_types.items() if t == 'ordinal']
    nom_feats = [c for c, t in feature_types.items() if t == 'nominal']

    # A. Cast Nominals (String -> Category).
    # NaN -> "__NA__" (literal string level). See train.py and README for rationale.
    for c in nom_feats:
        X[c] = df_raw[c].fillna("__NA__").astype(str).astype("category")

    # B. Cast Continuous (Float32)
    for c in con_feats:
        X[c] = pd.to_numeric(df_raw[c], errors='coerce').astype("float32")

    # C. Cast Ordinals (Values -> Integer Codes using saved levels)
    for c in ord_feats:
        levels = [_normalize_quotes(l) for l in feature_meta[c]['levels']]
        df_raw[c] = df_raw[c].map(lambda v: _normalize_quotes(v) if isinstance(v, str) else v)

        # Validate ordinal values against training levels (mirrors train.py)
        unique_vals = df_raw[c].dropna().unique()
        unknowns = [v for v in unique_vals if v not in levels]
        if unknowns:
            # Tier 1: unique-value fraction (hard error if >50% of distinct values are unknown)
            unknown_frac = len(unknowns) / len(unique_vals)
            if unknown_frac > 0.5:
                raise ValueError(
                    f"Feature '{c}': {unknown_frac:.0%} of unique values not in YAML levels "
                    f"{levels}. Check for case mismatches or missing level definitions."
                )
            print(f"[WARNING] Feature '{c}': {len(unknowns)} unique value(s) not in YAML levels: {unknowns}")
            # Tier 2: observation-level fraction (loud warning if >10% of observations are unknown)
            obs_vals = df_raw[c].dropna()
            n_unknown_obs = sum(v not in levels for v in obs_vals)
            obs_frac = n_unknown_obs / len(obs_vals) if len(obs_vals) > 0 else 0.0
            if obs_frac > 0.10:
                print(
                    f"[WARNING] Feature '{c}': {obs_frac:.1%} of non-missing observations "
                    f"({n_unknown_obs}/{len(obs_vals)}) have values not in YAML levels. "
                    f"This may indicate systematic data quality issues."
                )

        cat_type = pd.CategoricalDtype(categories=levels, ordered=True)
        # Explicitly set out-of-category values to NaN before casting to avoid
        # deprecated silent coercion (pandas 2.0+ FutureWarning)
        _src = df_raw[c].where(df_raw[c].isin(levels) | df_raw[c].isna(), other=pd.NA)
        X[c] = _src.astype(cat_type).cat.codes.astype("Int64")
        X.loc[X[c] == -1, c] = pd.NA

    # Reorder exactly as trained
    X = X[trained_features]

    # Warn on high missingness
    for c in trained_features:
        miss_rate = X[c].isna().mean()
        if miss_rate > 0.5:
            print(f"[WARNING] Feature '{c}' has {miss_rate:.1%} missing values in inference data.")

    print(f"[INFO] Feature matrix: {X.shape[0]} rows x {X.shape[1]} columns")

    # 7. Determine task and load labels
    task = detect_task(config)

    class_labels = None
    target_labels = None
    if task == "multiclass_classification":
        cl_path = os.path.join(train_dir, "class_labels.json")
        if os.path.exists(cl_path):
            with open(cl_path) as f:
                class_labels = json.load(f)
    elif task == "multi_regression":
        tl_path = os.path.join(train_dir, "target_labels.json")
        if os.path.exists(tl_path):
            with open(tl_path) as f:
                target_labels = json.load(f)

    print(f"[INFO] Task: {task.upper()}")

    # 8. Ensemble prediction across ALL K fold models on ALL N rows
    model_files = sorted(glob.glob(os.path.join(train_dir, "model_fold_*.cbm")))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {train_dir}")

    n_models = len(model_files)
    print(f"[INFO] Generating ensemble predictions using {n_models} fold models...")

    cat_features_indices = nom_feats
    pool_full = Pool(X, cat_features=cat_features_indices)

    # Accumulate predictions across models
    if task == "multiclass_classification":
        n_classes = len(class_labels)
        pred_accum = np.zeros((N, n_classes))
    elif task == "multi_regression":
        n_targets = len(outcome_cols)
        pred_accum = np.zeros((N, n_targets))
    else:
        pred_accum = np.zeros(N)

    per_model_metrics_rows: List[Dict[str, Any]] = []
    scaler_path = os.path.join(train_dir, "target_scaler.json")
    _scaler_info = None
    if task == "multi_regression" and os.path.exists(scaler_path):
        with open(scaler_path) as f:
            _scaler_info = json.load(f)

    for k, model_path in enumerate(model_files):
        if is_regression(task):
            model = CatBoostRegressor()
            model.load_model(model_path)
            preds = model.predict(pool_full)
        else:
            model = CatBoostClassifier()
            model.load_model(model_path)
            if task == "multiclass_classification":
                preds = model.predict_proba(pool_full)
            else:
                preds = model.predict_proba(pool_full)[:, 1]

        pred_accum += preds

        # --- Per-model performance metrics (if outcomes present) ---
        if has_outcomes and n_supervised > 0:
            fold_preds = preds
            # Inverse-transform multi-regression per-model predictions if scaler exists
            if task == "multi_regression" and _scaler_info is not None:
                fold_preds = preds * np.array(_scaler_info["scale"]) + np.array(_scaler_info["mean"])

            if task == "multi_regression":
                y_true_sup = df_raw[outcome_cols].values[supervised_mask]
                for t_idx, col in enumerate(outcome_cols):
                    for m_name in ["neg_rmse", "neg_mae", "r2"]:
                        fn = get_scoring_function(m_name)
                        raw = fn(y_true_sup[:, t_idx], fold_preds[supervised_mask][:, t_idx])
                        score = -raw if m_name.startswith("neg_") else raw
                        per_model_metrics_rows.append({
                            "fold": k, "metric": f"{m_name.replace('neg_', '').upper()}_{col}", "score": score
                        })
            elif task == "multiclass_classification":
                y_true_sup = df_raw[outcome_cols[0]].values[supervised_mask]
                preds_labels = np.argmax(fold_preds[supervised_mask], axis=1)
                for m_name in ["balanced_accuracy", "f1_weighted"]:
                    fn = get_scoring_function(m_name)
                    score = fn(y_true_sup, preds_labels)
                    per_model_metrics_rows.append({
                        "fold": k, "metric": m_name.upper(), "score": score
                    })
            else:
                y_true_sup = df_raw[outcome_cols[0]].values[supervised_mask]
                fold_sup = fold_preds[supervised_mask]
                if task == "binary_classification":
                    metric_list = ["roc_auc", "accuracy"]
                else:
                    metric_list = ["neg_rmse", "neg_mae", "r2"]
                for m_name in metric_list:
                    fn = get_scoring_function(m_name)
                    if task == "binary_classification" and m_name in ["accuracy", "f1"]:
                        raw = fn(y_true_sup, (fold_sup > 0.5).astype(int))
                    else:
                        raw = fn(y_true_sup, fold_sup)
                    score = -raw if m_name.startswith("neg_") else raw
                    per_model_metrics_rows.append({
                        "fold": k, "metric": m_name.replace("neg_", "").upper(), "score": score
                    })

    # Average across models (soft voting for classification)
    ensemble_preds = pred_accum / n_models

    # Inverse-transform multi-regression predictions if scaler exists
    if task == "multi_regression" and _scaler_info is not None:
        means = np.array(_scaler_info["mean"])
        scales = np.array(_scaler_info["scale"])
        ensemble_preds = ensemble_preds * scales + means
        print("[INFO] Inverse-transformed predictions to original target scale")

    # 9. Performance metrics (only if supervised subset exists)
    if has_outcomes and n_supervised > 0:
        print(f"\n--- Inference Performance (n={n_supervised}, 95% CI) ---")

        # Select metrics based on task type
        if task in ("regression", "multi_regression"):
            metrics_to_calc = ["neg_rmse", "neg_mae", "r2"]
        elif task == "multiclass_classification":
            metrics_to_calc = ["balanced_accuracy", "f1_weighted"]
        else:  # binary_classification
            metrics_to_calc = ["roc_auc", "accuracy"]

        boot_alpha = config["shap"]["bootstrapping"]["alpha"]
        n_boot = config["shap"]["bootstrapping"]["n_boot"]
        results = []

        # Extract supervised subset
        if task == "multi_regression":
            y_true_full = df_raw[outcome_cols].values
            y_true = y_true_full[supervised_mask]
            y_pred = ensemble_preds[supervised_mask]

            for t_idx, col in enumerate(outcome_cols):
                for m_name in metrics_to_calc:
                    fn = get_scoring_function(m_name)
                    raw_score, raw_low, raw_high = compute_bootstrap_ci(
                        y_true[:, t_idx], y_pred[:, t_idx], fn,
                        n_boot=n_boot, alpha=boot_alpha
                    )
                    if m_name.startswith("neg_"):
                        score, low, high = -raw_score, -raw_high, -raw_low
                    else:
                        score, low, high = raw_score, raw_low, raw_high
                    disp_name = f"{m_name.replace('neg_', '').upper()}_{col}"
                    print(f"  {disp_name}: {score:.4f} [{low:.4f}, {high:.4f}]")
                    results.append({"metric": disp_name, "score": score,
                                    "ci_low": low, "ci_high": high})
        elif task == "multiclass_classification":
            y_true = df_raw[outcome_cols[0]].values[supervised_mask]
            y_pred = ensemble_preds[supervised_mask]
            preds_labels = np.argmax(y_pred, axis=1)

            for m_name in metrics_to_calc:
                fn = get_scoring_function(m_name)
                raw_score, raw_low, raw_high = compute_bootstrap_ci(
                    y_true, preds_labels, fn, n_boot=n_boot, alpha=boot_alpha
                )
                disp_name = m_name.upper()
                print(f"  {disp_name}: {raw_score:.4f} [{raw_low:.4f}, {raw_high:.4f}]")
                results.append({"metric": disp_name, "score": raw_score,
                                "ci_low": raw_low, "ci_high": raw_high})
        else:
            # regression or binary_classification
            y_true = df_raw[outcome_cols[0]].values[supervised_mask]
            y_pred = ensemble_preds[supervised_mask]

            for m_name in metrics_to_calc:
                fn = get_scoring_function(m_name)
                if task == "binary_classification" and m_name in ["accuracy", "f1"]:
                    fn = lambda yt, yp, _fn=fn: _fn(yt, (yp > 0.5).astype(int))

                raw_score, raw_low, raw_high = compute_bootstrap_ci(
                    y_true, y_pred, fn, n_boot=n_boot, alpha=boot_alpha
                )

                if m_name.startswith("neg_"):
                    score, low, high = -raw_score, -raw_high, -raw_low
                else:
                    score, low, high = raw_score, raw_low, raw_high

                disp_name = m_name.replace("neg_", "").upper()
                print(f"  {disp_name:<5}: {score:.4f} [{low:.4f}, {high:.4f}]")
                results.append({"metric": disp_name, "score": score,
                                "ci_low": low, "ci_high": high})

        pd.DataFrame(results).to_csv(
            os.path.join(infer_dir, "performance_final.csv"), index=False
        )

        # Save per-model performance metrics
        if per_model_metrics_rows:
            pm_df = pd.DataFrame(per_model_metrics_rows)
            # Append summary rows (mean and std across folds)
            summary_rows = []
            for metric_name in pm_df["metric"].unique():
                scores = pm_df.loc[pm_df["metric"] == metric_name, "score"]
                summary_rows.append({"fold": "mean", "metric": metric_name, "score": scores.mean()})
                summary_rows.append({"fold": "std", "metric": metric_name, "score": scores.std()})
            pm_df = pd.concat([pm_df, pd.DataFrame(summary_rows)], ignore_index=True)
            pm_df.to_csv(os.path.join(infer_dir, "performance_per_model.csv"), index=False)
            print("\n--- Per-Model Performance (fold-level) ---")
            for metric_name in pm_df["metric"].unique():
                mean_val = pm_df.loc[(pm_df["metric"] == metric_name) & (pm_df["fold"] == "mean"), "score"].iloc[0]
                std_val = pm_df.loc[(pm_df["metric"] == metric_name) & (pm_df["fold"] == "std"), "score"].iloc[0]
                print(f"  {metric_name}: {mean_val:.4f} +/- {std_val:.4f}")

        # Permutation test
        print("\n--- Permutation Test (Model vs Chance) ---")
        n_perm = n_boot
        seed = config["execution"]["random_seed"]

        if task == "multi_regression":
            for t_idx, col in enumerate(outcome_cols):
                perm_fns = [get_scoring_function(m) for m in metrics_to_calc]
                perm_results = compute_permutation_test(
                    y_true[:, t_idx], y_pred[:, t_idx],
                    perm_fns, metrics_to_calc, n_perm, seed, infer_dir
                )
                for _, row in perm_results.iterrows():
                    sig = "*" if row["p_value"] < boot_alpha else ""
                    print(f"  {col}/{row['metric']}: observed={row['observed']:.4f}, "
                          f"p={row['p_value']:.4f} {sig}")
        else:
            if task == "multiclass_classification":
                perm_preds = np.argmax(ensemble_preds[supervised_mask], axis=1)
            else:
                perm_preds = ensemble_preds[supervised_mask]

            perm_fns = []
            perm_names = []
            for m_name in metrics_to_calc:
                fn = get_scoring_function(m_name)
                if task == "binary_classification" and m_name in ["accuracy", "f1"]:
                    fn = lambda yt, yp, _fn=fn: _fn(yt, (yp > 0.5).astype(int))
                perm_fns.append(fn)
                perm_names.append(m_name)

            perm_results = compute_permutation_test(
                y_true, perm_preds, perm_fns, perm_names, n_perm, seed, infer_dir
            )

            for _, row in perm_results.iterrows():
                sig_marker = "*" if row["p_value"] < boot_alpha else ""
                print(f"  {row['metric']:<5}: observed={row['observed']:.4f}, "
                      f"null={row['null_mean']:.4f} +/- {row['null_std']:.4f}, "
                      f"p={row['p_value']:.4f} {sig_marker}")

    # 10. Save predictions (all N rows)
    id_col = "id" if "id" in df_raw.columns else "index"
    ids = df_raw[id_col] if "id" in df_raw.columns else df_raw.index

    if task == "multiclass_classification":
        pred_df = pd.DataFrame({id_col: ids})
        if has_outcomes:
            pred_df["y_true"] = df_raw[outcome_cols[0]].values
        else:
            pred_df["y_true"] = np.nan
        for i, cl in enumerate(class_labels):
            pred_df[f"prob_{cl}"] = ensemble_preds[:, i]
    elif task == "multi_regression":
        pred_df = pd.DataFrame({id_col: ids})
        for i, col in enumerate(outcome_cols):
            if has_outcomes:
                pred_df[f"y_true_{col}"] = df_raw[col].values
            else:
                pred_df[f"y_true_{col}"] = np.nan
            pred_df[f"y_pred_{col}"] = ensemble_preds[:, i]
    else:
        pred_df = pd.DataFrame({id_col: ids, "y_pred": ensemble_preds})
        if has_outcomes:
            pred_df["y_true"] = df_raw[outcome_cols[0]].values
        else:
            pred_df["y_true"] = np.nan

    pred_df.to_csv(os.path.join(infer_dir, "predictions_ensemble.csv"), index=False)

    # 11. SHAP analysis in inference mode
    print("\n[INFO] Starting SHAP Pipeline (Inference Mode)...")

    # For SHAP, y is needed for cv splitter fallback but in inference mode
    # splits are synthetic (full dataset per fold), so y can be None
    y_for_shap = None
    if has_outcomes:
        if len(outcome_cols) > 1:
            y_for_shap = df_raw[outcome_cols].copy()
        else:
            y_for_shap = df_raw[outcome_cols[0]].copy()

    shap_ctx = {
        "run_dir": train_dir,
        "config": config,
        "task": task,
        "feature_names": trained_features,
        "feature_names_shadow": shadow_features,
        "cat_features": nom_feats,
        "feature_types": feature_types,
        "X": X,
        "y": y_for_shap,
        "X_raw": X_raw,
        "ids": ids,
        "class_labels": class_labels,
        "target_labels": target_labels,
        "inference_mode": True,
    }

    # Override run_dir so SHAP outputs go to the inference directory
    shap_ctx["run_dir"] = infer_dir

    run_shap_pipeline(shap_ctx)

    # 12. Save inference metadata
    metadata = {
        "training_dir": train_dir,
        "data_path": os.path.abspath(data_path),
        "output_dir": infer_dir,
        "timestamp": datetime.now().isoformat(),
        "n_models": n_models,
        "n_rows": N,
        "n_supervised": n_supervised,
        "has_outcomes": has_outcomes,
        "task": task,
    }
    save_json_atomic(metadata, os.path.join(infer_dir, "inference_metadata.json"))

    print(f"\n[SUCCESS] Inference complete. Outputs in {infer_dir}")


if __name__ == "__main__":
    main()
