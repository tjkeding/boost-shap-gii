#!/usr/bin/env python3
"""Prediction Inference for boost_shap_gii (training set)"""

from __future__ import annotationsß

import argparse
import json
import os
import sys
import glob
import warnings
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

from utils import (
    _normalize_quotes,
    load_config,
    save_json_atomic,
    detect_task,
    is_classification,
    is_regression,
    get_cv_splitter,
    get_scoring_function,
    compute_bootstrap_ci,
    compute_permutation_test,
)

# Local import for SHAP
try:
    from shap_utils import run_shap_pipeline
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from shap_utils import run_shap_pipeline

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------------------------------------------------------
# 1. Main Logic
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean Inference Driver")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # 1. Setup
    config = load_config(args.config)
    run_dir = config["paths"]["output_dir"]

    print(f"[INFO] Inference Run Directory: {run_dir}")

    # 2. Load Metadata (Strict Alignment with Train)
    try:
        with open(os.path.join(run_dir, "feature_names.json"), "r") as f:
            trained_features = json.load(f)
        with open(os.path.join(run_dir, "feature_types.json"), "r") as f:
            feature_types = json.load(f)
        with open(os.path.join(run_dir, "feature_metadata.json"), "r") as f:
            feature_meta = json.load(f)

        # NEW: Load Shadow Feature Names if available (for Noise Calibration)
        shadow_features_path = os.path.join(run_dir, "feature_names_shadow.json")
        shadow_features = []
        if os.path.exists(shadow_features_path):
            with open(shadow_features_path, "r") as f:
                shadow_features = json.load(f)
            print(f"[INFO] Found {len(shadow_features)} shadow features for noise calibration.")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing training artifacts in {run_dir}. Run train.py first.") from e

    # 3. Load Data
    data_path = config["paths"]["input_data"]
    print(f"[INFO] Loading data from {data_path}")
    if data_path.endswith('.csv'):
        try:
            df_raw = pd.read_csv(data_path)
        except (pd.errors.ParserError, ValueError, Exception):
            print("[WARNING] Standard CSV parsing failed. Attempting auto-detection (sep=None, engine='python')...")
            df_raw = pd.read_csv(data_path, sep=None, engine='python')
    else:
        df_raw = pd.read_parquet(data_path)

    # Replace whitespace-only strings with NaN across entire dataframe
    df_raw = df_raw.replace(r'^\s*$', pd.NA, regex=True)

    # 4. Feature Selection & Type Enforcement
    print("[INFO] Enforcing types and features from training artifacts...")

    missing = [f for f in trained_features if f not in df_raw.columns]
    if missing:
        raise KeyError(f"Input data is missing features used in training: {missing}")

    # Outcome Handling & Row Dropping (Strict Mirror of train.py)
    outcome_cfg = config["modeling"]["outcome"]
    if isinstance(outcome_cfg, list):
        outcome_cols = outcome_cfg
    else:
        outcome_cols = [outcome_cfg]

    for oc in outcome_cols:
        if oc not in df_raw.columns:
            raise ValueError(f"Outcome '{oc}' missing. OOF analysis requires target variable.")

    # Drop rows where any target is missing, just like train.py
    initial_len = len(df_raw)
    df_raw = df_raw.dropna(subset=outcome_cols)
    dropped = initial_len - len(df_raw)
    if dropped > 0:
        print(f"[INFO] Dropped {dropped} rows with missing outcome(s) (Mirroring train.py).")

    if len(outcome_cols) > 1:
        y = df_raw[outcome_cols].copy()
    else:
        y = df_raw[outcome_cols[0]].copy()

    # Determine ID column (Strict Mirror of train.py)
    id_col = "id" if "id" in df_raw.columns else "index"
    ids = df_raw[id_col] if "id" in df_raw.columns else df_raw.index

    # CAPTURE RAW DATA for Metadata (Before Encoding)
    # We only keep the trained features for X_raw to align with X columns
    X_raw = df_raw[trained_features].copy()

    X = pd.DataFrame(index=df_raw.index)

    # Identify lists from metadata
    con_feats = [c for c, t in feature_types.items() if t == 'continuous']
    ord_feats = [c for c, t in feature_types.items() if t == 'ordinal']
    nom_feats = [c for c, t in feature_types.items() if t == 'nominal']

    # A. Cast Nominals (String -> Category)
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
            unknown_frac = len(unknowns) / len(unique_vals)
            if unknown_frac > 0.5:
                raise ValueError(
                    f"Feature '{c}': {unknown_frac:.0%} of unique values not in YAML levels "
                    f"{levels}. Check for case mismatches or missing level definitions."
                )
            print(f"[WARNING] Feature '{c}': {len(unknowns)} value(s) not in YAML levels: {unknowns}")

        # Create categorical with exact training levels
        cat_type = pd.CategoricalDtype(categories=levels, ordered=True)
        # Convert to codes (0, 1, 2...). NaN becomes -1.
        X[c] = df_raw[c].astype(cat_type).cat.codes.astype("Int64")
        # Restore NaNs
        X.loc[X[c] == -1, c] = pd.NA

    # Reorder exactly as trained
    X = X[trained_features]

    print(f"[INFO] Evaluated on {len(X)} rows.")

    # 5. Determine Task
    task = detect_task(config)

    # Load class/target labels if available
    class_labels = None
    target_labels = None
    if task == "multiclass_classification":
        cl_path = os.path.join(run_dir, "class_labels.json")
        if os.path.exists(cl_path):
            with open(cl_path) as f:
                class_labels = json.load(f)
    elif task == "multi_regression":
        tl_path = os.path.join(run_dir, "target_labels.json")
        if os.path.exists(tl_path):
            with open(tl_path) as f:
                target_labels = json.load(f)

    print(f"[INFO] Task detected: {task.upper()} (based on config)")

    # 6. OOF Prediction Loop (Replaces Ensemble)
    model_files = glob.glob(os.path.join(run_dir, "model_fold_*.cbm"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {run_dir}")

    print(f"[INFO] Generating OOF Predictions using {len(model_files)} folds...")

    # OOF storage depends on task type
    if task == "multiclass_classification":
        n_classes = len(class_labels)
        oof_preds = np.full((len(X), n_classes), np.nan)
    elif task == "multi_regression":
        n_targets = len(outcome_cols)
        oof_preds = np.full((len(X), n_targets), np.nan)
    else:
        oof_preds = np.full(len(X), np.nan)

    counts = np.zeros(len(X))

    # Replicate Splitter from train.py
    y_for_split = y if isinstance(y, pd.Series) else y.iloc[:, 0]
    splitter = get_cv_splitter(config, y_for_split)

    cat_features_indices = nom_feats

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y_for_split)):
        model_path = os.path.join(run_dir, f"model_fold_{fold_idx}.cbm")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model file for fold {fold_idx}: {model_path}")

        X_val = X.iloc[val_idx]
        pool_val = Pool(X_val, cat_features=cat_features_indices)

        if is_regression(task):
            model = CatBoostRegressor()
            model.load_model(model_path)
            preds = model.predict(pool_val)
        else:
            model = CatBoostClassifier()
            model.load_model(model_path)
            if task == "multiclass_classification":
                preds = model.predict_proba(pool_val)
            else:
                preds = model.predict_proba(pool_val)[:, 1]

        oof_preds[val_idx] = preds
        counts[val_idx] += 1

    if np.any(counts == 0):
        n_missing = int(np.sum(counts == 0))
        raise RuntimeError(
            f"{n_missing} rows were never predicted in OOF loop. "
            f"CV fold structure may be corrupted."
        )
    if np.any(counts > 1):
        print("[WARNING] Some rows were predicted multiple times (counts>1).")

    # 6b. Inverse-transform multi-regression predictions if scaler exists
    scaler_path = os.path.join(run_dir, "target_scaler.json")
    if task == "multi_regression" and os.path.exists(scaler_path):
        with open(scaler_path) as f:
            scaler_info = json.load(f)
        means = np.array(scaler_info["mean"])
        scales = np.array(scaler_info["scale"])
        oof_preds = oof_preds * scales + means
        print("[INFO] Inverse-transformed predictions to original target scale")

    # 7. Metrics & Bootstrapping
    print("\n--- OOF Performance (95% CI) ---")

    # Select metrics based on task type
    if task == "regression":
        metrics_to_calc = ["neg_rmse", "neg_mae", "r2"]
    elif task == "multi_regression":
        metrics_to_calc = ["neg_rmse", "neg_mae", "r2"]
    elif task == "multiclass_classification":
        metrics_to_calc = ["balanced_accuracy", "f1_weighted"]
    else:  # binary_classification
        metrics_to_calc = ["roc_auc", "accuracy"]

    boot_alpha = config["shap"]["bootstrapping"]["alpha"]
    results = []

    if task == "multi_regression":
        # Per-target bootstrapped CIs
        y_vals = y.values
        for t_idx, col in enumerate(outcome_cols):
            for m_name in metrics_to_calc:
                fn = get_scoring_function(m_name)
                raw_score, raw_low, raw_high = compute_bootstrap_ci(
                    y_vals[:, t_idx], oof_preds[:, t_idx], fn,
                    n_boot=config["shap"]["bootstrapping"]["n_boot"],
                    alpha=boot_alpha
                )
                if m_name.startswith("neg_"):
                    score, low, high = -raw_score, -raw_high, -raw_low
                else:
                    score, low, high = raw_score, raw_low, raw_high
                disp_name = f"{m_name.replace('neg_', '').upper()}_{col}"
                print(f"  {disp_name}: {score:.4f} [{low:.4f}, {high:.4f}]")
                results.append({"metric": disp_name, "score": score, "ci_low": low, "ci_high": high})
    elif task == "multiclass_classification":
        # Multiclass: use argmax labels for hard metrics, proba for prob metrics
        y_vals = y.values
        preds_labels = np.argmax(oof_preds, axis=1)
        for m_name in metrics_to_calc:
            fn = get_scoring_function(m_name)
            raw_score, raw_low, raw_high = compute_bootstrap_ci(
                y_vals, preds_labels, fn,
                n_boot=config["shap"]["bootstrapping"]["n_boot"],
                alpha=boot_alpha
            )
            score, low, high = raw_score, raw_low, raw_high
            disp_name = m_name.upper()
            print(f"  {disp_name}: {score:.4f} [{low:.4f}, {high:.4f}]")
            results.append({"metric": disp_name, "score": score, "ci_low": low, "ci_high": high})
        # Also try AUC-OVR with probabilities
        try:
            from sklearn.metrics import roc_auc_score as _roc
            auc_fn = lambda yt, yp: _roc(yt, yp, multi_class='ovr', average='weighted')
            raw_score, raw_low, raw_high = compute_bootstrap_ci(
                y_vals, oof_preds, auc_fn,
                n_boot=config["shap"]["bootstrapping"]["n_boot"],
                alpha=boot_alpha
            )
            print(f"  ROC_AUC_OVR: {raw_score:.4f} [{raw_low:.4f}, {raw_high:.4f}]")
            results.append({"metric": "ROC_AUC_OVR", "score": raw_score, "ci_low": raw_low, "ci_high": raw_high})
        except Exception:
            print("  ROC_AUC_OVR: skipped (insufficient classes in bootstrap)")
    else:
        # regression or binary_classification
        y_vals = y.values
        for m_name in metrics_to_calc:
            fn = get_scoring_function(m_name)
            # Thresholding for classification hard metrics
            if task == "binary_classification" and m_name in ["accuracy", "f1"]:
                fn = lambda yt, yp, _fn=fn: _fn(yt, (yp > 0.5).astype(int))

            raw_score, raw_low, raw_high = compute_bootstrap_ci(
                y_vals, oof_preds, fn,
                n_boot=config["shap"]["bootstrapping"]["n_boot"],
                alpha=boot_alpha
            )

            if m_name.startswith("neg_"):
                score, low, high = -raw_score, -raw_high, -raw_low
            else:
                score, low, high = raw_score, raw_low, raw_high

            disp_name = m_name.replace("neg_", "").upper()
            print(f"  {disp_name:<5}: {score:.4f} [{low:.4f}, {high:.4f}]")
            results.append({"metric": disp_name, "score": score, "ci_low": low, "ci_high": high})

    pd.DataFrame(results).to_csv(os.path.join(run_dir, "performance_final.csv"), index=False)

    # 7b. Permutation Test (Null Model Comparison)
    print("\n--- Permutation Test (Model vs Chance) ---")

    n_perm = max(config["shap"]["bootstrapping"]["n_boot"], 1000)
    seed = config["execution"]["random_seed"]

    if task == "multi_regression":
        # Run permutation test per target
        for t_idx, col in enumerate(outcome_cols):
            perm_fns = [get_scoring_function(m) for m in metrics_to_calc]
            perm_results = compute_permutation_test(
                y.values[:, t_idx], oof_preds[:, t_idx],
                perm_fns, metrics_to_calc, n_perm, seed, run_dir
            )
            for _, row in perm_results.iterrows():
                sig = "*" if row["p_value"] < boot_alpha else ""
                print(f"  {col}/{row['metric']}: observed={row['observed']:.4f}, p={row['p_value']:.4f} {sig}")
    else:
        # For multiclass, use argmax labels for permutation test
        if task == "multiclass_classification":
            perm_preds = np.argmax(oof_preds, axis=1)
        else:
            perm_preds = oof_preds

        perm_fns = []
        perm_names = []
        for m_name in metrics_to_calc:
            fn = get_scoring_function(m_name)
            if task == "binary_classification" and m_name in ["accuracy", "f1"]:
                fn = lambda yt, yp, _fn=fn: _fn(yt, (yp > 0.5).astype(int))
            perm_fns.append(fn)
            perm_names.append(m_name)

        perm_results = compute_permutation_test(
            y_vals, perm_preds, perm_fns, perm_names, n_perm, seed, run_dir
        )

        for _, row in perm_results.iterrows():
            sig_marker = "*" if row["p_value"] < boot_alpha else ""
            print(f"  {row['metric']:<5}: observed={row['observed']:.4f}, "
                  f"null={row['null_mean']:.4f} +/- {row['null_std']:.4f}, "
                  f"p={row['p_value']:.4f} {sig_marker}")

    # 8. Save Predictions
    if task == "multiclass_classification":
        pred_df = pd.DataFrame({id_col: ids})
        pred_df["y_true"] = y.values
        for i, cl in enumerate(class_labels):
            pred_df[f"prob_{cl}"] = oof_preds[:, i]
    elif task == "multi_regression":
        pred_df = pd.DataFrame({id_col: ids})
        for i, col in enumerate(outcome_cols):
            pred_df[f"y_true_{col}"] = y.values[:, i]
            pred_df[f"y_pred_{col}"] = oof_preds[:, i]
    else:
        pred_df = pd.DataFrame({id_col: ids, "y_pred": oof_preds, "y_true": y.values})
    pred_df.to_csv(os.path.join(run_dir, "predictions_oof.csv"), index=False)

    # 9. Trigger SHAP (With OOF & Metadata Context)
    print("\n[INFO] Starting SHAP Pipeline (OOF Mode)...")

    shap_ctx = {
        "run_dir": run_dir,
        "config": config,
        "task": task,
        "feature_names": trained_features,
        "feature_names_shadow": shadow_features,
        "cat_features": nom_feats,
        "feature_types": feature_types,
        "X": X,
        "y": y,
        "X_raw": X_raw,
        "ids": ids,
        "class_labels": class_labels,
        "target_labels": target_labels,
    }

    run_shap_pipeline(shap_ctx)

if __name__ == "__main__":
    main()
