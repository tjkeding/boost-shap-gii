#!/usr/bin/env python3
"""Prediction Inference for boost-shap-gii (training set)"""

from __future__ import annotations

import argparse
import json
import os
import glob
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

from .utils import (
    _label_nominal,
    _validate_nominal_unseen,
    _load_sig_GII_from_shap_stats,
    load_config,
    load_dataframe,
    detect_task,
    is_regression,
    get_scoring_function,
    compute_bootstrap_ci,
    compute_permutation_test,
    validate_indiv_reports_config,
    load_transform_module,
    resolve_transform_path,
    coerce_ordinal_column,
)

from .shap_utils import run_shap_pipeline

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

        # Load shadow feature names if available (for noise calibration)
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
    df_raw = load_dataframe(data_path)

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

    _tx_config_path_early = os.path.join(run_dir, "transform_config.json")
    if os.path.exists(_tx_config_path_early):
        with open(_tx_config_path_early) as _f:
            _tx_info_early = json.load(_f)
        if _tx_info_early.get("active", False):
            _tx_req = _tx_info_early.get("required_cols", [])
            if _tx_req:
                _tx_missing = [c for c in _tx_req if c not in df_raw.columns]
                if _tx_missing:
                    raise KeyError(
                        f"[predict] transformations.required_cols references columns "
                        f"not in dataset: {_tx_missing}"
                    )
                pre_tx_len = len(df_raw)
                df_raw = df_raw.dropna(subset=_tx_req)
                tx_dropped = pre_tx_len - len(df_raw)
                if tx_dropped > 0:
                    print(f"[INFO] Dropped {tx_dropped} rows with missing "
                          f"transformations.required_cols value(s) "
                          f"(Mirroring train.py)")
                if len(df_raw) == 0:
                    raise ValueError(
                        "No data left after dropping rows with missing "
                        "transformations.required_cols (Mirroring train.py)."
                    )

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

    # A. Cast Nominals (String -> Category).
    # NaN -> "__NA__" (informative-missing sentinel); unseen level -> "__UNSEEN__" (OOD sentinel).
    # Levels are read from the training-time nominal codebook in feature_metadata.json.
    nominal_codebooks = feature_meta.get("nominal_codebooks", {})
    for c in nom_feats:
        # Distinguish NaN (informative-missing) from unseen-level (OOD) at predict-time.
        levels = set(nominal_codebooks.get(c, []))  # training-time codebook for column c
        if levels:
            _validate_nominal_unseen(df_raw[c], levels, column_name=c)
            X[c] = df_raw[c].apply(lambda v: _label_nominal(v, levels)).astype(str).astype("category")
        else:
            # Fallback: no codebook persisted (legacy training run); preserve prior behavior
            X[c] = df_raw[c].astype(object).fillna("__NA__").astype(str).astype("category")

    # B. Cast Continuous (Float32)
    for c in con_feats:
        X[c] = pd.to_numeric(df_raw[c], errors='coerce').astype("float32")

    # C. Cast Ordinals (Values -> Integer Codes using saved levels)
    for c in ord_feats:
        levels = feature_meta[c]['levels']
        X[c] = coerce_ordinal_column(df_raw[c], levels, c)

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

    # Load authoritative fold assignments from train.py output
    fold_assignments_path = os.path.join(run_dir, "fold_assignments.json")
    with open(fold_assignments_path) as f:
        fold_assignments = np.array(json.load(f))
    n_folds = int(fold_assignments.max()) + 1

    # Validate model file count against fold assignments
    if len(model_files) != n_folds:
        raise AssertionError(
            f"Found {len(model_files)} model file(s) in {run_dir} but fold_assignments.json "
            f"indicates {n_folds} fold(s). Re-run train.py or check output_dir."
        )

    cat_features_indices = nom_feats

    tx_config_path = os.path.join(run_dir, "transform_config.json")
    transform_module = None
    tx_info = None
    fold_shap_scale_factors = None
    if os.path.exists(tx_config_path):
        with open(tx_config_path) as f:
            tx_info = json.load(f)
        if tx_info.get("active", False):
            transform_module = load_transform_module(config)
            if transform_module is None:
                raise ValueError(
                    "transform_config.json indicates active transforms but "
                    "load_transform_module returned None. Verify the config "
                    "YAML contains a valid transformations block matching "
                    "the training config."
                )
            required_cols = tx_info.get("required_cols", [])
            if required_cols:
                missing = [c for c in required_cols if c not in df_raw.columns]
                if missing:
                    raise ValueError(
                        f"[predict] transformations.required_cols missing from "
                        f"dataframe: {missing}"
                    )
            if tx_info.get("back_transform_shap", False):
                if "fold_shap_scale_factors" not in tx_info:
                    raise ValueError(
                        "transform_config.json uses the legacy single-scalar "
                        "shap_scale_factor format. Re-run train.py with the current "
                        "version to regenerate per-fold scale factors."
                    )
                fold_shap_scale_factors = tx_info["fold_shap_scale_factors"]
            print(f"[INFO] Transformations active (file: {tx_info['file']})")
            ftm_path = os.path.join(run_dir, "fold_transform_metadata.json")
            with open(ftm_path) as f:
                _fold_transform_meta = json.load(f)
            if not is_regression(task):
                raise ValueError(
                    f"Outcome transformations are only supported for regression tasks "
                    f"(regression, multi_regression). Detected task: '{task}'. "
                    f"The input_transform/output_transform API contract assumes a "
                    f"continuous, invertible outcome transformation."
                )

    for fold_idx in range(n_folds):
        val_idx = np.where(fold_assignments == fold_idx)[0]
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
        if transform_module is not None:
            preds_bt = transform_module.output_transform(
                np.asarray(preds, dtype=float), _fold_transform_meta[fold_idx],
                tx_info.get("params", {}),
                df_raw=df_raw, row_indices=val_idx
            )
            oof_preds[val_idx] = preds_bt
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
    if task == "multi_regression" and os.path.exists(scaler_path) and transform_module is None:
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

    cv_strategy = config["modeling"].get("cv_strategy", "uniform")
    group_column = config["modeling"].get("group_column")
    if cv_strategy == "group" and group_column is not None and group_column in df_raw.columns:
        shap_ctx["groups"] = df_raw[group_column].values
        shap_ctx["cv_strategy"] = cv_strategy

    if fold_shap_scale_factors is not None:
        shap_ctx["fold_shap_scale_factors"] = fold_shap_scale_factors

    run_shap_pipeline(shap_ctx)

    # --- Per-individual SHAP reports (indiv_reports) ---
    validate_indiv_reports_config(config)

    nboot_indiv = int(config["shap"]["indiv_ci_nboot"])
    if nboot_indiv > 0:
        from .indiv_reports import orchestrate_bootstrap_cache, generate_indiv_reports

        n_jobs = config["execution"].get("n_jobs", 1)

        cluster_ids_indiv = None
        if cv_strategy == "group" and group_column is not None and group_column in df_raw.columns:
            cluster_ids_indiv = df_raw[group_column].values

        # Resolve transform module path for per-bootstrap transforms
        tx_module_path = None
        _outcome_col_boot = None
        if transform_module is not None:
            tx_module_path = resolve_transform_path(config)
            _outcome_col_boot = outcome_cols[0] if len(outcome_cols) == 1 else outcome_cols

        # 1. Build cache at run_dir/bootstrap_refits/
        cache_summary = orchestrate_bootstrap_cache(
            run_dir=run_dir,
            X_train=X,
            y_train=y,
            task=task,
            outcome_cols=outcome_cols,
            nom_feats=nom_feats,
            config=config,
            n_jobs=n_jobs,
            random_seed=config["execution"]["random_seed"],
            cluster_ids=cluster_ids_indiv,
            transform_module_path=tx_module_path,
            tx_params=tx_info.get("params", {}) if tx_info is not None else None,
            df_raw=df_raw if transform_module is not None else None,
            outcome_col=_outcome_col_boot,
            back_transform_shap=tx_info.get("back_transform_shap", False) if tx_info else False,
            is_affine=tx_info.get("is_affine", False) if tx_info else False,
        )
        print(
            f"[INFO] Bootstrap cache built: B={cache_summary['B']} iterations "
            f"across K={cache_summary['K']} folds "
            f"({cache_summary['total_refits']} total refits)."
        )

        # 2. Emit training-individual indiv_reports
        sig_GII_main, sig_GII_interaction = _load_sig_GII_from_shap_stats(run_dir)
        generate_indiv_reports(
            run_dir=run_dir,
            train_dir=run_dir,  # in predict.py, train_dir == run_dir
            X_target=X,
            ids_target=ids,
            X_train=X,
            y_target=y,
            task=task,
            outcome_cols=outcome_cols,
            nom_feats=nom_feats,
            config=config,
            mode="training",
            sig_GII_main=sig_GII_main,
            sig_GII_interaction=sig_GII_interaction,
            cluster_ids=cluster_ids_indiv,
            transform_module=transform_module,
            fold_transform_metadata=_fold_transform_meta if transform_module is not None else None,
            tx_params=tx_info.get("params", {}) if tx_info is not None else None,
            df_raw=df_raw,
            fold_shap_scale_factors=fold_shap_scale_factors,
        )
        print(f"[INFO] Training indiv_reports/ emitted to {run_dir}.")
    else:
        print("[INFO] shap.indiv_ci_nboot=0; skipping per-individual SHAP reports.")

if __name__ == "__main__":
    main()
