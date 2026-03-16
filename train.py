#!/usr/bin/env python3
"""Model Tuning and Training for boost-shap-gii pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, accuracy_score,
)

from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import optuna
from optuna.samplers import TPESampler

from utils import (
    _normalize_quotes,
    load_config,
    save_json_atomic,
    detect_task,
    is_classification,
    is_regression,
    get_cv_splitter,
    get_scoring_function,
    fill_config_defaults,
)

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# -----------------------------------------------------------------------------
# Custom Classes
# -----------------------------------------------------------------------------

class FeatureSelector:
    """Parse the YAML 'features' block and classify columns with conflict detection.

    Processes `continuous_groups`, `ordinal_groups`, and `nominal_groups` from the
    config. Raises `ValueError` if any column is claimed by more than one feature type.
    Column ordering in `final_columns` is deterministic (sorted), ensuring consistent
    train/predict/infer alignment.
    """
    def __init__(self, config_features):
        self.config = config_features
        self.selected_features = {}  # {col_name: feature_type}
        self.feature_metadata = {}   # {col_name: {levels: [], ...}}

    def _match(self, col_name, pattern, mode):
        """Return True if col_name matches pattern under the given mode."""
        if mode == "exact":
            return col_name == pattern
        elif mode == "prefix":
            return col_name.startswith(pattern)
        elif mode == "suffix":
            return col_name.endswith(pattern)
        # Default to substring
        return pattern in col_name

    def fit(self, all_columns):
        """Scan all_columns against the config and assign feature types.

        Parameters
        ----------
        all_columns : list[str]
            All candidate column names from the input DataFrame (outcomes excluded).

        Returns
        -------
        list[str]
            Sorted list of selected feature names.

        Raises
        ------
        ValueError
            If any column is claimed by more than one feature type.
        """
        # We process groups in this order, but we track claims globally to detect conflicts
        groups_map = {
            'continuous': self.config.get('continuous_groups', []),
            'ordinal': self.config.get('ordinal_groups', []),
            'nominal': self.config.get('nominal_groups', [])
        }

        # 1. COLLECT ALL CLAIMS
        # Structure: {col_name: [(feature_type, metadata_levels), ...]}
        raw_claims = {col: [] for col in all_columns}

        for f_type, group_defs in groups_map.items():
            if not group_defs: continue # Skip if empty in YAML

            for grp in group_defs:
                pattern = grp['pattern']
                mode = grp.get('match_mode', 'substring')
                exclusions = grp.get('exclude', [])

                # Scan all columns
                for col in all_columns:
                    # A. Check Match
                    if not self._match(col, pattern, mode):
                        continue

                    # B. Check Exclusions
                    if any(exc in col for exc in exclusions):
                        continue

                    # C. Register Claim
                    # We store the type and the levels (if ordinal) for later assignment
                    levels = grp.get('levels', []) if f_type == 'ordinal' else []
                    raw_claims[col].append((f_type, levels))

        # 2. VALIDATE CONFLICTS & ASSIGN
        conflicts = []

        for col, matches in raw_claims.items():
            if not matches:
                continue

            # Check for ambiguous types (e.g., claimed by both 'ordinal' and 'continuous')
            unique_types = set(m[0] for m in matches)

            if len(unique_types) > 1:
                conflicts.append(f"Column '{col}' claimed by distinct types: {unique_types}")
                continue

            # If valid, assign based on the unanimous type
            f_type = list(unique_types)[0]
            self.selected_features[col] = f_type

            # For ordinals, we grab the levels from the first match.
            # (Assuming sub-patterns within 'ordinal' are consistent if they overlap)
            if f_type == 'ordinal':
                # Matches is a list of tuples (type, levels). We take levels from the first match.
                self.feature_metadata[col] = {'levels': matches[0][1]}

        if conflicts:
            raise ValueError("Ambiguous Feature Definitions:\n" + "\n".join(conflicts))

        # Filter out columns that weren't selected at all
        self.final_columns = sorted(self.selected_features.keys())

        print(f"[INFO] Feature Selection Complete.")
        print(f"   - Total Columns Scanned: {len(all_columns)}")
        print(f"   - Features Selected:     {len(self.final_columns)}")
        return self.final_columns

    def get_feature_lists(self):
        """Return sorted (continuous, ordinal, nominal) feature name lists for CatBoost."""
        con = [c for c, t in self.selected_features.items() if t == 'continuous']
        ord_ = [c for c, t in self.selected_features.items() if t == 'ordinal']
        nom = [c for c, t in self.selected_features.items() if t == 'nominal']
        return sorted(con), sorted(ord_), sorted(nom)

# -----------------------------------------------------------------------------
# 1. Helper Utilities
# -----------------------------------------------------------------------------

def report_missingness(df: pd.DataFrame, features: list, outcome: str, run_dir: str):
    """Compute and save per-feature missingness rates to `missingness_report.csv`.

    Features with > 10% missing rate are individually flagged with a WARNING.
    """
    cols = [c for c in features + [outcome] if c in df.columns]
    miss_rates = df[cols].isnull().mean().sort_values(ascending=False)
    miss_counts = df[cols].isnull().sum().reindex(miss_rates.index)

    report = pd.DataFrame({
        "feature": miss_rates.index,
        "missing_rate": miss_rates.values,
        "missing_count": miss_counts.values,
        "total_count": len(df)
    })
    report.to_csv(os.path.join(run_dir, "missingness_report.csv"), index=False)

    n_any = (miss_rates > 0).sum()
    max_rate = miss_rates.max()
    print(f"[INFO] Missingness Report:")
    print(f"   - Features with any missing: {n_any}/{len(cols)}")
    print(f"   - Max missing rate: {max_rate:.1%}")
    if max_rate > 0.1:
        high = miss_rates[miss_rates > 0.1]
        for feat in high.index:
            print(f"   - WARNING: '{feat}' is {high[feat]:.1%} missing")


# -----------------------------------------------------------------------------
# 2. Core Optimization Logic
# -----------------------------------------------------------------------------

def run_optuna_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    cat_features: List[str],
    task: str,
    config: Dict,
    n_jobs: int,
    fold_idx: int = 0,
) -> Tuple[Dict[str, Any], int]:
    """Run Optuna TPE hyperparameter tuning on the inner CV.

    The inner CV seed is offset by `fold_idx + 1` relative to the outer seed,
    ensuring inner and outer folds use distinct random split patterns. The TPESampler
    seed is set identically to the inner CV seed for full consistency.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features for the current outer fold.
    y_train : pd.Series or pd.DataFrame
        Training labels (Series for single-output, DataFrame for multi_regression).
    cat_features : list[str]
        Nominal (categorical) feature names for CatBoost.
    task : str
        Task type (one of VALID_TASK_TYPES).
    config : dict
        Full pipeline config (read from YAML).
    n_jobs : int
        CPU threads for CatBoost.
    fold_idx : int
        Outer fold index (0-based). Used to offset the inner CV seed.

    Returns
    -------
    best_params : dict
        Winning hyperparameter set from Optuna (excludes `iterations`).
    tuned_iterations : int
        Mean `best_iteration_` across inner CV folds for the winning trial + 1.
    """
    tuning_cfg = config["modeling"]["tuning"]
    n_trials = tuning_cfg["n_iter"]
    scoring_name = tuning_cfg["scoring"]
    score_fn = get_scoring_function(scoring_name)
    space = tuning_cfg["search_space"]

    # Explicit Loss Function from YAML (No Hardcoding)
    loss_type = config["modeling"]["loss_function"]

    early_stopping = int(tuning_cfg["early_stopping_rounds"])

    # Define Inner CV
    inner_cv_folds = tuning_cfg["inner_cv_folds"]
    seed = config["execution"]["random_seed"]

    # Offset inner CV seed by fold_idx + 1 so inner and outer folds use distinct seeds.
    inner_seed = seed + fold_idx + 1
    inner_cv = KFold(n_splits=inner_cv_folds, shuffle=True, random_state=inner_seed)
    # For stratified splitting, need 1D y with limited unique values
    y_for_stratify = y_train if isinstance(y_train, pd.Series) else y_train.iloc[:, 0]
    if is_classification(task) and y_for_stratify.nunique() < 20:
        inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=inner_seed)

    def objective(trial):
        # 1. Parse YAML Search Space dynamically
        params = {
            "thread_count": n_jobs,
            "verbose": False,
            "allow_writing_files": False,
            "random_seed": seed
        }

        # Map YAML entries to Optuna suggestions
        for param, bounds in space.items():
            if isinstance(bounds, list):
                 params[param] = trial.suggest_categorical(param, bounds)
                 continue

            low = bounds.get("low")
            high = bounds.get("high")
            log = bounds.get("log", False)

            if isinstance(low, int) and isinstance(high, int):
                params[param] = trial.suggest_int(param, low, high, log=log)
            else:
                params[param] = trial.suggest_float(param, low, high, log=log)

        # 2. Inner CV Loop
        scores = []
        best_iters = []
        for t_idx, v_idx in inner_cv.split(X_train, y_for_stratify):
            X_t, X_v = X_train.iloc[t_idx], X_train.iloc[v_idx]
            y_t, y_v = y_train.iloc[t_idx], y_train.iloc[v_idx]

            pool_t = Pool(X_t, y_t, cat_features=cat_features)
            pool_v = Pool(X_v, y_v, cat_features=cat_features)

            if is_regression(task):
                model = CatBoostRegressor(**params, loss_function=loss_type)
            else:
                model = CatBoostClassifier(**params, loss_function=loss_type)

            model.fit(pool_t, eval_set=pool_v, early_stopping_rounds=early_stopping, verbose=False)
            best_iters.append(model.best_iteration_)

            # Task-aware prediction for scoring
            if task == "multiclass_classification":
                if scoring_name in ["roc_auc_ovr", "log_loss"]:
                    preds = model.predict_proba(X_v)
                else:
                    preds = np.argmax(model.predict_proba(X_v), axis=1)
            elif task == "binary_classification":
                if scoring_name in ["roc_auc", "log_loss"]:
                    preds = model.predict_proba(X_v)[:, 1]
                else:
                    preds = (model.predict_proba(X_v)[:, 1] > 0.5).astype(int)
            elif task == "multi_regression":
                preds = model.predict(X_v)
            else:  # regression
                preds = model.predict(X_v)

            scores.append(score_fn(y_v, preds))

        trial.set_user_attr("mean_best_iter", int(np.mean(best_iters)) + 1)
        return np.mean(scores)

    # Run Study — use inner_seed for full consistency with inner CV
    sampler = TPESampler(seed=inner_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    tuned_iters = study.best_trial.user_attrs["mean_best_iter"]
    return study.best_params, tuned_iters


# -----------------------------------------------------------------------------
# 3. Main Training Pipeline
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean Train Pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    # 1. Setup
    config = load_config(args.config)
    run_dir = config["paths"]["output_dir"]
    os.makedirs(run_dir, exist_ok=True)

    # 2. Load Data
    data_path = config["paths"]["input_data"]
    print(f"[INFO] Loading data from {data_path}")

    # Check extension
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

    outcome_cfg = config["modeling"]["outcome"]
    # multi_regression uses a list of outcome columns; all others use a single string
    if isinstance(outcome_cfg, list):
        outcome_cols = outcome_cfg
        for oc in outcome_cols:
            if oc not in df_raw.columns:
                raise KeyError(f"Outcome '{oc}' not found in dataset columns.")
    else:
        outcome_cols = [outcome_cfg]
        if outcome_cfg not in df_raw.columns:
            raise KeyError(f"Outcome '{outcome_cfg}' not found in dataset columns.")

    # Drop rows with missing target values (any outcome column)
    initial_len = len(df_raw)
    df_raw = df_raw.dropna(subset=outcome_cols)
    dropped = initial_len - len(df_raw)
    if dropped > 0:
        print(f"[INFO] Dropped {dropped} rows with missing outcome(s)")

    if len(df_raw) == 0:
        raise ValueError("No data left after dropping rows with missing target.")

    # 3. Feature Selection (THE NEW ENGINE)
    print("[INFO] Scanning and Selecting features based on YAML...")
    selector = FeatureSelector(config['features'])

    # Exclude outcome(s) from scan
    candidate_cols = [c for c in df_raw.columns if c not in outcome_cols]
    final_cols = selector.fit(candidate_cols)

    if len(final_cols) == 0:
        raise ValueError(
            "No features matched any pattern in config. "
            "Review features.continuous_groups / ordinal_groups / nominal_groups."
        )

    # 3b. Fill config defaults (now that we know n and p)
    n_rows = len(df_raw)
    n_features = len(final_cols)
    config, filled_defaults = fill_config_defaults(config, n_rows, n_features)

    if filled_defaults:
        print(f"[INFO] Auto-filled {len(filled_defaults)} config defaults:")
        for path, label in filled_defaults:
            print(f"  [DEFAULT] {path} = {label}")
    else:
        print("[INFO] All config fields provided by user (no defaults applied).")

    # Save fully-resolved config (with all defaults applied)
    with open(os.path.join(run_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(config, f)

    n_jobs = config["execution"]["n_jobs"]
    print(f"[INFO] Using {n_jobs} CPU cores for parallel processing.")

    con_feats, ord_feats, nom_feats = selector.get_feature_lists()

    # Missingness Characterization (saved before any imputation/type casting)
    report_missingness(df_raw, final_cols, outcome_cols[0], run_dir)

    # 3c. Drop all-missing columns
    all_missing = [c for c in final_cols if df_raw[c].isna().all()]
    if all_missing:
        print(f"[WARNING] Dropping {len(all_missing)} all-missing column(s): {all_missing}")
        final_cols = [c for c in final_cols if c not in all_missing]
        con_feats = [c for c in con_feats if c not in all_missing]
        ord_feats = [c for c in ord_feats if c not in all_missing]
        nom_feats = [c for c in nom_feats if c not in all_missing]

    # 4. Type Enforcement & Preprocessing
    X = df_raw[final_cols].copy()
    # For multi_regression, y is a DataFrame; otherwise a Series
    if len(outcome_cols) > 1:
        y = df_raw[outcome_cols].copy()
    else:
        y = df_raw[outcome_cols[0]].copy()

    # A. Force Nominal to String -> Category.
    # NaN is filled with the literal string "__NA__" before encoding. CatBoost treats
    # "__NA__" as a valid category level, allowing the model to learn whether missingness
    # is predictive. This is an implicit informativeness assumption — see README.
    for c in nom_feats:
        X[c] = X[c].fillna("__NA__").astype(str).astype("category")

    # B. Force Continuous to Float
    for c in con_feats:
        X[c] = pd.to_numeric(X[c], errors='coerce').astype("float32")

    # C. Force Ordinal (Map levels to integers)
    # This is critical so CatBoost treats them as numeric/ordinal, not string
    for c in ord_feats:
        levels = [_normalize_quotes(l) for l in selector.feature_metadata[c]['levels']]
        X[c] = X[c].map(lambda v: _normalize_quotes(v) if isinstance(v, str) else v)

        # Verify all data values exist in levels
        unique_vals = X[c].dropna().unique()
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
            obs_vals = X[c].dropna()
            n_unknown_obs = sum(v not in levels for v in obs_vals)
            obs_frac = n_unknown_obs / len(obs_vals) if len(obs_vals) > 0 else 0.0
            if obs_frac > 0.10:
                print(
                    f"[WARNING] Feature '{c}': {obs_frac:.1%} of non-missing observations "
                    f"({n_unknown_obs}/{len(obs_vals)}) have values not in YAML levels. "
                    f"This may indicate systematic data quality issues."
                )

        # Create ordered categorical then code
        cat_type = pd.CategoricalDtype(categories=levels, ordered=True)
        X[c] = X[c].astype(cat_type).cat.codes.astype("Int64")
        # Note: missing values become -1 in codes, we mask them back to NaN/Int64 NA
        X.loc[X[c] == -1, c] = pd.NA

    # CatBoost expects Nominal columns to be listed in `cat_features`
    # It handles Float and Int automatically.
    # We pass `nom_feats` as the categorical list.
    cat_features_indices = nom_feats

    print(f"[INFO] Feature Matrix: {X.shape[0]} rows x {X.shape[1]} columns")
    print(f"[INFO]   - Continuous Features: {len(con_feats)}")
    print(f"[INFO]   - Ordinal Features:    {len(ord_feats)}")
    print(f"[INFO]   - Nominal Features:    {len(nom_feats)}")

    # Determine Task
    task = detect_task(config)
    loss_func = config["modeling"]["loss_function"]

    if is_classification(task):
        y_class = y if isinstance(y, pd.Series) else y.iloc[:, 0]
        class_counts = y_class.value_counts()
        n_folds = config["modeling"]["cv_folds"]
        min_count = class_counts.min()
        if min_count < n_folds:
            raise ValueError(
                f"Minority class has {min_count} samples but {n_folds}-fold CV "
                f"requires at least {n_folds} per class. Reduce cv_folds or "
                f"resample data."
            )

    # Auto-scale multi-regression targets to common scale
    target_scaler = None
    if task == "multi_regression":
        from sklearn.preprocessing import StandardScaler
        target_scaler = StandardScaler()
        y_values = target_scaler.fit_transform(y.values)
        y = pd.DataFrame(y_values, columns=y.columns, index=y.index)
        save_json_atomic({
            "mean": target_scaler.mean_.tolist(),
            "scale": target_scaler.scale_.tolist(),
            "columns": outcome_cols,
        }, os.path.join(run_dir, "target_scaler.json"))
        print(f"[INFO] Auto-scaled {len(outcome_cols)} targets (z-score standardization)")

    print(f"[INFO] Task: {task.upper()} | Loss: {loss_func}")

    # Save Metadata for Predict.py / Shap Utils
    # 1. Feature Names list
    with open(os.path.join(run_dir, "feature_names.json"), "w") as f:
        json.dump(final_cols, f)

    # 2. Feature Type Map (Name -> Type)
    with open(os.path.join(run_dir, "feature_types.json"), "w") as f:
        json.dump(selector.selected_features, f, indent=2)

    # 3. Full Feature Metadata (levels, etc)
    with open(os.path.join(run_dir, "feature_metadata.json"), "w") as f:
        # Convert set/arrays to list for JSON serialization
        clean_meta = {}
        for k, v in selector.feature_metadata.items():
            clean_meta[k] = v
        json.dump(clean_meta, f, indent=2)

    # 4. Shadow Feature Names (Real + Permuted) for SHAP Utils
    shadow_names = final_cols + [f"shadow_{c}" for c in final_cols]
    with open(os.path.join(run_dir, "feature_names_shadow.json"), "w") as f:
        json.dump(shadow_names, f)

    # Save clean matrix for reproducibility
    X.to_parquet(os.path.join(run_dir, "train_matrix.parquet"))

    # Save class/target labels for downstream modules
    if task == "multiclass_classification":
        class_labels = sorted(y.unique().tolist())
        save_json_atomic(class_labels, os.path.join(run_dir, "class_labels.json"))
        print(f"[INFO] Classes: {class_labels}")
    elif task == "multi_regression":
        save_json_atomic(outcome_cols, os.path.join(run_dir, "target_labels.json"))
        print(f"[INFO] Targets: {outcome_cols}")

    # 5. Nested Cross-Validation Loop
    # For multi_regression, get_cv_splitter needs a 1D Series — use first target
    y_for_split = y if isinstance(y, pd.Series) else y.iloc[:, 0]
    splitter = get_cv_splitter(config, y_for_split)

    # OOF storage depends on task type
    if task == "multiclass_classification":
        n_classes = len(class_labels)
        oof_preds = pd.DataFrame(
            np.nan, index=X.index,
            columns=[f"prob_{c}" for c in class_labels]
        )
    elif task == "multi_regression":
        oof_preds = pd.DataFrame(
            np.nan, index=X.index, columns=outcome_cols
        )
    else:
        oof_preds = pd.Series(index=X.index, dtype=float)

    fold_metrics = []

    print(f"[INFO] Starting {splitter.get_n_splits()}-Fold Nested CV...")

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y_for_split)):
        print(f"\n--- Fold {fold_idx + 1} ---")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # --- PHASE 1: CLEAN TRAINING ---
        # A. Tune
        print("  > Tuning hyperparameters (Phase 1: Clean)...")
        # Note: CatBoost handles Ordinals as numeric if we don't list them in cat_features.
        # We ONLY pass nominals to cat_features argument.
        best_params, tuned_iters = run_optuna_tuning(X_train, y_train, nom_feats, task, config, n_jobs, fold_idx=fold_idx)
        print(f"  > Best Params: {best_params}")
        print(f"  > Tuned Iterations (inner CV mean): {tuned_iters}")

        # Inject global configs into best params
        best_params["thread_count"] = n_jobs
        best_params["loss_function"] = loss_func
        best_params["iterations"] = tuned_iters

        # B. Train (iteration count set by inner CV; no outer early stopping)
        print("  > Fitting final clean model...")
        pool_train = Pool(X_train, y_train, cat_features=nom_feats)

        if is_regression(task):
            model = CatBoostRegressor(**best_params)
        else:
            model = CatBoostClassifier(**best_params)

        model.fit(pool_train, verbose=False)

        # C. Predict & Store OOF
        if task == "multiclass_classification":
            proba = model.predict_proba(X_val)
            oof_preds.iloc[val_idx] = proba
            preds_labels = np.argmax(proba, axis=1)
        elif task == "binary_classification":
            preds = model.predict_proba(X_val)[:, 1]
            oof_preds.iloc[val_idx] = preds
        elif task == "multi_regression":
            preds = model.predict(X_val)
            if target_scaler is not None:
                preds = target_scaler.inverse_transform(
                    preds.reshape(1, -1) if preds.ndim == 1 else preds
                )
            oof_preds.iloc[val_idx] = preds
        else:  # regression
            preds = model.predict(X_val)
            oof_preds.iloc[val_idx] = preds

        # D. Log Metrics
        y_true_fold = y_val.values
        if task == "multi_regression" and target_scaler is not None:
            y_true_fold = target_scaler.inverse_transform(
                y_true_fold.reshape(1, -1) if y_true_fold.ndim == 1 else y_true_fold
            )
        if task == "regression":
            rmse = np.sqrt(mean_squared_error(y_true_fold, preds))
            mae = mean_absolute_error(y_true_fold, preds)
            r2 = r2_score(y_true_fold, preds)
            metrics = {"rmse": rmse, "mae": mae, "r2": r2}
            print(f"  > Scores: RMSE={rmse:.3f}, R2={r2:.3f}")
        elif task == "multi_regression":
            # Per-target RMSE + overall mean
            metrics = {}
            for i, col in enumerate(outcome_cols):
                t_rmse = np.sqrt(mean_squared_error(y_true_fold[:, i], preds[:, i]))
                t_r2 = r2_score(y_true_fold[:, i], preds[:, i])
                metrics[f"rmse_{col}"] = t_rmse
                metrics[f"r2_{col}"] = t_r2
            metrics["rmse_mean"] = np.mean([metrics[f"rmse_{c}"] for c in outcome_cols])
            print(f"  > Scores: Mean RMSE={metrics['rmse_mean']:.3f}")
        elif task == "multiclass_classification":
            from sklearn.metrics import balanced_accuracy_score as bas
            acc = bas(y_true_fold, preds_labels)
            metrics = {"balanced_accuracy": acc}
            try:
                auc = roc_auc_score(y_true_fold, proba, multi_class='ovr', average='weighted')
                metrics["roc_auc_ovr"] = auc
                print(f"  > Scores: Balanced Acc={acc:.3f}, AUC-OVR={auc:.3f}")
            except ValueError:
                print(f"  > Scores: Balanced Acc={acc:.3f} (AUC-OVR skipped)")
        else:  # binary_classification
            try:
                auc = roc_auc_score(y_true_fold, preds)
                print(f"  > Scores: AUC={auc:.3f}")
                metrics = {"auc": auc}
            except ValueError:
                acc = accuracy_score(y_true_fold, (np.array(preds) > 0.5).astype(int))
                print(f"  > Scores: ACC={acc:.3f} (AUC Failed)")
                metrics = {"acc": acc}

        fold_metrics.append(metrics)

        # E. Save Clean Model
        model_path = os.path.join(run_dir, f"model_fold_{fold_idx}.cbm")
        model.save_model(model_path)

        # --- PHASE 2: SHADOW TRAINING (NOISE CALIBRATION) ---
        print("  > Training Shadow Model (Phase 2: Calibration)...")

        # 1. Generate Shadow Data (Permutation)
        X_train_shadow = X_train.copy()
        X_val_shadow = X_val.copy()

        # Permute columns independently
        rng = np.random.default_rng(config["execution"]["random_seed"] + fold_idx)

        for c in X_train_shadow.columns:
            X_train_shadow[c] = rng.permutation(X_train_shadow[c].values)
        for c in X_val_shadow.columns:
            X_val_shadow[c] = rng.permutation(X_val_shadow[c].values)

        # Rename columns
        X_train_shadow.columns = [f"shadow_{c}" for c in X_train_shadow.columns]
        X_val_shadow.columns = [f"shadow_{c}" for c in X_val_shadow.columns]

        # 2. Concatenate (Real + Shadow)
        X_train_full = pd.concat([X_train, X_train_shadow], axis=1)
        X_val_full = pd.concat([X_val, X_val_shadow], axis=1)

        # 3. Define Full Categoricals
        # CatBoost needs the names of all categorical columns (Original + Shadow)
        shadow_nom_feats = [f"shadow_{c}" for c in nom_feats]
        full_cat_features = nom_feats + shadow_nom_feats

        # 4. Train Shadow Model with early stopping on outer validation fold.
        # Use tuned_iters * 2 as ceiling — the shadow model trains on 2p features
        # and requires more iterations to converge. Early stopping on X_val_full
        # is data-adaptive and introduces no leakage: shadow outputs are never
        # used for predictive evaluation, only for SHAP noise calibration.
        pool_train_full = Pool(X_train_full, y_train, cat_features=full_cat_features)
        pool_val_full = Pool(X_val_full, y_val, cat_features=full_cat_features)

        shadow_params = best_params.copy()
        shadow_params["iterations"] = tuned_iters * 2  # ceiling, not fixed count

        if is_regression(task):
            model_shadow = CatBoostRegressor(**shadow_params)
        else:
            model_shadow = CatBoostClassifier(**shadow_params)

        model_shadow.fit(
            pool_train_full,
            eval_set=pool_val_full,
            early_stopping_rounds=int(config["modeling"]["tuning"]["early_stopping_rounds"]),
            verbose=False,
        )

        # 5. Save Shadow Model
        shadow_model_path = os.path.join(run_dir, f"shadow_model_fold_{fold_idx}.cbm")
        model_shadow.save_model(shadow_model_path)

    # 6. Finalize
    print("\n[INFO] CV Complete. Saving Global Artifacts...")

    # We include ID if available in raw, else just index
    id_col = "id" if "id" in df_raw.columns else "index"
    ids = df_raw[id_col] if "id" in df_raw.columns else df_raw.index

    # Build OOF output based on task type
    if task == "multiclass_classification":
        oof_df = pd.DataFrame({id_col: ids})
        oof_df["y_true"] = y.values
        for col in oof_preds.columns:
            oof_df[col] = oof_preds[col].values
    elif task == "multi_regression":
        oof_df = pd.DataFrame({id_col: ids})
        for col in outcome_cols:
            oof_df[f"y_true_{col}"] = y[col].values
            oof_df[f"y_pred_{col}"] = oof_preds[col].values
    else:
        oof_df = pd.DataFrame({id_col: ids, "y_true": y, "y_pred": oof_preds})

    oof_df.to_csv(os.path.join(run_dir, "full_oof_predictions.csv"), index=False)

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.loc["mean"] = metrics_df.mean()
    metrics_df.to_csv(os.path.join(run_dir, "metrics_oof.csv"))

    save_json_atomic(fold_metrics, os.path.join(run_dir, "metrics_oof.json"))

    # Save task type for downstream modules
    save_json_atomic({"task_type": task}, os.path.join(run_dir, "task_info.json"))

    print(f"[SUCCESS] Training finished. Artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
