#!/usr/bin/env python3
"""Model Tuning and Training for boost-shap-gii pipeline."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, accuracy_score,
)

from scipy import stats as sp_stats

from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import optuna
from optuna.samplers import TPESampler

from .utils import (
    _block_permute_shadow,
    load_config,
    load_dataframe,
    save_json_atomic,
    detect_task,
    is_classification,
    is_regression,
    get_cv_splitter,
    get_scoring_function,
    fill_config_defaults,
    validate_cv_config,
    validate_bootstrap_config,
    load_transform_module,
    validate_transform_config,
    coerce_ordinal_column,
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


def _diagnose_outcome_distribution(y_series: pd.Series, col_name: str) -> None:
    """Emit advisory warnings for outcome distributions that may degrade RMSE loss.

    Computes zero-inflation rate, skewness, and excess kurtosis on the non-missing
    outcome values. If any threshold is exceeded, prints a ``[WARNING]`` with the
    diagnostic values, the recommended MAD-based Huber delta, and a reference to
    ``INPUT_SPECIFICATION.md``.

    Thresholds and their literature basis:
    - Zero-inflation >= 15%   (Olsen & Schafer, 2001; Tooze et al., 2002)
    - |Skewness| >= 2.0       (Groeneveld & Meeden, 1984; Kim, 2013)
    - Excess kurtosis >= 5.0  (Joanes & Gill, 1998, adjusted downward from 7.0 for
      gradient-boosting tail sensitivity; Kim, 2013)

    The recommended Huber delta uses MAD-based scale estimation (Huber, 1981;
    Maronna et al., 2006). When MAD = 0 (common with >50% zero-inflation),
    falls back to IQR / 1.3489, then to SD as a last resort.

    This function is advisory only -- the pipeline continues with whatever loss
    the user specified. Classification tasks should NOT call this function.

    Parameters
    ----------
    y_series : pd.Series
        Outcome values (may contain NaN; dropped internally).
    col_name : str
        Outcome column name for log messages.
    """
    y_vals = y_series.dropna().to_numpy(dtype=np.float64)
    n = len(y_vals)
    if n < 10:
        return  # Too few observations for meaningful diagnostics

    # ---- Compute diagnostics ----
    zero_frac = np.mean(y_vals == 0)
    skewness = float(sp_stats.skew(y_vals, bias=False))
    excess_kurt = float(sp_stats.kurtosis(y_vals, fisher=True, bias=False))

    # Huber delta via MAD scale estimate; IQR/1.3489 fallback when MAD=0 (Maronna et al., 2006).
    mad = float(np.median(np.abs(y_vals - np.median(y_vals))))
    if mad > 0:
        scale_est = 1.4826 * mad
        scale_method = "MAD"
    else:
        iqr = float(np.percentile(y_vals, 75) - np.percentile(y_vals, 25))
        scale_est = iqr / 1.3489 if iqr > 0 else float(np.std(y_vals))
        scale_method = "IQR" if iqr > 0 else "SD"
    delta = 1.345 * scale_est

    # ---- Check thresholds ----
    zero_flag = zero_frac >= 0.15
    skew_flag = abs(skewness) >= 2.0
    kurt_flag = excess_kurt >= 5.0

    if not (zero_flag or skew_flag or kurt_flag):
        return  # No distributional concerns

    # ---- Build warning message ----
    header = (
        f"[WARNING] Outcome distribution diagnostic for '{col_name}' "
        f"(n={n}):"
    )
    details = []
    if zero_flag:
        details.append(
            f"  Zero-inflation: {zero_frac:.1%} of observations are zero "
            f"(threshold: 15%)"
        )
    if skew_flag:
        direction = "right" if skewness > 0 else "left"
        details.append(
            f"  Skewness: {skewness:.3f} ({direction}-skewed; "
            f"threshold: |skew| >= 2.0)"
        )
    if kurt_flag:
        details.append(
            f"  Excess kurtosis: {excess_kurt:.3f} (heavy-tailed; "
            f"threshold: >= 5.0)"
        )

    # ---- Delta derivation string (adapts to scale method) ----
    if scale_method == "MAD":
        delta_detail = (
            f"delta = 1.345 * 1.4826 * MAD(y) = {delta:.4f}; "
            f"MAD = {mad:.4f}"
        )
    elif scale_method == "IQR":
        delta_detail = (
            f"delta = 1.345 * IQR(y) / 1.3489 = {delta:.4f}; "
            f"IQR = {iqr:.4f} (MAD = 0; IQR fallback)"
        )
    else:  # SD fallback
        delta_detail = (
            f"delta = 1.345 * SD(y) = {delta:.4f} "
            f"(MAD = 0, IQR = 0; SD fallback)"
        )

    # ---- Context-sensitive recommendation ----
    if skewness > 0 and zero_flag:
        recommendation = (
            "  This combination of right-skew and zero-inflation typically degrades\n"
            "  RMSE loss by inflating gradients for extreme residuals. Consider using\n"
            f"  Huber loss: loss_function: \"Huber:delta={delta:.4f}\"\n"
            f"  ({delta_detail})"
        )
    elif skewness < 0 and (zero_flag or skew_flag):
        recommendation = (
            "  Left-skewed outcome distribution detected. Huber loss may still reduce\n"
            "  outlier influence on RMSE gradients, but this pattern is less common\n"
            "  and warrants manual inspection of the outcome distribution.\n"
            f"  If using Huber: loss_function: \"Huber:delta={delta:.4f}\"\n"
            f"  ({delta_detail})"
        )
    else:
        recommendation = (
            "  Heavy tails or skewness may cause outlier-driven gradient inflation\n"
            "  under RMSE loss. Consider using Huber loss to cap residual influence.\n"
            f"  If using Huber: loss_function: \"Huber:delta={delta:.4f}\"\n"
            f"  ({delta_detail})"
        )

    recommendation += (
        "\n  See INPUT_SPECIFICATION.md Section 9 for the full derivation "
        "and literature."
    )

    print(header)
    for d in details:
        print(d)
    print(recommendation)


def _summarize_series(s: pd.Series) -> dict:
    """Return descriptive statistics for a numeric Series (unbiased SD, ddof=1)."""
    return {
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)),
        "min": float(s.min()),
        "max": float(s.max()),
        "q25": float(s.quantile(0.25)),
        "q50": float(s.quantile(0.50)),
        "q75": float(s.quantile(0.75)),
    }


def _write_train_outcome_stats(
    y: pd.Series | pd.DataFrame,
    task: str,
    outcome_cols: list,
    run_dir: str,
) -> None:
    """Write train_outcome_stats.json with training-outcome summary statistics.

    Called unconditionally after the CV fold loop. For regression and
    multi_regression tasks, the stats dict contains one entry per outcome
    column (mean, sd, min, max, q25, q50, q75). For classification tasks,
    stats is an empty dict but the file is still written (providing a stable
    artifact for downstream consumers regardless of task type).

    Parameters
    ----------
    y : pd.Series or pd.DataFrame
        Raw (unscaled) training-outcome values. For multi_regression this must
        be the pre-StandardScaler copy captured before target scaling.
    task : str
        Task type string (one of VALID_TASK_TYPES).
    outcome_cols : list[str]
        Outcome column name(s).
    run_dir : str
        Output directory; train_outcome_stats.json is written at its root.
    """
    stats: dict = {}
    if task in {"regression", "multi_regression"}:
        if task == "regression":
            series = y if isinstance(y, pd.Series) else y.iloc[:, 0]
            stats[outcome_cols[0]] = _summarize_series(series)
        else:  # multi_regression
            for col in outcome_cols:
                stats[col] = _summarize_series(y[col])
    payload = {
        "task_type": task,
        "outcome_columns": list(outcome_cols),
        "n": int(len(y)),
        "stats": stats,
    }
    save_json_atomic(payload, os.path.join(run_dir, "train_outcome_stats.json"))


def _validate_aggregate_shap(config: dict, final_cols: list, nom_feats: list) -> None:
    """Validate the aggregate_shap config block against the resolved feature set.

    Enforces five invariants before any model training occurs:
    (1) no group name collides with a resolved feature name,
    (2) each group has at least one member,
    (3) single-member groups emit a WARNING (no aggregation benefit),
    (4) every constituent exists in the resolved feature set,
    (5) nominal features are prohibited from aggregate groups (SHAP values for
        nominal/categorical features are not directly comparable across levels),
    (6) constituent membership is disjoint across groups.

    Parameters
    ----------
    config : dict
        Full pipeline config.
    final_cols : list[str]
        Resolved feature columns after all-missing-column drops.
    nom_feats : list[str]
        Resolved nominal feature names after all-missing-column drops.
    """
    agg_cfg = config.get("aggregate_shap", {})
    if not agg_cfg:
        return

    all_constituents: set = set()
    for group_name, members in agg_cfg.items():
        if group_name in final_cols:
            raise ValueError(
                f"aggregate_shap group name '{group_name}' collides with a "
                f"resolved feature column name. Rename the group."
            )
        if not isinstance(members, list) or len(members) == 0:
            raise ValueError(
                f"aggregate_shap group '{group_name}' has an empty or invalid "
                f"feature list. Provide at least one constituent feature name."
            )
        if len(members) == 1:
            print(
                f"[WARNING] aggregate_shap group '{group_name}' has only one "
                f"constituent feature; no aggregation benefit will be realized."
            )
        for feat in members:
            if feat not in final_cols:
                raise ValueError(
                    f"aggregate_shap constituent '{feat}' in group '{group_name}' "
                    f"is not in the resolved feature set. Check for typos or "
                    f"all-missing column drops."
                )
            if feat in nom_feats:
                raise ValueError(
                    f"aggregate_shap constituent '{feat}' in group '{group_name}' "
                    f"is a nominal feature. Nominal features are not permitted in "
                    f"aggregate groups because their SHAP values are not directly "
                    f"comparable across levels."
                )
            if feat in all_constituents:
                raise ValueError(
                    f"aggregate_shap constituent '{feat}' appears in multiple "
                    f"groups. Disjoint membership is required across all groups."
                )
            all_constituents.add(feat)

    print(
        f"[INFO] aggregate_shap: {len(agg_cfg)} group(s) validated, "
        f"{len(all_constituents)} constituent features."
    )


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
    groups: np.ndarray = None,
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
    n_inner_repeats = int(config["modeling"]["tuning"].get("n_inner_repeats", 1))

    inner_seed = seed + fold_idx + 1
    y_for_stratify = y_train if isinstance(y_train, pd.Series) else y_train.iloc[:, 0]

    cv_strategy = config["modeling"].get("cv_strategy", "uniform")

    inner_cv = get_cv_splitter(
        config, y_for_stratify, seed_override=inner_seed,
        groups=groups, n_repeats=n_inner_repeats,
        n_folds_override=inner_cv_folds,
    )

    if groups is not None and cv_strategy == "group":
        n_unique_inner = len(np.unique(groups))
        if n_unique_inner < 2 * inner_cv_folds:
            print(
                f"[WARNING] Inner CV has only {n_unique_inner} unique groups for "
                f"{inner_cv_folds} folds. Some folds may have very few groups, "
                f"producing unreliable tuning estimates."
            )

    if n_inner_repeats > 10:
        print("[WARNING] n_inner_repeats > 10: diminishing returns expected "
              "(Vanwinckelen & Blockeel 2012).")

    total_inner_fits = inner_cv_folds * n_inner_repeats * n_trials
    if total_inner_fits > 5000:
        print(f"[WARNING] Total inner fits per outer fold = {total_inner_fits} "
              f"({inner_cv_folds} folds x {n_inner_repeats} repeats x "
              f"{n_trials} trials). Consider reducing n_inner_repeats or n_iter.")

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
    df_raw = load_dataframe(data_path)

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

    _tx_required_cols = config.get("transformations", {}).get("required_cols", [])
    if _tx_required_cols:
        _tx_missing_cols = [c for c in _tx_required_cols if c not in df_raw.columns]
        if _tx_missing_cols:
            raise KeyError(
                f"transformations.required_cols references columns not in dataset: "
                f"{_tx_missing_cols}"
            )
        pre_tx_len = len(df_raw)
        df_raw = df_raw.dropna(subset=_tx_required_cols)
        tx_dropped = pre_tx_len - len(df_raw)
        if tx_dropped > 0:
            print(f"[INFO] Dropped {tx_dropped} rows with missing "
                  f"transformations.required_cols value(s)")
        if len(df_raw) == 0:
            raise ValueError(
                "No data left after dropping rows with missing "
                "transformations.required_cols."
            )

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
    validate_cv_config(config, df=df_raw)
    validate_bootstrap_config(config)

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

    # 3d. Validate aggregate_shap config against resolved feature set
    _validate_aggregate_shap(config, final_cols, nom_feats)

    cv_strategy = config["modeling"].get("cv_strategy", "uniform")
    group_column = config["modeling"].get("group_column")
    groups = None
    if cv_strategy == "group":
        if group_column in final_cols:
            final_cols = [c for c in final_cols if c != group_column]
            con_feats = [c for c in con_feats if c != group_column]
            ord_feats = [c for c in ord_feats if c != group_column]
            nom_feats = [c for c in nom_feats if c != group_column]
            print(f"[INFO] group_column '{group_column}' excluded from feature candidates.")
        groups = df_raw[group_column].values

    # 4. Type Enforcement & Preprocessing
    X = df_raw[final_cols].copy()
    # For multi_regression, y is a DataFrame; otherwise a Series
    if len(outcome_cols) > 1:
        y = df_raw[outcome_cols].copy()
    else:
        y = df_raw[outcome_cols[0]].copy()

    # Outcome Distribution Diagnostics (regression / multi_regression only)
    task_prelim = detect_task(config)
    if is_regression(task_prelim):
        if isinstance(y, pd.DataFrame):
            for col in y.columns:
                _diagnose_outcome_distribution(y[col], col)
        else:
            _diagnose_outcome_distribution(y, outcome_cols[0])

    # Load and validate transformations module (Site 2)
    transform_module = load_transform_module(config)
    if transform_module is not None:
        tx_cfg = config["transformations"]
        validate_transform_config(tx_cfg.get("required_cols", []), df_raw, "train")
        print(f"[INFO] Transformations module loaded: {tx_cfg['file']}")

    # Upfront smoke test (Site 3)
    if transform_module is not None:
        seed = config["execution"]["random_seed"]
        n_smoke = min(20, len(df_raw))
        rng = np.random.RandomState(seed)
        smoke_idx = rng.choice(len(df_raw), size=n_smoke, replace=False)
        smoke_train = smoke_idx[:n_smoke // 2]
        smoke_val = smoke_idx[n_smoke // 2:]
        tx_params = tx_cfg.get("params", {})
        outcome_col = outcome_cols[0] if len(outcome_cols) == 1 else outcome_cols

        try:
            y_sm_train, y_sm_val, sm_meta = transform_module.input_transform(
                df_raw, smoke_train, smoke_val, outcome_col, tx_params
            )
        except Exception as e:
            raise RuntimeError(
                f"Smoke test: input_transform failed on {n_smoke}-row subset: {e}"
            ) from e

        if len(y_sm_train) != len(smoke_train):
            raise ValueError(
                f"Smoke test: input_transform returned y_train with length "
                f"{len(y_sm_train)}, expected {len(smoke_train)}"
            )
        if len(y_sm_val) != len(smoke_val):
            raise ValueError(
                f"Smoke test: input_transform returned y_val with length "
                f"{len(y_sm_val)}, expected {len(smoke_val)}"
            )

        y_sm_all = np.concatenate([np.asarray(y_sm_train), np.asarray(y_sm_val)])
        if not np.all(np.isfinite(y_sm_all)):
            n_nonfinite = int(np.sum(~np.isfinite(y_sm_all)))
            raise ValueError(
                f"Smoke test: input_transform produced {n_nonfinite} non-finite "
                f"value(s) (NaN or Inf)"
            )

        try:
            json.dumps(sm_meta)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Smoke test: input_transform metadata is not JSON-serializable: {e}"
            ) from e

        try:
            y_sm_rt = transform_module.output_transform(
                np.asarray(y_sm_val, dtype=float), sm_meta, tx_params,
                df_raw=df_raw, row_indices=smoke_val
            )
        except Exception as e:
            raise RuntimeError(
                f"Smoke test: output_transform failed: {e}"
            ) from e
        if len(y_sm_rt) != len(smoke_val):
            raise ValueError(
                f"Smoke test: output_transform returned length {len(y_sm_rt)}, "
                f"expected {len(smoke_val)}"
            )

        n_probe = len(smoke_val)
        p1 = np.zeros(n_probe)
        p2 = np.ones(n_probe)
        p3 = np.full(n_probe, 2.0)
        o1 = np.asarray(transform_module.output_transform(
            p1, sm_meta, tx_params, df_raw=df_raw, row_indices=smoke_val
        ), dtype=float)
        o2 = np.asarray(transform_module.output_transform(
            p2, sm_meta, tx_params, df_raw=df_raw, row_indices=smoke_val
        ), dtype=float)
        o3 = np.asarray(transform_module.output_transform(
            p3, sm_meta, tx_params, df_raw=df_raw, row_indices=smoke_val
        ), dtype=float)

        expected_o3 = 2.0 * o2 - o1
        atol = 1e-6 * (np.abs(o2 - o1).max() + 1e-10)
        is_affine = np.allclose(o3, expected_o3, atol=atol, rtol=1e-6)

        if tx_cfg.get("back_transform_shap", False) and not is_affine:
            raise ValueError(
                "back_transform_shap=true but the output_transform is not affine. "
                "SHAP back-transformation requires output_transform(x) = alpha * x + beta "
                "for a constant alpha. Non-affine transforms break SHAP additivity."
            )

        print(f"[INFO] Smoke test passed ({n_smoke} rows). "
              f"Transform is {'affine' if is_affine else 'non-affine'}.")

        _tx_req_cols = tx_cfg.get("required_cols", [])
        for rc in _tx_req_cols:
            _rc_nan = int(df_raw[rc].isna().sum())
            if _rc_nan > 0:
                raise ValueError(
                    f"[INTERNAL] {_rc_nan} rows still have NaN in "
                    f"transformations.required_cols column '{rc}' after "
                    f"row-drop. This indicates a pipeline logic error."
                )

    # A. Force Nominal to String -> Category.
    # NaN is filled with the literal string "__NA__" before encoding. CatBoost treats
    # "__NA__" as a valid category level, allowing the model to learn whether missingness
    # is predictive. This is an implicit informativeness assumption — see README.
    for c in nom_feats:
        X[c] = X[c].astype(object).fillna("__NA__").astype(str).astype("category")

    # B. Force Continuous to Float
    for c in con_feats:
        X[c] = pd.to_numeric(X[c], errors='coerce').astype("float32")

    # C. Force Ordinal (Map levels to integers)
    for c in ord_feats:
        levels = selector.feature_metadata[c]['levels']
        X[c] = coerce_ordinal_column(X[c], levels, c)

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

    # Snapshot raw (unscaled) outcome for train_outcome_stats.json before any scaling
    y_raw = y.copy()

    # Auto-scale multi-regression targets to common scale
    target_scaler = None
    if task == "multi_regression" and transform_module is None:
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
        # Persist nominal-feature observed-level lists for predict/infer-time validation
        nominal_codebooks = {
            col: sorted(map(str, df_raw[col].dropna().unique().tolist()))
            for col in nom_feats
        }
        clean_meta["nominal_codebooks"] = nominal_codebooks
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
    splitter = get_cv_splitter(config, y_for_split, groups=groups)

    if cv_strategy == "group":
        fold_sizes = [len(val) for _, val in splitter.split(X, y_for_split)]
        ratio = max(fold_sizes) / max(min(fold_sizes), 1)
        if ratio > 2.0:
            print(f"[WARNING] GroupKFold folds are unbalanced: sizes {fold_sizes}. "
                  f"Max/min ratio = {ratio:.2f} (threshold: 2.0).")

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
    fold_assignments = np.full(len(X), -1, dtype=int)
    all_fold_transform_meta = []

    print(f"[INFO] Starting {splitter.get_n_splits()}-Fold Nested CV...")

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y_for_split)):
        print(f"\n--- Fold {fold_idx + 1} ---")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        fold_assignments[val_idx] = fold_idx
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_meta = None
        if transform_module is not None:
            y_train, y_val, fold_meta = transform_module.input_transform(
                df_raw, train_idx, val_idx, outcome_col, tx_cfg.get("params", {})
            )
            y_train = pd.Series(y_train, index=y.iloc[train_idx].index)
            y_val = pd.Series(y_val, index=y.iloc[val_idx].index)
            all_fold_transform_meta.append(fold_meta)

        if transform_module is not None and fold_idx == 0:
            if tx_cfg.get("back_transform_shap", False) and is_affine:
                probe_0 = np.zeros(len(val_idx))
                probe_1 = np.ones(len(val_idx))
                ot_0 = np.asarray(transform_module.output_transform(
                    probe_0, fold_meta, tx_cfg.get("params", {}),
                    df_raw=df_raw, row_indices=val_idx
                ), dtype=float)
                ot_1 = np.asarray(transform_module.output_transform(
                    probe_1, fold_meta, tx_cfg.get("params", {}),
                    df_raw=df_raw, row_indices=val_idx
                ), dtype=float)
                alpha_vec = ot_1 - ot_0
                if not np.allclose(alpha_vec, alpha_vec[0], rtol=1e-6):
                    raise ValueError(
                        "back_transform_shap=true but output_transform has "
                        "non-constant slope across samples in fold 0. "
                        "This indicates a sample-dependent scale factor."
                    )
                shap_scale_factor = float(alpha_vec[0])
                print(f"[INFO] SHAP scale factor (alpha) = {shap_scale_factor:.6f}")
            else:
                shap_scale_factor = 1.0

        # --- PHASE 1: CLEAN TRAINING ---
        # A. Tune
        print("  > Tuning hyperparameters (Phase 1: Clean)...")
        # Note: CatBoost handles Ordinals as numeric if we don't list them in cat_features.
        # We ONLY pass nominals to cat_features argument.
        inner_groups = groups[train_idx] if groups is not None else None
        best_params, tuned_iters = run_optuna_tuning(X_train, y_train, nom_feats, task, config, n_jobs, fold_idx=fold_idx, groups=inner_groups)
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

        agg_groups = config.get("aggregate_shap", {})
        _block_permute_shadow(X_train_shadow, agg_groups, rng)
        _block_permute_shadow(X_val_shadow, agg_groups, rng)

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

        # 4. Phase-2 shadow training uses a fixed iteration ceiling of 2 * tuned_iters
        # with NO early stopping. The 2x ceiling preserves the original rationale that
        # shadow models need additional iterations to converge with 2p shadow features
        # added to the feature space (Boruta-style stratified shadow features;
        # Kursa & Rudnicki 2010). Removing eval_set=pool_val_full closes an
        # outer-validation-pool leakage path in the shadow-model fit.
        pool_train_full = Pool(X_train_full, y_train, cat_features=full_cat_features)

        shadow_params = best_params.copy()
        shadow_params["iterations"] = tuned_iters * 2  # fixed ceiling, no early stopping

        if is_regression(task):
            model_shadow = CatBoostRegressor(**shadow_params)
        else:
            model_shadow = CatBoostClassifier(**shadow_params)

        model_shadow.fit(
            pool_train_full,
            verbose=False,
        )

        # 5. Save Shadow Model
        shadow_model_path = os.path.join(run_dir, f"shadow_model_fold_{fold_idx}.cbm")
        model_shadow.save_model(shadow_model_path)

    # 6. Finalize
    print("\n[INFO] CV Complete. Saving Global Artifacts...")
    save_json_atomic(fold_assignments.tolist(), os.path.join(run_dir, "fold_assignments.json"))

    if transform_module is not None:
        tx_artifact = {
            "active": True,
            "file": tx_cfg["file"],
            "params": tx_cfg.get("params", {}),
            "required_cols": tx_cfg.get("required_cols", []),
            "is_affine": is_affine,
            "back_transform_shap": tx_cfg.get("back_transform_shap", False),
            "shap_scale_factor": shap_scale_factor,
        }
        save_json_atomic(tx_artifact, os.path.join(run_dir, "transform_config.json"))
        print(f"[INFO] Saved transform_config.json")
        save_json_atomic(all_fold_transform_meta, os.path.join(run_dir, "fold_transform_metadata.json"))
        print(f"[INFO] Saved fold_transform_metadata.json ({len(all_fold_transform_meta)} folds)")

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

    # --- Training-outcome statistics artifact (consumed by indiv_reports at predict / infer time) ---
    _write_train_outcome_stats(y_raw, task, outcome_cols, run_dir)

    print(f"[SUCCESS] Training finished. Artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
