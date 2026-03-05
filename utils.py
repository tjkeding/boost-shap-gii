"""Shared Utilities for boost-shap-gii pipeline."""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, log_loss, accuracy_score, f1_score,
    balanced_accuracy_score
)
from sklearn.utils import resample

# Valid task types
VALID_TASK_TYPES = {
    "regression",
    "binary_classification",
    "multiclass_classification",
    "multi_regression",
}


def _normalize_quotes(s):
    """Replace common Unicode curly quotes with ASCII equivalents."""
    if not isinstance(s, str):
        return s
    return s.replace('\u2018', "'").replace('\u2019', "'").replace('\u201C', '"').replace('\u201D', '"')


def load_config(path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration without defaults."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Expand path variables like ${paths.output_dir}
    def expand(d):
        if isinstance(d, dict):
            return {k: expand(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [expand(v) for v in d]
        elif isinstance(d, str):
            for k, v in config["paths"].items():
                if isinstance(v, str):
                    d = d.replace(f"${{paths.{k}}}", v)
            return d
        return d

    def _check_unresolved(d, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                _check_unresolved(v, f"{path}.{k}")
        elif isinstance(d, str) and "${" in d:
            print(f"[WARNING] Unresolved variable in config at {path}: {d}")

    result = expand(config)
    _check_unresolved(result)
    return result


def save_json_atomic(data: Any, path: str):
    """Save JSON atomically to prevent corruption."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def detect_task(config: Dict) -> str:
    """Determine task type from config.

    Returns one of: 'regression', 'binary_classification',
    'multiclass_classification', 'multi_regression'.

    If config has explicit 'task_type', uses that.
    Otherwise falls back to inference from scoring metric
    (legacy behavior: regression or binary_classification).
    """
    explicit = config["modeling"].get("task_type", None)
    if explicit is not None:
        if explicit not in VALID_TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {sorted(VALID_TASK_TYPES)}, got '{explicit}'"
            )
        return explicit

    # Legacy fallback: infer from scoring string
    scoring = config["modeling"]["tuning"]["scoring"]
    is_regression = scoring.startswith("neg_") or scoring == "r2"
    return "regression" if is_regression else "binary_classification"


def is_classification(task: str) -> bool:
    """Check if a task type is any form of classification."""
    return task in ("binary_classification", "multiclass_classification")


def is_regression(task: str) -> bool:
    """Check if a task type is any form of regression."""
    return task in ("regression", "multi_regression")


def get_cv_splitter(config: Dict, y: pd.Series):
    """Return a KFold or StratifiedKFold splitter based on task."""
    n_folds = int(config["modeling"]["cv_folds"])
    seed = int(config["execution"]["random_seed"])
    task = detect_task(config)

    if is_regression(task):
        return KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        if y.nunique() < 20:
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            return KFold(n_splits=n_folds, shuffle=True, random_state=seed)


def get_scoring_function(metric_name: str):
    """Map config metric string to callable.

    Raises ValueError for unknown metric names.
    """
    metrics = {
        # Regression
        "neg_mae": lambda y, p: -mean_absolute_error(y, p),
        "neg_rmse": lambda y, p: -np.sqrt(mean_squared_error(y, p)),
        "r2": r2_score,
        # Binary classification
        "roc_auc": roc_auc_score,
        "accuracy": accuracy_score,
        "f1": f1_score,
        "log_loss": lambda y, p: -log_loss(y, p),
        # Multiclass classification
        "balanced_accuracy": balanced_accuracy_score,
        "f1_weighted": lambda y, p: f1_score(y, p, average='weighted'),
        "roc_auc_ovr": lambda y, p: roc_auc_score(
            y, p, multi_class='ovr', average='weighted'
        ),
    }
    if metric_name not in metrics:
        raise ValueError(
            f"Unknown scoring metric: '{metric_name}'. "
            f"Valid options: {sorted(metrics.keys())}"
        )
    return metrics[metric_name]


# =============================================================================
# Config defaults — "minimal mode" auto-fill
# =============================================================================

# Deterministic mapping from task_type to loss_function and scoring metric
_TASK_LOSS_SCORING = {
    "regression":                 ("RMSE",       "neg_rmse"),
    "binary_classification":      ("Logloss",    "roc_auc"),
    "multiclass_classification":  ("MultiClass", "balanced_accuracy"),
    "multi_regression":           ("MultiRMSE",  "neg_rmse"),
}


def _default_cv_folds(n: int) -> int:
    """Outer CV folds: {3, 5, 10}. Minimum 30 samples per validation fold."""
    if n // 30 >= 10:
        return 10
    elif n // 30 >= 5:
        return 5
    else:
        return 3


def _default_inner_cv_folds(n: int, outer_folds: int) -> int:
    """Inner CV folds: {3, 5, 10}. Minimum 20 samples per inner validation fold."""
    n_train = n - (n // outer_folds)
    if n_train // 20 >= 10:
        return 10
    elif n_train // 20 >= 5:
        return 5
    else:
        return min(3, outer_folds)


def _default_search_space(n: int, p: int) -> Dict[str, Any]:
    """Data-driven hyperparameter search space. Always 10 parameters."""
    return {
        "iterations":          {"low": 100,   "high": 5000},
        "learning_rate":       {"low": 0.001, "high": 0.3,   "log": True},
        "depth":               {"low": 2,     "high": min(10, int(np.log2(max(n / 5, 4))))},
        "l2_leaf_reg":         {"low": 0.01,  "high": 100.0, "log": True},
        "min_data_in_leaf":    {"low": 1,     "high": max(2, min(200, n // 50))},
        "random_strength":     {"low": 0.001, "high": 10.0,  "log": True},
        "bagging_temperature": {"low": 0.1,   "high": 1.0},
        "border_count":        {"low": 32,    "high": 255},
        "colsample_bylevel":   {"low": 0.05,  "high": 1.0},
        "one_hot_max_size":    {"low": 2,     "high": min(25, max(p, 2))},
    }


def _default_n_boot(n: int) -> int:
    """Bootstrap iterations scaled by sample size."""
    if n < 100:
        return 2000
    elif n < 500:
        return 5000
    else:
        return 10000


def _infer_task_type(config: Dict) -> str:
    """Infer task_type when omitted. Uses outcome shape and scoring if available."""
    outcome = config["modeling"]["outcome"]
    if isinstance(outcome, list):
        return "multi_regression"
    # If scoring is present, use existing detect_task logic
    scoring = (config.get("modeling", {})
               .get("tuning", {})
               .get("scoring", None))
    if scoring is not None:
        is_reg = scoring.startswith("neg_") or scoring == "r2"
        return "regression" if is_reg else "binary_classification"
    # No scoring either — default to regression for single outcome
    return "regression"


def _setdefault_nested(d: Dict, keys: List[str], value: Any) -> bool:
    """Set a nested key only if it doesn't exist. Returns True if value was set."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    if keys[-1] not in d:
        d[keys[-1]] = value
        return True
    return False


def fill_config_defaults(
    config: Dict[str, Any],
    n_rows: int,
    n_features: int,
) -> Tuple[Dict[str, Any], List[str]]:
    """Fill omitted config fields with data-driven defaults.

    User-provided values are NEVER overwritten (uses setdefault semantics).

    Parameters
    ----------
    config : dict
        The parsed YAML config (may be incomplete).
    n_rows : int
        Number of rows in the dataset (after dropping missing outcomes).
    n_features : int
        Number of selected features.

    Returns
    -------
    config : dict
        The same dict, mutated in-place with defaults filled.
    filled : list[str]
        Dotted-path names of fields that were auto-filled.
    """
    filled = []

    def _set(keys, value, label=None):
        """Helper: set nested key if absent, track what was filled."""
        if _setdefault_nested(config, keys, value):
            path = ".".join(keys)
            filled.append((path, label or str(value)))

    # -- execution --
    _set(["execution", "n_jobs"], os.cpu_count(), f"{os.cpu_count()} (auto-detected CPUs)")
    _set(["execution", "random_seed"], 42)

    # -- modeling.task_type (needed for loss/scoring inference) --
    if "task_type" not in config.get("modeling", {}):
        task = _infer_task_type(config)
        _set(["modeling", "task_type"], task, f"{task} (inferred)")
    task_type = config["modeling"]["task_type"]

    # -- modeling.loss_function & tuning.scoring --
    loss, scoring = _TASK_LOSS_SCORING[task_type]
    _set(["modeling", "loss_function"], loss, f"{loss} (from task_type={task_type})")
    _set(["modeling", "tuning", "scoring"], scoring, f"{scoring} (from task_type={task_type})")

    # -- CV folds --
    outer = _default_cv_folds(n_rows)
    _set(["modeling", "cv_folds"], outer, f"{outer} (n={n_rows})")

    # Resolve actual outer folds for inner calc (might be user-provided)
    actual_outer = config["modeling"]["cv_folds"]
    inner = _default_inner_cv_folds(n_rows, actual_outer)
    _set(["modeling", "tuning", "inner_cv_folds"], inner, f"{inner} (n={n_rows}, outer={actual_outer})")

    # -- tuning parameters --
    _set(["modeling", "tuning", "n_iter"], 300, "300 (10 params × 30/dim; Bergstra et al. 2011)")
    _set(["modeling", "tuning", "early_stopping_rounds"], 250)

    # -- search space --
    space = _default_search_space(n_rows, n_features)
    _set(["modeling", "tuning", "search_space"], space, f"data-driven (n={n_rows}, p={n_features})")

    # -- shap --
    _set(["shap", "output_microdata_n"], 10)

    # -- shap.bootstrapping --
    n_boot = _default_n_boot(n_rows)
    _set(["shap", "bootstrapping", "n_boot"], n_boot, f"{n_boot} (n={n_rows})")
    _set(["shap", "bootstrapping", "alpha"], 0.05)
    _set(["shap", "bootstrapping", "fdr_correct"], True)
    _set(["shap", "bootstrapping", "stab_thresh"], 2)
    _set(["shap", "bootstrapping", "output_boots_n"], 10)

    # -- shap.splines --
    _set(["shap", "splines", "n_knots"], 4)
    _set(["shap", "splines", "degree"], 3)
    _set(["shap", "splines", "discrete_threshold"], 15)

    # Validate search space bounds (low < high)
    space = config["modeling"]["tuning"]["search_space"]
    for param, bounds in space.items():
        if isinstance(bounds, dict) and "low" in bounds and "high" in bounds:
            if bounds["low"] >= bounds["high"]:
                raise ValueError(
                    f"Search space '{param}': low ({bounds['low']}) >= high ({bounds['high']})"
                )

    return config, filled


# =============================================================================
# Statistical helpers (used by predict.py and infer.py)
# =============================================================================

def compute_permutation_test(y_true, y_pred, metric_fns, metric_names, n_perm, seed, run_dir):
    """
    Permutation test for OOF model performance.
    Shuffles y_true relative to fixed y_pred to build null distributions.
    All scoring functions follow the convention: higher = better.
    P-value = (# permuted >= observed + 1) / (n_perm + 1).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    # Observed scores (raw, not abs — preserves direction for one-sided test)
    observed = {}
    for name, fn in zip(metric_names, metric_fns):
        try:
            observed[name] = fn(y_true, y_pred)
        except Exception:
            observed[name] = np.nan

    # Null distributions
    null_dists = {name: np.full(n_perm, np.nan) for name in metric_names}

    for i in range(n_perm):
        y_perm = y_true[rng.permutation(n)]
        for name, fn in zip(metric_names, metric_fns):
            try:
                null_dists[name][i] = fn(y_perm, y_pred)
            except Exception:
                pass

    # P-values and summary
    def _to_display(name, val):
        """Sign-aware conversion: negate neg_* metrics for display, leave others."""
        if np.isnan(val):
            return np.nan
        return -val if name.startswith("neg_") else val

    results = []
    for name in metric_names:
        obs = observed[name]
        null = null_dists[name]
        null_clean = null[~np.isnan(null)]

        if np.isnan(obs) or len(null_clean) == 0:
            p_val = np.nan
        else:
            # One-sided: higher is better for all scoring functions
            p_val = (np.sum(null_clean >= obs) + 1) / (len(null_clean) + 1)

        disp_name = name.replace("neg_", "").upper()
        results.append({
            "metric": disp_name,
            "observed": _to_display(name, obs),
            "null_mean": _to_display(name, np.nanmean(null)),
            "null_std": np.nanstd(null),
            "p_value": p_val
        })

    # Save null distributions (sign-aware for plotting)
    null_df = pd.DataFrame({
        name.replace("neg_", "").upper(): (
            -null_dists[name] if name.startswith("neg_") else null_dists[name]
        )
        for name in metric_names
    })
    null_df.to_parquet(os.path.join(run_dir, "permutation_null_distributions.parquet"), index=False)

    # Save summary
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(run_dir, "permutation_test_results.csv"), index=False)

    return res_df


def compute_bootstrap_ci(y_true, y_pred, metric_fn, n_boot=2000, alpha=0.05):
    """Compute Confidence Interval for a metric via bootstrapping (raw scale)."""
    scores = []
    indices = np.arange(len(y_true))

    # Baseline
    try:
        base_score = metric_fn(y_true, y_pred)
    except Exception:
        return np.nan, np.nan, np.nan

    for _ in range(n_boot):
        idx = resample(indices, replace=True, n_samples=len(indices))
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
            scores.append(score)
        except Exception:
            continue

    if not scores:
        return base_score, base_score, base_score

    lower = np.percentile(scores, 100 * (alpha / 2))
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return base_score, lower, upper
