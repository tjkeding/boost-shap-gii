"""Shared Utilities for boost-shap-gii pipeline."""

from __future__ import annotations

import copy
import json
import os
import warnings
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
    """Build data-driven CatBoost hyperparameter search space (10 parameters).

    Notable bounds:
    - `depth.high`: floored at 3 to guarantee at least a [2, 3] range even for n < 20.
    - `one_hot_max_size`: fixed [2, 25], independent of feature count p.
    """
    return {
        "iterations":          {"low": 100,   "high": 5000},
        "learning_rate":       {"low": 0.001, "high": 0.3,   "log": True},
        "depth":               {"low": 2,     "high": max(3, min(10, int(np.log2(max(n / 5, 4)))))},
        "l2_leaf_reg":         {"low": 0.01,  "high": 100.0, "log": True},
        "min_data_in_leaf":    {"low": 1,     "high": max(2, min(200, n // 50))},
        "random_strength":     {"low": 0.001, "high": 10.0,  "log": True},
        "bagging_temperature": {"low": 0.1,   "high": 1.0},
        "border_count":        {"low": 32,    "high": 255},
        "colsample_bylevel":   {"low": 0.05,  "high": 1.0},
        "one_hot_max_size":    {"low": 2,     "high": 25},
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
    _set(["shap", "compute_global_on_inference"], False)

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

    # Validate spline config after defaults are filled
    validate_spline_config(config)

    return config, filled


# =============================================================================
# Indiv-reports and plot config validators
# =============================================================================

_VALID_INDIV_SCALING_MODES = {"raw", "sd", "custom_value"}
_REGRESSION_TASKS = {"regression", "multi_regression"}


def validate_spline_config(config: dict) -> None:
    """Validate shap.splines configuration.

    Spline stability requires at least n_knots + degree + 2 unique x values to
    support the basis (Wood 2017, Generalized Additive Models, ch. 4). Below
    this threshold, the basis is rank-deficient and fits become unstable.
    """
    splines = config.get("shap", {}).get("splines", {})
    n_knots = splines.get("n_knots")
    degree = splines.get("degree")
    discrete_threshold = splines.get("discrete_threshold")
    if n_knots is None or degree is None or discrete_threshold is None:
        return  # fill_config_defaults will set these; nothing to validate yet
    lower_bound = n_knots + degree + 2
    if discrete_threshold < lower_bound:
        raise ValueError(
            f"shap.splines.discrete_threshold ({discrete_threshold}) must be "
            f">= n_knots + degree + 2 ({n_knots} + {degree} + 2 = {lower_bound}). "
            f"Spline basis is rank-deficient below this lower bound (Wood 2017, "
            f"Generalized Additive Models, ch. 4)."
        )


def validate_indiv_reports_config(config: dict) -> None:
    """Validate shap.indiv_* and shap.compute_global_on_inference keys.

    Raises ValueError with precise messages on any violation:
      - shap.indiv_ci_nboot missing, non-integer, or negative
      - shap.indiv_scaling_mode missing or not in {raw, sd, custom_value}
      - shap.indiv_scaling_mode == 'sd' but task_type not in {regression, multi_regression}
      - shap.indiv_scaling_mode == 'custom_value' but shap.indiv_scaling_value missing or <= 0
      - shap.compute_global_on_inference present but not bool
    """
    shap_cfg = config.get("shap", {})

    # -- shap.indiv_ci_nboot --
    nboot = shap_cfg.get("indiv_ci_nboot")
    if nboot is None:
        raise ValueError(
            "shap.indiv_ci_nboot is required but missing from config. "
            "Set to 0 to disable per-individual CI computation, or to a positive "
            "integer (minimum recommended 2500) to enable it."
        )
    if not isinstance(nboot, int):
        raise ValueError(
            f"shap.indiv_ci_nboot must be an integer, got {type(nboot).__name__}: {nboot!r}."
        )
    if nboot < 0:
        raise ValueError(
            f"shap.indiv_ci_nboot must be >= 0, got {nboot}. "
            "Set to 0 to disable or to a positive integer to enable."
        )

    # -- shap.indiv_scaling_mode --
    scaling_mode = shap_cfg.get("indiv_scaling_mode")
    if scaling_mode is None:
        raise ValueError(
            "shap.indiv_scaling_mode is required but missing from config. "
            f"Must be one of: {sorted(_VALID_INDIV_SCALING_MODES)}."
        )
    if scaling_mode not in _VALID_INDIV_SCALING_MODES:
        raise ValueError(
            f"shap.indiv_scaling_mode='{scaling_mode}' is not valid. "
            f"Must be one of: {sorted(_VALID_INDIV_SCALING_MODES)}."
        )

    # -- sd mode requires regression task --
    if scaling_mode == "sd":
        task_type = config.get("modeling", {}).get("task_type")
        if task_type not in _REGRESSION_TASKS:
            raise ValueError(
                f"shap.indiv_scaling_mode='sd' requires a regression task; "
                f"got task_type='{task_type}'. Use 'raw' or 'custom_value' instead."
            )

    # -- custom_value mode requires indiv_scaling_value > 0 --
    if scaling_mode == "custom_value":
        scaling_value = shap_cfg.get("indiv_scaling_value")
        if scaling_value is None:
            raise ValueError(
                "shap.indiv_scaling_value is required when shap.indiv_scaling_mode='custom_value' "
                "but is missing from config. Provide a positive number (e.g., outcome theoretical "
                "maximum, minimum-meaningful-difference threshold, or any domain-specific anchor)."
            )
        if not isinstance(scaling_value, (int, float)) or scaling_value <= 0:
            raise ValueError(
                f"shap.indiv_scaling_value must be a positive number, got {scaling_value!r}."
            )

    # -- shap.compute_global_on_inference (optional, must be bool if present) --
    cgi = shap_cfg.get("compute_global_on_inference")
    if cgi is not None and not isinstance(cgi, bool):
        raise ValueError(
            f"shap.compute_global_on_inference must be a bool (true/false), "
            f"got {type(cgi).__name__}: {cgi!r}."
        )


def validate_plot_config(config: dict) -> None:
    """Validate plot.* required keys. Called only from cmd_plot (not from train/predict/infer).

    Raises ValueError with precise messages on missing or wrong-typed keys:
      - plot.outcome_max missing or non-positive number
      - plot.negate_shap missing or not bool
      - plot.gii_y_label / plot.gii_y_sublabel / plot.indiv_y_label / plot.indiv_y_sublabel
        missing or empty string
    """
    plot_cfg = config.get("plot", {})

    # -- plot.outcome_max --
    outcome_max = plot_cfg.get("outcome_max")
    if outcome_max is None:
        raise ValueError(
            "plot.outcome_max is required for the plot subcommand but is missing from config. "
            "Provide a positive number representing the theoretical maximum of the outcome."
        )
    if not isinstance(outcome_max, (int, float)) or outcome_max <= 0:
        raise ValueError(
            f"plot.outcome_max must be a positive number, got {outcome_max!r}."
        )

    # -- plot.negate_shap --
    negate_shap = plot_cfg.get("negate_shap")
    if negate_shap is None:
        raise ValueError(
            "plot.negate_shap is required for the plot subcommand but is missing from config. "
            "Set to true to sign-flip SHAP y-axis values, or false to display them as-is."
        )
    if not isinstance(negate_shap, bool):
        raise ValueError(
            f"plot.negate_shap must be a bool (true/false), got {type(negate_shap).__name__}: {negate_shap!r}."
        )

    # -- label strings --
    required_labels = [
        ("gii_y_label", "plot.gii_y_label"),
        ("gii_y_sublabel", "plot.gii_y_sublabel"),
        ("indiv_y_label", "plot.indiv_y_label"),
        ("indiv_y_sublabel", "plot.indiv_y_sublabel"),
    ]
    for key, dotted in required_labels:
        val = plot_cfg.get(key)
        if val is None:
            raise ValueError(
                f"{dotted} is required for the plot subcommand but is missing from config."
            )
        if not isinstance(val, str) or not val.strip():
            raise ValueError(
                f"{dotted} must be a non-empty string, got {val!r}."
            )


# =============================================================================
# Nominal feature helpers (used by predict.py and infer.py)
# =============================================================================

def _label_nominal(value, levels: set) -> str:
    """Map a nominal value to a sentinel-aware label.

    Returns "__NA__" if value is NaN (training-time signal: missingness may
    itself be informative). Returns "__UNSEEN__" if value is non-NaN but not
    in the training-time codebook `levels` (out-of-distribution; routes to
    CatBoost prior-mean fallback). Returns the value unchanged otherwise.
    """
    if pd.isna(value):
        return "__NA__"
    if value not in levels:
        return "__UNSEEN__"
    return str(value)


def _validate_nominal_unseen(
    series: pd.Series,
    levels: set,
    column_name: str,
    *,
    tier1_unique_threshold: float = 0.50,
    tier2_obs_threshold: float = 0.10,
) -> None:
    """Two-tier validation for nominal feature values not in the training codebook.

    Mirrors the ordinal validation pattern at predict.py:148-169.

    Tier 1 (hard error): if > 50% of unique observed values are absent from
    `levels`, raises ValueError. Indicates misconfigured codebook or systematic
    naming mismatch between training and inference data.

    Tier 2 (loud warning): if > 10% of observations (non-NaN) have values
    absent from `levels`, prints a warning with the exact fraction.
    """
    non_na = series.dropna()
    if len(non_na) == 0:
        return  # all-NaN column; no unseen levels possible
    unique_observed = set(non_na.unique())
    unseen_unique = unique_observed - levels
    unseen_obs = non_na.isin(unseen_unique)

    unique_unseen_frac = len(unseen_unique) / max(len(unique_observed), 1)
    obs_unseen_frac = float(unseen_obs.mean())

    if unique_unseen_frac > tier1_unique_threshold:
        raise ValueError(
            f"Nominal feature '{column_name}': "
            f"{unique_unseen_frac:.1%} of unique observed values are absent "
            f"from the training-time codebook (threshold: "
            f"{tier1_unique_threshold:.0%}). This indicates a misconfigured "
            f"codebook or systematic naming mismatch between training and "
            f"inference data. Retrain with an expanded codebook or correct "
            f"the inference data."
        )
    if obs_unseen_frac > tier2_obs_threshold:
        warnings.warn(
            f"Nominal feature '{column_name}': "
            f"{obs_unseen_frac:.1%} of observations have values absent from "
            f"the training-time codebook (threshold: "
            f"{tier2_obs_threshold:.0%}). These will route to CatBoost "
            f"prior-mean fallback via the '__UNSEEN__' sentinel.",
            UserWarning,
        )


# =============================================================================
# Statistical helpers (used by predict.py and infer.py)
# =============================================================================

def compute_permutation_test(y_true, y_pred, metric_fns, metric_names, n_perm, seed, run_dir):
    """Run a one-sided permutation test (model vs. chance) for each metric.

    Shuffles y_true relative to fixed y_pred to build null distributions. A while-loop
    guarantees exactly n_perm successful iterations (capped at 2 * n_perm total attempts).
    Permutation failures are rare numerical artifacts, not diagnostic events — retry is
    statistically valid because permutations preserve the full y distribution (no class-loss
    issue). This contrasts with bootstrap drops, which are diagnostic of class imbalance.

    P-value uses the Davison & Hinkley (1997) +1 correction:
    p = (sum(null >= observed) + 1) / (n_perm_effective + 1).

    Parameters
    ----------
    y_true : array-like
        Observed outcome values.
    y_pred : array-like
        Model predictions (fixed across all permutations).
    metric_fns : list[callable]
        Scoring functions (higher = better convention).
    metric_names : list[str]
        Corresponding metric names; `neg_` prefix triggers sign-aware display.
    n_perm : int
        Target number of successful permutation iterations.
    seed : int
        Random seed for the permutation RNG.
    run_dir : str
        Directory to save `permutation_test_results.csv` and
        `permutation_null_distributions.parquet`.

    Returns
    -------
    pd.DataFrame
        Columns: metric, observed, null_mean, null_std, p_value.
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

    # Null distributions — while-loop guarantees n_perm successful iterations.
    # Unlike bootstraps (where single-class failure is diagnostic), permutation
    # failures are rare numerical artifacts with no diagnostic value: retry is
    # statistically valid because permutations preserve the full y distribution.
    null_dists = {name: [] for name in metric_names}
    n_attempts = 0
    max_attempts = 2 * n_perm

    while min(len(v) for v in null_dists.values()) < n_perm and n_attempts < max_attempts:
        n_attempts += 1
        y_perm = y_true[rng.permutation(n)]
        for name, fn in zip(metric_names, metric_fns):
            try:
                val = fn(y_perm, y_pred)
                if not np.isnan(val):
                    null_dists[name].append(val)
            except Exception:
                pass

    n_perm_effective = min(len(v) for v in null_dists.values())
    if n_attempts >= max_attempts and n_perm_effective < n_perm:
        print(
            f"[WARNING] compute_permutation_test: reached {max_attempts} attempt cap. "
            f"Effective permutation count: {n_perm_effective}/{n_perm}."
        )

    # Convert lists to arrays, truncated to minimum effective count for alignment
    null_dists = {name: np.array(v[:n_perm_effective]) for name, v in null_dists.items()}

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
    """Compute a bootstrapped confidence interval for a metric (raw scale).

    Bootstrap iterations where all resampled y_true values share a single class are
    dropped (the metric is undefined for single-class samples). This can occur with
    severely imbalanced classification on small n. n_boot_effective tracks the number
    of valid iterations and is reported as a confidence proxy when the drop rate
    exceeds 5%.

    Note: dropped iterations are NOT replaced. The failure rate is diagnostic of
    sample size vs. class imbalance and should not be suppressed via retry. This
    contrasts with compute_permutation_test(), where retry is statistically valid.

    Parameters
    ----------
    y_true : array-like
        True outcome values.
    y_pred : array-like
        Model predictions.
    metric_fn : callable
        Scoring function (y_true, y_pred) -> float (higher = better).
    n_boot : int
        Number of bootstrap iterations.
    alpha : float
        Significance level; CI = [alpha/2, 1-alpha/2] percentiles.

    Returns
    -------
    base_score : float
        Score on the full (non-resampled) data.
    lower : float
        Lower CI bound (alpha/2 percentile).
    upper : float
        Upper CI bound (1 - alpha/2 percentile).
    """
    scores = []
    indices = np.arange(len(y_true))
    n_dropped = 0

    # Baseline
    try:
        base_score = metric_fn(y_true, y_pred)
    except Exception:
        return np.nan, np.nan, np.nan

    for _ in range(n_boot):
        idx = resample(indices, replace=True, n_samples=len(indices))
        if len(np.unique(y_true[idx])) < 2:
            n_dropped += 1
            continue
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
            scores.append(score)
        except Exception:
            continue

    n_boot_effective = len(scores)
    drop_rate = n_dropped / n_boot if n_boot > 0 else 0.0
    if drop_rate > 0.05:
        print(
            f"[WARNING] compute_bootstrap_ci: {drop_rate:.1%} of bootstrap iterations "
            f"dropped (single-class resample). n_boot_effective={n_boot_effective}/{n_boot}. "
            f"CIs may be unreliable for severely imbalanced data."
        )

    if not scores:
        warnings.warn(
            f"compute_bootstrap_ci: all bootstrap iterations dropped for "
            f"'<unknown effect>' (n_boot_effective = 0). Returning point estimate "
            f"with NaN CI bounds. This indicates severe data sparsity or class "
            f"imbalance for this effect; CI is undefined.",
            RuntimeWarning,
        )
        return base_score, float("nan"), float("nan")

    lower = np.percentile(scores, 100 * (alpha / 2))
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return base_score, lower, upper


# =============================================================================
# sig_GII loader (shared by predict.py and infer.py for indiv_reports)
# =============================================================================

def _load_sig_GII_from_shap_stats(run_dir: str) -> Tuple[Dict[str, bool], Dict[Tuple[str, str], bool]]:
    """Load sig_GII flags from the single shap_stats_global.csv.

    Returns (sig_GII_main, sig_GII_interaction):
      sig_GII_main:        {feature_name: bool}
      sig_GII_interaction: {(feature_a, feature_b): bool}  # order as written

    File path resolution:
      - Single-output mode: read run_dir/shap_analysis/shap_stats_global.csv.
      - Multi-output mode (no shap_analysis/ present, but shap_{slice}/ dirs exist):
        v1 scope-limited to the first shap_*/ slice found by glob; emits an INFO log.

    Raises RuntimeError if no shap_stats_global.csv can be located.
    """
    import glob as _glob

    csv_path = os.path.join(run_dir, "shap_analysis", "shap_stats_global.csv")

    if not os.path.exists(csv_path):
        # Multi-output mode: look for shap_{slice}/ subdirectories
        slice_dirs = sorted(_glob.glob(os.path.join(run_dir, "shap_*/")))
        csv_path = None
        for sd in slice_dirs:
            candidate = os.path.join(sd, "shap_stats_global.csv")
            if os.path.exists(candidate):
                slice_label = os.path.basename(sd.rstrip("/"))
                print(f"[INFO] indiv_reports using representative slice: {slice_label}")
                csv_path = candidate
                break

    if csv_path is None or not os.path.exists(csv_path):
        raise RuntimeError(
            f"shap_stats_global.csv not found in run_dir '{run_dir}'; "
            "run full predict with shap computation enabled before indiv_reports."
        )

    df = pd.read_csv(csv_path)

    main_df = df[df["type"] == "Singleton"]
    int_df = df[df["type"] != "Singleton"]

    sig_GII_main: Dict[str, bool] = {
        str(row["effect"]): bool(row["sig_GII"])
        for _, row in main_df.iterrows()
    }

    sig_GII_interaction: Dict[Tuple[str, str], bool] = {}
    for _, row in int_df.iterrows():
        eff = str(row["effect"])
        if " x " in eff:
            a, b = eff.split(" x ", 1)
            sig_GII_interaction[(a, b)] = bool(row["sig_GII"])

    return sig_GII_main, sig_GII_interaction
