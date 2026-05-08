"""Per-individual SHAP reports with bootstrap confidence intervals for individual-case model inspection.

Training mode: coupled bootstrap. The user config
shap.indiv_ci_nboot = B specifies the number of coupled iterations. Per iteration b:
draw ONE shared bootstrap sample s_b (size N, with replacement, cluster-aware when
cluster_ids present); for each fold k in range(K) refit CatBoost on (X_train[s_b],
y_train[s_b]) using params_k = get_all_params(model_fold_{k}.cbm). Total refits = K * B.

Inference mode: bootstrap-of-CV design. Per iteration b: draw one bootstrap sample
s_b (size N_train, with replacement, cluster-aware when cluster_ids present); generate a
fresh K-fold split on s_b (KFold/StratifiedKFold, seed = random_seed + b + 1); for each
fold k refit CatBoost on the fold-train portion of s_b using frozen fold_hyperparameters[k]
(no HP retuning); compute SHAP on the inference pool from each of the K refits and average
to produce ONE ensemble-replicate per b. CIs are basic/reverse-percentile intervals:
  ci_lo = 2 * hat - q_hi,  ci_hi = 2 * hat - q_lo
where hat is the deployed ensemble-mean point estimate and (q_lo, q_hi) are bootstrap
percentiles. This aligns the bootstrap estimand with the deployed ensemble estimand.

Point estimates (deployed-product SHAP; NOT bootstrap-distribution statistics):
  - Training individual i (assigned to fold k_i): OOF single-model SHAP and prediction
    from model_fold_{k_i}.cbm (leakage-free).
  - Inference individual: ensemble-mean SHAP and prediction across all K original
    model_fold_{k}.cbm files (matching infer.py ensemble logic).

Per-individual CIs (estimand-matched to the point estimate):
  - Training individual i: aggregate iterations where i not in s_b, using ONLY the
    fold-k_i refit from each such iteration (single-model estimand match to OOF point).
    Expected effective count ~= 0.368 * B (Breiman 2001 OOB rate).
    Individuals with effective count < OOB_FLOOR_MIN emit NaN CI bounds with oob_count preserved.
  - Inference individual: bootstrap-of-CV basic/reverse-percentile interval anchored on
    the original ensemble point estimate (all B replicates used; no OOB filtering needed).

Scaling: raw (unscaled) | sd (divide by training-outcome SD from train_outcome_stats.json)
         | custom_value (divide by user-supplied value).

Output structure:
  {run_dir}/indiv_reports/
    main_effects.parquet   -- long format, ALL features, sig_GII column
    interactions.parquet   -- long format, HARD-FILTERED to sig_GII=True
    predictions.parquet    -- per-individual predicted outcome (raw + scaled) with CIs
    indiv_reports_metadata.json

Called from:
  predict.py: generate_indiv_reports(..., mode='training')  # after bootstrap cache built
  infer.py:   generate_indiv_reports(..., mode='inference') # uses live bootstrap-of-CV
"""

from __future__ import annotations

import concurrent.futures
import datetime
import glob
import json
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import KFold, StratifiedKFold

from .utils import get_cv_splitter, save_json_atomic

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

OOB_FLOOR_MIN: int = 50   # individuals with fewer OOB refits emit NaN CI bounds
CI_LO_PCT: float = 2.5
CI_HI_PCT: float = 97.5

# Allowlist of user-facing CatBoost HP keys that are safe to pass on refit.
# Runtime/internal keys are excluded to avoid conflicts with Pool cat_features.
_CATBOOST_USER_PARAM_ALLOWLIST = {
    "iterations",
    "depth",
    "learning_rate",
    "loss_function",
    "border_count",
    "l2_leaf_reg",
    "bagging_temperature",
    "random_strength",
    "random_seed",
    "min_data_in_leaf",
    "colsample_bylevel",
    "one_hot_max_size",
    "eval_metric",
    "class_weights",
    "scale_pos_weight",
    "custom_loss",
    "boosting_type",
    "bootstrap_type",
    "subsample",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_user_level_params(all_params: dict) -> dict:
    """Return only the allowlisted user-facing HP subset from get_all_params()."""
    return {k: v for k, v in all_params.items() if k in _CATBOOST_USER_PARAM_ALLOWLIST}


def _load_one_model(model_path: str, task: str) -> Union[CatBoostRegressor, CatBoostClassifier]:
    """Load a CatBoost model from disk."""
    if task in ("regression", "multi_regression"):
        m = CatBoostRegressor()
    else:
        m = CatBoostClassifier()
    m.load_model(model_path)
    return m


def _predict_single(model, pool: Pool, task: str) -> np.ndarray:
    """Return predictions for a Pool. Shape depends on task."""
    if task in ("regression", "multi_regression"):
        return np.asarray(model.predict(pool), dtype=np.float64)
    elif task == "binary_classification":
        return np.asarray(model.predict_proba(pool)[:, 1], dtype=np.float64)
    else:  # multiclass_classification
        return np.asarray(model.predict_proba(pool), dtype=np.float64)


def _output_dim(model) -> int:
    """Return the number of output classes/coordinates for a fitted CatBoost estimator.

    For multiclass classification, returns model.n_classes_ (n >= 2).
    For binary classification, regression, and multi_regression, returns 1
    (these task types use 1D output along the class dimension; the singleton
    class axis is collapsed in tensor handling for these tasks).
    """
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) > 2:
        return int(len(classes))
    return 1


def _shap_single(model, pool: Pool, task: str, n_individuals: int) -> np.ndarray:
    """Return SHAP main-effect values with normalized class axis.

    Always returns a 3D tensor of shape (N, C, F) where:
      - N is the number of rows in `pool`,
      - C is the number of output classes (1 for non-multiclass tasks),
      - F is the number of features (bias column trimmed off).
    For non-multiclass tasks, the class axis is inserted as a singleton
    dimension so downstream consumers can use uniform 3D indexing.
    """
    sv = model.get_feature_importance(pool, type="ShapValues")
    # ShapValues returns shape (N, F+1) for non-multiclass or (N, C, F+1) for multiclass.
    if sv.ndim == 3:
        # multiclass: (N, C, F+1) -> trim bias -> (N, C, F)
        return sv[:, :, :-1].astype(np.float32)
    # non-multiclass: (N, F+1) -> trim bias -> (N, F) -> add singleton C axis -> (N, 1, F)
    trimmed = sv[:, :-1].astype(np.float32)
    return trimmed[:, np.newaxis, :]


def _shap_interaction_single(model, pool: Pool) -> np.ndarray:
    """Return SHAP interaction values with bias-trim and a normalized class axis.

    Always returns a 4D tensor of shape (N, C, F, F) where:
      - N is the number of rows in `pool`,
      - C is the number of output classes (1 for non-multiclass tasks),
      - F is the number of features (bias column trimmed off both feature axes).
    For non-multiclass tasks, the class axis is inserted as a singleton
    dimension so downstream consumers can use uniform 4D indexing.
    """
    sv = model.get_feature_importance(pool, type="ShapInteractionValues")
    sv = np.asarray(sv, dtype=np.float32)
    if sv.ndim == 4:
        # Multiclass: shape (N, C, F+1, F+1) -> trim bias on both feature axes.
        return sv[:, :, :-1, :-1]
    if sv.ndim == 3:
        # Non-multiclass: shape (N, F+1, F+1) -> trim bias, then add singleton C axis.
        trimmed = sv[:, :-1, :-1]
        return trimmed[:, np.newaxis, :, :]
    raise ValueError(
        f"_shap_interaction_single: unexpected SHAP interaction tensor "
        f"with ndim={sv.ndim} (expected 3 or 4)."
    )


def _reconstruct_fold_assignments(
    config: dict,
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, pd.DataFrame],
) -> np.ndarray:
    """Reconstruct per-individual fold assignments from saved config.

    Mirrors predict.py:222-236 exactly. Raises AssertionError if partition is incomplete.

    Returns
    -------
    fold_of : np.ndarray of shape (N_train,), dtype int32
        fold_of[i] = fold index that individual i was assigned to as the validation fold.
    """
    y_for_split = y_train if isinstance(y_train, pd.Series) else y_train.iloc[:, 0]
    splitter = get_cv_splitter(config, y_for_split)
    N = len(X_train)
    fold_of = np.full(N, -1, dtype=np.int32)
    for fold_idx, (_, val_idx) in enumerate(splitter.split(X_train, y_for_split)):
        fold_of[val_idx] = fold_idx
    assert (fold_of >= 0).all(), (
        "Fold assignment reconstruction failed: not all individuals were assigned to a "
        "validation fold. This indicates the CV splitter did not produce a full partition. "
        "Verify that config random_seed and cv_folds match the training run."
    )
    return fold_of


def _bootstrap_sample_indices(
    N: int,
    rng: np.random.Generator,
    cluster_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw one bootstrap sample of row indices (size N, with replacement).

    When cluster_ids is provided, resamples clusters (matching shap_utils.py convention).
    Returns a sorted array of indices of length N (unsorted if cluster-aware, may differ in length).
    """
    if cluster_ids is None:
        return rng.integers(0, N, size=N, dtype=np.int32)
    # Cluster-aware: resample clusters, expand to member rows
    unique_clusters = np.unique(cluster_ids)
    sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
    indices = []
    for c in sampled_clusters:
        indices.append(np.where(cluster_ids == c)[0])
    return np.concatenate(indices).astype(np.int32)


# ---------------------------------------------------------------------------
# Bootstrap-of-CV inference CI routine
# ---------------------------------------------------------------------------

def _bootstrap_of_cv_inference(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    inference_pool: Pool,
    fold_hyperparameters: List[dict],
    B: int,
    K: int,
    random_seed: int,
    cluster_ids: Optional[np.ndarray],
    task: str,
    nom_feats: List[str],
    point_shap_main: np.ndarray,
    point_shap_int: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Bootstrap-of-CV with basic/reverse-percentile intervals for inference mode.

    For each iteration b in {1, ..., B}:
      1. Draw a bootstrap sample s_b of size N_train from (X_train, y_train).
      2. Generate a fresh K-fold split on s_b (independent of the original training split;
         seed = random_seed + b + 1) using KFold for regression and StratifiedKFold for
         classification tasks.
      3. For each fold k: refit a CatBoost model on the fold-train portion of s_b using
         fold_hyperparameters[k] (no HP retuning).
      4. Compute SHAP main and (when point_shap_int is not None) interaction values from
         each refitted fold model on inference_pool, then average across K fold models to
         produce one ensemble-replicate.
    After B iterations, compute the basic/reverse-percentile interval at each cell:
      ci_lo = 2 * hat - q_hi
      ci_hi = 2 * hat - q_lo
    where (q_lo, q_hi) are the (2.5, 97.5) percentiles of the bootstrap distribution and
    hat is the original deployed ensemble-mean point estimate.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (N_train rows).
    y_train : np.ndarray
        Training outcome array (1D or 2D).
    inference_pool : Pool
        CatBoost Pool for the inference target individuals.
    fold_hyperparameters : list of dict
        Frozen fold-specific HPs extracted from the K deployed model_fold_k.cbm files.
    B : int
        Number of bootstrap iterations.
    K : int
        Number of CV folds (must match len(fold_hyperparameters)).
    random_seed : int
        Base random seed for the bootstrap RNG.
    cluster_ids : np.ndarray or None
        Cluster membership vector (length N_train). When provided, resampling is
        cluster-aware (mirrors _bootstrap_sample_indices convention).
    task : str
        One of 'regression', 'multi_regression', 'binary_classification',
        'multiclass_classification'.
    nom_feats : list of str
        Nominal feature names for CatBoost Pool construction.
    point_shap_main : np.ndarray
        Deployed ensemble-mean SHAP main values, shape (N_target, C, F).
    point_shap_int : np.ndarray or None
        Deployed ensemble-mean SHAP interaction values, shape (N_target, C, F, F), or
        None when interaction CIs are not requested.

    Returns
    -------
    main_ci_lo, main_ci_hi : np.ndarray, each shape (N_target, C, F)
    int_ci_lo, int_ci_hi : np.ndarray or None, each shape (N_target, C, F, F)
    """
    rng = np.random.default_rng(random_seed)
    N_train = len(X_train)
    N_target = point_shap_main.shape[0]
    C = point_shap_main.shape[1]
    F = point_shap_main.shape[2]

    main_replicates = np.full((B, N_target, C, F), np.nan, dtype=np.float32)
    compute_interactions = point_shap_int is not None
    if compute_interactions:
        int_replicates = np.full((B, N_target, C, F, F), np.nan, dtype=np.float32)
    else:
        int_replicates = None

    is_cls = task in ("binary_classification", "multiclass_classification")

    for b in range(B):
        s_b_idx = _bootstrap_sample_indices(N=N_train, rng=rng, cluster_ids=cluster_ids)
        X_b = X_train.iloc[s_b_idx].reset_index(drop=True)
        if y_train.ndim == 1:
            y_b = y_train[s_b_idx]
        else:
            y_b = y_train[s_b_idx, :]

        # Fresh K-fold split on s_b; seed decoupled from s_b draw.
        fold_seed = random_seed + b + 1
        if is_cls and y_b.ndim == 1:
            splitter = StratifiedKFold(n_splits=K, shuffle=True, random_state=fold_seed)
            fold_iter = list(splitter.split(X_b, y_b))
        else:
            splitter = KFold(n_splits=K, shuffle=True, random_state=fold_seed)
            fold_iter = list(splitter.split(X_b))

        per_fold_main = np.full((K, N_target, C, F), np.nan, dtype=np.float32)
        if compute_interactions:
            per_fold_int = np.full((K, N_target, C, F, F), np.nan, dtype=np.float32)

        for k, (train_idx, _) in enumerate(fold_iter):
            X_fold_train = X_b.iloc[train_idx].reset_index(drop=True)
            if y_b.ndim == 1:
                y_fold_train = y_b[train_idx]
            else:
                y_fold_train = y_b[train_idx, :]

            refit_params = dict(fold_hyperparameters[k])
            refit_params["thread_count"] = 1
            refit_params["verbose"] = False
            refit_params["allow_writing_files"] = False

            fold_train_pool = Pool(X_fold_train, label=y_fold_train, cat_features=nom_feats)

            if task in ("regression", "multi_regression"):
                refit_model = CatBoostRegressor(**refit_params)
            else:
                refit_model = CatBoostClassifier(**refit_params)
            refit_model.fit(fold_train_pool)

            per_fold_main[k] = _shap_single(refit_model, inference_pool, task, N_target)
            if compute_interactions:
                per_fold_int[k] = _shap_interaction_single(refit_model, inference_pool)

        main_replicates[b] = np.nanmean(per_fold_main, axis=0)
        if compute_interactions:
            int_replicates[b] = np.nanmean(per_fold_int, axis=0)

        # Release per-iteration fold arrays
        del per_fold_main
        if compute_interactions:
            del per_fold_int

        if (b + 1) % max(1, B // 10) == 0:
            print(f"[INFO] Bootstrap-of-CV inference: {b + 1}/{B} iterations complete.")

    alpha = 0.05
    q_lo_pct = (alpha / 2) * 100.0
    q_hi_pct = (1.0 - alpha / 2) * 100.0

    main_q_lo = np.nanpercentile(main_replicates, q_lo_pct, axis=0)
    main_q_hi = np.nanpercentile(main_replicates, q_hi_pct, axis=0)
    main_ci_lo = 2.0 * point_shap_main - main_q_hi
    main_ci_hi = 2.0 * point_shap_main - main_q_lo

    if compute_interactions:
        int_q_lo = np.nanpercentile(int_replicates, q_lo_pct, axis=0)
        int_q_hi = np.nanpercentile(int_replicates, q_hi_pct, axis=0)
        int_ci_lo = 2.0 * point_shap_int - int_q_hi
        int_ci_hi = 2.0 * point_shap_int - int_q_lo
    else:
        int_ci_lo = None
        int_ci_hi = None

    return main_ci_lo, main_ci_hi, int_ci_lo, int_ci_hi


# ---------------------------------------------------------------------------
# Worker function for ProcessPoolExecutor (must be module-level for pickling)
# ---------------------------------------------------------------------------

def _fit_and_save_refit(
    b: int,
    k: int,
    sample_indices: np.ndarray,
    params_k: dict,
    X_train_parquet_path: str,
    y_train_path: str,
    nom_feats: list,
    task: str,
    out_path: str,
) -> None:
    """Fit one bootstrap refit model and save it.

    Loads X_train and y_train from parquet (avoids passing large DataFrames to workers).
    """
    X_train = pd.read_parquet(X_train_parquet_path)
    y_arr = np.load(y_train_path, allow_pickle=True)
    if y_arr.ndim == 1:
        y_boot = y_arr[sample_indices]
    else:
        y_boot = y_arr[sample_indices, :]

    X_boot = X_train.iloc[sample_indices]

    refit_params = dict(params_k)
    refit_params["thread_count"] = 1
    refit_params["verbose"] = False
    refit_params["allow_writing_files"] = False

    pool = Pool(X_boot, label=y_boot, cat_features=nom_feats)

    if task in ("regression", "multi_regression"):
        m = CatBoostRegressor(**refit_params)
    else:
        m = CatBoostClassifier(**refit_params)

    m.fit(pool)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    m.save_model(out_path)


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------

def _load_bootstrap_cache_or_fail(train_dir: str) -> dict:
    """Load bootstrap_metadata.json from train_dir/bootstrap_refits/.

    Raises FileNotFoundError with remediation hint if cache missing.
    """
    meta_path = os.path.join(train_dir, "bootstrap_refits", "bootstrap_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"bootstrap_metadata.json not found at {meta_path}. "
            "Run predict.py with shap.indiv_ci_nboot > 0 to build the bootstrap cache "
            "before invoking infer.py indiv_reports."
        )
    with open(meta_path) as f:
        return json.load(f)


def _load_train_outcome_stats_or_fail(train_dir: str) -> dict:
    """Load train_outcome_stats.json from train_dir.

    Raises FileNotFoundError with remediation hint if missing.
    """
    path = os.path.join(train_dir, "train_outcome_stats.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"train_outcome_stats.json not found at {path}. "
            "Re-run train.py to regenerate this artifact. "
            "This file is required for per-individual SHAP report scaling."
        )
    with open(path) as f:
        return json.load(f)


def _resolve_scaling_divisor(
    mode: str,
    value: float,
    task: str,
    outcome_cols: list,
    stats: dict,
) -> Union[float, Dict[str, float]]:
    """Return scalar divisor or per-outcome dict {col -> divisor}.

    mode in {'raw', 'sd', 'custom_value'}
    For multi_regression with mode='sd', returns {col: sd_for_col}.
    """
    if mode == "raw":
        if task == "multi_regression":
            return {col: 1.0 for col in outcome_cols}
        return 1.0
    elif mode == "custom_value":
        if value is None or value <= 0:
            raise ValueError(
                f"shap.indiv_scaling_value must be a positive number for mode='custom_value'; "
                f"got {value!r}."
            )
        if task == "multi_regression":
            return {col: float(value) for col in outcome_cols}
        return float(value)
    elif mode == "sd":
        # Restricted to regression tasks (validated upstream)
        outcome_stats = stats.get("stats", {})
        if task == "multi_regression":
            divisors = {}
            for col in outcome_cols:
                col_stats = outcome_stats.get(col, {})
                sd = col_stats.get("sd")
                if sd is None or sd <= 0:
                    raise ValueError(
                        f"SD for outcome column '{col}' is missing or non-positive in "
                        f"train_outcome_stats.json (sd={sd!r}). "
                        "Use scaling_mode='raw' or 'custom_value' instead."
                    )
                divisors[col] = float(sd)
            return divisors
        else:
            # regression: single outcome
            col = outcome_cols[0]
            col_stats = outcome_stats.get(col, {})
            sd = col_stats.get("sd")
            if sd is None or sd <= 0:
                raise ValueError(
                    f"SD for outcome '{col}' is missing or non-positive in "
                    f"train_outcome_stats.json (sd={sd!r}). "
                    "Use scaling_mode='raw' or 'custom_value' instead."
                )
            return float(sd)
    else:
        raise ValueError(f"Unknown scaling mode: '{mode}'.")


# ---------------------------------------------------------------------------
# Parquet / JSON emitters
# ---------------------------------------------------------------------------

def _emit_main_effects_parquet(run_dir: str, rows: List[dict]) -> None:
    """Write main_effects.parquet to run_dir/indiv_reports/."""
    out_dir = os.path.join(run_dir, "indiv_reports")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(out_dir, "main_effects.parquet"), index=False)


def _emit_interactions_parquet(run_dir: str, rows: List[dict]) -> None:
    """Write interactions.parquet to run_dir/indiv_reports/.

    If rows is empty, emits a header-only parquet with the correct schema.
    """
    out_dir = os.path.join(run_dir, "indiv_reports")
    os.makedirs(out_dir, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
    else:
        # Header-only empty parquet with expected schema (non-multiclass)
        df = pd.DataFrame(columns=[
            "id", "feature_a", "feature_b",
            "feature_a_value_raw", "feature_b_value_raw",
            "feature_a_type", "feature_b_type",
            "shap_value_raw", "shap_value_scaled",
            "shap_value_ci_lo", "shap_value_ci_hi", "oob_count",
        ])
    df.to_parquet(os.path.join(out_dir, "interactions.parquet"), index=False)


def _emit_predictions_parquet(run_dir: str, rows: List[dict]) -> None:
    """Write predictions.parquet to run_dir/indiv_reports/."""
    out_dir = os.path.join(run_dir, "indiv_reports")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(out_dir, "predictions.parquet"), index=False)


def _emit_metadata_json(
    run_dir: str,
    scaling_mode: str,
    scaling_divisor: Union[float, dict],
    B: int,
    K: int,
    oob_floor: int,
    outcome_names: list,
    mode: str,
    timestamp: str,
) -> None:
    """Write indiv_reports_metadata.json to run_dir/indiv_reports/."""
    out_dir = os.path.join(run_dir, "indiv_reports")
    os.makedirs(out_dir, exist_ok=True)
    point_source = "OOF_single_model" if mode == "training" else "ensemble_mean"
    ci_agg = (
        "OOB_single_model"
        if mode == "training"
        else "ensemble_replicates_basic_percentile"
    )
    payload = {
        "design": "coupled",
        "scaling_mode": scaling_mode,
        "scaling_divisor": scaling_divisor,
        "B": B,
        "K": K,
        "total_refits": K * B,
        "point_estimate_source": point_source,
        "ci_aggregation": ci_agg,
        "oob_count_floor": oob_floor,
        "outcome_columns": outcome_names,
        "mode": mode,
        "timestamp": timestamp,
    }
    save_json_atomic(payload, os.path.join(out_dir, "indiv_reports_metadata.json"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def orchestrate_bootstrap_cache(
    run_dir: str,
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, pd.DataFrame],
    task: str,
    outcome_cols: list,
    nom_feats: list,
    config: dict,
    n_jobs: int,
    random_seed: int,
) -> dict:
    """Build the coupled bootstrap-refit cache at run_dir/bootstrap_refits/.

    No-ops and returns {} if config['shap']['indiv_ci_nboot'] == 0.

    Returns a summary dict: {"K": K, "B": B, "total_refits": K*B,
                              "effective_B": B, "B_per_fold": B, "cache_dir": path}.
    """
    nboot = int(config["shap"]["indiv_ci_nboot"])
    if nboot == 0:
        return {}

    B = nboot
    cache_dir = os.path.join(run_dir, "bootstrap_refits")
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Discover K from saved model files
    model_files = sorted(glob.glob(os.path.join(run_dir, "model_fold_*.cbm")))
    if not model_files:
        raise FileNotFoundError(
            f"No model_fold_*.cbm files found in {run_dir}. "
            "Run train.py before building bootstrap cache."
        )
    K = len(model_files)
    N_train = len(X_train)

    print(f"[INFO] orchestrate_bootstrap_cache: K={K}, B={B}, total_refits={K * B}")

    # 2. Preload all K original fold models and extract user-level HP dicts
    params: List[dict] = []
    for k, mpath in enumerate(model_files):
        if task in ("regression", "multi_regression"):
            m = CatBoostRegressor()
        else:
            m = CatBoostClassifier()
        m.load_model(mpath)
        params.append(_extract_user_level_params(m.get_all_params()))
    print(f"[INFO] Loaded HP from {K} fold models.")

    # 3. Draw shared bootstrap index vectors for all B iterations
    cluster_ids = None
    cluster_col = config.get("data", {}).get("cluster_id_col")
    if cluster_col and cluster_col in X_train.columns:
        cluster_ids = X_train[cluster_col].values

    rng = np.random.default_rng(random_seed)
    shared_indices_list = []
    for b in range(B):
        idx = _bootstrap_sample_indices(N_train, rng, cluster_ids)
        shared_indices_list.append(idx)

    # Pad to rectangular array (cluster-aware samples can differ in length)
    # Store as an object array if lengths differ, else as int32 2D array
    lengths = [len(s) for s in shared_indices_list]
    if len(set(lengths)) == 1:
        shared_indices = np.stack(shared_indices_list).astype(np.int32)
        ragged = False
    else:
        shared_indices = np.empty(B, dtype=object)
        for b, s in enumerate(shared_indices_list):
            shared_indices[b] = s
        ragged = True

    npz_path = os.path.join(cache_dir, "shared_indices.npz")
    if ragged:
        np.savez_compressed(npz_path, indices=shared_indices)
    else:
        np.savez_compressed(npz_path, indices=shared_indices)
    print(f"[INFO] Saved shared bootstrap indices ({B} iterations) to {npz_path}.")

    # Serialize X_train and y_train to temp files for worker processes.
    # y_train is also persisted permanently as y_train.npy for inference-mode bootstrap-of-CV.
    x_tmp = os.path.join(cache_dir, "_X_train_tmp.parquet")
    y_tmp = os.path.join(cache_dir, "_y_train_tmp.npy")
    y_persistent = os.path.join(cache_dir, "y_train.npy")
    X_train.to_parquet(x_tmp)
    if isinstance(y_train, pd.DataFrame):
        np.save(y_tmp, y_train.values)
        np.save(y_persistent, y_train.values)
    else:
        np.save(y_tmp, y_train.values)
        np.save(y_persistent, y_train.values)
    print(f"[INFO] Serialized training data for worker processes.")

    # 4. Dispatch K * B refits to process pool
    max_workers = max(1, int(n_jobs))

    def _make_tasks():
        for b in range(B):
            s = shared_indices_list[b]
            for k in range(K):
                out_path = os.path.join(cache_dir, f"iter_{b:05d}", f"fold_{k}.cbm")
                yield (b, k, s, params[k], x_tmp, y_tmp, nom_feats, task, out_path)

    tasks = list(_make_tasks())
    total = len(tasks)
    print(f"[INFO] Dispatching {total} bootstrap refit tasks with max_workers={max_workers}...")

    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fit_and_save_refit, b, k, s, p, x_tmp, y_tmp, nom_feats, task, out
            ): (b, k)
            for b, k, s, p, _, _, _, _, out in tasks
        }
        for fut in concurrent.futures.as_completed(futures):
            b_idx, k_idx = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Bootstrap refit failed at iteration b={b_idx}, fold k={k_idx}: {exc}"
                ) from exc
            completed += 1
            if completed % max(1, total // 10) == 0:
                print(f"[INFO] Bootstrap refits: {completed}/{total} complete.")

    print(f"[INFO] All {total} bootstrap refits complete.")

    # Cleanup temp files
    for tmp in (x_tmp, y_tmp):
        if os.path.exists(tmp):
            os.remove(tmp)

    # 5. Write bootstrap_metadata.json
    meta = {
        "design": "coupled",
        "training_ci_design": "shared_sample_oob_single_model",
        "inference_ci_design": "bootstrap_of_cv_basic_percentile",
        "K": K,
        "B": B,
        "total_refits": K * B,
        "random_seed": random_seed,
        "cluster_aware": cluster_ids is not None,
        "fold_hp_summary": {str(k): params[k] for k in range(K)},
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    save_json_atomic(meta, os.path.join(cache_dir, "bootstrap_metadata.json"))
    print(f"[INFO] bootstrap_metadata.json written to {cache_dir}.")

    return {
        "K": K,
        "B": B,
        "effective_B": B,
        "B_per_fold": B,
        "total_refits": K * B,
        "cache_dir": cache_dir,
    }


def generate_indiv_reports(
    run_dir: str,
    train_dir: str,
    X_target: pd.DataFrame,
    ids_target: pd.Series,
    X_train: pd.DataFrame,
    y_target: Optional[Union[pd.Series, pd.DataFrame]],
    task: str,
    outcome_cols: list,
    nom_feats: list,
    config: dict,
    mode: Literal["training", "inference"],
    sig_GII_main: Dict[str, bool],
    sig_GII_interaction: Dict[Tuple[str, str], bool],
) -> None:
    """Compute per-individual SHAP point estimates + coupled-bootstrap CIs and emit parquets.

    No-ops and returns immediately if config['shap']['indiv_ci_nboot'] == 0.

    Preconditions:
      - train_dir/bootstrap_refits/ exists (built by orchestrate_bootstrap_cache).
      - train_dir/train_outcome_stats.json exists.
      - train_dir/model_fold_*.cbm exist.
      In training mode: X_train and y_target (== y_train) must be provided.
    """
    nboot = int(config["shap"]["indiv_ci_nboot"])
    if nboot == 0:
        return

    scaling_mode = config["shap"]["indiv_scaling_mode"]
    scaling_value = config["shap"].get("indiv_scaling_value")
    feature_types_cfg = config.get("data", {}).get("feature_types", {})

    # --- Validate preconditions ---
    cache_meta = _load_bootstrap_cache_or_fail(train_dir)
    outcome_stats = _load_train_outcome_stats_or_fail(train_dir)

    K = cache_meta["K"]
    B = cache_meta["B"]
    cache_dir = os.path.join(train_dir, "bootstrap_refits")

    N_target = len(X_target)
    feature_names = list(X_target.columns)
    N_features = len(feature_names)

    # --- Load feature_types from artifact ---
    feat_types_path = os.path.join(train_dir, "feature_types.json")
    if os.path.exists(feat_types_path):
        with open(feat_types_path) as f:
            feature_type_map: Dict[str, str] = json.load(f)
    else:
        feature_type_map = {}

    # --- Load shared bootstrap indices ---
    npz_path = os.path.join(cache_dir, "shared_indices.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"shared_indices.npz not found at {npz_path}. "
            "Re-run predict.py with indiv_ci_nboot > 0 to rebuild the cache."
        )
    npz = np.load(npz_path, allow_pickle=True)
    shared_indices_arr = npz["indices"]
    # Reconstruct list (handles both ragged and rectangular)
    if shared_indices_arr.dtype == object:
        shared_indices_list = [shared_indices_arr[b] for b in range(B)]
    else:
        shared_indices_list = [shared_indices_arr[b] for b in range(B)]

    # --- Load original K fold models for point estimates ---
    orig_model_files = sorted(glob.glob(os.path.join(train_dir, "model_fold_*.cbm")))
    if len(orig_model_files) != K:
        raise FileNotFoundError(
            f"Expected {K} model_fold_*.cbm files in {train_dir}, "
            f"found {len(orig_model_files)}."
        )
    orig_models = []
    for mpath in orig_model_files:
        orig_models.append(_load_one_model(mpath, task))

    # Resolve number of output classes once; used for all buffer allocations throughout.
    n_outputs = _output_dim(orig_models[0])
    if task == "multiclass_classification":
        n_classes = orig_models[0].classes_count_
    else:
        n_classes = 1  # C=1 singleton class axis for all non-multiclass task types

    cat_feats = nom_feats

    # --- Memory guard ---
    N_pairs_sig = sum(1 for v in sig_GII_interaction.values() if v)
    try:
        import psutil as _psutil
        avail_bytes = _psutil.virtual_memory().available
    except ImportError:
        raise ImportError(
            "psutil is required for the indiv_reports memory guard but is not installed. "
            "Install it with: pip install psutil  (or add psutil to environment.yaml)."
        )
    budget_bytes = int(0.5 * avail_bytes)
    effective_B_max = B  # upper bound for both training and inference
    # Main CI buffer: (N_target, B, C, F) float32 -- 4 bytes per element.
    projected_main = N_target * effective_B_max * n_outputs * N_features * 4
    # Interaction CI buffer: (N_target, B, C, F, F) float32 -- 4 bytes per element.
    projected_inter = N_target * effective_B_max * n_outputs * N_features * N_features * 4
    projected_total = projected_main + projected_inter
    if projected_total > budget_bytes:
        raise MemoryError(
            f"indiv_reports CI accumulation would require ~{projected_total / 1e9:.2f} GB "
            f"but the 50% budget is {budget_bytes / 1e9:.2f} GB "
            f"(of {avail_bytes / 1e9:.2f} GB available). "
            "Reduce shap.indiv_ci_nboot or run on a higher-memory node. "
            "No streaming fallback is provided (err-on-kill)."
        )

    # --- Resolve scaling divisor ---
    divisor = _resolve_scaling_divisor(
        scaling_mode, scaling_value, task, outcome_cols, outcome_stats
    )

    # --- STEP 3: Point-estimate SHAP and predictions ---
    print(f"[INFO] Computing point-estimate SHAP for {N_target} individuals (mode={mode})...")

    if mode == "training":
        # Reconstruct fold assignments
        fold_of = _reconstruct_fold_assignments(config, X_train, y_target)

        # OOF single-model SHAP: each individual i uses model_fold_{fold_of[i]}.
        # _shap_single always returns (N, C, F) with C=1 for non-multiclass.
        point_shap = np.full((N_target, n_classes, N_features), np.nan, dtype=np.float32)
        if task == "multiclass_classification":
            point_y_pred = np.full((N_target, n_classes), np.nan, dtype=np.float64)
        elif task == "multi_regression":
            point_y_pred = np.full((N_target, len(outcome_cols)), np.nan, dtype=np.float64)
        else:
            point_y_pred = np.full(N_target, np.nan, dtype=np.float64)

        # Batch by fold for efficiency
        for k in range(K):
            fold_mask = fold_of == k
            if not fold_mask.any():
                continue
            X_fold = X_target.iloc[np.where(fold_mask)[0]]
            pool_fold = Pool(X_fold, cat_features=cat_feats)
            sv = _shap_single(orig_models[k], pool_fold, task, X_fold.shape[0])  # (n_fold, C, F)
            pv = _predict_single(orig_models[k], pool_fold, task)
            point_shap[fold_mask] = sv
            point_y_pred[fold_mask] = pv

    else:  # inference
        # Ensemble-mean SHAP across all K original fold models.
        # _shap_single always returns (N, C, F) with C=1 for non-multiclass.
        pool_all = Pool(X_target, cat_features=cat_feats)

        shap_accum = np.zeros((N_target, n_classes, N_features), dtype=np.float64)
        if task == "multiclass_classification":
            pred_accum = np.zeros((N_target, n_classes), dtype=np.float64)
        elif task == "multi_regression":
            pred_accum = np.zeros((N_target, len(outcome_cols)), dtype=np.float64)
        else:
            pred_accum = np.zeros(N_target, dtype=np.float64)

        for k in range(K):
            sv = _shap_single(orig_models[k], pool_all, task, N_target)  # (N, C, F)
            pv = _predict_single(orig_models[k], pool_all, task)
            shap_accum += sv
            pred_accum += pv

        point_shap = (shap_accum / K).astype(np.float32)
        point_y_pred = pred_accum / K

    # --- Interaction point estimates ---
    compute_interactions = N_pairs_sig > 0
    if compute_interactions:
        print(f"[INFO] Computing interaction-tensor point estimates for {N_pairs_sig} sig pairs...")
        sig_int_pairs = [pair for pair, v in sig_GII_interaction.items() if v]

        # _shap_interaction_single always returns (N, C, F, F) with C=1 for non-multiclass.
        if mode == "training":
            point_shap_int = np.full(
                (N_target, n_classes, N_features, N_features), np.nan, dtype=np.float32
            )
            for k in range(K):
                fold_mask = fold_of == k
                if not fold_mask.any():
                    continue
                X_fold = X_target.iloc[np.where(fold_mask)[0]]
                pool_fold = Pool(X_fold, cat_features=cat_feats)
                sv_int = _shap_interaction_single(orig_models[k], pool_fold)  # (n_fold, C, F, F)
                point_shap_int[fold_mask] = sv_int
        else:
            int_accum = np.zeros((N_target, n_classes, N_features, N_features), dtype=np.float64)
            pool_all = Pool(X_target, cat_features=cat_feats)
            for k in range(K):
                sv_int = _shap_interaction_single(orig_models[k], pool_all)  # (N, C, F, F)
                int_accum += sv_int
            point_shap_int = (int_accum / K).astype(np.float32)
    else:
        point_shap_int = None
        sig_int_pairs = []

    # --- STEP 4: CI computation (design differs by mode) ---

    if mode == "inference":
        # Bootstrap-of-CV: draw fresh bootstrap samples and K-fold splits per iteration.
        # CIs are basic/reverse-percentile intervals anchored on the original ensemble
        # point estimate. No pre-built refit cache is used for inference CIs.
        print(f"[INFO] Bootstrap-of-CV inference CI: {B} iterations, K={K} folds each...")

        # Load y_train from the persistent cache artifact written by orchestrate_bootstrap_cache.
        y_train_npy_path = os.path.join(cache_dir, "y_train.npy")
        if not os.path.exists(y_train_npy_path):
            raise FileNotFoundError(
                f"y_train.npy not found at {y_train_npy_path}. "
                "Re-run predict.py with shap.indiv_ci_nboot > 0 to rebuild the bootstrap cache. "
                "This artifact is required for inference-mode bootstrap-of-CV CI computation."
            )
        y_train_arr = np.load(y_train_npy_path, allow_pickle=False)

        # Resolve cluster_ids from config and X_train column (mirrors orchestrate_bootstrap_cache).
        infer_cluster_ids: Optional[np.ndarray] = None
        cluster_col = config.get("data", {}).get("cluster_id_col")
        if cluster_col and cluster_col in X_train.columns:
            infer_cluster_ids = X_train[cluster_col].values

        # Frozen fold-specific HPs from the K deployed model_fold_k.cbm files.
        frozen_hps: List[dict] = [
            _extract_user_level_params(m.get_all_params()) for m in orig_models
        ]

        # Build the inference Pool for SHAP computation inside bootstrap-of-CV.
        pool_infer_tgt = Pool(X_target, cat_features=cat_feats)

        random_seed_cfg = int(config.get("execution", {}).get("random_seed", 42))

        main_ci_lo_inf, main_ci_hi_inf, int_ci_lo_inf, int_ci_hi_inf = (
            _bootstrap_of_cv_inference(
                X_train=X_train,
                y_train=y_train_arr,
                inference_pool=pool_infer_tgt,
                fold_hyperparameters=frozen_hps,
                B=B,
                K=K,
                random_seed=random_seed_cfg,
                cluster_ids=infer_cluster_ids,
                task=task,
                nom_feats=nom_feats,
                point_shap_main=point_shap,
                point_shap_int=point_shap_int if compute_interactions else None,
            )
        )
        print("[INFO] Bootstrap-of-CV inference CI computation complete.")

        # Wrap CI arrays in per-individual accessor helpers compatible with
        # the output-row construction block below.  oob_counts = B for all individuals
        # (basic/reverse-percentile uses all B replicates; no OOB filtering needed).
        oob_counts = np.full(N_target, B, dtype=np.int32)

        def _compute_ci_inf_main(i: int):
            """Return (ci_lo, ci_hi) for individual i's main SHAP values."""
            return main_ci_lo_inf[i], main_ci_hi_inf[i]  # each shape (C, F)

        def _compute_ci_inf_int(i: int):
            """Return (ci_lo, ci_hi) for individual i's interaction values, or (None, None)."""
            if int_ci_lo_inf is None:
                return None, None
            return int_ci_lo_inf[i], int_ci_hi_inf[i]  # each shape (C, F, F)

        # Placeholder pred CI: inference-mode does not compute prediction CIs via
        # bootstrap-of-CV (SHAP CIs are the primary deliverable per the CR plan).
        # Emit NaN bounds consistently so the predictions parquet schema is preserved.
        def _compute_ci_inf_pred(i: int):
            return None, None

    else:
        # Training mode: coupled bootstrap using pre-built refit cache.
        # Load shared bootstrap indices from cache.
        print(f"[INFO] Accumulating bootstrap CI distributions from {B} iterations (training mode)...")

        # Pre-allocate ragged accumulators as flat numpy arrays (more memory-efficient).
        # shap_ci_buf shape: (N_target, B, C, F) uniformly across all task types.
        # C = n_classes for multiclass, C = 1 for all other task types.
        shap_ci_buf = np.full((N_target, B, n_classes, N_features), np.nan, dtype=np.float32)
        if task == "multiclass_classification":
            pred_ci_buf = np.full((N_target, B, n_classes), np.nan, dtype=np.float32)
        elif task == "multi_regression":
            pred_ci_buf = np.full((N_target, B, len(outcome_cols)), np.nan, dtype=np.float32)
        else:
            pred_ci_buf = np.full((N_target, B), np.nan, dtype=np.float32)

        oob_counts = np.zeros(N_target, dtype=np.int32)  # per-individual effective OOB count

        if compute_interactions:
            # int_ci_buf shape: (N_target, B, C, F, F) uniformly across all task types.
            int_ci_buf = np.full(
                (N_target, B, n_classes, N_features, N_features), np.nan, dtype=np.float32
            )
        else:
            int_ci_buf = None

        for b in range(B):
            s_b = shared_indices_list[b]
            # Determine OOB membership for training individuals
            in_sample = np.zeros(N_target, dtype=bool)
            in_sample[s_b] = True
            oob_mask = ~in_sample

            # Load K coupled refits from cache
            boot_models = []
            for k in range(K):
                bp = os.path.join(cache_dir, f"iter_{b:05d}", f"fold_{k}.cbm")
                boot_models.append(_load_one_model(bp, task))

            # Compute SHAP for each refit on X_target.
            # _shap_single returns (N, C, F) uniformly; C=1 for non-multiclass.
            shap_iter_folds = np.zeros((K, N_target, n_classes, N_features), dtype=np.float32)
            if task == "multiclass_classification":
                pred_iter_folds = np.zeros((K, N_target, n_classes), dtype=np.float32)
            elif task == "multi_regression":
                pred_iter_folds = np.zeros((K, N_target, len(outcome_cols)), dtype=np.float32)
            else:
                pred_iter_folds = np.zeros((K, N_target), dtype=np.float32)

            pool_tgt = Pool(X_target, cat_features=cat_feats)
            for k in range(K):
                sv_k = _shap_single(boot_models[k], pool_tgt, task, N_target)  # (N, C, F)
                pv_k = _predict_single(boot_models[k], pool_tgt, task).astype(np.float32)
                shap_iter_folds[k] = sv_k
                pred_iter_folds[k] = pv_k

            if compute_interactions:
                # _shap_interaction_single returns (N, C, F, F) uniformly; C=1 for non-multiclass.
                int_iter_folds = np.zeros(
                    (K, N_target, n_classes, N_features, N_features), dtype=np.float32
                )
                for k in range(K):
                    sv_int_k = _shap_interaction_single(boot_models[k], pool_tgt)  # (N, C, F, F)
                    int_iter_folds[k] = sv_int_k
            else:
                int_iter_folds = None

            # For each OOB individual: use ONLY their fold-k_i refit (OOF estimand match)
            for i in np.where(oob_mask)[0]:
                k_i = int(fold_of[i])
                slot = oob_counts[i]
                shap_ci_buf[i, slot] = shap_iter_folds[k_i, i]
                pred_ci_buf[i, slot] = pred_iter_folds[k_i, i]
                if compute_interactions and int_ci_buf is not None and int_iter_folds is not None:
                    int_ci_buf[i, slot] = int_iter_folds[k_i, i]
                oob_counts[i] += 1

            # Release boot_models and per-fold arrays from memory
            del boot_models

            if (b + 1) % max(1, B // 10) == 0:
                print(f"[INFO] CI accumulation: {b + 1}/{B} iterations complete.")

        print("[INFO] CI accumulation complete. Computing percentile CIs...")

    # --- STEP 5: Per-individual aggregation ---
    # Helper: compute percentile CIs with NaN for below-floor individuals (training mode).
    # In inference mode, CIs are already computed as basic/reverse-percentile intervals.

    def _compute_ci(buf_i: np.ndarray, count: int):
        """buf_i has shape (B, ...). Only first `count` slots are valid. Training mode only."""
        if count < OOB_FLOOR_MIN:
            return None, None  # caller fills NaN
        valid = buf_i[:count]
        lo = np.nanpercentile(valid, CI_LO_PCT, axis=0)
        hi = np.nanpercentile(valid, CI_HI_PCT, axis=0)
        return lo, hi

    # --- STEP 6-8: Build output rows ---

    # Resolve class labels for multiclass
    class_labels = None
    if task == "multiclass_classification":
        cl_path = os.path.join(train_dir, "class_labels.json")
        if os.path.exists(cl_path):
            with open(cl_path) as f:
                class_labels = json.load(f)
        else:
            class_labels = [str(c) for c in range(n_classes)]

    ids_list = list(ids_target)
    X_raw = X_target  # raw feature values (already in pre-encoding form or we use as-is)

    main_rows = []
    pred_rows = []

    for i in range(N_target):
        ind_id = str(ids_list[i])
        eff_count = int(oob_counts[i])
        if mode == "inference":
            ci_lo_shap, ci_hi_shap = _compute_ci_inf_main(i)
            ci_lo_pred, ci_hi_pred = _compute_ci_inf_pred(i)
        else:
            ci_lo_shap, ci_hi_shap = _compute_ci(shap_ci_buf[i], eff_count)
            ci_lo_pred, ci_hi_pred = _compute_ci(pred_ci_buf[i], eff_count)

        # Predictions row(s)
        y_true_val = None
        if y_target is not None:
            if isinstance(y_target, pd.DataFrame):
                y_true_val = y_target.iloc[i]
            else:
                y_true_val = y_target.iloc[i]

        if task == "multiclass_classification":
            for c_idx, cl in enumerate(class_labels):
                pt_prob = float(point_y_pred[i, c_idx])
                lo_prob = float(ci_lo_pred[c_idx]) if ci_lo_pred is not None else float("nan")
                hi_prob = float(ci_hi_pred[c_idx]) if ci_hi_pred is not None else float("nan")
                y_true_str = str(y_true_val) if y_true_val is not None else float("nan")
                pred_rows.append({
                    "id": ind_id,
                    "class": str(cl),
                    "prob": pt_prob,
                    "prob_ci_lo": lo_prob,
                    "prob_ci_hi": hi_prob,
                    "prob_oob_count": eff_count,
                    "y_true": y_true_str,
                })
        elif task == "multi_regression":
            row = {"id": ind_id}
            for t_idx, col in enumerate(outcome_cols):
                col_div = divisor[col] if isinstance(divisor, dict) else float(divisor)
                pt_raw = float(point_y_pred[i, t_idx])
                pt_scaled = pt_raw / col_div
                lo_raw = float(ci_lo_pred[t_idx]) if ci_lo_pred is not None else float("nan")
                hi_raw = float(ci_hi_pred[t_idx]) if ci_hi_pred is not None else float("nan")
                lo_scaled = lo_raw / col_div if not np.isnan(lo_raw) else float("nan")
                hi_scaled = hi_raw / col_div if not np.isnan(hi_raw) else float("nan")
                y_true_col = float(y_true_val[col]) if y_true_val is not None else float("nan")
                row[f"y_true_{col}"] = y_true_col
                row[f"y_pred_raw_{col}"] = pt_raw
                row[f"y_pred_scaled_{col}"] = pt_scaled
                row[f"y_pred_ci_lo_{col}"] = lo_scaled
                row[f"y_pred_ci_hi_{col}"] = hi_scaled
            row["y_pred_oob_count"] = eff_count
            pred_rows.append(row)
        else:
            # regression or binary_classification
            div_scalar = float(divisor) if not isinstance(divisor, dict) else list(divisor.values())[0]
            pt_raw = float(point_y_pred[i])
            pt_scaled = pt_raw / div_scalar
            lo_raw = float(ci_lo_pred) if ci_lo_pred is not None else float("nan")
            hi_raw = float(ci_hi_pred) if ci_hi_pred is not None else float("nan")
            lo_scaled = lo_raw / div_scalar if not np.isnan(lo_raw) else float("nan")
            hi_scaled = hi_raw / div_scalar if not np.isnan(hi_raw) else float("nan")
            y_true_flt = float(y_true_val) if y_true_val is not None else float("nan")
            pred_rows.append({
                "id": ind_id,
                "y_pred_raw": pt_raw,
                "y_pred_scaled": pt_scaled,
                "y_pred_ci_lo": lo_scaled,
                "y_pred_ci_hi": hi_scaled,
                "y_pred_oob_count": eff_count,
                "y_true": y_true_flt,
            })

        # Main effects rows
        for f_idx, feat in enumerate(feature_names):
            feat_type = feature_type_map.get(feat, "continuous")
            try:
                feat_val_raw = str(X_raw.iloc[i][feat])
            except Exception:
                feat_val_raw = ""

            is_sig = bool(sig_GII_main.get(feat, False))
            div_scalar = float(divisor) if not isinstance(divisor, dict) else float(divisor.get(outcome_cols[0], 1.0))

            # point_shap is (N_target, C, F) uniformly; ci_lo_shap/ci_hi_shap are (C, F) or None.
            # For multiclass, emit one row per class.  For non-multiclass, C=1; collapse class axis.
            if n_outputs > 1:
                for c_idx, cl in enumerate(class_labels):
                    pt_shap_raw = float(point_shap[i, c_idx, f_idx])
                    pt_shap_scaled = pt_shap_raw / div_scalar
                    if ci_lo_shap is not None:
                        lo_s = float(ci_lo_shap[c_idx, f_idx]) / div_scalar
                        hi_s = float(ci_hi_shap[c_idx, f_idx]) / div_scalar
                    else:
                        lo_s = float("nan")
                        hi_s = float("nan")
                    main_rows.append({
                        "id": ind_id,
                        "feature": feat,
                        "class": str(cl),
                        "feature_value_raw": feat_val_raw,
                        "feature_type": feat_type,
                        "shap_value_raw": pt_shap_raw,
                        "shap_value_scaled": pt_shap_scaled,
                        "shap_value_ci_lo": lo_s,
                        "shap_value_ci_hi": hi_s,
                        "oob_count": eff_count,
                        "sig_GII": is_sig,
                    })
            else:
                # Non-multiclass: C=1 singleton axis; collapse to scalar by indexing c=0.
                pt_shap_raw = float(point_shap[i, 0, f_idx])
                pt_shap_scaled = pt_shap_raw / div_scalar
                if ci_lo_shap is not None:
                    lo_s = float(ci_lo_shap[0, f_idx]) / div_scalar
                    hi_s = float(ci_hi_shap[0, f_idx]) / div_scalar
                else:
                    lo_s = float("nan")
                    hi_s = float("nan")
                main_rows.append({
                    "id": ind_id,
                    "feature": feat,
                    "feature_value_raw": feat_val_raw,
                    "feature_type": feat_type,
                    "shap_value_raw": pt_shap_raw,
                    "shap_value_scaled": pt_shap_scaled,
                    "shap_value_ci_lo": lo_s,
                    "shap_value_ci_hi": hi_s,
                    "oob_count": eff_count,
                    "sig_GII": is_sig,
                })

    # --- Interaction rows ---
    int_rows = []
    if compute_interactions and point_shap_int is not None:
        feat_idx_map = {feat: f_idx for f_idx, feat in enumerate(feature_names)}
        for i in range(N_target):
            ind_id = str(ids_list[i])
            eff_count = int(oob_counts[i])
            if mode == "inference":
                ci_lo_int, ci_hi_int = _compute_ci_inf_int(i)
            else:
                ci_lo_int, ci_hi_int = _compute_ci(int_ci_buf[i], eff_count) if int_ci_buf is not None else (None, None)

            div_scalar = float(divisor) if not isinstance(divisor, dict) else float(divisor.get(outcome_cols[0], 1.0))

            for (fa, fb) in sig_int_pairs:
                fa_idx = feat_idx_map.get(fa)
                fb_idx = feat_idx_map.get(fb)
                if fa_idx is None or fb_idx is None:
                    continue

                fa_type = feature_type_map.get(fa, "continuous")
                fb_type = feature_type_map.get(fb, "continuous")
                try:
                    fa_val = str(X_raw.iloc[i][fa])
                except Exception:
                    fa_val = ""
                try:
                    fb_val = str(X_raw.iloc[i][fb])
                except Exception:
                    fb_val = ""

                # point_shap_int is (N_target, C, F, F) uniformly; ci_lo_int/ci_hi_int are
                # (C, F, F) or None.  Emit one row per class for multiclass; collapse to scalar
                # for non-multiclass (C=1).
                if n_outputs > 1:
                    # Genuine per-class interaction values from the 4D tensor.
                    for c_idx, cl in enumerate(class_labels):
                        pt_int_raw = float(
                            point_shap_int[i, c_idx, fa_idx, fb_idx]
                            + point_shap_int[i, c_idx, fb_idx, fa_idx]
                        )
                        pt_int_scaled = pt_int_raw / div_scalar
                        if ci_lo_int is not None:
                            lo_i = float(
                                ci_lo_int[c_idx, fa_idx, fb_idx]
                                + ci_lo_int[c_idx, fb_idx, fa_idx]
                            ) / div_scalar
                            hi_i = float(
                                ci_hi_int[c_idx, fa_idx, fb_idx]
                                + ci_hi_int[c_idx, fb_idx, fa_idx]
                            ) / div_scalar
                        else:
                            lo_i = float("nan")
                            hi_i = float("nan")
                        int_rows.append({
                            "id": ind_id,
                            "feature_a": fa,
                            "feature_b": fb,
                            "class": str(cl),
                            "feature_a_value_raw": fa_val,
                            "feature_b_value_raw": fb_val,
                            "feature_a_type": fa_type,
                            "feature_b_type": fb_type,
                            "shap_value_raw": pt_int_raw,
                            "shap_value_scaled": pt_int_scaled,
                            "shap_value_ci_lo": lo_i,
                            "shap_value_ci_hi": hi_i,
                            "oob_count": eff_count,
                        })
                else:
                    # Non-multiclass: C=1 singleton axis; collapse by indexing c=0.
                    pt_int_raw = float(
                        point_shap_int[i, 0, fa_idx, fb_idx]
                        + point_shap_int[i, 0, fb_idx, fa_idx]
                    )
                    pt_int_scaled = pt_int_raw / div_scalar
                    if ci_lo_int is not None:
                        lo_i = float(
                            ci_lo_int[0, fa_idx, fb_idx] + ci_lo_int[0, fb_idx, fa_idx]
                        ) / div_scalar
                        hi_i = float(
                            ci_hi_int[0, fa_idx, fb_idx] + ci_hi_int[0, fb_idx, fa_idx]
                        ) / div_scalar
                    else:
                        lo_i = float("nan")
                        hi_i = float("nan")
                    int_rows.append({
                        "id": ind_id,
                        "feature_a": fa,
                        "feature_b": fb,
                        "feature_a_value_raw": fa_val,
                        "feature_b_value_raw": fb_val,
                        "feature_a_type": fa_type,
                        "feature_b_type": fb_type,
                        "shap_value_raw": pt_int_raw,
                        "shap_value_scaled": pt_int_scaled,
                        "shap_value_ci_lo": lo_i,
                        "shap_value_ci_hi": hi_i,
                        "oob_count": eff_count,
                    })

    # --- Emit outputs ---
    print(f"[INFO] Emitting indiv_reports/ parquets to {run_dir}...")
    _emit_main_effects_parquet(run_dir, main_rows)
    _emit_interactions_parquet(run_dir, int_rows)
    _emit_predictions_parquet(run_dir, pred_rows)

    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    _emit_metadata_json(
        run_dir=run_dir,
        scaling_mode=scaling_mode,
        scaling_divisor=divisor if not isinstance(divisor, dict) else divisor,
        B=B,
        K=K,
        oob_floor=OOB_FLOOR_MIN,
        outcome_names=outcome_cols,
        mode=mode,
        timestamp=timestamp,
    )

    print(
        f"[INFO] indiv_reports/ complete: {len(main_rows)} main-effect rows, "
        f"{len(int_rows)} interaction rows, {len(pred_rows)} prediction rows."
    )
