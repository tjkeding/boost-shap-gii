#!/usr/bin/env python3
"""
SHAP Analysis Library for boost-shap-gii
- Tier 1 (Singleton) & Tier 2 (Interaction) Analysis
- Decomposes SHAP into M (Magnitude) and V (Variability)
- V-Calculation: Hybrid Approach (Splines vs Group Means) based on Type/Resolution
- V-Stability A: "Dynamic Resolution" guarantees non-exploding V via Mean Fallback (Density Check).
- V-Stability B: "Energy Ratio" guarantees non-exploding V via Total Variation Check (Physics Check).
- Statistics: Stratified Max Boruta Exceedance Test (per measurement-type stratum).
- Output: 3 Separate Bootstrap Parquets + Microdata + Stratified Noise Distributions.

Statistical Assumptions
-----------------------
1. Bootstrap: Observations are approximately exchangeable within the
   SHAP matrix (not time-series dependent). Default B>=2000 provides
   stable percentile CIs at alpha=0.05.
2. Boruta Exceedance: Null hypothesis for each effect is that its bootstrap
   distribution does not exceed the stratum max-shadow distribution.
   One-sided test: higher = more important.
   Exceedance p-values use the Davison & Hinkley (1997) +1 correction:
   p = (sum(boot <= noise) + 1) / (n_boot + 1). Minimum achievable p = 1/(n_boot+1).
3. Splines (V component): SHAP-vs-feature relationship is smooth enough
   for configured n_knots/degree. Energy conservation gates reject
   overfitting. Auto-downgrade handles sparse unique values.
4. GII = sqrt(M * V): Geometric mean requires both magnitude AND
   structured variability. High M + flat V (no trend) -> low GII.
   High V + tiny M (weak effect) -> low GII.
5. Interaction Scale: Singletons use the diagonal Phi[i,i] (full scale).
   Interactions use Phi[i,j] + Phi[j,i] (both off-diagonal cells). CatBoost
   divides the total interaction contribution by 2 per cell, so the summed
   convention recovers the true Shapley interaction index. This ensures
   singleton and interaction GII are on the same prediction-contribution scale.
6. Boruta Noise Baseline: Shadow SHAP values are conditioned on real features
   (shadow model trained jointly on 2p features). The noise baseline is therefore
   model-adaptive. This is standard Boruta behavior (Kursa & Rudnicki, 2010).
7. NaN handling: V spline failures return NaN (not 0.0). Downstream aggregation
   (nanmean, nanpercentile) excludes failed iterations natively. The v_failure_rate
   diagnostic warns when > 5% of iterations fail for a given effect.
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
import joblib

from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from scipy.interpolate import LSQUnivariateSpline, LSQBivariateSpline
from scipy import stats
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

from .utils import get_cv_splitter, detect_task, is_regression

# -----------------------------------------------------------------------------
# 1. IO & Helpers
# -----------------------------------------------------------------------------

def _to_csv_atomic(df: pd.DataFrame, path: str, **kwargs):
    """Atomic CSV write."""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, **kwargs)
    os.replace(tmp, path)

def _to_parquet_atomic(df: pd.DataFrame, path: str, **kwargs):
    """Atomic Parquet write."""
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False, **kwargs)
    os.replace(tmp, path)

def _discover_fold_models(run_dir: str) -> List[str]:
    import glob
    return sorted(glob.glob(os.path.join(run_dir, "model_fold_*.cbm")))

def _discover_shadow_models(run_dir: str) -> List[str]:
    import glob
    return sorted(glob.glob(os.path.join(run_dir, "shadow_model_fold_*.cbm")))

def _to_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Converts mixed types to numeric for Spline/Stats calculations.
    Categories -> Codes (NaN -> max_code+1 sentinel), Int64 -> Float.
    Continuous NaN -> 0.0 placeholder (routing handled by nan_mask in bootstrap).
    """
    df_num = df.copy()
    for col in df_num.columns:
        if df_num[col].dtype.name == 'category':
            # Already categorical; extract codes directly (no redundant re-cast)
            codes = df_num[col].cat.codes
            max_code = codes.max()
            # NaN (-1 in cat.codes) -> max_code + 1 as distinct sentinel level
            codes = codes.where(codes != -1, max_code + 1)
            df_num[col] = codes
        elif df_num[col].dtype == object:
            codes = df_num[col].astype('category').cat.codes
            max_code = codes.max()
            # NaN (-1 in cat.codes) -> max_code + 1 as distinct sentinel level
            codes = codes.where(codes != -1, max_code + 1)
            df_num[col] = codes
        df_num[col] = df_num[col].astype(float)

        if df_num[col].isnull().any():
            df_num[col] = df_num[col].fillna(0.0)

    return df_num.values

def _get_effect_stratum(effect_name: str, effect_type: str, feature_types: Dict[str, str]) -> str:
    """
    Determine the measurement-type stratum for an effect.
    Singletons: singleton_{continuous|ordinal|nominal}
    Interactions: interaction_{typeA}_{typeB} (alphabetically sorted)
    """
    if effect_type == "Singleton":
        # Strip shadow_ prefix if present to look up base type
        base_name = effect_name[7:] if effect_name.startswith("shadow_") else effect_name
        feat_type = feature_types.get(effect_name, feature_types.get(base_name, "continuous"))
        return f"singleton_{feat_type}"
    else:  # Interaction
        parts = effect_name.split(" x ")
        base_a = parts[0][7:] if parts[0].startswith("shadow_") else parts[0]
        base_b = parts[1][7:] if parts[1].startswith("shadow_") else parts[1]
        type_a = feature_types.get(parts[0], feature_types.get(base_a, "continuous"))
        type_b = feature_types.get(parts[1], feature_types.get(base_b, "continuous"))
        sorted_types = sorted([type_a, type_b])
        return f"interaction_{'_'.join(sorted_types)}"

# -----------------------------------------------------------------------------
# 2. V-Component Logic (Unified Signal Variance)
# -----------------------------------------------------------------------------

def _get_adaptive_knots_and_degree(arr: np.ndarray, target_knots: int, target_degree: int) -> Tuple[np.ndarray, int]:
    """
    Generates knots using percentiles, prunes duplicates (zero-inflation),
    and downgrades degree if too few unique knots exist.
    """
    quantiles = np.linspace(0, 100, target_knots + 2)[1:-1]
    knots = np.percentile(arr, quantiles)
    knots = np.unique(knots)

    min_x, max_x = np.min(arr), np.max(arr)
    knots = knots[(knots > min_x) & (knots < max_x)]

    if len(knots) < 4:
        if target_degree > 1:
            print(f"[SHAP] Spline degree downgraded from {target_degree} to 1 "
                  f"(only {len(knots)} unique interior knots)")
        return knots, 1  # Downgrade to Linear

    return knots, target_degree

def _check_spline_energy_stability_1d(y_raw: np.ndarray, y_spline: np.ndarray) -> bool:
    """
    Gate 2 (1D): Physics Check (Total Variation)
    A smoothed signal cannot have more variation (arc length) than the raw signal.
    """
    # y_raw must be passed in the same sort order as y_spline
    tv_raw = np.sum(np.abs(np.diff(y_raw)))
    tv_spline = np.sum(np.abs(np.diff(y_spline)))

    if tv_raw == 0:
        return True if tv_spline < 1e-9 else False

    # Energy conservation: spline total variation must not exceed raw data's.
    # 1.001 multiplier = floating-point tolerance for scipy splev rounding,
    # not a smoothing allowance. Fallback: group means.
    if tv_spline > (tv_raw * 1.001):
        return False

    return True

def _check_spline_energy_stability_2d(x1: np.ndarray, x2: np.ndarray, y_raw: np.ndarray, y_spline: np.ndarray) -> bool:
    """
    Gate 2 (2D): Physics Check (Axis-Wise Total Variation)
    Extends the 1D concept to surfaces by checking variation along both axes.
    If the spline oscillates in ANY direction more than the data, it fails.
    """
    # 1. Variation along X1 axis
    idx1 = np.argsort(x1)
    tv_raw_1 = np.sum(np.abs(np.diff(y_raw[idx1])))
    tv_spline_1 = np.sum(np.abs(np.diff(y_spline[idx1])))

    # 2. Variation along X2 axis
    idx2 = np.argsort(x2)
    tv_raw_2 = np.sum(np.abs(np.diff(y_raw[idx2])))
    tv_spline_2 = np.sum(np.abs(np.diff(y_spline[idx2])))

    # Sum the energies
    tv_raw_total = tv_raw_1 + tv_raw_2
    tv_spline_total = tv_spline_1 + tv_spline_2

    if tv_raw_total == 0:
        return True if tv_spline_total < 1e-9 else False

    # Energy conservation: spline total variation must not exceed raw data's.
    # 1.001 multiplier = floating-point tolerance for scipy splev rounding,
    # not a smoothing allowance. Fallback: group means.
    if tv_spline_total > (tv_raw_total * 1.001):
        return False

    return True

# --- A. Discrete Logic (Group Means) ---

def calculate_v_group_means_1d(x: np.ndarray, y: np.ndarray) -> float:
    """V = StdDev of Group Means."""
    df_tmp = pd.DataFrame({'x': x, 'y': y})
    signal = df_tmp.groupby('x')['y'].transform('mean').values
    return np.std(signal)

def calculate_v_group_means_2d(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> float:
    """V = StdDev of Group Means (Heatmap)."""
    df_tmp = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    signal = df_tmp.groupby(['x1', 'x2'])['y'].transform('mean').values
    return np.std(signal)

# --- B. Continuous Logic (Splines) with Double Guardrails ---

def calculate_v_spline_1d(x: np.ndarray, y: np.ndarray, n_knots: int, degree: int, discrete_threshold: int) -> float:
    """V = StdDev of 1D Spline Fit (with Density & Energy Checks)."""

    # Gate 1: Density Check (Prevent Overfitting on small N)
    if len(x) <= discrete_threshold:
        return calculate_v_group_means_1d(x, y)

    try:
        order = np.argsort(x)
        xs, ys = x[order], y[order]

        knots, k_adj = _get_adaptive_knots_and_degree(xs, n_knots, degree)

        if len(knots) == 0:
            if len(np.unique(xs)) > 1:
                z = np.polyfit(xs, ys, 1)
                p = np.poly1d(z)
                signal = p(xs)
                return np.std(signal)
            else:
                return 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spline = LSQUnivariateSpline(xs, ys, t=knots, k=k_adj)
            signal = spline(xs)

        # Gate 2: Energy Check (1D)
        if not _check_spline_energy_stability_1d(ys, signal):
            return calculate_v_group_means_1d(x, y)

        return np.std(signal)
    except Exception:
        return np.nan  # NaN distinguishes "spline failure" from "genuinely zero V"

def calculate_v_spline_2d(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, n_knots: int, degree: int,
                          discrete_threshold: int = 15) -> float:
    """V = StdDev of 2D Bivariate Spline (with Variance Inflation Check).

    When one axis lacks sufficient knot resolution, routes to stacked-spline
    (treating the low-resolution axis as a discrete grouping variable). Falls back
    to group-means only when BOTH axes lack resolution (quasi-discrete x quasi-discrete).
    """
    try:
        tx, kx_adj = _get_adaptive_knots_and_degree(x1, n_knots, degree)
        ty, ky_adj = _get_adaptive_knots_and_degree(x2, n_knots, degree)

        x1_ok = len(tx) >= 2
        x2_ok = len(ty) >= 2

        if not x1_ok and not x2_ok:
            # Both axes lack resolution: use 2D group means (appropriate for
            # quasi-discrete x quasi-discrete interactions)
            return calculate_v_group_means_2d(x1, x2, y)
        elif not x1_ok:
            # x1 lacks resolution: treat x1 as discrete grouping, fit 1D splines along x2
            return calculate_v_stacked_spline(x2, x1, y, n_knots, degree, discrete_threshold)
        elif not x2_ok:
            # x2 lacks resolution: treat x2 as discrete grouping, fit 1D splines along x1
            return calculate_v_stacked_spline(x1, x2, y, n_knots, degree, discrete_threshold)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spline = LSQBivariateSpline(x1, x2, y, tx, ty, kx=kx_adj, ky=ky_adj)
            signal = spline.ev(x1, x2)

        # Gate 2: Energy Check (2D Axis-Wise)
        if not _check_spline_energy_stability_2d(x1, x2, y, signal):
            return calculate_v_group_means_2d(x1, x2, y)

        return np.std(signal)
    except Exception:
        return np.nan  # NaN distinguishes "spline failure" from "genuinely zero V"

# --- C. Mixed Logic (Stacked Splines) with Double Guardrails ---

def calculate_v_stacked_spline(x_cont: np.ndarray, x_disc: np.ndarray, y: np.ndarray,
                               n_knots: int, degree: int, discrete_threshold: int) -> float:
    """
    Splits data by x_disc. Fits 1D Spline or Mean to x_cont for each group.
    """
    df = pd.DataFrame({'xc': x_cont, 'xd': x_disc, 'y': y})
    df['signal'] = 0.0

    groups = df['xd'].unique()
    for g in groups:
        mask = df['xd'] == g
        sub = df[mask]

        # Gate 1: Density Check
        if len(sub) <= discrete_threshold:
            df.loc[mask, 'signal'] = np.mean(sub['y'].values)
            continue

        # Fit 1D Spline
        xs = sub['xc'].values
        ys = sub['y'].values

        try:
            order = np.argsort(xs)
            xs_sort, ys_sort = xs[order], ys[order]

            knots, k_adj = _get_adaptive_knots_and_degree(xs_sort, n_knots, degree)

            if len(knots) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    spline = LSQUnivariateSpline(xs_sort, ys_sort, t=knots, k=k_adj)
                    pred = spline(xs)

                    # Gate 2: Energy Check (1D Check on this slice)
                    pred_sorted = spline(xs_sort)
                    if not _check_spline_energy_stability_1d(ys_sort, pred_sorted):
                        pred = np.mean(ys) # Fallback
            else:
                if len(np.unique(xs)) > 1:
                    z = np.polyfit(xs, ys, 1)
                    p = np.poly1d(z)
                    pred = p(xs)
                else:
                    pred = np.mean(ys)

            df.loc[mask, 'signal'] = pred

        except Exception:
            df.loc[mask, 'signal'] = np.mean(ys)

    return np.std(df['signal'].values)

# -----------------------------------------------------------------------------
# 3. Core SHAP Computation
# -----------------------------------------------------------------------------

def _compute_interaction_values(
    model, X: pd.DataFrame, cat_idx: List[int], slice_idx: Optional[int] = None
) -> np.ndarray:
    """Compute SHAP interaction values.

    For single-output models (3D tensor), returns (N, F, F).
    For multi-output models (4D tensor: N x K x F+1 x F+1):
      - If slice_idx is None, returns the first slice (backward compat).
      - If slice_idx is given, returns that class/target slice.
    """
    pool = Pool(X, cat_features=cat_idx if cat_idx else None)
    raw = model.get_feature_importance(data=pool, type="ShapInteractionValues")

    if raw.ndim == 3:
        return raw[:, :-1, :-1]
    elif raw.ndim == 4:
        idx = slice_idx if slice_idx is not None else 0
        return raw[:, idx, :-1, :-1]
    else:
        raise ValueError(f"Unexpected SHAP shape: {raw.shape}")

def _flatten_interaction_matrix(
    phi_inter: np.ndarray,
    feature_names: List[str],
    effect_filter: str = "real"
) -> Tuple[pd.DataFrame, Dict[str, Tuple[Tuple[int, int], str]]]:
    """Flatten the SHAP interaction matrix into a per-effect DataFrame.

    Scale convention:
    - Singletons: Phi[i,i] (diagonal) — stored at full scale.
    - Interactions: Phi[i,j] + Phi[j,i] (both off-diagonal cells). CatBoost's
      ShapInteractionValues divides the total interaction contribution equally across
      both off-diagonal cells, so the sum recovers the true Shapley interaction index.
      This ensures singletons and interactions are on the same prediction-contribution
      scale, making cross-type GII comparisons valid.

    Parameters
    ----------
    phi_inter : np.ndarray, shape (N, M, M)
        SHAP interaction matrix for N observations and M features.
    feature_names : list[str]
        Feature names corresponding to the M columns of phi_inter.
    effect_filter : str
        "real"        — singletons/interactions where ALL features are real (no shadow_)
        "shadow_pure" — singletons where feature IS shadow; interactions where BOTH are shadow

    Returns
    -------
    df_flat : pd.DataFrame
        Columns are effect names; rows are observations.
    metadata : dict
        Maps effect name to ((i, j), type_str) where type_str is "Singleton" or "Interaction".
    """
    N, M, _ = phi_inter.shape
    data = {}
    metadata = {}

    # 1. Singletons
    for i in range(M):
        name = feature_names[i]
        is_shadow = name.startswith("shadow_")

        if effect_filter == "real" and is_shadow:
            continue
        if effect_filter == "shadow_pure" and not is_shadow:
            continue

        data[name] = phi_inter[:, i, i]
        metadata[name] = ((i, i), "Singleton")

    # 2. Interactions — deterministic filter: include if any sample is non-zero
    for i, j in combinations(range(M), 2):
        name_i = feature_names[i]
        name_j = feature_names[j]
        is_shadow_i = name_i.startswith("shadow_")
        is_shadow_j = name_j.startswith("shadow_")

        if effect_filter == "real" and (is_shadow_i or is_shadow_j):
            continue
        if effect_filter == "shadow_pure" and not (is_shadow_i and is_shadow_j):
            continue

        val = phi_inter[:, i, j] + phi_inter[:, j, i]
        if np.any(val != 0):
            name = f"{name_i} x {name_j}"
            data[name] = val
            metadata[name] = ((i, j), "Interaction")

    df_flat = pd.DataFrame(data)
    return df_flat, metadata

# -----------------------------------------------------------------------------
# 4. Bootstrap Engine & Output Handlers
# -----------------------------------------------------------------------------

def _bootstrap_worker_chunk(
    indices_chunk: np.ndarray,
    X_vals: np.ndarray,
    SHAP_vals: np.ndarray,
    effect_indices: List[Tuple[int, int]],
    spline_cfg: Dict[str, int],
    feature_names: List[str],
    feature_types: Dict[str, str],
    discrete_threshold: int,
    nan_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute M, V, and GII for a chunk of bootstrap iterations.

    For each bootstrap resample:
      - M (Magnitude) = mean(|SHAP|) across observations
      - V (Variability) = std(signal), where signal is the fitted trend
        of SHAP values as a function of feature values (spline for continuous,
        group means for discrete)
      - GII (Global Importance Index) = sqrt(M * V), the geometric mean
        of magnitude and variability

    Returns (chunk_mag, chunk_var, chunk_gii), each shaped (n_iter, n_effects).
    """
    n_iter, n_samples = indices_chunk.shape
    n_effects = SHAP_vals.shape[1]

    chunk_mag = np.zeros((n_iter, n_effects), dtype=np.float32)
    chunk_var = np.zeros((n_iter, n_effects), dtype=np.float32)
    chunk_gii = np.zeros((n_iter, n_effects), dtype=np.float32)

    n_knots = spline_cfg["n_knots"]
    degree = spline_cfg["degree"]

    feat_props = []
    for i, name in enumerate(feature_names):
        ftype = feature_types.get(name, "continuous")
        is_nominal = (ftype == "nominal")
        feat_props.append(is_nominal)

    # NOTE: get_nature() evaluates per bootstrap resample, so a feature
    # near discrete_threshold may be treated as continuous in one iteration
    # and discrete in another. This is intentional — it reflects genuine
    # uncertainty. The stab_thresh stability gate filters effects whose
    # estimates are unstable from this switching.
    def get_nature(feat_idx, vec):
        is_nom = feat_props[feat_idx]
        if is_nom:
            return "discrete"
        if len(np.unique(vec)) <= discrete_threshold:
            return "discrete"
        return "continuous"

    for i in range(n_iter):
        idx = indices_chunk[i]
        s_res = SHAP_vals[idx]
        x_res = X_vals[idx]
        nan_res = nan_mask[idx] if nan_mask is not None else None

        # M uses ALL samples (SHAP exists for every observation)
        m = np.mean(np.abs(s_res), axis=0)

        v = np.zeros(n_effects)
        for e in range(n_effects):
            idx_a, idx_b = effect_indices[e]
            shap_vec = s_res[:, e]

            if idx_a == idx_b:  # Singleton
                vec_a = x_res[:, idx_a]
                nature_a = get_nature(idx_a, vec_a)
                has_nan = nan_res is not None and np.any(nan_res[:, idx_a])

                if has_nan and nature_a == "continuous":
                    # Approach A: stacked-spline with NaN indicator as discrete axis
                    nan_ind = nan_res[:, idx_a].astype(float)
                    v[e] = calculate_v_stacked_spline(
                        vec_a, nan_ind, shap_vec, n_knots, degree, discrete_threshold
                    )
                elif nature_a == "discrete":
                    # Discrete: NaN encoded as max_code+1 sentinel → group means handle it
                    v[e] = calculate_v_group_means_1d(vec_a, shap_vec)
                else:
                    v[e] = calculate_v_spline_1d(vec_a, shap_vec, n_knots, degree, discrete_threshold)
            else:  # Interaction
                vec_a = x_res[:, idx_a]
                vec_b = x_res[:, idx_b]
                shap_v = shap_vec

                # Exclude NaN rows for interactions
                if nan_res is not None:
                    valid = ~(nan_res[:, idx_a] | nan_res[:, idx_b])
                    if np.sum(valid) < 3:
                        v[e] = 0.0
                        continue
                    vec_a = vec_a[valid]
                    vec_b = vec_b[valid]
                    shap_v = shap_v[valid]

                nature_a = get_nature(idx_a, vec_a)
                nature_b = get_nature(idx_b, vec_b)

                if nature_a == "discrete" and nature_b == "discrete":
                    v[e] = calculate_v_group_means_2d(vec_a, vec_b, shap_v)
                elif nature_a == "continuous" and nature_b == "continuous":
                    v[e] = calculate_v_spline_2d(vec_a, vec_b, shap_v, n_knots, degree, discrete_threshold)
                else:
                    if nature_a == "continuous":
                        v[e] = calculate_v_stacked_spline(vec_a, vec_b, shap_v, n_knots, degree, discrete_threshold)
                    else:
                        v[e] = calculate_v_stacked_spline(vec_b, vec_a, shap_v, n_knots, degree, discrete_threshold)

        chunk_mag[i] = m
        chunk_var[i] = v
        chunk_gii[i] = np.sqrt(m * v)

    return chunk_mag, chunk_var, chunk_gii

def _process_and_save_bootstraps(
    boot_matrix: np.ndarray,
    effect_names: List[str],
    significance_mask: np.ndarray,
    p_values: np.ndarray,
    output_n: int,
    out_path: str
):
    """Saves bootstrap distributions for Sig + Top N Non-Sig features."""
    n_effects = len(effect_names)
    df_meta = pd.DataFrame({
        "idx": range(n_effects), "sig": significance_mask, "p": p_values
    })

    df_meta = df_meta.sort_values(by=["sig", "p"], ascending=[False, True])
    n_save = significance_mask.sum() + output_n
    indices = df_meta["idx"].iloc[:n_save].values

    if len(indices) > 0:
        data = {effect_names[idx]: boot_matrix[:, idx] for idx in indices}
        _to_parquet_atomic(pd.DataFrame(data), out_path)

def _process_and_save_microdata(
    df_shap: pd.DataFrame,
    df_X: pd.DataFrame,
    effect_names: List[str],
    significance_mask: np.ndarray,
    p_values: np.ndarray,
    output_n: int,
    out_path: str,
    df_X_raw: pd.DataFrame,  # NEW
    ids: pd.Series,          # NEW
    feature_types: Dict[str, str] # NEW
):
    n_effects = len(effect_names)
    df_meta = pd.DataFrame({
        "idx": range(n_effects), "sig": significance_mask, "p": p_values
    })

    df_meta = df_meta.sort_values(by=["sig", "p"], ascending=[False, True])
    n_save = significance_mask.sum() + output_n
    indices = df_meta["idx"].iloc[:n_save].values

    if len(indices) == 0: return

    micro_rows = []

    # Pre-extract ID list to array
    id_vals = ids.values if hasattr(ids, 'values') else np.array(ids)

    for idx in indices:
        eff = effect_names[idx]
        phi = df_shap[eff].values

        # Build block
        block = pd.DataFrame({
            "id": id_vals,
            "effect_name": eff,
            "shap_value": phi
        })

        if " x " in eff:
            parts = eff.split(" x ")
            fa, fb = parts[0], parts[1]

            block["main_feature"] = fa
            block["interaction_partner"] = fb

            block["feature_value"] = df_X[fa].astype(str).values
            block["partner_value"] = df_X[fb].astype(str).values

            # NEW Additive Columns
            block["main_feature_raw"] = df_X_raw[fa].astype(str).values
            block["main_feature_type"] = feature_types.get(fa, "unknown")
            block["partner_feature_raw"] = df_X_raw[fb].astype(str).values
            block["partner_feature_type"] = feature_types.get(fb, "unknown")

        else:
            block["main_feature"] = eff
            block["interaction_partner"] = None

            block["feature_value"] = df_X[eff].astype(str).values
            block["partner_value"] = None

            # NEW Additive Columns
            block["main_feature_raw"] = df_X_raw[eff].astype(str).values
            block["main_feature_type"] = feature_types.get(eff, "unknown")
            block["partner_feature_raw"] = None
            block["partner_feature_type"] = None

        micro_rows.append(block)

    if micro_rows:
        _to_parquet_atomic(pd.concat(micro_rows, ignore_index=True), out_path)

def _run_bootstrap_pipeline(
    df_shap: pd.DataFrame,
    df_shap_shadow: pd.DataFrame,
    X_full: pd.DataFrame,
    effect_indices: List[Tuple[int, int]],
    shadow_effect_indices: List[Tuple[int, int]],
    effect_names: List[str],
    shadow_effect_names: List[str],
    effect_types: List[str],
    config: Dict[str, Any],
    out_dir: str,
    feature_names: List[str],
    feature_types: Dict[str, str],
    nan_mask: np.ndarray,
    X_raw_metadata: pd.DataFrame,
    ids: pd.Series,
    cluster_ids: Optional[np.ndarray] = None,
    X_display: Optional[pd.DataFrame] = None
) -> pd.DataFrame:

    # Prep Matrices — single numeric conversion for both real and shadow
    SHAP_vals = df_shap.values
    X_vals = _to_numeric_matrix(X_full)

    SHAP_vals_shadow = df_shap_shadow.values if not df_shap_shadow.empty else None

    # For microdata: extract real-feature columns from X_full
    real_feature_names = [n for n in feature_names if not n.startswith("shadow_")]
    X_real_for_micro = X_full[real_feature_names]
    real_feature_types = {k: v for k, v in feature_types.items() if not k.startswith("shadow_")}

    # Config
    n_jobs = config["execution"]["n_jobs"]
    seed = config["execution"]["random_seed"]
    n_boot = config["shap"]["bootstrapping"]["n_boot"]
    alpha = config["shap"]["bootstrapping"]["alpha"]
    fdr_correct = bool(config["shap"]["bootstrapping"]["fdr_correct"])
    output_boots_n = config["shap"]["bootstrapping"]["output_boots_n"]
    output_micro_n = config["shap"]["output_microdata_n"]
    spline_cfg = config["shap"]["splines"]
    discrete_threshold = config["shap"]["splines"]["discrete_threshold"]
    stab_thresh = float(config["shap"]["bootstrapping"]["stab_thresh"])

    n_samples = SHAP_vals.shape[0]
    n_features = SHAP_vals.shape[1]

    # 1. Bootstrap Execution (Real)
    rng = np.random.default_rng(seed)

    if cluster_ids is not None:
        # Cluster-aware bootstrap: resample at observation level, expand to
        # all K rows per cluster.  This preserves fold-to-fold variation
        # within each observation and produces correct CIs.
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)
        cluster_to_rows: Dict[Any, np.ndarray] = {}
        for cid in unique_clusters:
            cluster_to_rows[cid] = np.where(cluster_ids == cid)[0]
        cluster_sizes = np.array([len(cluster_to_rows[c]) for c in unique_clusters])
        assert np.all(cluster_sizes == cluster_sizes[0]), (
            f"Cluster bootstrap requires equal cluster sizes (K folds per obs), "
            f"but found sizes {np.unique(cluster_sizes)}"
        )
        rows_per_cluster = cluster_sizes[0]
        # For each bootstrap iteration, sample N cluster IDs with replacement,
        # then expand to all K rows per cluster → K*N rows per resample.
        sampled_cluster_idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
        all_indices = np.empty((n_boot, n_clusters * rows_per_cluster), dtype=np.intp)
        for b in range(n_boot):
            expanded = np.concatenate(
                [cluster_to_rows[unique_clusters[c]] for c in sampled_cluster_idx[b]]
            )
            all_indices[b] = expanded
    else:
        all_indices = rng.integers(0, n_samples, size=(n_boot, n_samples))

    indices_split = np.array_split(all_indices, max(1, n_jobs))

    print(f"[SHAP] Bootstrapping {n_features} Real effects (B={n_boot})...")

    res_real = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_bootstrap_worker_chunk)(
            chunk, X_vals, SHAP_vals, effect_indices, spline_cfg,
            feature_names, feature_types, discrete_threshold, nan_mask
        ) for chunk in indices_split
    )

    boot_mag = np.concatenate([r[0] for r in res_real], axis=0)
    boot_var = np.concatenate([r[1] for r in res_real], axis=0)
    boot_gii = np.concatenate([r[2] for r in res_real], axis=0)

    # Track V spline failure rate before downstream aggregation.
    v_failure_rate = np.isnan(boot_var).mean(axis=0)
    for e_idx, e_name in enumerate(effect_names):
        if v_failure_rate[e_idx] > 0.05:
            print(f"[WARNING] Effect '{e_name}': V spline failed in "
                  f"{v_failure_rate[e_idx]*100:.1f}% of bootstrap iterations")

    # NaN values in boot_var/boot_gii (from spline failures) are intentionally
    # preserved. Downstream aggregation (nanmean, nanpercentile) excludes them
    # natively. Replacing with 0.0 would conflate computational failure with
    # genuinely zero V, biasing GII estimates downward and inflating p-values.

    # 2. Bootstrap Execution (Shadow/Noise) & Stratified Max Boruta
    stratified_noise_m = np.zeros((n_boot, n_features))
    stratified_noise_v = np.zeros((n_boot, n_features))
    stratified_noise_gii = np.zeros((n_boot, n_features))
    real_strata = [_get_effect_stratum(n, t, feature_types)
                   for n, t in zip(effect_names, effect_types)]

    if SHAP_vals_shadow is not None:
        n_shadow = SHAP_vals_shadow.shape[1]
        print(f"[SHAP] Bootstrapping {n_shadow} Shadow effects (Stratified Max Noise)...")
        res_shadow = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_bootstrap_worker_chunk)(
                chunk, X_vals, SHAP_vals_shadow, shadow_effect_indices, spline_cfg,
                feature_names, feature_types, discrete_threshold, nan_mask
            ) for chunk in indices_split
        )

        s_boot_mag = np.concatenate([r[0] for r in res_shadow], axis=0)
        s_boot_var = np.concatenate([r[1] for r in res_shadow], axis=0)
        s_boot_gii = np.concatenate([r[2] for r in res_shadow], axis=0)

        # Determine stratum for each shadow effect
        shadow_types = ["Singleton" if " x " not in n else "Interaction"
                        for n in shadow_effect_names]
        shadow_strata = [_get_effect_stratum(n, t, feature_types)
                         for n, t in zip(shadow_effect_names, shadow_types)]

        # Per-stratum max across shadow effects (per bootstrap iteration)
        unique_strata = sorted(set(real_strata))

        # Compute global (un-stratified) max as fallback for empty strata.
        # This occurs when a stratum has no shadow effects of that type.
        # Using global max is conservative: real effects must exceed the
        # strongest noise from ANY stratum, making significance harder
        # (not easier) to achieve.
        global_noise_mag = (
            np.nanmax(s_boot_mag, axis=1) if s_boot_mag.shape[1] > 0 else np.zeros(n_boot)
        )
        global_noise_var = (
            np.nanmax(s_boot_var, axis=1) if s_boot_var.shape[1] > 0 else np.zeros(n_boot)
        )
        global_noise_gii = (
            np.nanmax(s_boot_gii, axis=1) if s_boot_gii.shape[1] > 0 else np.zeros(n_boot)
        )

        stratum_noise = {}
        for stratum in unique_strata:
            s_indices = [i for i, s in enumerate(shadow_strata) if s == stratum]
            if s_indices:
                stratum_noise[stratum] = (
                    np.nanmax(s_boot_mag[:, s_indices], axis=1),
                    np.nanmax(s_boot_var[:, s_indices], axis=1),
                    np.nanmax(s_boot_gii[:, s_indices], axis=1),
                )
            else:
                print(f"[SHAP]   WARNING: Stratum '{stratum}' has 0 shadow effects; "
                      f"falling back to global max noise")
                stratum_noise[stratum] = (
                    global_noise_mag.copy(),
                    global_noise_var.copy(),
                    global_noise_gii.copy(),
                )

        # Log stratum counts
        for stratum in unique_strata:
            n_real_in = sum(1 for s in real_strata if s == stratum)
            n_shadow_in = sum(1 for s in shadow_strata if s == stratum)
            print(f"[SHAP]   Stratum '{stratum}': {n_real_in} real, {n_shadow_in} shadow")

        # Assign per-stratum max noise to each real effect
        for e in range(n_features):
            stratum = real_strata[e]
            stratified_noise_m[:, e] = stratum_noise[stratum][0]
            stratified_noise_v[:, e] = stratum_noise[stratum][1]
            stratified_noise_gii[:, e] = stratum_noise[stratum][2]

        # Save noise distributions (per-effect columns for R plotting compatibility)
        _to_parquet_atomic(
            pd.DataFrame({effect_names[e]: stratified_noise_gii[:, e] for e in range(n_features)}),
            os.path.join(out_dir, "stratified_noise_distributions_GII.parquet"))
        _to_parquet_atomic(
            pd.DataFrame({effect_names[e]: stratified_noise_m[:, e] for e in range(n_features)}),
            os.path.join(out_dir, "stratified_noise_distributions_M.parquet"))
        _to_parquet_atomic(
            pd.DataFrame({effect_names[e]: stratified_noise_v[:, e] for e in range(n_features)}),
            os.path.join(out_dir, "stratified_noise_distributions_V.parquet"))

    # 3. Stats & Hypothesis Testing (Stratified Max Exceedance)
    # NaN-aware aggregation: rare V spline failures (< 0.1% of iterations) should
    # not propagate to summary stats. np.nanmean/nanpercentile/nanmedian ignore them.
    obs_mag = np.nanmean(boot_mag, axis=0)
    obs_var = np.nanmean(boot_var, axis=0)
    obs_gii = np.nanmean(boot_gii, axis=0)

    # Percentile-Based Bootstrap Stability (median / CI_width)
    _pct_lo = 100 * (alpha / 2.0)
    _pct_hi = 100 * (1.0 - alpha / 2.0)
    ci_w_m = np.nanpercentile(boot_mag, _pct_hi, axis=0) - np.nanpercentile(boot_mag, _pct_lo, axis=0)
    ci_w_v = np.nanpercentile(boot_var, _pct_hi, axis=0) - np.nanpercentile(boot_var, _pct_lo, axis=0)
    ci_w_gii = np.nanpercentile(boot_gii, _pct_hi, axis=0) - np.nanpercentile(boot_gii, _pct_lo, axis=0)
    # Stability = median / CI_width. When CI width ≈ 0 (degenerate: all bootstrap
    # iterations identical), stab = 0 — a zero-variance estimate is NOT "stable",
    # it's uninformative. Only a non-zero median with a meaningfully narrow
    # (but non-zero) CI should produce a high stability score.
    def _safe_stability(median_arr, ci_w_arr):
        stab = np.zeros_like(median_arr)
        valid = ci_w_arr > 1e-12
        stab[valid] = median_arr[valid] / ci_w_arr[valid]
        return stab

    stab_m = _safe_stability(np.nanmedian(boot_mag, axis=0), ci_w_m)
    stab_v = _safe_stability(np.nanmedian(boot_var, axis=0), ci_w_v)
    stab_gii = _safe_stability(np.nanmedian(boot_gii, axis=0), ci_w_gii)

    # Stratified Max Exceedance P-Values (Real <= Stratum Max Shadow)
    # +1 correction (Davison & Hinkley 1997; Phipson & Smyth 2010): prevents
    # p=0 and is consistent with compute_permutation_test() in utils.py.
    # Minimum achievable p = 1 / (n_boot + 1).
    p_exceed_m = (np.nansum(boot_mag <= stratified_noise_m, axis=0) + 1) / (n_boot + 1)
    p_exceed_v = (np.nansum(boot_var <= stratified_noise_v, axis=0) + 1) / (n_boot + 1)
    p_exceed_gii = (np.nansum(boot_gii <= stratified_noise_gii, axis=0) + 1) / (n_boot + 1)

    # FDR Correction — NaN-safe: multipletests() returns all-NaN q-values if ANY
    # input p-value is NaN. Mask NaN→1.0 (conservative: "not significant"), run
    # correction, then restore NaN for affected positions.
    def _nan_safe_fdr(p_vals, alpha_val):
        nan_mask_p = np.isnan(p_vals)
        if nan_mask_p.any():
            p_clean = p_vals.copy()
            p_clean[nan_mask_p] = 1.0  # conservative placeholder
            q_vals = multipletests(p_clean, alpha=alpha_val, method='fdr_bh')[1]
            q_vals[nan_mask_p] = np.nan  # restore NaN for failed effects
            return q_vals
        return multipletests(p_vals, alpha=alpha_val, method='fdr_bh')[1]

    q_exceed_m = _nan_safe_fdr(p_exceed_m, alpha)
    q_exceed_v = _nan_safe_fdr(p_exceed_v, alpha)
    q_exceed_gii = _nan_safe_fdr(p_exceed_gii, alpha)

    # Strict Significance: q < alpha AND percentile stability > stab_thresh
    if fdr_correct:
        sig_m = (q_exceed_m < alpha) & (stab_m > stab_thresh)
        sig_v = (q_exceed_v < alpha) & (stab_v > stab_thresh)
        sig_gii = (q_exceed_gii < alpha) & (stab_gii > stab_thresh)
    else:
        sig_m = (p_exceed_m < alpha) & (stab_m > stab_thresh)
        sig_v = (p_exceed_v < alpha) & (stab_v > stab_thresh)
        sig_gii = (p_exceed_gii < alpha) & (stab_gii > stab_thresh)

    # CIs (NaN-aware)
    pct_low, pct_high = 100 * (alpha / 2.0), 100 * (1.0 - alpha / 2.0)
    ci_l_m, ci_h_m = np.nanpercentile(boot_mag, pct_low, axis=0), np.nanpercentile(boot_mag, pct_high, axis=0)
    ci_l_v, ci_h_v = np.nanpercentile(boot_var, pct_low, axis=0), np.nanpercentile(boot_var, pct_high, axis=0)
    ci_l_g, ci_h_g = np.nanpercentile(boot_gii, pct_low, axis=0), np.nanpercentile(boot_gii, pct_high, axis=0)

    # 4. CSV Output
    df_res = pd.DataFrame({
        "effect": effect_names,
        "type": effect_types,
        "noise_stratum": real_strata,

        "GII": obs_gii, "GII_ci_low": ci_l_g, "GII_ci_high": ci_h_g,
        "p_exceed_GII": p_exceed_gii, "q_exceed_GII": q_exceed_gii,
        "stab_pctl_GII": stab_gii, "sig_GII": sig_gii,

        "M": obs_mag, "M_ci_low": ci_l_m, "M_ci_high": ci_h_m,
        "p_exceed_M": p_exceed_m, "q_exceed_M": q_exceed_m,
        "stab_pctl_M": stab_m, "sig_M": sig_m,

        "V": obs_var, "V_ci_low": ci_l_v, "V_ci_high": ci_h_v,
        "p_exceed_V": p_exceed_v, "q_exceed_V": q_exceed_v,
        "stab_pctl_V": stab_v, "sig_V": sig_v,

        # calc_failed: True when any point estimate is NaN (couldn't be calculated).
        # Distinguishes "not significant" from "calculation failed."
        "calc_failed": np.isnan(obs_gii) | np.isnan(obs_mag) | np.isnan(obs_var),

        # v_failure_rate: fraction of bootstrap iterations where V spline fitting
        # failed (returned NaN). High rates (>5%) indicate unreliable V estimates.
        "v_failure_rate": v_failure_rate,
    })

    # Rank by Q-value then GII
    df_res["_rank"] = df_res["q_exceed_GII"]
    df_res.loc[~df_res["sig_GII"], "_rank"] += 1000
    df_res = df_res.sort_values(["_rank", "GII"], ascending=[True, False]).drop(columns=["_rank"])

    _to_csv_atomic(df_res, os.path.join(out_dir, "shap_stats_global.csv"))

    # 5. Save Bootstraps (Real)
    print(f"[SHAP] Saving Bootstrap Parquets (Top {output_boots_n} extra)...")
    _process_and_save_bootstraps(boot_mag, effect_names, sig_gii, q_exceed_gii, output_boots_n,
                                 os.path.join(out_dir, "bootstrap_distributions_M.parquet"))
    _process_and_save_bootstraps(boot_var, effect_names, sig_v, q_exceed_v, output_boots_n,
                                 os.path.join(out_dir, "bootstrap_distributions_V.parquet"))
    _process_and_save_bootstraps(boot_gii, effect_names, sig_gii, q_exceed_gii, output_boots_n,
                                 os.path.join(out_dir, "bootstrap_distributions_GII.parquet"))

    # 6. Save Microdata (Real)
    # When cluster_ids is present (inference mode), microdata should show one row
    # per observation (averaged across K folds), not K duplicates.
    print(f"[SHAP] Saving Microdata Parquets (Top {output_micro_n} extra)...")
    if cluster_ids is not None:
        df_shap_micro = df_shap.copy()
        df_shap_micro.index = cluster_ids
        df_shap_micro = df_shap_micro.groupby(level=0).mean()
        X_micro = X_display[real_feature_names] if X_display is not None else X_real_for_micro.iloc[:len(df_shap_micro)]
        X_micro.index = df_shap_micro.index
        ids_micro = ids
    else:
        df_shap_micro = df_shap
        X_micro = X_real_for_micro
        ids_micro = ids
    _process_and_save_microdata(df_shap_micro, X_micro, effect_names, sig_gii, q_exceed_gii, output_micro_n,
                                os.path.join(out_dir, "microdata_M.parquet"), X_raw_metadata, ids_micro, real_feature_types)
    _process_and_save_microdata(df_shap_micro, X_micro, effect_names, sig_v, q_exceed_v, output_micro_n,
                                os.path.join(out_dir, "microdata_V.parquet"), X_raw_metadata, ids_micro, real_feature_types)
    _process_and_save_microdata(df_shap_micro, X_micro, effect_names, sig_gii, q_exceed_gii, output_micro_n,
                                os.path.join(out_dir, "microdata_GII.parquet"), X_raw_metadata, ids_micro, real_feature_types)

    return df_res

# -----------------------------------------------------------------------------
# 5. Main Driver
# -----------------------------------------------------------------------------

def _run_shap_for_slice(
    ctx: Dict[str, Any],
    shap_dir: str,
    shadow_paths: List[str],
    splits: list,
    all_feature_types: Dict[str, str],
    slice_idx: Optional[int] = None,
    slice_label: Optional[str] = None,
    inference_mode: bool = False,
) -> None:
    """Run the full SHAP pipeline for a single class/target slice (or the whole model).

    When inference_mode=True, every fold uses the full dataset as val_idx and
    SHAP values are averaged across folds (instead of concatenated by OOF index).
    """
    config = ctx["config"]
    task = ctx["task"]
    X_aligned = ctx["X"]
    feature_names = ctx["feature_names"]
    cat_feats = ctx["cat_features"]
    feature_types = ctx.get("feature_types", {})
    X_raw = ctx.get("X_raw", None)
    ids = ctx.get("ids", None)
    n_jobs = config["execution"]["n_jobs"]

    label_str = f" [slice={slice_label}]" if slice_label else ""
    mode_str = "Inference" if inference_mode else "OOF"

    # --- Boruta Fold Processor ---
    def _process_boruta_fold(fold_data):
        fold_idx, (model_path, (_, val_idx)) = fold_data

        X_val_real = X_aligned.iloc[val_idx]

        rng = np.random.default_rng(config["execution"]["random_seed"] + fold_idx + 1000)
        X_val_shadow = X_val_real.copy()
        for c in X_val_shadow.columns:
            X_val_shadow[c] = rng.permutation(X_val_shadow[c].values)
        X_val_shadow.columns = [f"shadow_{c}" for c in X_val_shadow.columns]

        X_full = pd.concat([X_val_real, X_val_shadow], axis=1)
        all_names = list(X_full.columns)

        if is_regression(task):
            m = CatBoostRegressor()
        else:
            m = CatBoostClassifier()
        m.load_model(model_path)

        full_cat_names = cat_feats + [f"shadow_{c}" for c in cat_feats]
        full_cat_idx = [X_full.columns.get_loc(c) for c in full_cat_names if c in X_full.columns]

        phi = _compute_interaction_values(m, X_full, full_cat_idx, slice_idx=slice_idx)

        flat_real, meta_real = _flatten_interaction_matrix(phi, all_names, effect_filter="real")
        flat_shadow, meta_shadow = _flatten_interaction_matrix(phi, all_names, effect_filter="shadow_pure")

        return flat_real, meta_real, flat_shadow, meta_shadow, val_idx, X_full

    # --- Process All Folds ---
    print(f"[SHAP] Processing {len(shadow_paths)} Boruta Folds ({mode_str} Mode){label_str}...")
    fold_iter = enumerate(zip(shadow_paths, splits))
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_boruta_fold)(item) for item in fold_iter
    )

    # Merge across folds
    meta_real = {}
    meta_shadow_all = {}
    chunks_real = []
    chunks_shadow = []
    chunks_X = []

    for r in results:
        flat_r, m_r, flat_s, m_s, vidx, xfull = r
        meta_real.update(m_r)
        meta_shadow_all.update(m_s)
        flat_r.index = vidx
        flat_s.index = vidx
        xfull.index = vidx
        chunks_real.append(flat_r)
        chunks_shadow.append(flat_s)
        chunks_X.append(xfull)

    cluster_ids: Optional[np.ndarray] = None

    if inference_mode:
        # Keep all K*N rows for cluster bootstrap (each observation appears K times).
        # cluster_ids maps each row to its original observation index so the
        # bootstrap can resample at the observation level.
        df_shap_real_full = pd.concat(chunks_real).fillna(0.0)
        df_shap_shadow_full = pd.concat(chunks_shadow).fillna(0.0)
        cluster_ids = df_shap_real_full.index.values
        df_shap_real_full = df_shap_real_full.reset_index(drop=True)
        df_shap_shadow_full = df_shap_shadow_full.reset_index(drop=True)

        # X_stacked: tile the single X matrix K times to match K*N SHAP rows
        n_folds = len(chunks_X)
        X_stacked = pd.concat([chunks_X[0]] * n_folds, ignore_index=True)

        # Save averaged version to parquet for human readability (one row per obs)
        df_shap_real_avg = pd.concat(chunks_real).groupby(level=0).mean().fillna(0.0)
        df_shap_shadow_avg = pd.concat(chunks_shadow).groupby(level=0).mean().fillna(0.0)
        _to_parquet_atomic(df_shap_real_avg, os.path.join(shap_dir, "real_shap_interaction_matrix.parquet"))
        _to_parquet_atomic(df_shap_shadow_avg, os.path.join(shap_dir, "shadow_shap_interaction_matrix.parquet"))

        # Use the full K*N matrices for bootstrap
        df_shap_real = df_shap_real_full
        df_shap_shadow = df_shap_shadow_full
    else:
        df_shap_real = pd.concat(chunks_real).sort_index().fillna(0.0)
        df_shap_shadow = pd.concat(chunks_shadow).sort_index().fillna(0.0)
        X_stacked = pd.concat(chunks_X).sort_index()

        _to_parquet_atomic(df_shap_real, os.path.join(shap_dir, "real_shap_interaction_matrix.parquet"))
        _to_parquet_atomic(df_shap_shadow, os.path.join(shap_dir, "shadow_shap_interaction_matrix.parquet"))

    nan_mask = X_stacked.isnull().values

    eff_names_real = list(df_shap_real.columns)
    eff_idx_real = [meta_real[c][0] for c in eff_names_real]
    eff_type_real = [meta_real[c][1] for c in eff_names_real]

    eff_names_shadow = list(df_shap_shadow.columns)
    eff_idx_shadow = [meta_shadow_all[c][0] for c in eff_names_shadow]

    all_feature_names = list(X_stacked.columns)

    _run_bootstrap_pipeline(
        df_shap_real, df_shap_shadow,
        X_stacked,
        eff_idx_real, eff_idx_shadow,
        eff_names_real, eff_names_shadow,
        eff_type_real,
        config, shap_dir,
        all_feature_names,
        all_feature_types,
        nan_mask,
        X_raw, ids,
        cluster_ids=cluster_ids,
        X_display=chunks_X[0] if inference_mode else None
    )


def run_shap_pipeline(ctx: Dict[str, Any]) -> None:
    """Execute the full SHAP GII pipeline for all output slices.

    Entry point called by predict.py (OOF mode) and infer.py (inference mode).
    For single-output models, one SHAP directory is created (`shap_analysis/`).
    For multiclass or multi-regression models, one directory per class/target
    (`shap_<label>/`).

    Parameters
    ----------
    ctx : dict
        Pipeline context dictionary with keys:
        - run_dir (str): output directory for SHAP artifacts.
        - config (dict): full pipeline config.
        - task (str): task type string.
        - feature_names (list[str]): real feature names (no shadow_ prefix).
        - feature_names_shadow (list[str]): real + shadow feature names.
        - cat_features (list[str]): nominal feature names.
        - feature_types (dict): {name: type} map.
        - X (pd.DataFrame): encoded feature matrix.
        - y (pd.Series or pd.DataFrame or None): outcome values.
        - X_raw (pd.DataFrame): pre-encoding feature matrix for microdata.
        - ids (pd.Series): row identifiers.
        - class_labels (list or None): for multiclass tasks.
        - target_labels (list or None): for multi_regression tasks.
        - inference_mode (bool, optional): if True, use full-dataset splits per fold.
    """
    run_dir = ctx["run_dir"]
    config = ctx["config"]
    task = ctx["task"]
    X_aligned = ctx["X"]
    feature_names = ctx["feature_names"]
    feature_types = ctx.get("feature_types", {})
    y = ctx.get("y", None)
    class_labels = ctx.get("class_labels", None)
    target_labels = ctx.get("target_labels", None)
    inference_mode = ctx.get("inference_mode", False)

    shadow_paths = _discover_shadow_models(run_dir)
    if not shadow_paths:
        print("[SHAP] No shadow (Boruta) models found. Skipping SHAP analysis.")
        return

    # Replicate Splitter for OOF SHAP (or full-dataset splits for inference)
    if inference_mode:
        # In inference mode, every fold uses the full dataset
        splits = [(None, np.arange(len(X_aligned)))] * len(shadow_paths)
    elif y is not None:
        y_for_split = y if isinstance(y, pd.Series) else y.iloc[:, 0]
        splitter = get_cv_splitter(config, y_for_split)
        splits = list(splitter.split(X_aligned, y_for_split))
    else:
        splits = [(None, np.arange(len(X_aligned)))] * len(shadow_paths)

    # Build shadow feature type map
    all_feature_types = dict(feature_types)
    for f in feature_names:
        shadow_f = f"shadow_{f}"
        if f in feature_types:
            all_feature_types[shadow_f] = feature_types[f]

    # Determine slices to process
    if task == "multiclass_classification" and class_labels:
        slices = [(i, str(cl)) for i, cl in enumerate(class_labels)]
    elif task == "multi_regression" and target_labels:
        slices = [(i, tl) for i, tl in enumerate(target_labels)]
    else:
        slices = [(None, None)]  # single output

    for slice_idx, slice_label in slices:
        if slice_label is not None:
            shap_dir = os.path.join(run_dir, f"shap_{slice_label}")
        else:
            shap_dir = os.path.join(run_dir, "shap_analysis")
        os.makedirs(shap_dir, exist_ok=True)

        _run_shap_for_slice(
            ctx, shap_dir, shadow_paths, splits,
            all_feature_types, slice_idx, slice_label,
            inference_mode=inference_mode,
        )

    print(f"[SUCCESS] SHAP Analysis Complete. Outputs in {run_dir}")
