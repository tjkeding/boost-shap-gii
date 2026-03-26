<?xml version="1.0" encoding="UTF-8"?>
<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-03-16T02:00:00Z" />
  <input_reports>
    <report path="boost-shap-gii_brainstorm_20260316_000000.md" mode="brainstorm" key_items="15" />
  </input_reports>
  <changes>

    <!-- ===== P0 ===== -->
    <change id="C1" priority="P0" source_item="T1 / F1">
      <file path="predict.py" action="modify" />
      <description>Remove trailing ß character from the `from __future__ import annotations` line at predict.py:4. This is a syntax error that blocks execution of predict.py entirely.</description>
      <spec>Line 4: change `from __future__ import annotationsß` to `from __future__ import annotations`</spec>
      <dependencies></dependencies>
      <risk>low - single character deletion, trivial</risk>
      <rollback>Re-add the ß character (not recommended)</rollback>
    </change>

    <!-- ===== P1 ===== -->
    <change id="C2" priority="P1" source_item="T2 / F2">
      <file path="train.py" action="modify" />
      <description>Offset inner CV random_state by fold_idx + 1 so inner and outer folds use distinct seeds. Eliminates theoretical seed correlation in nested CV.</description>
      <spec>
        In run_optuna_tuning(), the inner_cv seed is currently `seed` (line 212/216).
        The function does not receive fold_idx. Two options:
        (a) Pass fold_idx into run_optuna_tuning() and offset seed there, or
        (b) Offset at the call site in main().
        Option (b) is simpler. Change the call in main() at line 540:
          run_optuna_tuning(X_train, y_train, nom_feats, task, config, n_jobs)
        to pass fold_idx, and add fold_idx parameter to run_optuna_tuning().
        Inside run_optuna_tuning(), replace:
          inner_cv = KFold(n_splits=inner_cv_folds, shuffle=True, random_state=seed)
          inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=seed)
        with:
          inner_cv = KFold(n_splits=inner_cv_folds, shuffle=True, random_state=seed + fold_idx + 1)
          inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=seed + fold_idx + 1)
        Also update the Optuna sampler seed to seed + fold_idx + 1 for full consistency.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - one-line arithmetic change to a seed value</risk>
      <rollback>Remove fold_idx parameter, revert to random_state=seed</rollback>
    </change>

    <change id="C3" priority="P1" source_item="T3 / F3">
      <file path="train.py" action="modify" />
      <file path="predict.py" action="modify" />
      <file path="infer.py" action="modify" />
      <description>Add two-tier ordinal unknown-value check: (1) existing unique-value fraction check (hard error at >50% unknown unique values); (2) new observation-level fraction check (loud warning at >10% of non-missing observations have unknown values). Both metrics reported in messages.</description>
      <spec>
        All three files have an identical ordinal validation block:
          unique_vals = ...dropna().unique()
          unknowns = [v for v in unique_vals if v not in levels]
          if unknowns:
              unknown_frac = len(unknowns) / len(unique_vals)
              if unknown_frac > 0.5: raise ValueError(...)
              print(f"[WARNING] ...")

        Extend the `if unknowns:` block in all three files to also compute
        observation-level fraction:
          obs_vals = df_raw[c].dropna()   # or X[c].dropna() for train.py
          n_unknown_obs = sum(v not in levels for v in obs_vals)
          obs_frac = n_unknown_obs / len(obs_vals) if len(obs_vals) > 0 else 0.0
          if obs_frac > 0.10:
              print(f"[WARNING] Feature '{c}': {obs_frac:.1%} of non-missing observations "
                    f"({n_unknown_obs}/{len(obs_vals)}) have values not in YAML levels. "
                    f"This may indicate systematic data quality issues.")

        Note: In train.py the ordinal loop operates on X[c] (already mapped), while
        unique_vals comes from X[c].dropna(). Use the same X[c] series for obs_frac.
        In predict.py and infer.py, the loop uses df_raw[c] after mapping.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - additive warning, does not change any existing logic</risk>
      <rollback>Remove the obs_frac block</rollback>
    </change>

    <change id="C4" priority="P1" source_item="T4 / F4">
      <file path="train.py" action="modify" />
      <description>Shadow model currently trains with `best_params` which includes `iterations=tuned_iters` (tuned for p features). Replace with early stopping on the outer validation fold, using tuned_iters * 2 as a ceiling and the same early_stopping_rounds patience.</description>
      <spec>
        In the shadow training block (after line ~656), replace:
          model_shadow.fit(pool_train_full, verbose=False)
        with:
          pool_val_full = Pool(X_val_full, y_val, cat_features=full_cat_features)
          shadow_params = best_params.copy()
          shadow_params["iterations"] = tuned_iters * 2   # ceiling, not fixed count
          if is_regression(task):
              model_shadow = CatBoostRegressor(**shadow_params)
          else:
              model_shadow = CatBoostClassifier(**shadow_params)
          model_shadow.fit(
              pool_train_full,
              eval_set=pool_val_full,
              early_stopping_rounds=int(config["modeling"]["tuning"]["early_stopping_rounds"]),
              verbose=False
          )
        Remove the existing model_shadow instantiation lines (they appear just before .fit).
        Note: X_val_full and y_val are available in scope; X_val_full is already constructed
        above as pd.concat([X_val, X_val_shadow], axis=1).
      </spec>
      <dependencies>none</dependencies>
      <risk>medium - changes shadow model training significantly; no data leakage because shadow outputs are never used for predictive evaluation</risk>
      <rollback>Revert to model_shadow.fit(pool_train_full, verbose=False) with original best_params</rollback>
    </change>

    <change id="C5" priority="P1" source_item="T5 / F5">
      <file path="predict.py" action="modify" />
      <description>Add assertion that the number of discovered model files matches the expected CV fold count from the splitter.</description>
      <spec>
        After line ~196 (model_files discovery), add:
          expected_folds = splitter.get_n_splits()  # splitter is created at line ~215
        BUT splitter is currently constructed AFTER model_files check. Re-order:
          1. Construct splitter (move get_cv_splitter call before model_files check), or
          2. Add the assertion after the splitter is created (line ~215), using the already-loaded model_files.
        Option 2 is less invasive. After `splitter = get_cv_splitter(config, y_for_split)` (line ~215):
          expected_folds = splitter.get_n_splits()
          if len(model_files) != expected_folds:
              raise AssertionError(
                  f"Found {len(model_files)} model file(s) in {run_dir} but CV splitter "
                  f"expects {expected_folds} fold(s). Re-run train.py or check output_dir."
              )
      </spec>
      <dependencies>C1 (predict.py must be syntactically valid first)</dependencies>
      <risk>low - additive guard, will only raise if there is a genuine mismatch</risk>
      <rollback>Remove the assertion block</rollback>
    </change>

    <change id="C6" priority="P1" source_item="T15 / F15">
      <file path="shap_utils.py" action="modify" />
      <description>Replace `return 0.0` in calculate_v_spline_2d (when either axis has fewer than 2 interior knots) with stacked-spline routing: treat the low-resolution axis as a discrete grouping variable and fit 1D splines along the well-resolved axis. Fall back to group-means only when BOTH axes lack resolution.</description>
      <spec>
        Current code in calculate_v_spline_2d (lines ~244-261):
          tx, kx_adj = _get_adaptive_knots_and_degree(x1, n_knots, degree)
          ty, ky_adj = _get_adaptive_knots_and_degree(x2, n_knots, degree)
          if len(tx) < 2 or len(ty) < 2:
              return 0.0

        Replace the early-return block:
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
              # discrete_threshold must be passed through — add it as a parameter
              return calculate_v_stacked_spline(x2, x1, y, n_knots, degree, discrete_threshold)
          elif not x2_ok:
              # x2 lacks resolution: treat x2 as discrete grouping, fit 1D splines along x1
              return calculate_v_stacked_spline(x1, x2, y, n_knots, degree, discrete_threshold)
          # Both axes have adequate resolution: proceed with 2D spline (existing code)

        Function signature change: add discrete_threshold parameter:
          def calculate_v_spline_2d(x1, x2, y, n_knots, degree, discrete_threshold) -> float:

        Update all call sites of calculate_v_spline_2d in _bootstrap_worker_chunk:
          v[e] = calculate_v_spline_2d(vec_a, vec_b, shap_v, n_knots, degree, discrete_threshold)
      </spec>
      <dependencies>none</dependencies>
      <risk>medium - changes fallback behavior for low-resolution interactions; stacked-spline already used for mixed interactions so code path is tested</risk>
      <rollback>Revert to `return 0.0` for the insufficient-knots case</rollback>
    </change>

    <change id="C7" priority="P1" source_item="T16 / F16">
      <file path="shap_utils.py" action="modify" />
      <description>Remove NaN-to-zero replacement for boot_var and boot_gii (lines 709-711). Downstream nanmean/nanpercentile calls already handle NaN exclusion natively. Preserves diagnostic distinction between "genuinely zero V" and "spline failed".</description>
      <spec>
        Remove lines 707-711:
          # Replace NaN in boot_var/boot_gii with 0.0 so downstream nanmean/nanpercentile
          # treat failed iterations conservatively (V=0 → GII=0 for that iteration).
          nan_in_var = np.isnan(boot_var)
          boot_var[nan_in_var] = 0.0
          boot_gii[nan_in_var] = 0.0
        The comment and three executable lines are all removed. The v_failure_rate tracking
        (lines 699-705) is preserved as-is — it already runs before the NaN replacement.
        The downstream np.nanmean / np.nanpercentile calls at lines 803-812 are already NaN-safe.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - downstream aggregation already uses nan-aware functions</risk>
      <rollback>Re-add the three lines and comment</rollback>
    </change>

    <change id="C8" priority="P1" source_item="T18 / F18">
      <file path="shap_utils.py" action="modify" />
      <description>Apply Davison-Hinkley +1 correction to all exceedance p-values (M, V, GII). Changes minimum achievable p-value from 0 to 1/(n_boot+1), consistent with the permutation test already implemented in utils.py.</description>
      <spec>
        Lines 828-830:
          p_exceed_m = np.nanmean(boot_mag <= stratified_noise_m, axis=0)
          p_exceed_v = np.nanmean(boot_var <= stratified_noise_v, axis=0)
          p_exceed_gii = np.nanmean(boot_gii <= stratified_noise_gii, axis=0)

        Replace with:
          # +1 correction (Davison & Hinkley 1997; Phipson & Smyth 2010):
          # prevents p=0 and is consistent with compute_permutation_test() in utils.py.
          p_exceed_m = (np.nansum(boot_mag <= stratified_noise_m, axis=0) + 1) / (n_boot + 1)
          p_exceed_v = (np.nansum(boot_var <= stratified_noise_v, axis=0) + 1) / (n_boot + 1)
          p_exceed_gii = (np.nansum(boot_gii <= stratified_noise_gii, axis=0) + 1) / (n_boot + 1)
      </spec>
      <dependencies>none</dependencies>
      <risk>low - makes p-values slightly more conservative; minimum p = 1/(n_boot+1) > 0</risk>
      <rollback>Revert to np.nanmean(... <= ...) formula</rollback>
    </change>

    <!-- ===== P2 ===== -->
    <change id="C9" priority="P2" source_item="T6 / F6">
      <file path="utils.py" action="modify" />
      <description>Add floor to depth search space high bound to guarantee [2, 3] minimum range for small n, preventing the search space from collapsing to [2, 2].</description>
      <spec>
        In _default_search_space(), line ~194:
          "depth": {"low": 2, "high": min(10, int(np.log2(max(n / 5, 4))))},
        Replace with:
          "depth": {"low": 2, "high": max(3, min(10, int(np.log2(max(n / 5, 4)))))},
      </spec>
      <dependencies>none</dependencies>
      <risk>low - single max() wrapper around existing expression</risk>
      <rollback>Remove the max(3, ...) wrapper</rollback>
    </change>

    <change id="C10" priority="P2" source_item="T7 / F7">
      <file path="utils.py" action="modify" />
      <description>Fix one_hot_max_size search space to fixed [2, 25] range, removing semantically incorrect dependency on total feature count p.</description>
      <spec>
        In _default_search_space(), line ~201:
          "one_hot_max_size": {"low": 2, "high": min(25, max(p, 2))},
        Replace with:
          "one_hot_max_size": {"low": 2, "high": 25},
        Remove reference to p in this line.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - expands the upper bound for large feature sets (no narrowing possible)</risk>
      <rollback>Revert to the original expression</rollback>
    </change>

    <change id="C11" priority="P2" source_item="T8 / F8">
      <file path="check_env.py" action="modify" />
      <description>Add joblib and statsmodels to PYTHON_DEPS in check_env.py. Both are imported by shap_utils.py but absent from the pre-flight check.</description>
      <spec>
        Line 9:
          PYTHON_DEPS = [
              "catboost", "optuna", "shap", "pyarrow", "sklearn", "scipy", "pandas", "yaml"
          ]
        Replace with:
          PYTHON_DEPS = [
              "catboost", "optuna", "shap", "pyarrow", "sklearn", "scipy",
              "pandas", "yaml", "joblib", "statsmodels"
          ]
      </spec>
      <dependencies>none</dependencies>
      <risk>low - additive list entries only</risk>
      <rollback>Remove joblib and statsmodels from the list</rollback>
    </change>

    <change id="C12" priority="P2" source_item="T10 / F10">
      <file path="utils.py" action="modify" />
      <description>Add n_boot_effective tracking and >5% drop-rate warning to compute_bootstrap_ci. Reports effective sample count as a transparent confidence proxy.</description>
      <spec>
        Current signature: compute_bootstrap_ci(y_true, y_pred, metric_fn, n_boot=2000, alpha=0.05) -> tuple
        Returns: (base_score, lower, upper)

        Add n_dropped counter and warning:
          n_dropped = 0
          for _ in range(n_boot):
              idx = resample(indices, replace=True, n_samples=len(indices))
              if len(np.unique(y_true[idx])) < 2:
                  n_dropped += 1
                  continue
              ...

          n_boot_effective = len(scores)
          drop_rate = n_dropped / n_boot
          if drop_rate > 0.05:
              print(
                  f"[WARNING] compute_bootstrap_ci: {drop_rate:.1%} of bootstrap iterations "
                  f"dropped (single-class resample). n_boot_effective={n_boot_effective}/{n_boot}. "
                  f"CIs may be unreliable for severely imbalanced data."
              )

        Return value unchanged (base_score, lower, upper) — n_boot_effective is
        diagnostic output only (logged, not returned), to avoid breaking all call sites.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - additive counter and conditional print, no logic change to return values</risk>
      <rollback>Remove n_dropped counter and warning block</rollback>
    </change>

    <change id="C13" priority="P2" source_item="T11 / F11">
      <file path="plot.R" action="modify" />
      <description>Cap N_CORES at min(config value, detected physical cores), with NA fallback to config value if detectCores() returns NA.</description>
      <spec>
        Line 65:
          N_CORES <- cfg$execution$n_jobs
        Replace with:
          N_CORES <- {
            detected <- parallel::detectCores(logical = FALSE)
            requested <- cfg$execution$n_jobs
            if (is.na(detected)) requested else min(requested, detected)
          }
        But R does not support block assignments this way; use:
          N_CORES <- local({
            detected <- parallel::detectCores(logical = FALSE)
            requested <- cfg$execution$n_jobs
            if (is.na(detected)) requested else min(requested, detected)
          })
        Add `library(parallel)` to suppressPackageStartupMessages block if not already present.
        Check: parallel is not in the library() calls at lines 40-53. Add it.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - falls back to requested count if detectCores returns NA</risk>
      <rollback>Revert to `N_CORES <- cfg$execution$n_jobs`</rollback>
    </change>

    <change id="C14" priority="P2" source_item="T23 / F23">
      <file path="utils.py" action="modify" />
      <description>Convert permutation test from for-loop to while-loop with 2*n_perm attempt cap. Guarantees full n_perm effective iterations; warns if cap reached.</description>
      <spec>
        In compute_permutation_test(), current loop:
          null_dists = {name: np.full(n_perm, np.nan) for name in metric_names}
          for i in range(n_perm):
              y_perm = y_true[rng.permutation(n)]
              for name, fn in zip(metric_names, metric_fns):
                  try:
                      null_dists[name][i] = fn(y_perm, y_pred)
                  except Exception:
                      pass

        Replace with:
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

          n_boot_effective_perm = min(len(v) for v in null_dists.values())
          if n_attempts >= max_attempts and n_boot_effective_perm < n_perm:
              print(
                  f"[WARNING] compute_permutation_test: reached {max_attempts} attempt cap. "
                  f"Effective permutation count: {n_boot_effective_perm}/{n_perm}."
              )

          # Convert lists to arrays (truncate to minimum effective count for alignment)
          null_dists = {name: np.array(v[:n_boot_effective_perm]) for name, v in null_dists.items()}

        Update downstream references to null_dists[name] that use index slicing:
          null_clean = null[~np.isnan(null)]  -> null_clean = null  (already NaN-free)
          len(null_clean) -> len(null)
        BUT keep the NaN-guard (null_clean = null[~np.isnan(null)]) for safety — it will
        be a no-op since we now only append non-NaN values.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - permutation failures are rare; while-loop converges in near-n_perm iterations</risk>
      <rollback>Revert to for-loop over range(n_perm)</rollback>
    </change>

    <change id="C15" priority="P2" source_item="T25 / F25">
      <file path="plot.R" action="modify" />
      <description>Change left density panel from GII distribution to M distribution. Load bootstrap_distributions_M.parquet and stratified_noise_distributions_M.parquet. Update x-axis label to "Importance Magnitude (M)". Update get_global_x_limit_dir to use M parquets for x-axis scaling.</description>
      <spec>
        1. Function get_global_x_limit_dir (lines ~247-260):
           Replace references to GII parquets with M parquets:
             p1 <- file.path(shap_dir, "bootstrap_distributions_GII.parquet")
             p2 <- file.path(shap_dir, "stratified_noise_distributions_GII.parquet")
           Change to:
             p1 <- file.path(shap_dir, "bootstrap_distributions_M.parquet")
             p2 <- file.path(shap_dir, "stratified_noise_distributions_M.parquet")

        2. Data loading (lines ~274-284):
           boot_path currently points to bootstrap_distributions_GII.parquet
           noise_path currently points to stratified_noise_distributions_GII.parquet
           Change:
             boot_path <- file.path(SHAP_DIR, "bootstrap_distributions_M.parquet")
             noise_path <- file.path(SHAP_DIR, "stratified_noise_distributions_M.parquet")

        3. In the plotting loop, Panel 1 x-axis label (line ~342):
             labs(x = "GII Magnitude", y = "Density")
           Change to:
             labs(x = "Importance Magnitude (M)", y = "Density")

        4. File existence check at line ~277:
             micro_path, boot_path, noise_path
           boot_path and noise_path are now M parquets — no logic change needed.

        Note: The plot output filename (line ~539) uses feat_rank and clean_name with
        "_GII.png" suffix — keep that unchanged as it refers to the GII significance
        criterion (features are selected and ranked by sig_GII), not the left panel content.
      </spec>
      <dependencies>C13 (both are plot.R changes; can be applied in sequence)</dependencies>
      <risk>low - data files already exist (bootstrap_distributions_M.parquet is written by shap_utils.py); purely cosmetic/data-source change</risk>
      <rollback>Revert all three substitutions back to GII parquet references and original x-axis label</rollback>
    </change>

  </changes>

  <execution_order>C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C15, C14</execution_order>
  <!-- Note: C13 and C15 both modify plot.R; apply sequentially. C14 moved last among utils.py changes for clarity. -->
</implement_plan>
