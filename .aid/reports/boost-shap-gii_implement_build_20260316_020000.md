<?xml version="1.0" encoding="UTF-8"?>
<implement_report>
  <meta project="boost-shap-gii" mode="implement" submodule="build" timestamp="2026-03-16T02:00:00Z" />
  <spec_ref>boost-shap-gii_implement_plan_20260316_020000.md</spec_ref>
  <changes_applied>

    <change id="C1" status="done">
      <files_modified>
        <file path="predict.py" lines_changed="1" />
      </files_modified>
      <notes>Removed trailing ß from `from __future__ import annotations` at line 4. Verified with py_compile.</notes>
    </change>

    <change id="C2" status="done">
      <files_modified>
        <file path="train.py" lines_changed="8" />
      </files_modified>
      <notes>
        Added fold_idx=0 parameter to run_optuna_tuning(). Computed inner_seed = seed + fold_idx + 1.
        Applied inner_seed to both KFold/StratifiedKFold and TPESampler for full consistency.
        Updated call site in main() to pass fold_idx=fold_idx.
      </notes>
    </change>

    <change id="C3" status="done">
      <files_modified>
        <file path="train.py" lines_changed="10" />
        <file path="predict.py" lines_changed="10" />
        <file path="infer.py" lines_changed="10" />
      </files_modified>
      <notes>
        Added Tier 2 observation-level fraction check (>10% threshold, loud warning) to all
        three ordinal validation blocks. Tier 1 unique-value fraction check (>50% error) retained
        unchanged. Both tiers report distinct messages with quantitative detail.
      </notes>
    </change>

    <change id="C4" status="done">
      <files_modified>
        <file path="train.py" lines_changed="14" />
      </files_modified>
      <notes>
        Replaced unconditional model_shadow.fit(pool_train_full) with early-stopped fit using
        pool_val_full as eval_set. Ceiling set to tuned_iters * 2. Early stopping patience from
        config["modeling"]["tuning"]["early_stopping_rounds"]. shadow_params is a copy of
        best_params with iterations overridden — original best_params not mutated.
      </notes>
    </change>

    <change id="C5" status="done">
      <files_modified>
        <file path="predict.py" lines_changed="8" />
      </files_modified>
      <notes>
        Assertion inserted immediately after splitter construction (which comes after model_files
        discovery). Raises AssertionError with actionable message if counts diverge.
      </notes>
    </change>

    <change id="C6" status="done">
      <files_modified>
        <file path="shap_utils.py" lines_changed="20" />
      </files_modified>
      <notes>
        Added discrete_threshold parameter to calculate_v_spline_2d() (default=15 for backward
        compatibility). Replaced `return 0.0` early exit with three-way routing: both-axes-fail
        -> group_means_2d; x1-fails -> stacked_spline(x2, x1); x2-fails -> stacked_spline(x1, x2).
        Updated call site in _bootstrap_worker_chunk to pass discrete_threshold.
      </notes>
    </change>

    <change id="C7" status="done">
      <files_modified>
        <file path="shap_utils.py" lines_changed="5" />
      </files_modified>
      <notes>
        Removed nan_in_var = ...; boot_var[nan_in_var] = 0.0; boot_gii[nan_in_var] = 0.0.
        Replaced with explanatory comment. v_failure_rate tracking block (lines 699-705) preserved.
        Downstream nanmean/nanpercentile calls confirmed NaN-safe.
      </notes>
    </change>

    <change id="C8" status="done">
      <files_modified>
        <file path="shap_utils.py" lines_changed="7" />
      </files_modified>
      <notes>
        Changed p_exceed_m/v/gii from np.nanmean(... &lt;= ...) to
        (np.nansum(...) + 1) / (n_boot + 1). Added inline citation comment.
        Consistent with the existing +1 correction already present in compute_permutation_test().
      </notes>
    </change>

    <change id="C9" status="done">
      <files_modified>
        <file path="utils.py" lines_changed="1" />
      </files_modified>
      <notes>Wrapped existing depth high expression in max(3, ...). Guarantees [2,3] minimum range.</notes>
    </change>

    <change id="C10" status="done">
      <files_modified>
        <file path="utils.py" lines_changed="1" />
      </files_modified>
      <notes>Replaced min(25, max(p, 2)) with fixed 25. Removes incorrect dependency on feature count.</notes>
    </change>

    <change id="C11" status="done">
      <files_modified>
        <file path="check_env.py" lines_changed="3" />
      </files_modified>
      <notes>Added "joblib" and "statsmodels" to PYTHON_DEPS list.</notes>
    </change>

    <change id="C12" status="done">
      <files_modified>
        <file path="utils.py" lines_changed="18" />
      </files_modified>
      <notes>
        Added n_dropped counter. Added n_boot_effective and drop_rate computation after loop.
        Warning printed if drop_rate > 5%. Return signature unchanged (base_score, lower, upper).
      </notes>
    </change>

    <change id="C13" status="done">
      <files_modified>
        <file path="plot.R" lines_changed="7" />
      </files_modified>
      <notes>
        Added library(parallel) to suppressPackageStartupMessages block.
        Replaced N_CORES assignment with local({}) block using parallel::detectCores(logical=FALSE)
        with NA fallback to cfg$execution$n_jobs.
      </notes>
    </change>

    <change id="C15" status="done">
      <files_modified>
        <file path="plot.R" lines_changed="6" />
      </files_modified>
      <notes>
        get_global_x_limit_dir: GII parquet paths changed to M parquet paths.
        boot_path and noise_path data loading: changed to bootstrap_distributions_M.parquet
        and stratified_noise_distributions_M.parquet.
        Panel 1 x-axis label: "GII Magnitude" -> "Importance Magnitude (M)".
        Output filename suffix (_GII.png) retained — refers to GII significance criterion,
        not left panel content.
      </notes>
    </change>

    <change id="C14" status="done">
      <files_modified>
        <file path="utils.py" lines_changed="22" />
      </files_modified>
      <notes>
        Replaced for-loop over range(n_perm) with while-loop. Loop terminates when all metrics
        reach n_perm successful iterations or 2*n_perm total attempts, whichever comes first.
        null_dists converted from dict of np.full arrays to dict of lists, then truncated to
        n_perm_effective after loop. Downstream null_clean = null[~np.isnan(null)] retained as
        a no-op safety guard (arrays are already NaN-free by construction).
      </notes>
    </change>

  </changes_applied>

  <summary>
    <total_changes>15</total_changes>
    <completed>15</completed>
    <syntax_check>All 6 modified Python files pass py_compile with no errors.</syntax_check>
  </summary>

  <next_steps>Recommended: run /test to validate all changes. No existing test suite coverage was verified for the shadow model early-stopping path (C4) or the stacked-spline routing in calculate_v_spline_2d (C6) — these are the highest-priority candidates for new test cases.</next_steps>
</implement_report>
