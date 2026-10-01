<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-09-29T14:00:00-04:00" />
  <input_reports>
    <report path="boost-shap-gii_brainstorm_20260929_134500.md" mode="brainstorm" key_items="3" />
  </input_reports>
  <changes>
    <change id="C1" priority="P0" source_item="brainstorm T1/A1">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Restructure the foreach %dopar% loop (lines 593-1108) into two phases to eliminate fork-unsafe ggsave() and Rcpp (ggtext/gridtext) execution in forked children. Phase 1 (%dopar%) computes data only; Phase 2 (sequential for loop) constructs ggplots and saves.</description>
      <spec>
**Phase 1 refactor: the foreach body returns a data-only list**

Replace the current foreach body (lines 593-1108) so that each worker returns a structured list containing all computed data, with NO ggplot construction, NO arrangeGrob(), and NO ggsave(). The return list schema:

```r
list(
  feat_name = feat_name,        # character
  feat_rank = feat_rank,        # integer
  feat_type = feat_type,        # "Main" or "Interaction"
  is_v_only = is_v_only,        # logical
  status = "ok",                # "ok" or "skip"
  skip_msg = NULL,              # character if status=="skip"

  # Panel 1 data (NULL if is_v_only)
  p1_data = list(
    df_p1 = df_p1,              # data.frame(val, type)
    noise_label = noise_label,  # character (markdown)
    signal_label = signal_label,# character (markdown)
    local_xmax = local_xmax     # numeric
  ),

  # For interaction features: a list of per-orientation results
  # For singleton features: NULL
  orientations = list(
    list(
      status = "ok",            # "ok" or "skip"
      skip_msg = NULL,
      mod_name = mod_name,      # character
      focal_label = focal_label,# character
      legend_title = legend_title,# character
      df_ori = df_ori,          # data.frame (filtered, with x_plot, stratum columns)
      is_focal_discrete = is_focal_discrete, # logical
      strata_colors = strata_colors,  # named character vector
      strat_levels = strat$levels,    # character vector
      n_strata = n_strata,     # integer
      x_labels = x_labels,     # character vector (discrete only, NULL otherwise)
      n_lev = n_lev,           # integer (discrete only, NULL otherwise)
      trend_list = trend_list, # list of data.frames (continuous only, NULL otherwise)
      boot_sd_list = boot_sd_list, # named list of bootstrap SD data.frames (continuous only)
      means_df = means_df,     # data.frame with sd column (discrete only, NULL otherwise)
      fname = fname_ori,       # output filename
      fpath = fpath_ori        # output filepath
    ),
    ... # second orientation
  ),

  # For singleton features: direct plot data
  # For interaction features: NULL
  singleton = list(
    is_discrete = is_discrete,   # logical
    df_m = df_m,                 # data.frame (filtered, with x_plot)
    m_type = m_type,             # character
    legend_title = legend_title, # character
    # Continuous singleton fields:
    trend_data = trend_data,     # data.frame (NULL if discrete)
    boot_sd = boot_sd_singleton, # data.frame (NULL if discrete)
    # Discrete singleton fields:
    fac_levels = levels(fac),    # character vector (NULL if continuous)
    x_labels = x_labels,         # character vector (NULL if continuous)
    n_lev = n_lev,               # integer (NULL if continuous)
    df_means = df_means,         # data.frame with sd column (NULL if continuous)
    fname = fname,               # output filename
    fpath = fpath                # output filepath
  )
)
```

The foreach `.packages` argument can be reduced: remove `"ggplot2"`, `"gridExtra"`, `"grid"`, `"grDevices"` (no longer needed in workers). Keep `"dplyr"`, `"splines"`, `"stringr"` (still used for data computation and string operations).

All data filtering, SHAP transformation, stratification, spline fitting, bootstrap SD computation, factor construction, x-axis label generation, filename construction, and group-mean computation remain in Phase 1. Everything that calls ggplot(), arrangeGrob(), ggsave(), textGrob(), gpar(), or ggtext::element_markdown() moves to Phase 2.

**Phase 2: sequential render loop**

After the foreach completes, iterate over the results list sequentially. For each result with `status == "ok"`:

1. Construct the y-axis grob (textGrob + arrangeGrob) using the existing code (current lines 711-717).
2. If `is_v_only == FALSE`: construct Panel 1 (density plot) from `p1_data`, including `ggtext::element_markdown()` theme.
3. For interaction features: loop over `orientations`, construct Panel 2 from orientation data, assemble via arrangeGrob, ggsave.
4. For singleton features: construct Panel 2 from `singleton` data, assemble via arrangeGrob, ggsave.
5. Emit per-feature progress: `cat(sprintf("[INFO] %s\n", msg))` after each save.

The render loop uses the EXACT SAME ggplot construction code as the current foreach body (lines 631-665 for Panel 1, lines 857-944 for interaction Panel 2, lines 981-1057 for singleton Panel 2, lines 1062-1084 for common theme), just relocated from the foreach body to the sequential loop.

**Early-return handling**: Features that return `status == "skip"` (from the current `return(sprintf("Skipped %s: ..."))` paths at lines 671, 692) are handled by checking `status` in Phase 2 and emitting the skip message without attempting plot construction.

**Post-render message loop**: The existing post-foreach message loop (lines 1110-1113) is replaced by inline progress output in Phase 2 (each feature's message is printed immediately after its plot is saved or skipped).
      </spec>
      <dependencies>none</dependencies>
      <risk>medium - Large refactor (~500 lines restructured). Functional behavior must be identical; the only change is WHERE code executes (parent vs. forked child). Risk is mitigated by the fact that all data flows and ggplot construction logic are preserved verbatim.</risk>
      <rollback>git checkout -- src/boost_shap_gii/scripts/plot.R</rollback>
    </change>

    <change id="C2" priority="P1" source_item="brainstorm T2/B1">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Add bootstrap subsampling to bootstrap_spline_sd for large-dataset efficiency. Subsample to n_sub = min(n, max_subsample_n) before the bootstrap loop; apply sqrt(n_sub / n) correction to the final SD.</description>
      <spec>
Modify `bootstrap_spline_sd` (line 317) to accept a new parameter `max_subsample_n` (default NULL, meaning no subsampling):

```r
bootstrap_spline_sd <- function(x, y, cfg, B, min_boot_n, max_subsample_n = NULL) {
  valid_idx <- which(!is.na(x) & !is.na(y) & !is.nan(x) & !is.nan(y))
  if (length(valid_idx) < min_boot_n) return(NULL)

  x_valid <- x[valid_idx]
  y_valid <- y[valid_idx]
  n_full <- length(x_valid)

  # Reference spline: fitted to ALL data (no subsampling)
  ref <- calc_v_spline_pred(x_valid, y_valid, cfg)
  if (nrow(ref) == 0 || all(is.na(ref$y_pred))) return(NULL)

  x_eval <- ref$x

  # Subsample for bootstrap iterations (computational optimization;
  # Politis, Romano, & Wolf 1999)
  if (!is.null(max_subsample_n) && n_full > max_subsample_n) {
    sub_idx <- sample(n_full, max_subsample_n)
    x_boot_src <- x_valid[sub_idx]
    y_boot_src <- y_valid[sub_idx]
    n_boot_src <- max_subsample_n
  } else {
    x_boot_src <- x_valid
    y_boot_src <- y_valid
    n_boot_src <- n_full
  }

  boot_preds <- matrix(NA_real_, nrow = B, ncol = length(x_eval))

  for (b in 1:B) {
    idx_boot <- sample(n_boot_src, replace = TRUE)
    sp_boot <- calc_v_spline_pred(x_boot_src[idx_boot], y_boot_src[idx_boot], cfg)
    if (nrow(sp_boot) > 1) {
      boot_preds[b, ] <- tryCatch(
        approx(sp_boot$x, sp_boot$y_pred, xout = x_eval)$y,
        error = function(e) rep(NA_real_, length(x_eval))
      )
    }
  }

  sd_vals <- apply(boot_preds, 2, stats::sd, na.rm = TRUE)

  # m-out-of-n correction (Bickel & Sakov 2008)
  if (n_boot_src < n_full) {
    sd_vals <- sd_vals * sqrt(n_boot_src / n_full)
  }

  data.frame(x = x_eval, y_pred = ref$y_pred, sd = sd_vals)
}
```

Update all call sites (4 total: lines 872, 976 in the current code, plus their equivalents after C1 restructuring) to pass `MAX_BOOT_SUBSAMPLE_N` as the new argument.

Read `MAX_BOOT_SUBSAMPLE_N` from config near line 89:
```r
MAX_BOOT_SUBSAMPLE_N <- cfg$plot$bootstrap_ribbons$max_subsample_n %||% 5000L
```
      </spec>
      <dependencies>C1 (call sites move during restructure)</dependencies>
      <risk>low - Additive change to an existing function. NULL default preserves current behavior if config key is absent. The m-out-of-n correction is a single multiplication.</risk>
      <rollback>git checkout -- src/boost_shap_gii/scripts/plot.R</rollback>
    </change>

    <change id="C3" priority="P2" source_item="brainstorm T3/C1">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Replace bootstrap_group_mean_sd's replicate loop with analytical SE formula: sd(subset_y) / sqrt(length(subset_y)).</description>
      <spec>
Replace the body of `bootstrap_group_mean_sd` (lines 347-360). The function signature changes to drop the `B` parameter (no longer needed):

```r
group_mean_sd <- function(x_factor, y, min_boot_n) {
  levels_x <- levels(x_factor)
  sd_out <- rep(NA_real_, length(levels_x))

  for (i in seq_along(levels_x)) {
    subset_y <- y[!is.na(x_factor) & x_factor == levels_x[i]]
    subset_y <- subset_y[!is.na(subset_y)]
    if (length(subset_y) < min_boot_n) next
    sd_out[i] <- sd(subset_y) / sqrt(length(subset_y))
  }

  data.frame(x_plot = seq_along(levels_x), level = levels_x, sd = sd_out)
}
```

Rename from `bootstrap_group_mean_sd` to `group_mean_sd` since it no longer uses the bootstrap.

Update all call sites (3 total: lines 895, 1030 in the current code, plus their equivalents after C1 restructuring) to use the new name and drop the `B` argument:
- Old: `bootstrap_group_mean_sd(factor(...), y, BOOT_RIBBON_B, MIN_BOOT_N)`
- New: `group_mean_sd(factor(...), y, MIN_BOOT_N)`
      </spec>
      <dependencies>C1 (call sites move during restructure)</dependencies>
      <risk>low - The analytical formula is the exact limit of the bootstrap estimate. The function body simplifies. The rename prevents accidental use of the old name.</risk>
      <rollback>git checkout -- src/boost_shap_gii/scripts/plot.R</rollback>
    </change>

    <change id="C4" priority="P1" source_item="brainstorm T2/B1">
      <file path="example_config_advanced.yaml" action="modify" />
      <description>Add the max_subsample_n config key to the bootstrap_ribbons section in the advanced example config.</description>
      <spec>
In the `plot.bootstrap_ribbons` block (line 157-158), add after `n_boot`:

```yaml
  bootstrap_ribbons:
    n_boot: 2000           # Bootstrap resamples for uncertainty ribbons on V-component plots (default: 2000)
    max_subsample_n: 5000  # Cap on data points per bootstrap spline fit; larger datasets are
                           # subsampled with m-out-of-n correction (Bickel & Sakov 2008). Set to
                           # null to disable subsampling and use all data. (default: 5000)
```
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Config-only change. Does not affect behavior unless C2 is also applied.</risk>
      <rollback>git checkout -- example_config_advanced.yaml</rollback>
    </change>

    <change id="C5" priority="P0" source_item="brainstorm T1/A1 (implicit)">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Add per-feature progress output in the sequential render phase (Phase 2 of C1), and a pre-loop banner indicating parallel computation is running.</description>
      <spec>
Before the foreach loop (after the "Found N significant features to plot" message at line 564), add:

```r
cat(sprintf("[INFO] Computing feature data in parallel (%d features, %d cores)...\n",
            nrow(df_sig), N_CORES))
```

In the Phase 2 sequential render loop, after each ggsave or skip, emit immediately:

```r
cat(sprintf("[INFO] %s\n", msg))
```

This replaces the post-foreach batch output loop (current lines 1110-1113).
      </spec>
      <dependencies>C1 (Phase 2 loop is created by C1)</dependencies>
      <risk>low - Output-only change. No functional impact.</risk>
      <rollback>git checkout -- src/boost_shap_gii/scripts/plot.R</rollback>
    </change>
  </changes>
  <execution_order>C1, C5, C2, C3, C4</execution_order>
</implement_plan>
