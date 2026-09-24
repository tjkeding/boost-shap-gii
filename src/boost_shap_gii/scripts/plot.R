#!/usr/bin/env Rscript

# -----------------------------------------------------------------------------
# SHAP Visualization Engine (GII Density + V-Component Splines + indiv_reports)
# -----------------------------------------------------------------------------
# Dependencies: ggplot2, dplyr, nanoparquet, tidyr, foreach, doParallel, gridExtra,
#               splines, stringr, yaml, ggtext (M-panel legend markdown formatting)
# -----------------------------------------------------------------------------
# calc_v_spline_pred uses splines::splineDesign with adaptive-knot LSQ fitting
# that mirrors scipy.interpolate.LSQUnivariateSpline as used in
# shap_utils.py:146-164. Visualization fits are consistent with the
# V-statistic shown beneath them.
# -----------------------------------------------------------------------------

# --- 1. USER CONFIGURATION ---------------------------------------------------
# Get the command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if at least one argument is provided
if (length(args) < 1) {
  stop("At least 1 argument must be supplied: CONFIG_PATH", call. = FALSE)
}

# Get config path (required)
CONFIG_PATH <- args[1]

# -----------------------------------------------------------------------------
# 2. SETUP & LIBRARIES
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(nanoparquet)
  library(tidyr)
  library(foreach)
  library(doParallel)
  library(gridExtra)
  library(splines)
  library(stringr)
  library(grid)
  library(grDevices)
  library(yaml)
  library(parallel)
})

# Path to the YAML config used for this run
cfg <- yaml::read_yaml(CONFIG_PATH)

# Validate required plot.* keys from config
plot_cfg <- cfg$plot
required_keys <- c("outcome_max", "negate_shap", "gii_y_label", "gii_y_sublabel",
                   "indiv_y_label", "indiv_y_sublabel")
missing_keys <- setdiff(required_keys, names(plot_cfg))
if (length(missing_keys) > 0) {
  stop(sprintf("Missing required plot.* config keys: %s",
               paste(missing_keys, collapse = ", ")), call. = FALSE)
}

OUTCOME_MAX    <- as.numeric(plot_cfg$outcome_max)
NEGATE_SHAP    <- as.logical(plot_cfg$negate_shap)
GII_Y_LABEL    <- plot_cfg$gii_y_label
GII_Y_SUBLABEL <- plot_cfg$gii_y_sublabel
INDIV_Y_LABEL    <- plot_cfg$indiv_y_label
INDIV_Y_SUBLABEL <- plot_cfg$indiv_y_sublabel

# The directory of the current run to plot (can be overridden by 2nd arg for inference)
RUN_DIR <- cfg$paths$output_dir
if (length(args) >= 2) {
  RUN_DIR <- args[2]
}

# Read available cores for parallel processing; cap at physical core count.
# detectCores(logical = FALSE) returns NA on some systems — fall back to config value.
N_CORES <- local({
  detected <- parallel::detectCores(logical = FALSE)
  requested <- cfg$execution$n_jobs
  if (is.na(detected)) requested else min(requested, detected)
})

# Read spline parameters from YAML config
SPLINE_K_KNOTS <- cfg$shap$splines$n_knots
SPLINE_DEGREE <- cfg$shap$splines$degree
SPLINE_DISC_THRESH <- cfg$shap$splines$discrete_threshold

# Null-coalescing operator (not in base R; mirrors rlang::`%||%`)
`%||%` <- function(a, b) if (!is.null(a)) a else b

BOOT_RIBBON_B <- cfg$plot$bootstrap_ribbons$n_boot %||% 2000L
MIN_BOOT_N <- 10L
MAX_INTERACTION_STRATA <- as.integer(cfg$plot$max_interaction_strata %||% 3L)
if (MAX_INTERACTION_STRATA < 2L) stop("plot.max_interaction_strata must be >= 2", call. = FALSE)

cat(sprintf("[INFO] Bootstrap ribbon params: B=%d, min_n=%d\n", BOOT_RIBBON_B, MIN_BOOT_N))

cat(sprintf("[INFO] Spline params from config: knots=%d, degree=%d, disc_thresh=%d\n",
            SPLINE_K_KNOTS, SPLINE_DEGREE, SPLINE_DISC_THRESH))

# Discover SHAP output directories (shap_analysis for single-output, shap_<label> for multi-output)
shap_dirs <- c()
default_shap <- file.path(RUN_DIR, "shap_analysis")
if (dir.exists(default_shap) && file.exists(file.path(default_shap, "shap_stats_global.csv"))) {
  shap_dirs <- c(shap_dirs, default_shap)
}
# Look for shap_<label> subdirectories (multiclass/multi-regression)
all_subdirs <- list.dirs(RUN_DIR, recursive = FALSE, full.names = TRUE)
for (sd in all_subdirs) {
  bn <- basename(sd)
  if (startsWith(bn, "shap_") && bn != "shap_analysis" &&
      file.exists(file.path(sd, "shap_stats_global.csv"))) {
    shap_dirs <- c(shap_dirs, sd)
  }
}

if (length(shap_dirs) == 0) {
  stop("No SHAP output directories found in run directory.", call. = FALSE)
}
cat(sprintf("[INFO] Found %d SHAP output director%s to plot.\n",
            length(shap_dirs), ifelse(length(shap_dirs) == 1, "y", "ies")))

registerDoParallel(cores = N_CORES)
cat(sprintf("[INFO] Parallel backend registered with %d cores.\n", N_CORES))

# Flag to ensure performance plot is only generated once across SHAP dirs
perf_plotted_flag <- FALSE

# OOB floor constant (must match indiv_reports.py OOB_FLOOR_MIN)
OOB_FLOOR_MIN <- 50L

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

# Python equivalent: shap_utils.py:_get_adaptive_knots_and_degree (lines 181-196)
# Returns a list with `interior_knots` (vector) and `degree` (integer).
get_adaptive_knots_and_degree <- function(x_values, n_knots_target, degree_target) {
  x_valid <- x_values[!is.na(x_values)]
  x_unique <- sort(unique(x_valid))
  n_unique <- length(x_unique)

  if (n_unique < 2) {
    return(list(interior_knots = numeric(0), degree = 1L))
  }

  # Percentile-based interior knots on FULL array (matches shap_utils.py:187
  # np.percentile(arr, quantiles); type=7 matches numpy's default interpolation)
  probs <- seq(0, 1, length.out = n_knots_target + 2)
  probs <- probs[2:(length(probs) - 1)]
  candidate_knots <- quantile(x_valid, probs = probs, type = 7, names = FALSE)

  interior_knots <- unique(candidate_knots)

  x_min <- min(x_unique)
  x_max <- max(x_unique)
  interior_knots <- interior_knots[interior_knots > x_min & interior_knots < x_max]

  effective_degree <- ifelse(length(interior_knots) < 4, 1L, as.integer(degree_target))

  # Rank-sufficiency reduction: the B-spline system requires
  # n_unique >= degree + n_interior_knots + 1 (the number of basis functions)
  # for qr.solve to have full column rank.
  n_basis <- effective_degree + length(interior_knots) + 1L
  if (n_unique < n_basis) {
    effective_degree <- 1L
    max_knots <- n_unique - 2L
    if (max_knots < 0L) max_knots <- 0L
    if (length(interior_knots) > max_knots) {
      if (max_knots == 0L) {
        interior_knots <- numeric(0)
      } else {
        idx <- round(seq(1, length(interior_knots), length.out = max_knots))
        interior_knots <- interior_knots[idx]
      }
    }
  }

  return(list(interior_knots = interior_knots, degree = effective_degree))
}

# Spline fit for plotting trend lines. Uses splines::splineDesign with
# adaptive-knot LSQ fitting that mirrors shap_utils.py:calculate_v_spline_1d
# (lines 269-306). Includes the Python pipeline's fallback chain: zero-knots
# fallback to linear fit (shap_utils.py:282-289), solver-failure fallback to
# linear fit (shap_utils.py:305-306), and Gate 2 energy stability check
# (shap_utils.py:198-216) with fallback on overshooting.
calc_v_spline_pred <- function(x, y, cfg) {
  n_knots_target <- cfg$shap$splines$n_knots %||% 4L
  degree_target <- cfg$shap$splines$degree %||% 3L

  valid <- !is.na(x) & !is.na(y) & !is.nan(x) & !is.nan(y)
  if (sum(valid) < 2) return(data.frame(x = numeric(0), y_pred = numeric(0)))

  xs <- x[valid]
  ys <- y[valid]
  ord <- order(xs)
  xs <- xs[ord]
  ys <- ys[ord]

  linear_fallback <- function(xs, ys) {
    if (length(unique(xs)) < 2) return(data.frame(x = xs, y_pred = rep(mean(ys), length(xs))))
    data.frame(x = xs, y_pred = predict(lm(ys ~ xs)))
  }

  knot_info <- get_adaptive_knots_and_degree(xs, n_knots_target, degree_target)
  interior_knots <- knot_info$interior_knots
  degree <- knot_info$degree

  if (length(interior_knots) == 0) {
    return(linear_fallback(xs, ys))
  }

  if (length(xs) < degree + length(interior_knots) + 2L) {
    return(linear_fallback(xs, ys))
  }

  x_min <- min(xs, na.rm = TRUE)
  x_max <- max(xs, na.rm = TRUE)
  knot_seq <- c(rep(x_min, degree + 1L), interior_knots, rep(x_max, degree + 1L))

  basis <- tryCatch(
    splines::splineDesign(knots = knot_seq, x = xs, ord = degree + 1L, outer.ok = TRUE),
    error = function(e) NULL
  )
  if (is.null(basis)) return(linear_fallback(xs, ys))

  fit <- tryCatch(qr.solve(basis, ys), error = function(e) NULL)
  if (is.null(fit)) return(linear_fallback(xs, ys))

  preds <- as.vector(basis %*% fit)

  # Gate 2: total-variation energy stability (shap_utils.py:198-216).
  # A smoothed signal cannot have more variation than the raw signal.
  # 0.1% empirical tolerance (Higham 2002 ch. 1).
  tv_raw <- sum(abs(diff(ys)))
  tv_spline <- sum(abs(diff(preds)))
  energy_ok <- if (tv_raw == 0) tv_spline < 1e-9 else tv_spline <= tv_raw * 1.001
  if (!energy_ok) return(linear_fallback(xs, ys))

  return(data.frame(x = xs, y_pred = preds))
}

stratify_moderator <- function(mod_values, mod_raw_values, mod_type, disc_thresh, max_strata) {
  n <- length(mod_values)
  strata <- rep(NA_character_, n)
  is_ordered_type <- mod_type %in% c("ordinal", "binary")
  is_nominal_type <- identical(mod_type, "nominal")

  if (is_nominal_type || is_ordered_type) {
    valid <- !is.na(mod_raw_values)
    if (is_ordered_type) {
      fac <- create_ordered_factor(mod_raw_values[valid], mod_values[valid])
    } else {
      fac <- factor(as.character(mod_raw_values[valid]))
    }
    strata[valid] <- as.character(fac)
    levels_out <- levels(fac)
    ordered_out <- is_ordered_type
    method_out <- "natural_levels"
  } else {
    valid <- !is.na(mod_values) & !is.nan(mod_values)
    n_unique <- n_distinct(mod_values[valid])

    if (n_unique <= disc_thresh) {
      fac <- create_ordered_factor(mod_raw_values[valid], mod_values[valid])
      strata[valid] <- as.character(fac)
      levels_out <- levels(fac)
      ordered_out <- TRUE
      method_out <- "natural_levels"
    } else {
      n_valid <- sum(valid)
      n_bins <- min(floor(n_valid / disc_thresh), n_unique, max_strata)
      n_bins <- max(n_bins, 1L)
      probs <- seq(0, 1, length.out = n_bins + 1)
      breaks <- quantile(mod_values[valid], probs = probs, type = 7, names = FALSE)
      breaks <- unique(breaks)
      if (length(breaks) < 2) {
        strata[valid] <- "all"
        levels_out <- "all"
      } else {
        bin_idx <- cut(mod_values[valid], breaks = breaks, include.lowest = TRUE, labels = FALSE)
        bin_labels <- sprintf("[%.2f, %.2f]", breaks[-length(breaks)], breaks[-1])
        strata[valid] <- bin_labels[bin_idx]
        levels_out <- bin_labels
      }
      ordered_out <- TRUE
      method_out <- "quantile_bins"
    }
  }

  list(strata = strata, levels = levels_out, ordered = ordered_out, method = method_out)
}

fit_per_stratum_splines <- function(df, x_col, y_col, strata_col, cfg) {
  out <- list()
  strata_vals <- unique(df[[strata_col]])
  strata_vals <- strata_vals[!is.na(strata_vals)]
  for (s in strata_vals) {
    df_sub <- df[df[[strata_col]] == s & !is.na(df[[strata_col]]), ]
    trend <- calc_v_spline_pred(df_sub[[x_col]], df_sub[[y_col]], cfg)
    out[[as.character(s)]] <- trend
  }
  out
}

compute_per_stratum_group_means <- function(df, x_col, y_col, strata_col) {
  df_valid <- df[!is.na(df[[strata_col]]), ]
  df_valid %>%
    group_by(across(all_of(c(x_col, strata_col)))) %>%
    summarise(mean_shap = mean(.data[[y_col]], na.rm = TRUE), n = n(), .groups = "drop") %>%
    rename(x_plot = all_of(x_col), stratum = all_of(strata_col))
}

# Pointwise bootstrap SD for a 1D V spline fit (visualization uncertainty ribbon).
# Resamples (x, y) pairs with replacement B times, refits calc_v_spline_pred per
# resample, and computes the pointwise SD across resamples at the reference
# spline's evaluation x-coordinates.
bootstrap_spline_sd <- function(x, y, cfg, B, min_boot_n) {
  valid_idx <- which(!is.na(x) & !is.na(y) & !is.nan(x) & !is.nan(y))
  if (length(valid_idx) < min_boot_n) return(NULL)

  x_valid <- x[valid_idx]
  y_valid <- y[valid_idx]

  ref <- calc_v_spline_pred(x_valid, y_valid, cfg)
  if (nrow(ref) == 0 || all(is.na(ref$y_pred))) return(NULL)

  x_eval <- ref$x
  boot_preds <- matrix(NA_real_, nrow = B, ncol = length(x_eval))

  for (b in 1:B) {
    idx_boot <- sample(length(x_valid), replace = TRUE)
    sp_boot <- calc_v_spline_pred(x_valid[idx_boot], y_valid[idx_boot], cfg)
    if (nrow(sp_boot) > 1) {
      boot_preds[b, ] <- tryCatch(
        approx(sp_boot$x, sp_boot$y_pred, xout = x_eval)$y,
        error = function(e) rep(NA_real_, length(x_eval))
      )
    }
  }

  sd_vals <- apply(boot_preds, 2, stats::sd, na.rm = TRUE)
  data.frame(x = x_eval, y_pred = ref$y_pred, sd = sd_vals)
}

# Bootstrap SD of the group mean at each discrete level (visualization error bars).
# x_factor: factor of discrete level labels; y: numeric SHAP values.
bootstrap_group_mean_sd <- function(x_factor, y, B, min_boot_n) {
  levels_x <- levels(x_factor)
  sd_out <- rep(NA_real_, length(levels_x))

  for (i in seq_along(levels_x)) {
    subset_y <- y[!is.na(x_factor) & x_factor == levels_x[i]]
    subset_y <- subset_y[!is.na(subset_y)]
    if (length(subset_y) < min_boot_n) next
    boot_means <- replicate(B, mean(sample(subset_y, replace = TRUE)))
    sd_out[i] <- sd(boot_means)
  }

  data.frame(x_plot = seq_along(levels_x), level = levels_x, sd = sd_out)
}

create_ordered_factor <- function(raw_vec, enc_vec, na_sentinel = "NA") {
  df_map <- data.frame(raw = as.character(raw_vec), enc = as.numeric(enc_vec)) %>%
    distinct() %>%
    arrange(enc)
  lvls <- df_map$raw
  if (na_sentinel %in% lvls) {
    lvls <- c(lvls[lvls != na_sentinel], na_sentinel)
  }
  return(factor(as.character(raw_vec), levels = lvls))
}

get_red_blue_palette <- function(n) {
  if (n < 1) return(character(0))
  if (n == 1) return("#2166ac")
  return(colorRampPalette(c("#b2182b", "#2166ac"))(n))
}


# -----------------------------------------------------------------------------
# 4. DATA LOADING & GII PLOTTING (per SHAP directory)
# -----------------------------------------------------------------------------

for (SHAP_DIR in shap_dirs) {

shap_label <- basename(SHAP_DIR)
cat(sprintf("\n[INFO] Processing SHAP directory: %s\n", shap_label))

PLOT_DIR <- file.path(SHAP_DIR, "plots")
if (!dir.exists(PLOT_DIR)) dir.create(PLOT_DIR, recursive = TRUE)

# ---------------------------------------------------------------------------
# 4a. MODEL PERFORMANCE PLOT (once per RUN_DIR)
# ---------------------------------------------------------------------------
if (!perf_plotted_flag) {
  perf_file   <- file.path(RUN_DIR, "performance_final.csv")
  perm_file   <- file.path(RUN_DIR, "permutation_test_results.csv")
  null_file   <- file.path(RUN_DIR, "permutation_null_distributions.parquet")

  if (file.exists(perf_file) && file.exists(perm_file) && file.exists(null_file)) {
    cat("[INFO] Generating model performance plot.\n")

    df_perf <- read.csv(perf_file)
    df_perm <- read.csv(perm_file)
    df_null <- read_parquet(null_file)

    boot_perf_file <- file.path(RUN_DIR, "bootstrap_distributions_perf.parquet")
    has_boot_perf <- file.exists(boot_perf_file)
    if (has_boot_perf) {
      df_boot_perf <- read_parquet(boot_perf_file)
      df_boot_long <- df_boot_perf %>%
        pivot_longer(everything(), names_to = "metric", values_to = "boot_value") %>%
        filter(!is.na(boot_value))
    }

    # Reshape null distributions for faceting
    df_null_long <- df_null %>%
      pivot_longer(everything(), names_to = "metric", values_to = "null_value")

    # Per-metric summary statistics
    df_null_stats <- df_null_long %>%
      group_by(metric) %>%
      summarise(null_mean = mean(null_value, na.rm = TRUE), null_sd = sd(null_value, na.rm = TRUE), .groups = "drop")

    # Merge observed stats with p-values and null statistics
    df_obs <- df_perf %>%
      left_join(df_perm %>% select(metric, p_value), by = "metric") %>%
      left_join(df_null_stats, by = "metric")

    # Enforce metric display order: RMSE, MAE, R² (Unicode superscript)
    metric_levels <- c("RMSE", "MAE", "R²")
    df_obs$metric <- factor(
      ifelse(df_obs$metric == "R2", "R²", df_obs$metric),
      levels = metric_levels
    )
    df_null_long$metric <- factor(
      ifelse(df_null_long$metric == "R2", "R²", df_null_long$metric),
      levels = metric_levels
    )

    if (has_boot_perf) {
      df_boot_long$metric <- factor(
        ifelse(df_boot_long$metric == "R2", "R²", df_boot_long$metric),
        levels = metric_levels
      )
      df_boot_stats <- df_boot_long %>%
        group_by(metric) %>%
        summarise(boot_mean = mean(boot_value, na.rm = TRUE), boot_sd = sd(boot_value, na.rm = TRUE), .groups = "drop")
      df_obs <- df_obs %>% left_join(df_boot_stats, by = "metric")

      df_dist <- bind_rows(
        df_null_long %>% transmute(metric, value = null_value, source = "Permutation Null"),
        df_boot_long %>% transmute(metric, value = boot_value, source = "Trained")
      ) %>%
        mutate(source = factor(source, levels = c("Permutation Null", "Trained")))

      # Build faceted performance plot (shared legend across Null/Bootstrap distributions)
      p_perf <- ggplot() +
        geom_density(data = df_dist, aes(x = value, fill = source, color = source),
                     alpha = 0.35, linewidth = 0.3) +
        scale_fill_manual(values = c("Permutation Null" = "#CCCCCC", "Trained" = "#377eb8"), name = NULL) +
        scale_color_manual(values = c("Permutation Null" = "#666666", "Trained" = "#08306b"), name = NULL) +
        # Trained model score as vertical line
        geom_vline(data = df_obs, aes(xintercept = score),
                   color = "#377eb8", linewidth = 0.7) +
        # Permutation Null distribution mean as vertical line
        geom_vline(data = df_obs, aes(xintercept = null_mean),
                   color = "#666666", linewidth = 0.5) +
        # Trained mean+SD annotation (centered below trained mean line)
        geom_label(data = df_obs, aes(x = boot_mean, y = -Inf,
                   label = sprintf("%.2f (%.2f)", boot_mean, boot_sd)),
                   vjust = 0.5, hjust = 0.5, size = 1.5, color = "black",
                   fill = "white", label.size = NA, label.padding = unit(0.15, "lines")) +
        # Permutation Null mean+SD annotation (centered below null mean line)
        geom_label(data = df_obs, aes(x = null_mean, y = -Inf,
                   label = sprintf("%.2f (%.2f)", null_mean, null_sd)),
                   vjust = 0.5, hjust = 0.5, size = 1.5, color = "black",
                   fill = "white", label.size = NA, label.padding = unit(0.15, "lines")) +
        coord_cartesian(clip = "off") +
        # Facet by metric
        facet_wrap(~ metric, scales = "free", ncol = 1) +
        labs(x = "Score", y = "Density") +
        theme_minimal(base_size = 7) +
        theme(
          strip.text = element_text(size = 6, face = "bold"),
          axis.title = element_text(size = 5, face = "bold"),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          panel.grid.major = element_line(color = "grey92"),
          panel.grid.minor = element_blank(),
          plot.background = element_rect(fill = "transparent", color = NA),
          plot.margin = margin(5.5, 5.5, 8, 5.5),
          legend.position = "bottom",
          legend.text = element_text(size = 5),
          legend.title = element_blank(),
          legend.key.size = unit(0.3, "cm")
        )
    } else {
      # Build faceted performance plot (CI band fallback, no shared legend)
      p_perf <- ggplot() +
        # Null distribution density
        geom_density(data = df_null_long, aes(x = null_value),
                     fill = "#CCCCCC", color = "#666666", alpha = 0.4, linewidth = 0.3) +
        # Bootstrap CI band fallback
        geom_rect(data = df_obs, aes(xmin = ci_low, xmax = ci_high, ymin = -Inf, ymax = Inf),
                  fill = "#377eb8", alpha = 0.15) +
        # Trained model score as vertical line
        geom_vline(data = df_obs, aes(xintercept = score),
                   color = "#377eb8", linewidth = 0.7) +
        # Permutation Null distribution mean as vertical line
        geom_vline(data = df_obs, aes(xintercept = null_mean),
                   color = "#666666", linewidth = 0.5) +
        # Permutation Null mean+SD annotation (centered below null mean line)
        geom_label(data = df_obs, aes(x = null_mean, y = -Inf,
                   label = sprintf("%.2f (%.2f)", null_mean, null_sd)),
                   vjust = 0.5, hjust = 0.5, size = 1.5, color = "black",
                   fill = "white", label.size = NA, label.padding = unit(0.15, "lines")) +
        coord_cartesian(clip = "off") +
        # Facet by metric
        facet_wrap(~ metric, scales = "free", ncol = 1) +
        labs(x = "Score", y = "Density") +
        theme_minimal(base_size = 7) +
        theme(
          strip.text = element_text(size = 6, face = "bold"),
          axis.title = element_text(size = 5, face = "bold"),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          panel.grid.major = element_line(color = "grey92"),
          panel.grid.minor = element_blank(),
          plot.background = element_rect(fill = "transparent", color = NA),
          plot.margin = margin(5.5, 5.5, 8, 5.5)
        )
    }

    # Dynamic sizing: 1 col, height scales with number of metrics
    n_metrics <- nrow(df_obs)
    fig_w <- 2.75
    fig_h <- max(1.275, n_metrics * 1.275)

    ggsave(file.path(PLOT_DIR, "0_model_performance.png"),
           p_perf, width = fig_w, height = fig_h, dpi = 300, bg = "transparent")
    cat("[INFO] Saved 0_model_performance.png\n")
  } else {
    cat("[INFO] No performance files found, skipping performance plot.\n")
  }
  perf_plotted_flag <- TRUE
}

stats_path <- file.path(SHAP_DIR, "shap_stats_global.csv")
df_stats <- read.csv(stats_path)
df_stats$sig_GII <- toupper(as.character(df_stats$sig_GII)) == "TRUE"
df_stats$sig_V   <- toupper(as.character(df_stats$sig_V))   == "TRUE"
df_sig <- df_stats %>%
  filter(sig_GII | sig_V) %>%
  mutate(
    v_only = sig_V & !sig_GII,
    rank_metric = ifelse(v_only, -V, -GII),
    rank_tier = ifelse(v_only, 1L, 0L)
  ) %>%
  arrange(rank_tier, rank_metric) %>%
  mutate(rank = row_number()) %>%
  select(-rank_metric, -rank_tier)

cat(sprintf("[INFO] Found %d significant features to plot.\n", nrow(df_sig)))

micro_path <- file.path(SHAP_DIR, "microdata_GII.parquet")
micro_v_path <- file.path(SHAP_DIR, "microdata_V.parquet")
boot_path <- file.path(SHAP_DIR, "bootstrap_distributions_M.parquet")
noise_path <- file.path(SHAP_DIR, "stratified_noise_distributions_M.parquet")

if (!file.exists(micro_path) || !file.exists(boot_path) || !file.exists(noise_path)) {
  cat(sprintf("[WARNING] Missing parquet files in %s, skipping.\n", shap_label))
  next
}

df_micro <- read_parquet(micro_path)
df_micro_v <- if (file.exists(micro_v_path)) read_parquet(micro_v_path) else NULL
df_boot <- read_parquet(boot_path)
df_noise <- read_parquet(noise_path)

micro_gii_effects <- unique(df_micro$effect_name)
micro_v_effects <- if (!is.null(df_micro_v)) unique(df_micro_v$effect_name) else character(0)

# -----------------------------------------------------------------------------
# 5. GII PLOTTING LOOP
# -----------------------------------------------------------------------------

if (nrow(df_sig) == 0) {
  cat("[INFO] No significant features to plot. Skipping.\n")
  next
}

results <- foreach(i = 1:nrow(df_sig), .packages = c("ggplot2", "dplyr", "splines", "gridExtra", "grid", "stringr", "grDevices")) %dopar% {

  row <- df_sig[i, ]
  feat_name <- row$effect
  feat_rank <- row$rank
  feat_type <- row$type
  is_v_only <- isTRUE(row$v_only)

  # --- MICRODATA DISPATCH ---
  if (is_v_only && feat_name %in% micro_v_effects) {
    df_m_source <- df_micro_v
  } else {
    df_m_source <- df_micro
  }

  # --- PANEL 1: DENSITY (skipped for V-sig-only effects; they did not pass the M test) ---
  p1 <- NULL
  if (!is_v_only) {
    vec_signal <- df_boot[[feat_name]]
    vec_noise <- df_noise[[feat_name]]

    signal_m <- mean(vec_signal, na.rm = TRUE)
    signal_sd <- sd(vec_signal, na.rm = TRUE)
    noise_m <- mean(vec_noise, na.rm = TRUE)
    noise_sd <- sd(vec_noise, na.rm = TRUE)
    local_xmax <- max(c(vec_noise, vec_signal), na.rm = TRUE) * 1.05

    # Stats are embedded directly into the legend text (rendered via ggtext::element_markdown()
    # below) rather than as spatial plot annotations, avoiding label-to-label collision when
    # the noise/signal distributions sit close together.
    noise_label <- sprintf("**Noise:** <span style='font-size:3.6pt'>%.2f (%.2f)</span>", noise_m, noise_sd)
    signal_label <- sprintf("**Signal:** <span style='font-size:3.6pt'>%.2f (%.2f)</span>", signal_m, signal_sd)

    df_p1 <- data.frame(
      val = c(vec_noise, vec_signal),
      type = rep(c("Noise", "Signal"), c(length(vec_noise), length(vec_signal)))
    )

    p1 <- ggplot(df_p1, aes(x = val, fill = type, color = type)) +
      geom_density(aes(alpha = type), linewidth = 0.4) +
      scale_fill_manual(values = c("Noise" = "lightgray", "Signal" = "#377eb8"),
                         labels = c("Noise" = noise_label, "Signal" = signal_label)) +
      scale_color_manual(values = c("Noise" = "#404040", "Signal" = "#08306b"),
                          labels = c("Noise" = noise_label, "Signal" = signal_label)) +
      scale_alpha_manual(values = c("Noise" = 0.5, "Signal" = 1.0),
                          labels = c("Noise" = noise_label, "Signal" = signal_label)) +
      scale_x_continuous(limits = c(0, local_xmax), expand = c(0, 0)) +
      scale_y_continuous(expand = c(0, 0)) +

      theme_minimal(base_size = 7) +
      theme(
        legend.position = "bottom",
        legend.direction = "horizontal",
        legend.background = element_rect(fill = "transparent", color = NA, linewidth = 0),
        legend.key.size = unit(0.2, "cm"),
        legend.text = ggtext::element_markdown(size = 4.2),
        legend.title = element_blank(),
        legend.spacing.x = unit(3, "mm"),
        legend.margin = margin(0, 0, 0, 0),
        legend.box.margin = margin(0, 0, 0, 0),

        axis.title.y = element_text(size = 5, angle = 90, vjust = 1, face = "bold"),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.title.x = element_text(size = 5, face = "bold"),

        panel.grid.major = element_line(color = "grey92"),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        plot.background = element_rect(fill = "transparent", color = NA),
        plot.margin = unit(c(1, 0.5, 1, 1), "mm")
      ) +
      labs(x = "Importance Magnitude (M)", y = "Density")
  }

  # --- PANEL 2: V-COMPONENT ---
  df_m <- df_m_source %>% filter(effect_name == feat_name)

  if (nrow(df_m) == 0) return(sprintf("Skipped %s: No valid data", feat_name))

  m_type <- unique(df_m$main_feature_type)[1]
  is_main_discrete <- m_type %in% c("nominal", "ordinal", "binary")

  # Type-aware filtering: preserve MISSING as valid level for discrete features
  if (is_main_discrete) {
    # Recode NaN/NA raw labels as "MISSING" factor level
    df_m <- df_m %>% mutate(
      main_feature_raw = ifelse(
        is.na(main_feature_raw) | main_feature_raw == "nan" | main_feature_raw == "NaN",
        "NA", as.character(main_feature_raw)
      )
    )
  } else {
    # Continuous: convert to numeric and drop NaN rows
    df_m <- df_m %>%
      mutate(feature_value = as.numeric(feature_value)) %>%
      filter(!is.na(feature_value) & !is.nan(feature_value))
  }

  if (nrow(df_m) == 0) return(sprintf("Skipped %s: No valid data after filtering", feat_name))

  # TRANSFORM SHAP (sign-flip only; color/ordering anchored to raw signed SHAP)
  if (NEGATE_SHAP){
    df_m$shap_value <- -df_m$shap_value
  }
  # Skip OUTCOME_MAX scaling for multi_regression: SHAP values are on z-scaled
  # targets (StandardScaler applied in train.py), so percent-of-max rescaling
  # would create a unit mismatch.
  task_type <- cfg$modeling$task_type
  if (!identical(task_type, "multi_regression")) {
    df_m$shap_value <- (df_m$shap_value / OUTCOME_MAX) * 100
  }

  p2 <- NULL
  legend_title <- "Feature Value"

  # --- CUSTOM Y-AXIS LABEL GROB ---
  # Transparent background implicit in textGrob
  y_grob_title <- textGrob(GII_Y_LABEL, rot = 90,
                           gp = gpar(fontsize = 5.5, fontface = "bold", col = "black"))
  y_grob_sub   <- textGrob(GII_Y_SUBLABEL, rot = 90,
                           gp = gpar(fontsize = 4.5, fontface = "plain", col = "black"))

  y_axis_grob <- arrangeGrob(y_grob_title, y_grob_sub, ncol = 2,
                             widths = unit(c(2.5, 2.0), "mm"))

  if (feat_type == "Interaction") {
    main_feat_name <- unique(df_m$main_feature)[1]
    partner_feat_name <- unique(df_m$interaction_partner)[1]

    orientations <- list(
      list(focal_val_col = "feature_value", focal_raw_col = "main_feature_raw",
           focal_type_col = "main_feature_type", mod_val_col = "partner_value",
           mod_raw_col = "partner_feature_raw", mod_type_col = "partner_feature_type",
           mod_name = partner_feat_name, focal_name = main_feat_name),
      list(focal_val_col = "partner_value", focal_raw_col = "partner_feature_raw",
           focal_type_col = "partner_feature_type", mod_val_col = "feature_value",
           mod_raw_col = "main_feature_raw", mod_type_col = "main_feature_type",
           mod_name = main_feat_name, focal_name = partner_feat_name)
    )

    suffix <- if (is_v_only) "Vsig" else "GII"
    clean_name <- str_replace_all(feat_name, "[^a-zA-Z0-9_]", "")

    ori_msgs <- c()

    for (ori in orientations) {
      legend_title <- ori$mod_name
      focal_label <- ori$focal_name

      df_ori <- df_m %>%
        mutate(
          focal_value_enc = suppressWarnings(as.numeric(.data[[ori$focal_val_col]])),
          focal_raw = as.character(.data[[ori$focal_raw_col]]),
          mod_value_enc = suppressWarnings(as.numeric(.data[[ori$mod_val_col]])),
          mod_raw = as.character(.data[[ori$mod_raw_col]])
        )

      focal_type <- unique(df_ori[[ori$focal_type_col]])[1]
      mod_type <- unique(df_ori[[ori$mod_type_col]])[1]
      is_focal_type_discrete <- focal_type %in% c("nominal", "ordinal", "binary")

      # Recode NA raw labels for discrete focal; drop NaN rows for continuous focal
      if (is_focal_type_discrete) {
        df_ori <- df_ori %>% mutate(
          focal_raw = ifelse(is.na(focal_raw) | focal_raw == "nan" | focal_raw == "NaN",
                              "NA", focal_raw)
        )
      } else {
        df_ori <- df_ori %>% filter(!is.na(focal_value_enc) & !is.nan(focal_value_enc))
      }

      if (nrow(df_ori) == 0) {
        ori_msgs <- c(ori_msgs, sprintf("Skipped %s (mod=%s): No valid focal data", feat_name, ori$mod_name))
        next
      }

      is_focal_discrete <- is_focal_type_discrete |
        (!is_focal_type_discrete & n_distinct(df_ori$focal_value_enc) <= SPLINE_DISC_THRESH)

      # --- Moderator stratification (type-aware; both orientations) ---
      strat <- stratify_moderator(df_ori$mod_value_enc, df_ori$mod_raw, mod_type, SPLINE_DISC_THRESH, MAX_INTERACTION_STRATA)
      df_ori$stratum <- strat$strata
      df_ori <- df_ori[!is.na(df_ori$stratum), ]

      if (nrow(df_ori) == 0 || length(strat$levels) == 0) {
        ori_msgs <- c(ori_msgs, sprintf("Skipped %s (mod=%s): No valid stratified data", feat_name, ori$mod_name))
        next
      }

      df_ori$stratum <- factor(df_ori$stratum, levels = strat$levels)
      n_strata <- length(strat$levels)

      if (n_strata > MAX_INTERACTION_STRATA) {
        if (identical(mod_type, "nominal")) {
          grand_mean_shap <- mean(df_ori$shap_value, na.rm = TRUE)
          strata_contrib <- df_ori %>%
            filter(!is.na(stratum)) %>%
            group_by(stratum) %>%
            summarise(
              n_k = n(),
              contribution = n() * (mean(shap_value, na.rm = TRUE) - grand_mean_shap)^2,
              .groups = "drop"
            ) %>%
            arrange(desc(contribution))
          keep_strata <- as.character(strata_contrib$stratum[1:MAX_INTERACTION_STRATA])
          df_ori <- df_ori %>% filter(as.character(stratum) %in% keep_strata)
          strat$levels <- keep_strata
        } else {
          valid_mod <- !is.na(df_ori$mod_value_enc) & !is.nan(df_ori$mod_value_enc)
          probs <- seq(0, 1, length.out = MAX_INTERACTION_STRATA + 1)
          breaks <- unique(quantile(df_ori$mod_value_enc[valid_mod], probs = probs, type = 7, names = FALSE))
          if (length(breaks) >= 2) {
            bin_idx <- cut(df_ori$mod_value_enc, breaks = breaks, include.lowest = TRUE, labels = FALSE)
            bin_labels <- sprintf("[%.2f, %.2f]", breaks[-length(breaks)], breaks[-1])
            df_ori$stratum <- bin_labels[bin_idx]
            strat$levels <- bin_labels
            strat$method <- "quantile_bins_capped"
          }
        }
        df_ori$stratum <- factor(df_ori$stratum, levels = strat$levels)
        df_ori <- df_ori[!is.na(df_ori$stratum), ]
        n_strata <- length(strat$levels)
      }

      strata_colors <- get_red_blue_palette(n_strata)
      names(strata_colors) <- strat$levels

      # --- Focal x-axis construction ---
      if (is_focal_discrete) {
        fac <- create_ordered_factor(df_ori$focal_raw, df_ori$focal_value_enc)

        # V-contribution-ranked top-5 selection (NOMINAL focal only), mirrors singleton logic.
        if (identical(focal_type, "nominal") && nlevels(fac) > 5) {
          grand_mean_shap <- mean(df_ori$shap_value, na.rm = TRUE)
          level_contrib <- df_ori %>%
            group_by(focal_value_enc) %>%
            summarise(
              n_k = n(),
              contribution = n() * (mean(shap_value, na.rm = TRUE) - grand_mean_shap) ^ 2,
              .groups = "drop"
            ) %>%
            arrange(desc(contribution))
          top <- as.character(level_contrib$focal_value_enc[1:5])
          df_ori <- df_ori %>% filter(as.character(focal_value_enc) %in% top)
          fac <- create_ordered_factor(df_ori$focal_raw, df_ori$focal_value_enc)
        }

        df_ori$x_plot <- as.integer(fac)
        x_labels <- gsub("_", "\n", levels(fac))
        n_lev <- length(x_labels)
        capacity <- n_lev + 0.5
        x_scale <- scale_x_continuous(breaks = 1:n_lev, labels = x_labels, limits = c(0.5, capacity))
        axis_seg <- annotate("segment", x = 1, xend = n_lev, y = -Inf, yend = -Inf,
                              color = "black", linewidth = 0.2)
      } else {
        df_ori$x_plot <- df_ori$focal_value_enc
        x_scale <- scale_x_continuous(guide = guide_axis(check.overlap = TRUE))
        axis_seg <- annotate("segment", x = min(df_ori$x_plot), xend = max(df_ori$x_plot), y = -Inf, yend = -Inf,
                              color = "black", linewidth = 0.2)
      }

      # --- Scatter, colored by moderator stratum ---
      pos <- if (is_focal_discrete) position_jitter(width = 0.1) else position_identity()
      p2 <- ggplot(df_ori, aes(x = x_plot, y = shap_value, color = stratum)) +
        geom_hline(yintercept = 0, color = "gray50", linewidth = 0.3, linetype = "dashed") +
        geom_point(alpha = 0.35, size = 0.9, position = pos) +
        scale_color_manual(values = strata_colors, name = legend_title,
                            guide = guide_legend(override.aes = list(alpha = 1))) +
        x_scale + axis_seg

      # --- Per-stratum overlay (spline for continuous focal, group means for discrete focal) ---
      if (!is_focal_discrete) {
        trend_list <- fit_per_stratum_splines(df_ori, "x_plot", "shap_value", "stratum", cfg)
        for (s in strat$levels) {
          trend_s <- trend_list[[s]]
          if (is.null(trend_s) || nrow(trend_s) == 0) next

          df_stratum_sub <- df_ori[as.character(df_ori$stratum) == s, ]
          boot_sd_stratum <- bootstrap_spline_sd(df_stratum_sub$x_plot, df_stratum_sub$shap_value, cfg, BOOT_RIBBON_B, MIN_BOOT_N)
          if (!is.null(boot_sd_stratum)) {
            p2 <- p2 + geom_ribbon(data = boot_sd_stratum, aes(x = x, ymin = y_pred - sd, ymax = y_pred + sd),
                                   fill = strata_colors[[s]], alpha = 0.15, inherit.aes = FALSE)
          }

          p2 <- p2 +
            geom_line(data = trend_s, aes(x = x, y = y_pred), color = "white", linewidth = 1.0, inherit.aes = FALSE) +
            geom_line(data = trend_s, aes(x = x, y = y_pred), color = strata_colors[[s]], linewidth = 0.5, inherit.aes = FALSE)
        }
      } else {
        # DISCRETE FOCAL: per-stratum group means at each focal level, offset to avoid overplot.
        means_df <- compute_per_stratum_group_means(df_ori, "x_plot", "shap_value", "stratum")
        strat_idx <- as.integer(factor(means_df$stratum, levels = strat$levels))
        offset_width <- 0.6
        means_df$x_offset <- means_df$x_plot +
          (strat_idx - (n_strata + 1) / 2) * (offset_width / n_strata)

        for (s in strat$levels) {
          means_s <- means_df[as.character(means_df$stratum) == s, ]
          if (nrow(means_s) == 0) next

          df_stratum_sub <- df_ori[as.character(df_ori$stratum) == s, ]
          boot_sd_stratum_disc <- bootstrap_group_mean_sd(
            factor(df_stratum_sub$x_plot, levels = sort(unique(df_stratum_sub$x_plot))),
            df_stratum_sub$shap_value, BOOT_RIBBON_B, MIN_BOOT_N
          )
          means_s <- means_s %>%
            left_join(boot_sd_stratum_disc %>% mutate(x_plot = as.numeric(level)) %>% select(x_plot, sd),
                      by = "x_plot")

          if (any(!is.na(means_s$sd))) {
            p2 <- p2 + geom_errorbar(data = means_s, aes(x = x_offset, y = mean_shap, ymin = mean_shap - sd, ymax = mean_shap + sd),
                                     color = strata_colors[[s]], alpha = 0.6, width = 0.05, linewidth = 0.5,
                                     inherit.aes = FALSE, na.rm = TRUE)
          }

          p2 <- p2 +
            geom_errorbar(data = means_s, aes(x = x_offset, y = mean_shap, ymin = mean_shap, ymax = mean_shap),
                         color = strata_colors[[s]], width = 0.20, linewidth = 1.5, inherit.aes = FALSE)

          if (nrow(means_s) > 1) {
            p2 <- p2 + geom_line(
              data = means_s, aes(x = x_offset, y = mean_shap),
              color = strata_colors[[s]], linewidth = 0.6, alpha = 1.0,
              inherit.aes = FALSE
            )
          }
        }
      }

      p2 <- p2 +
        theme_minimal(base_size = 7) +
        theme(
          axis.title.x = element_text(size = 5, face = "bold"),
          axis.title.y = element_blank(),

          legend.position = "right",
          legend.key.height = unit(0.2, "cm"),
          legend.key.width = unit(0.2, "cm"),
          legend.title = element_text(size = 5, face = "bold"),
          legend.text = element_text(size = 4.5),
          legend.margin = margin(0,0,0,0),

          plot.margin = unit(c(1, 10, 1, 1), "mm"),

          panel.border = element_blank(),
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.grid.major = element_line(color = "grey92"),
          panel.grid.minor = element_blank(),
          axis.line.x = element_blank()
        ) +
        labs(y = NULL, x = focal_label)

      clean_mod_name <- str_replace_all(ori$mod_name, "[^a-zA-Z0-9_]", "")
      fname_ori <- sprintf("%d_%s_%s_mod_%s.png", feat_rank, clean_name, suffix, clean_mod_name)
      fpath_ori <- file.path(PLOT_DIR, fname_ori)

      msg_ori <- tryCatch({
        p2_with_axis <- arrangeGrob(p2, left = y_axis_grob)
        if (is.null(p1)) {
          g <- p2_with_axis
          save_width_ori <- 5.1 * (3.25 / 4.25)
        } else {
          g <- arrangeGrob(p1, p2_with_axis, ncol = 2, widths = unit(c(1, 3.25), "null"))
          save_width_ori <- 5.1
        }
        ggsave(fpath_ori, g, width = save_width_ori, height = 1.5, dpi = 300, bg = "transparent")
        sprintf("Saved: %s", fname_ori)
      }, error = function(e) {
        sprintf("Error plotting %s (mod=%s): %s", feat_name, ori$mod_name, e$message)
      })
      ori_msgs <- c(ori_msgs, msg_ori)
    }

    return(paste(ori_msgs, collapse = "; "))

  } else {
    is_discrete <- is_main_discrete | (!is_main_discrete & n_distinct(df_m$feature_value) <= SPLINE_DISC_THRESH)

    if (!is_discrete) {
      # CONTINUOUS
      df_m$x_plot <- as.numeric(df_m$feature_value)
      trend_data <- calc_v_spline_pred(df_m$x_plot, df_m$shap_value, cfg)
      boot_sd_singleton <- bootstrap_spline_sd(df_m$x_plot, df_m$shap_value, cfg, BOOT_RIBBON_B, MIN_BOOT_N)

      axis_seg <- annotate("segment", x = min(df_m$x_plot), xend = max(df_m$x_plot), y = -Inf, yend = -Inf,
                            color = "black", linewidth = 0.2)

      p2 <- ggplot(df_m, aes(x = x_plot, y = shap_value)) +
        geom_hline(yintercept = 0, color="gray50", linewidth=0.3, linetype="dashed") +
        geom_point(aes(color = x_plot), alpha = 0.5, size = 0.9)

      if (!is.null(boot_sd_singleton)) {
        p2 <- p2 + geom_ribbon(data = boot_sd_singleton, aes(x = x, ymin = y_pred - sd, ymax = y_pred + sd),
                               fill = "black", alpha = 0.15, inherit.aes = FALSE)
      }

      p2 <- p2 +
        geom_line(data = trend_data, aes(x = x, y = y_pred), color = "white", linewidth = 1.0) +
        geom_line(data = trend_data, aes(x = x, y = y_pred), color = "black", linewidth = 0.5) +
        scale_color_gradient(low = "#b2182b", high = "#2166ac", name = legend_title,
                           guide = guide_colorbar(reverse = TRUE)) +
        scale_x_continuous(guide = guide_axis(check.overlap = TRUE)) +
        axis_seg

    } else {
      # DISCRETE SINGLETON
      fac <- create_ordered_factor(df_m$main_feature_raw, df_m$feature_value)
      # V-contribution-ranked top-5 selection (NOMINAL only).
      # V_nominal is the ANOVA between-group SS contribution per level:
      #   contribution_k = count_k * (mean_SHAP_k - grand_mean_SHAP)^2
      # Ranking by this contribution exactly matches the per-level contribution to
      # the V-statistic shown in the plot.
      if (m_type == "nominal" && nlevels(fac) > 5) {
        grand_mean_shap <- mean(df_m$shap_value, na.rm = TRUE)
        level_contrib <- df_m %>%
          group_by(feature_value) %>%
          summarise(
            n_k = n(),
            mean_shap_k = mean(shap_value, na.rm = TRUE),
            contribution = n() * (mean(shap_value, na.rm = TRUE) - grand_mean_shap) ^ 2,
            .groups = "drop"
          ) %>%
          arrange(desc(contribution))

        top <- as.character(level_contrib$feature_value[1:5])
        df_m <- df_m %>% filter(as.character(feature_value) %in% top)
        fac <- create_ordered_factor(df_m$main_feature_raw, df_m$feature_value)
      }

      df_m$x_plot <- as.integer(fac)
      x_labels <- gsub("_", "\n", levels(fac))
      n_lev <- length(x_labels)
      capacity <- n_lev + 0.5

      df_means <- df_m %>% group_by(x_plot) %>% summarize(m = mean(shap_value), .groups='drop') %>% arrange(x_plot)

      boot_sd_disc_singleton <- bootstrap_group_mean_sd(
        factor(df_m$x_plot, levels = sort(unique(df_m$x_plot))), df_m$shap_value, BOOT_RIBBON_B, MIN_BOOT_N
      )
      df_means <- df_means %>%
        left_join(boot_sd_disc_singleton %>% mutate(x_plot = as.numeric(level)) %>% select(x_plot, sd),
                  by = "x_plot")

      axis_seg <- annotate("segment", x = 1, xend = nlevels(fac), y = -Inf, yend = -Inf,
                            color = "black", linewidth = 0.2)

      p2 <- ggplot(df_m, aes(x = x_plot, y = shap_value)) +
        geom_hline(yintercept = 0, color="gray50", linewidth=0.3, linetype="dashed") +
        geom_point(aes(color = fac), alpha = 0.5, size = 0.9,
                   position = position_jitter(width = 0.1))

      if (any(!is.na(df_means$sd))) {
        p2 <- p2 + geom_errorbar(data = df_means, aes(x = x_plot, y = m, ymin = m - sd, ymax = m + sd),
                                 color = "gray40", alpha = 0.4, width = 0.3, linewidth = 0.3,
                                 inherit.aes = FALSE, na.rm = TRUE)
      }

      p2 <- p2 +
        geom_errorbar(data = df_means, aes(y = m, ymin = m, ymax = m),
                      color = "black", width = 0.5, linewidth = 0.5) +
        scale_color_manual(values = get_red_blue_palette(n_lev), name = legend_title,
                           guide = guide_legend(override.aes = list(alpha = 1))) +
        scale_x_continuous(breaks = 1:n_lev, labels = x_labels, limits = c(0.5, capacity)) +
        axis_seg
    }
  }

  # Common Theme Panel 2
  p2 <- p2 +
    theme_minimal(base_size = 7) +
    theme(
      axis.title.x = element_text(size = 5, face = "bold"),
      axis.title.y = element_blank(), # Handled by Custom Grob

      legend.position = "right",
      legend.key.height = unit(0.2, "cm"),
      legend.key.width = unit(0.2, "cm"),
      legend.title = element_text(size = 5, face = "bold"),
      legend.text = element_text(size = 4.5),
      legend.margin = margin(0,0,0,0),

      plot.margin = unit(c(1, 10, 1, 1), "mm"),

      panel.border = element_blank(),
      # TRANSPARENT PLOT BACKGROUND
      plot.background = element_rect(fill = "transparent", color = NA),
      panel.grid.major = element_line(color = "grey92"),
      panel.grid.minor = element_blank(),
      axis.line.x = element_blank()
    ) +
    labs(y = NULL, x = feat_name)

  # --- SAVE ---
  suffix <- if (is_v_only) "Vsig" else "GII"
  clean_name <- str_replace_all(feat_name, "[^a-zA-Z0-9_]", "")
  fname <- sprintf("%d_%s_%s.png", feat_rank, clean_name, suffix)
  fpath <- file.path(PLOT_DIR, fname)

  tryCatch({
    p2_with_axis <- arrangeGrob(p2, left = y_axis_grob)
    if (is.null(p1)) {
      g <- p2_with_axis
      save_width <- 5.1 * (3.25 / 4.25)
    } else {
      g <- arrangeGrob(p1, p2_with_axis, ncol = 2, widths = unit(c(1, 3.25), "null"))
      save_width <- 5.1
    }

    # SAVE WITH TRANSPARENT BG
    ggsave(fpath, g, width = save_width, height = 1.5, dpi = 300, bg = "transparent")
    return(sprintf("Saved: %s", fname))
  }, error = function(e) {
    return(sprintf("Error plotting %s: %s", feat_name, e$message))
  })
}

for (msg in results) {
  prefix <- if (grepl("Error", msg, fixed = TRUE)) "[WARN]" else "[INFO]"
  cat(sprintf("%s %s\n", prefix, msg))
}

cat(sprintf("[INFO] Done plotting for %s.\n", shap_label))

}  # end for (SHAP_DIR in shap_dirs)

# -----------------------------------------------------------------------------
# 6. PER-INDIVIDUAL SHAP PLOTS (indiv_reports)
# -----------------------------------------------------------------------------

render_indiv_main_effects_plots <- function(path, out_dir, y_label, y_sublabel, negate_flag, n_cores) {
  # Load long-format parquet (one row per individual x feature, all features)
  df_all <- tryCatch(
    read_parquet(path),
    error = function(e) {
      cat(sprintf("[WARNING] Could not read main_effects.parquet: %s\n", e$message))
      return(NULL)
    }
  )
  if (is.null(df_all) || nrow(df_all) == 0) {
    cat("[INFO] main_effects.parquet is empty or unreadable; skipping individual main-effects plots.\n")
    return(invisible(NULL))
  }

  # Create output directory
  indiv_plot_dir <- file.path(out_dir, "plots")
  if (!dir.exists(indiv_plot_dir)) dir.create(indiv_plot_dir, recursive = TRUE)

  # Filter to sig_GII=TRUE features
  if ("sig_GII" %in% names(df_all)) {
    df_sig <- df_all %>% filter(sig_GII == TRUE | sig_GII == "True" | sig_GII == "TRUE")
  } else {
    df_sig <- df_all
  }

  if (nrow(df_sig) == 0) {
    cat("[INFO] No sig_GII=TRUE features in main_effects.parquet; skipping individual main-effects plots.\n")
    return(invisible(NULL))
  }

  # Get list of unique individual IDs
  ids <- unique(df_sig$id)
  cat(sprintf("[INFO] Rendering per-individual main-effects plots for %d individuals.\n", length(ids)))

  # Build y-axis label grob constructor (used inside mclapply)
  make_y_grob <- function(y_lbl, y_sub) {
    y_grob_title <- textGrob(y_lbl, rot = 90,
                             gp = gpar(fontsize = 7, fontface = "bold", col = "black"))
    if (nchar(trimws(y_sub)) > 0) {
      y_grob_sub <- textGrob(y_sub, rot = 90,
                             gp = gpar(fontsize = 5.5, fontface = "plain", col = "black"))
      arrangeGrob(y_grob_title, y_grob_sub, ncol = 2,
                  widths = unit(c(3.0, 2.5), "mm"))
    } else {
      y_grob_title
    }
  }

  # Detect multiclass schema: main_effects.parquet has a 'class' column when n_outputs > 1.
  is_multiclass_main <- "class" %in% names(df_sig)

  plot_one_individual <- function(indiv_id) {
    tryCatch({
      df_i <- df_sig %>% filter(id == indiv_id)

      if (nrow(df_i) == 0) {
        return(sprintf("[SKIP] %s: no sig_GII features", indiv_id))
      }

      # For multiclass tasks, split by class and produce one plot per (individual, class).
      # For non-multiclass tasks, produce a single plot per individual.
      class_levels <- if (is_multiclass_main) unique(as.character(df_i[["class"]])) else NA_character_

      msgs <- c()
      for (cl_val in class_levels) {
        if (is_multiclass_main) {
          df_c <- df_i %>% filter(as.character(.data[["class"]]) == cl_val)
        } else {
          df_c <- df_i
        }

        if (nrow(df_c) == 0) next

        # Determine below-OOB-floor status: if ANY feature has oob_count < OOB_FLOOR_MIN
        below_floor <- FALSE
        if ("oob_count" %in% names(df_c)) {
          below_floor <- any(!is.na(df_c$oob_count) & df_c$oob_count < OOB_FLOOR_MIN)
        }

        # x-axis ordering: features ordered by RAW signed SHAP (descending, most positive left)
        if ("shap_value_raw" %in% names(df_c)) {
          df_c <- df_c %>% arrange(desc(shap_value_raw))
          raw_order <- df_c$feature
          df_c$feature <- factor(df_c$feature, levels = raw_order)
          color_col <- df_c$shap_value_raw
        } else {
          df_c <- df_c %>% arrange(desc(shap_value_scaled))
          raw_order <- df_c$feature
          df_c$feature <- factor(df_c$feature, levels = raw_order)
          color_col <- df_c$shap_value_scaled
        }

        # y-axis values: shap_value_scaled with optional sign-flip
        y_vals <- df_c$shap_value_scaled
        if (negate_flag) y_vals <- -y_vals
        df_c$y_plot <- y_vals

        # CI bounds with sign-flip applied (bounds swap when negated)
        if ("shap_value_ci_lo" %in% names(df_c) && "shap_value_ci_hi" %in% names(df_c)) {
          if (negate_flag) {
            ci_lo <- -df_c$shap_value_ci_hi
            ci_hi <- -df_c$shap_value_ci_lo
          } else {
            ci_lo <- df_c$shap_value_ci_lo
            ci_hi <- df_c$shap_value_ci_hi
          }
        } else {
          ci_lo <- rep(NA_real_, nrow(df_c))
          ci_hi <- rep(NA_real_, nrow(df_c))
        }
        df_c$ci_lo_plot <- ci_lo
        df_c$ci_hi_plot <- ci_hi
        df_c$color_raw  <- color_col

        # Build plot
        p <- ggplot(df_c, aes(x = feature, y = y_plot, color = color_raw)) +
          geom_hline(yintercept = 0, color = "gray50", linewidth = 0.3, linetype = "dashed") +
          geom_point(size = 2.0) +
          scale_color_gradient2(
            low = "#b2182b", mid = "white", high = "#2166ac",
            midpoint = 0, guide = "none"
          ) +
          scale_x_discrete() +
          theme_minimal(base_size = 8) +
          theme(
            axis.title.x = element_text(size = 6, face = "bold"),
            axis.title.y = element_blank(),
            axis.text.x = element_text(size = 5.5, angle = 45, hjust = 1),
            panel.grid.major = element_line(color = "grey92"),
            panel.grid.minor = element_blank(),
            panel.border = element_blank(),
            plot.background = element_rect(fill = "transparent", color = NA),
            plot.margin = unit(c(2, 2, 2, 2), "mm"),
            plot.caption = element_text(size = 5, hjust = 0, margin = margin(t = 4))
          ) +
          labs(x = "Feature", y = NULL)

        # Add whiskers only for compliant plots (not below-floor)
        if (!below_floor) {
          p <- p + geom_errorbar(
            aes(ymin = ci_lo_plot, ymax = ci_hi_plot),
            width = 0.2, linewidth = 0.5, na.rm = TRUE
          )
        } else {
          p <- p + labs(
            caption = "CI unavailable (oob_count < 50); point estimate shown only."
          )
        }

        # Y-axis grob
        y_axis_grob <- make_y_grob(y_label, y_sublabel)

        # Save: multiclass uses <id>_main_effects_<label>.png; non-multiclass uses <id>_main_effects.png
        safe_id <- str_replace_all(as.character(indiv_id), "[^a-zA-Z0-9_\\-]", "_")
        if (is_multiclass_main) {
          safe_cl <- str_replace_all(as.character(cl_val), "[^a-zA-Z0-9_\\-]", "_")
          fname <- sprintf("%s_main_effects_%s.png", safe_id, safe_cl)
        } else {
          fname <- sprintf("%s_main_effects.png", safe_id)
        }
        fpath <- file.path(indiv_plot_dir, fname)

        p_with_axis <- arrangeGrob(p, left = y_axis_grob)
        ggsave(fpath, p_with_axis, width = 10, height = 5, dpi = 300, bg = "transparent")
        msgs <- c(msgs, sprintf("Saved: %s", fname))
      }
      return(paste(msgs, collapse = "; "))
    }, error = function(e) {
      return(sprintf("[ERROR] individual %s: %s", indiv_id, e$message))
    })
  }

  # Parallelize over individuals using mclapply (Unix) or lapply (Windows fallback)
  if (.Platform$OS.type == "unix") {
    results_indiv <- parallel::mclapply(ids, plot_one_individual, mc.cores = n_cores)
  } else {
    results_indiv <- lapply(ids, plot_one_individual)
  }

  for (msg in results_indiv) {
    cat(sprintf("[INFO] %s\n", msg))
  }
  invisible(NULL)
}


render_indiv_interactions_plots <- function(path, out_dir, y_label, y_sublabel, negate_flag, n_cores) {
  # Load long-format parquet for interactions (already filtered to sig_GII=TRUE at emission)
  df_all <- tryCatch(
    read_parquet(path),
    error = function(e) {
      cat(sprintf("[WARNING] Could not read interactions.parquet: %s\n", e$message))
      return(NULL)
    }
  )

  if (is.null(df_all) || nrow(df_all) == 0) {
    cat("[INFO] interactions.parquet is empty or unreadable; skipping individual interactions plots.\n")
    return(invisible(NULL))
  }

  # Create output directory
  indiv_plot_dir <- file.path(out_dir, "plots")
  if (!dir.exists(indiv_plot_dir)) dir.create(indiv_plot_dir, recursive = TRUE)

  # Construct composite x-axis label: feature_a x feature_b
  if ("feature_a" %in% names(df_all) && "feature_b" %in% names(df_all)) {
    df_all <- df_all %>%
      mutate(pair_label = paste0(feature_a, " × ", feature_b))
  } else if ("feature" %in% names(df_all)) {
    # Fall back if parquet uses a single composite column
    df_all <- df_all %>% mutate(pair_label = feature)
  } else {
    cat("[WARNING] interactions.parquet has unexpected schema; skipping individual interactions plots.\n")
    return(invisible(NULL))
  }

  ids <- unique(df_all$id)
  cat(sprintf("[INFO] Rendering per-individual interactions plots for %d individuals.\n", length(ids)))

  # Detect multiclass schema: interactions.parquet has a 'class' column when n_outputs > 1.
  is_multiclass_int <- "class" %in% names(df_all)

  make_y_grob <- function(y_lbl, y_sub) {
    y_grob_title <- textGrob(y_lbl, rot = 90,
                             gp = gpar(fontsize = 7, fontface = "bold", col = "black"))
    if (nchar(trimws(y_sub)) > 0) {
      y_grob_sub <- textGrob(y_sub, rot = 90,
                             gp = gpar(fontsize = 5.5, fontface = "plain", col = "black"))
      arrangeGrob(y_grob_title, y_grob_sub, ncol = 2,
                  widths = unit(c(3.0, 2.5), "mm"))
    } else {
      y_grob_title
    }
  }

  plot_one_individual_int <- function(indiv_id) {
    tryCatch({
      df_i <- df_all %>% filter(id == indiv_id)

      if (nrow(df_i) == 0) {
        return(sprintf("[SKIP] %s: no interactions data", indiv_id))
      }

      # For multiclass tasks, split by class and produce one plot per (individual, class).
      class_levels_int <- if (is_multiclass_int) unique(as.character(df_i[["class"]])) else NA_character_

      msgs <- c()
      for (cl_val in class_levels_int) {
        if (is_multiclass_int) {
          df_c <- df_i %>% filter(as.character(.data[["class"]]) == cl_val)
        } else {
          df_c <- df_i
        }

        if (nrow(df_c) == 0) next

        # Determine below-OOB-floor status
        below_floor <- FALSE
        if ("oob_count" %in% names(df_c)) {
          below_floor <- any(!is.na(df_c$oob_count) & df_c$oob_count < OOB_FLOOR_MIN)
        }

        # Order pairs by RAW signed SHAP (descending)
        if ("shap_value_raw" %in% names(df_c)) {
          df_c <- df_c %>% arrange(desc(shap_value_raw))
          color_col <- df_c$shap_value_raw
        } else {
          df_c <- df_c %>% arrange(desc(shap_value_scaled))
          color_col <- df_c$shap_value_scaled
        }
        df_c$pair_label <- factor(df_c$pair_label, levels = unique(df_c$pair_label))

        # y-axis values with optional sign-flip
        y_vals <- df_c$shap_value_scaled
        if (negate_flag) y_vals <- -y_vals
        df_c$y_plot   <- y_vals
        df_c$color_raw <- color_col

        # CI bounds
        if ("shap_value_ci_lo" %in% names(df_c) && "shap_value_ci_hi" %in% names(df_c)) {
          if (negate_flag) {
            ci_lo <- -df_c$shap_value_ci_hi
            ci_hi <- -df_c$shap_value_ci_lo
          } else {
            ci_lo <- df_c$shap_value_ci_lo
            ci_hi <- df_c$shap_value_ci_hi
          }
        } else {
          ci_lo <- rep(NA_real_, nrow(df_c))
          ci_hi <- rep(NA_real_, nrow(df_c))
        }
        df_c$ci_lo_plot <- ci_lo
        df_c$ci_hi_plot <- ci_hi

        p <- ggplot(df_c, aes(x = pair_label, y = y_plot, color = color_raw)) +
          geom_hline(yintercept = 0, color = "gray50", linewidth = 0.3, linetype = "dashed") +
          geom_point(size = 2.0) +
          scale_color_gradient2(
            low = "#b2182b", mid = "white", high = "#2166ac",
            midpoint = 0, guide = "none"
          ) +
          scale_x_discrete() +
          theme_minimal(base_size = 8) +
          theme(
            axis.title.x = element_text(size = 6, face = "bold"),
            axis.title.y = element_blank(),
            axis.text.x = element_text(size = 5.5, angle = 45, hjust = 1),
            panel.grid.major = element_line(color = "grey92"),
            panel.grid.minor = element_blank(),
            panel.border = element_blank(),
            plot.background = element_rect(fill = "transparent", color = NA),
            plot.margin = unit(c(2, 2, 2, 2), "mm"),
            plot.caption = element_text(size = 5, hjust = 0, margin = margin(t = 4))
          ) +
          labs(x = "Feature Pair", y = NULL)

        if (!below_floor) {
          p <- p + geom_errorbar(
            aes(ymin = ci_lo_plot, ymax = ci_hi_plot),
            width = 0.2, linewidth = 0.5, na.rm = TRUE
          )
        } else {
          p <- p + labs(
            caption = "CI unavailable (oob_count < 50); point estimate shown only."
          )
        }

        y_axis_grob <- make_y_grob(y_label, y_sublabel)

        # Save: multiclass uses <id>_interactions_<label>.png; non-multiclass uses <id>_interactions.png
        safe_id <- str_replace_all(as.character(indiv_id), "[^a-zA-Z0-9_\\-]", "_")
        if (is_multiclass_int) {
          safe_cl <- str_replace_all(as.character(cl_val), "[^a-zA-Z0-9_\\-]", "_")
          fname <- sprintf("%s_interactions_%s.png", safe_id, safe_cl)
        } else {
          fname <- sprintf("%s_interactions.png", safe_id)
        }
        fpath <- file.path(indiv_plot_dir, fname)

        p_with_axis <- arrangeGrob(p, left = y_axis_grob)
        ggsave(fpath, p_with_axis, width = 10, height = 5, dpi = 300, bg = "transparent")
        msgs <- c(msgs, sprintf("Saved: %s", fname))
      }
      return(paste(msgs, collapse = "; "))
    }, error = function(e) {
      return(sprintf("[ERROR] individual %s: %s", indiv_id, e$message))
    })
  }

  if (.Platform$OS.type == "unix") {
    results_indiv <- parallel::mclapply(ids, plot_one_individual_int, mc.cores = n_cores)
  } else {
    results_indiv <- lapply(ids, plot_one_individual_int)
  }

  for (msg in results_indiv) {
    cat(sprintf("[INFO] %s\n", msg))
  }
  invisible(NULL)
}


# --- Auto-discover and render per-individual plots ---
indiv_dir <- file.path(RUN_DIR, "indiv_reports")
if (dir.exists(indiv_dir)) {
  cat(sprintf("\n[INFO] indiv_reports/ found at %s; rendering per-individual plots.\n", indiv_dir))
  main_path <- file.path(indiv_dir, "main_effects.parquet")
  int_path  <- file.path(indiv_dir, "interactions.parquet")
  if (file.exists(main_path)) {
    render_indiv_main_effects_plots(main_path, indiv_dir, INDIV_Y_LABEL,
                                    INDIV_Y_SUBLABEL, NEGATE_SHAP, N_CORES)
  }
  if (file.exists(int_path)) {
    render_indiv_interactions_plots(int_path, indiv_dir, INDIV_Y_LABEL,
                                    INDIV_Y_SUBLABEL, NEGATE_SHAP, N_CORES)
  }
} else {
  cat("[INFO] No indiv_reports/ directory found; skipping per-individual plots.\n")
}

cat("\n[INFO] plot.R complete.\n")
