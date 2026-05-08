#!/usr/bin/env Rscript

# -----------------------------------------------------------------------------
# SHAP Visualization Engine (GII Density + V-Component Splines + indiv_reports)
# -----------------------------------------------------------------------------
# Dependencies: ggplot2, dplyr, nanoparquet, tidyr, foreach, doParallel, gridExtra, splines
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

# Null-coalescing operator (not in base R; mirrors rlang::`%||%`)
`%||%` <- function(a, b) if (!is.null(a)) a else b

# Python equivalent: shap_utils.py:_get_adaptive_knots_and_degree (lines 146-164)
# Returns a list with `interior_knots` (vector) and `degree` (integer).
get_adaptive_knots_and_degree <- function(x_values, n_knots_target, degree_target) {
  # Sort and uniquify
  x_unique <- sort(unique(x_values[!is.na(x_values)]))
  n_unique <- length(x_unique)

  if (n_unique < 2) {
    return(list(interior_knots = numeric(0), degree = 1L))
  }

  # Percentile-based interior knots; type=7 matches numpy's default
  probs <- seq(0, 1, length.out = n_knots_target + 2)
  probs <- probs[2:(length(probs) - 1)]  # exclude 0 and 1 (boundary exclusion)
  candidate_knots <- quantile(x_unique, probs = probs, type = 7, names = FALSE)

  # Drop duplicate knots (occurs when the data is highly discrete)
  interior_knots <- unique(candidate_knots)

  # Boundary exclusion: drop knots at min/max of x_unique
  x_min <- min(x_unique)
  x_max <- max(x_unique)
  interior_knots <- interior_knots[interior_knots > x_min & interior_knots < x_max]

  # Degree downgrade: if fewer than 4 unique interior knots, downgrade to linear (degree=1)
  effective_degree <- ifelse(length(interior_knots) < 4, 1L, as.integer(degree_target))
  return(list(interior_knots = interior_knots, degree = effective_degree))
}

# VISUALIZATION-ONLY spline for plotting trend lines.
# Uses splines::splineDesign with adaptive-knot LSQ fitting that mirrors
# scipy.interpolate.LSQUnivariateSpline as used in shap_utils.py:146-164.
# Knot parameters are read from cfg$shap$splines at call time.
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

  knot_info <- get_adaptive_knots_and_degree(xs, n_knots_target, degree_target)
  interior_knots <- knot_info$interior_knots
  degree <- knot_info$degree

  if (length(xs) < degree + length(interior_knots) + 2L) {
    # Insufficient unique x for stable LSQ; return NA-filled predictions
    return(data.frame(x = xs, y_pred = rep(NA_real_, length(xs))))
  }

  # Construct knot sequence with degree+1 multiplicity at boundaries
  x_min <- min(xs, na.rm = TRUE)
  x_max <- max(xs, na.rm = TRUE)
  knot_seq <- c(rep(x_min, degree + 1L), interior_knots, rep(x_max, degree + 1L))

  # Build B-spline design matrix
  basis <- tryCatch(
    splines::splineDesign(knots = knot_seq, x = xs, ord = degree + 1L, outer.ok = TRUE),
    error = function(e) NULL
  )
  if (is.null(basis)) return(data.frame(x = xs, y_pred = rep(NA_real_, length(xs))))

  # LSQ fit (mirrors scipy.interpolate.LSQUnivariateSpline)
  # Solve: coef = (B^T B)^-1 B^T y
  fit <- tryCatch(qr.solve(basis, ys), error = function(e) NULL)
  if (is.null(fit)) return(data.frame(x = xs, y_pred = rep(NA_real_, length(xs))))

  preds <- as.vector(basis %*% fit)
  return(data.frame(x = xs, y_pred = preds))
}

find_zero_crossing <- function(df_trend) {
  crossings <- c()
  for(i in 1:(nrow(df_trend)-1)) {
    y1 <- df_trend$y_pred[i]
    y2 <- df_trend$y_pred[i+1]
    if (sign(y1) != sign(y2) && y1 != 0) {
      x1 <- df_trend$x[i]
      x2 <- df_trend$x[i+1]
      x_cross <- x1 - y1 * (x2 - x1) / (y2 - y1)
      crossings <- c(crossings, x_cross)
    }
  }
  if(length(crossings) > 0) return(crossings[1])
  return(NULL)
}

create_ordered_factor <- function(raw_vec, enc_vec) {
  df_map <- data.frame(raw = as.character(raw_vec), enc = as.numeric(enc_vec)) %>%
    distinct() %>%
    arrange(enc)
  return(factor(as.character(raw_vec), levels = df_map$raw))
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

    # Reshape null distributions for faceting
    df_null_long <- df_null %>%
      pivot_longer(everything(), names_to = "metric", values_to = "null_value")

    # Merge observed stats with p-values
    df_obs <- df_perf %>%
      left_join(df_perm %>% select(metric, p_value), by = "metric")

    # Build faceted performance plot
    p_perf <- ggplot() +
      # Null distribution density
      geom_density(data = df_null_long, aes(x = null_value),
                   fill = "#CCCCCC", color = "#666666", alpha = 0.4, linewidth = 0.3) +
      # Bootstrap CI as shaded vertical band
      geom_rect(data = df_obs, aes(xmin = ci_low, xmax = ci_high, ymin = -Inf, ymax = Inf),
                fill = "#377eb8", alpha = 0.15) +
      # Observed score as vertical line
      geom_vline(data = df_obs, aes(xintercept = score),
                 color = "#377eb8", linewidth = 0.7) +
      # P-value annotation
      geom_text(data = df_obs, aes(x = score, y = Inf,
                label = sprintf("p = %.3f", p_value)),
                vjust = 1.5, hjust = -0.1, size = 2, color = "#08306b") +
      # Facet by metric
      facet_wrap(~ metric, scales = "free", ncol = 2) +
      labs(x = "Score", y = "Density") +
      theme_minimal(base_size = 7) +
      theme(
        strip.text = element_text(size = 6, face = "bold"),
        axis.title = element_text(size = 5, face = "bold"),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        panel.grid.major = element_line(color = "grey92"),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "transparent", color = NA)
      )

    # Dynamic sizing: 2 cols, height scales with number of metric rows
    n_metrics <- nrow(df_obs)
    n_rows <- ceiling(n_metrics / 2)
    fig_w <- 5.1
    fig_h <- max(1.275, n_rows * 1.275)

    ggsave(file.path(PLOT_DIR, "0_model_performance.png"),
           p_perf, width = fig_w, height = fig_h, dpi = 300, bg = "transparent")
    cat("[INFO] Saved 0_model_performance.png\n")
  } else {
    cat("[INFO] No performance files found, skipping performance plot.\n")
  }
  perf_plotted_flag <- TRUE
}

get_global_x_limit_dir <- function(shap_dir) {
  global_max <- 0
  p1 <- file.path(shap_dir, "bootstrap_distributions_M.parquet")
  if (file.exists(p1)) {
    m <- max(as.matrix(read_parquet(p1)), na.rm = TRUE)
    if (m > global_max) global_max <- m
  }
  p2 <- file.path(shap_dir, "stratified_noise_distributions_M.parquet")
  if (file.exists(p2)) {
    m <- max(as.matrix(read_parquet(p2)), na.rm = TRUE)
    if (m > global_max) global_max <- m
  }
  return(global_max * 1.05)
}

GLOBAL_X_MAX <- get_global_x_limit_dir(SHAP_DIR)

stats_path <- file.path(SHAP_DIR, "shap_stats_global.csv")
df_stats <- read.csv(stats_path)
df_sig <- df_stats %>%
  filter(sig_GII == "True" | sig_GII == "TRUE" | sig_GII == TRUE) %>%
  mutate(rank = rank(-GII, ties.method = "first")) %>%
  arrange(rank)

cat(sprintf("[INFO] Found %d significant features to plot.\n", nrow(df_sig)))

micro_path <- file.path(SHAP_DIR, "microdata_GII.parquet")
boot_path <- file.path(SHAP_DIR, "bootstrap_distributions_M.parquet")
noise_path <- file.path(SHAP_DIR, "stratified_noise_distributions_M.parquet")

if (!file.exists(micro_path) || !file.exists(boot_path) || !file.exists(noise_path)) {
  cat(sprintf("[WARNING] Missing parquet files in %s, skipping.\n", shap_label))
  next
}

df_micro <- read_parquet(micro_path)
df_boot <- read_parquet(boot_path)
df_noise <- read_parquet(noise_path)

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

  # --- PANEL 1: DENSITY ---
  vec_signal <- df_boot[[feat_name]]
  vec_noise <- df_noise[[feat_name]]

  df_p1 <- data.frame(
    val = c(vec_noise, vec_signal),
    type = rep(c("Noise", "Signal"), c(length(vec_noise), length(vec_signal)))
  )

  p1 <- ggplot(df_p1, aes(x = val, fill = type, color = type)) +
    geom_density(aes(alpha = type), linewidth = 0.4) +
    scale_fill_manual(values = c("Noise" = "lightgray", "Signal" = "#377eb8")) +
    scale_color_manual(values = c("Noise" = "#404040", "Signal" = "#08306b")) +
    scale_alpha_manual(values = c("Noise" = 0.5, "Signal" = 1.0)) +
    scale_x_continuous(limits = c(0, GLOBAL_X_MAX), expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +

    theme_minimal(base_size = 7) +
    theme(
      legend.position = c(0.98, 0.98),
      legend.justification = c(1, 1),
      # TRANSPARENT LEGEND BACKGROUND
      legend.background = element_rect(fill = "transparent", color = NA, linewidth = 0),
      legend.key.size = unit(0.2, "cm"),
      legend.text = element_text(size = 4.5),
      legend.title = element_blank(),
      legend.margin = margin(1, 1, 1, 1),

      axis.title.y = element_text(size = 5, angle = 90, vjust = 1, face = "bold"),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.title.x = element_text(size = 5, face = "bold"),

      panel.grid.major = element_line(color = "grey92"),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      # TRANSPARENT PLOT BACKGROUND
      plot.background = element_rect(fill = "transparent", color = NA),
      plot.margin = unit(c(1, 0.5, 1, 1), "mm")
    ) +
    labs(x = "Importance Magnitude (M)", y = "Density")

  # --- PANEL 2: V-COMPONENT ---
  df_m <- df_micro %>% filter(effect_name == feat_name)

  if (nrow(df_m) == 0) return(sprintf("Skipped %s: No valid data", feat_name))

  m_type <- unique(df_m$main_feature_type)[1]
  is_main_discrete <- m_type %in% c("nominal", "ordinal", "binary")

  # Type-aware filtering: preserve MISSING as valid level for discrete features
  if (is_main_discrete) {
    # Recode NaN/NA raw labels as "MISSING" factor level
    df_m <- df_m %>% mutate(
      main_feature_raw = ifelse(
        is.na(main_feature_raw) | main_feature_raw == "nan" | main_feature_raw == "NaN",
        "__NA__", as.character(main_feature_raw)
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
    legend_title <- "Moderator Value"
    p_type <- unique(df_m$partner_feature_type)[1]
    is_partner_discrete <- p_type %in% c("nominal", "ordinal", "binary")

    if (is_partner_discrete) {
      df_m <- df_m %>% mutate(
        partner_feature_raw = ifelse(
          is.na(partner_feature_raw) | partner_feature_raw == "nan" | partner_feature_raw == "NaN",
          "__NA__", as.character(partner_feature_raw)
        )
      )
    } else {
      df_m <- df_m %>%
        mutate(partner_value = as.numeric(partner_value)) %>%
        filter(!is.na(partner_value) & !is.nan(partner_value))
    }

    if (m_type %in% c("nominal", "ordinal", "binary")) {
      fac <- create_ordered_factor(df_m$main_feature_raw, df_m$feature_value)
      df_m$x_plot <- as.integer(fac)
      x_labels <- levels(fac)
      n_lev <- length(x_labels)
      capacity <- if(n_lev == 2) 7 else if(n_lev == 3) 6 else if(n_lev == 4) 5.5 else n_lev + 0.5
      x_scale <- scale_x_continuous(breaks = 1:n_lev, labels = x_labels, limits = c(0.5, capacity))
      pos <- position_jitter(width = 0.1)
    } else {
      df_m$x_plot <- as.numeric(df_m$feature_value)
      x_scale <- scale_x_continuous()
      pos <- position_identity()
    }

    if (is_partner_discrete || n_distinct(df_m$partner_value) <= 5) {
      df_m$col_plot <- create_ordered_factor(df_m$partner_feature_raw, df_m$partner_value)
      p2 <- ggplot(df_m, aes(x = x_plot, y = shap_value, color = col_plot)) +
        geom_hline(yintercept = 0, color="black", linewidth=0.2, linetype="dashed", alpha=0.5) +
        geom_point(alpha = 0.7, size = 0.9, position = pos) +
        scale_color_manual(values = get_red_blue_palette(nlevels(df_m$col_plot)), name = legend_title,
                           guide = guide_legend(reverse = TRUE, override.aes = list(alpha = 1)))
    } else {
      df_m$col_plot <- as.numeric(df_m$partner_value)
      p2 <- ggplot(df_m, aes(x = x_plot, y = shap_value, color = col_plot)) +
        geom_hline(yintercept = 0, color="black", linewidth=0.2, linetype="dashed", alpha=0.5) +
        geom_point(alpha = 0.7, size = 0.9) +
        scale_color_gradient(low = "#b2182b", high = "#2166ac", name = legend_title)
    }
    p2 <- p2 + x_scale

  } else {
    is_discrete <- is_main_discrete | (!is_main_discrete & n_distinct(df_m$feature_value) <= 5)

    if (!is_discrete) {
      # CONTINUOUS
      df_m$x_plot <- as.numeric(df_m$feature_value)
      trend_data <- calc_v_spline_pred(df_m$x_plot, df_m$shap_value, cfg)
      x_cross <- find_zero_crossing(trend_data)

      axis_seg <- geom_segment(aes(x = min(df_m$x_plot), xend = max(df_m$x_plot), y = -Inf, yend = -Inf),
                               color = "black", linewidth = 0.2, inherit.aes = FALSE)

      p2 <- ggplot(df_m, aes(x = x_plot, y = shap_value)) +
        geom_hline(yintercept = 0, color="gray50", linewidth=0.3, linetype="dashed") +
        geom_point(aes(color = x_plot), alpha = 0.5, size = 0.9) +
        geom_line(data = trend_data, aes(x = x, y = y_pred), color = "white", linewidth = 1.0) +
        geom_line(data = trend_data, aes(x = x, y = y_pred), color = "black", linewidth = 0.5) +
        scale_color_gradient(low = "#b2182b", high = "#2166ac", name = legend_title) +
        scale_x_continuous() +
        axis_seg

      if (!is.null(x_cross)) {
        p2 <- p2 +
          geom_vline(xintercept = x_cross, color = "red", linetype = "dashed", linewidth = 0.4) +
          annotate("text", x = x_cross, y = -Inf,
                   label = sprintf("x=%.1f", x_cross),
                   color = "black", size = 1.8, fontface = "plain",
                   vjust = -0.5, hjust = 1.1)
      }

    } else {
      # DISCRETE SINGLETON
      fac <- create_ordered_factor(df_m$main_feature_raw, df_m$feature_value)
      # V-contribution-ranked top-5 selection (NOMINAL only).
      # V_nominal is the ANOVA between-group SS contribution per level:
      #   contribution_k = count_k * (mean_SHAP_k - grand_mean_SHAP)^2
      # Ranking by this contribution exactly matches the per-level contribution to
      # the V-statistic shown in the plot.
      level_label_lookup <- NULL
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

        # Annotate N_k below each surviving level for transparency.
        level_labels_nk <- df_m %>%
          group_by(feature_value) %>%
          summarise(n_k = n(), .groups = "drop")
        level_label_lookup <- setNames(
          paste0(level_labels_nk$feature_value, "\n(N=", level_labels_nk$n_k, ")"),
          as.character(level_labels_nk$feature_value)
        )
      }

      df_m$x_plot <- as.integer(fac)
      x_labels <- if (!is.null(level_label_lookup)) {
        # Map ordered factor levels through the N_k lookup; fall back to bare
        # level name if a level is not found (defensive; should not occur).
        lvls <- levels(fac)
        ifelse(lvls %in% names(level_label_lookup), level_label_lookup[lvls], lvls)
      } else {
        levels(fac)
      }
      n_lev <- length(x_labels)
      capacity <- if(n_lev == 2) 7 else if(n_lev == 3) 6 else if(n_lev == 4) 5.5 else n_lev + 0.5

      df_means <- df_m %>% group_by(x_plot) %>% summarize(m = mean(shap_value), .groups='drop') %>% arrange(x_plot)

      cross_points <- c()
      for(k in 1:(nrow(df_means)-1)) {
        m1 <- df_means$m[k]
        m2 <- df_means$m[k+1]
        if (m1 != 0 && m2 != 0 && sign(m1) != sign(m2)) {
          cross_points <- c(cross_points, (df_means$x_plot[k] + df_means$x_plot[k+1])/2)
        }
      }

      axis_seg <- geom_segment(aes(x = 1, xend = nlevels(fac), y = -Inf, yend = -Inf),
                               color = "black", linewidth = 0.2, inherit.aes = FALSE)

      p2 <- ggplot(df_m, aes(x = x_plot, y = shap_value)) +
        geom_hline(yintercept = 0, color="gray50", linewidth=0.3, linetype="dashed") +
        geom_point(aes(color = fac), alpha = 0.5, size = 0.9,
                   position = position_jitter(width = 0.1)) +
        geom_errorbar(data = df_means, aes(y = m, ymin = m, ymax = m),
                      color = "black", width = 0.5, linewidth = 0.5) +
        scale_color_manual(values = get_red_blue_palette(n_lev), name = legend_title,
                           guide = guide_legend(reverse = TRUE, override.aes = list(alpha = 1))) +
        scale_x_continuous(breaks = 1:n_lev, labels = x_labels, limits = c(0.5, capacity)) +
        axis_seg

      if(length(cross_points) > 0) {
        p2 <- p2 + geom_vline(xintercept = cross_points, color = "red", linetype = "dashed", linewidth = 0.4)
      }
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
    labs(y = NULL, x = "Feature Value")

  # --- SAVE ---
  clean_name <- str_replace_all(feat_name, "[^a-zA-Z0-9_]", "")
  fname <- sprintf("%d_%s_GII.png", feat_rank, clean_name)
  fpath <- file.path(PLOT_DIR, fname)

  tryCatch({
    p2_with_axis <- arrangeGrob(p2, left = y_axis_grob)
    g <- arrangeGrob(p1, p2_with_axis, ncol = 2, widths = unit(c(1, 3.25), "null"))

    # SAVE WITH TRANSPARENT BG
    ggsave(fpath, g, width = 5.1, height = 1.275, dpi = 300, bg = "transparent")
    return(sprintf("Saved: %s", fname))
  }, error = function(e) {
    return(sprintf("Error plotting %s: %s", feat_name, e$message))
  })
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
