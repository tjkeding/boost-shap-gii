#!/usr/bin/env Rscript

# -----------------------------------------------------------------------------
# SHAP Visualization Engine (GII Density + V-Component Splines)
# -----------------------------------------------------------------------------
# Dependencies: ggplot2, dplyr, arrow, tidyr, foreach, doParallel, gridExtra, splines
# -----------------------------------------------------------------------------

# --- 1. USER CONFIGURATION ---------------------------------------------------
# Get the command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if at least one argument is provided
if (length(args) < 4) {
  stop("At least 4 arguments must be supplied: CONFIG_PATH, OUTCOME_RANGE, NEGATE_SHAP, Y_AXIS_LABEL.", call. = FALSE)
}

# Get config
CONFIG_PATH <- args[1]

# Max score of the outcome for this run (used for scaling)
OUTCOME_RANGE <- as.numeric(args[2])

# Flag to negate SHAP values for interpretability
NEGATE_SHAP <- as.logical(args[3])

# Y-axis labels that will need to be changed based on outcome
Y_AXIS_LABEL <- paste0(args[4])
if (NEGATE_SHAP){
    Y_AXIS_SUBLABEL <- "(Negative SHAP %)"
}else{
    Y_AXIS_SUBLABEL <- "(SHAP %)"
}


# -----------------------------------------------------------------------------
# 2. SETUP & LIBRARIES
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(arrow)
  library(tidyr)
  library(foreach)
  library(doParallel)
  library(gridExtra)
  library(splines)
  library(stringr)
  library(grid)
  library(grDevices)
  library(yaml)
})

# Path to the YAML config used for this run
cfg <- yaml::read_yaml(CONFIG_PATH)

# The directory of the current run to plot (can be overridden by 5th arg for inference)
RUN_DIR <- cfg$paths$output_dir
if (length(args) >= 5) {
  RUN_DIR <- args[5]
}

# Read available cores for parallel processing
N_CORES <- cfg$execution$n_jobs

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

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

# VISUALIZATION-ONLY spline for plotting trend lines.
# Uses R's bs() (B-spline basis via lm), NOT the same spline used for V-component
# calculation in shap_utils.py (which uses scipy LSQUnivariateSpline).
# Different fitting methods → different curves. The V values in shap_stats_global.csv
# are authoritative; this function only approximates the trend for display purposes.
calc_v_spline_pred <- function(x, y, k, degree) {
  valid <- !is.na(x) & !is.na(y) & !is.nan(x) & !is.nan(y)
  if (sum(valid) < 2) return(data.frame(x = numeric(0), y_pred = numeric(0)))

  xs <- x[valid]
  ys <- y[valid]
  ord <- order(xs)
  xs <- xs[ord]
  ys <- ys[ord]

  if (length(unique(xs)) < (degree + 1)) {
    mod <- lm(ys ~ xs)
    pred <- predict(mod)
  } else {
    probs <- seq(0, 1, length.out = k + 2)[-c(1, k + 2)]
    knots <- unique(quantile(xs, probs = probs, names = FALSE))
    tryCatch({
      mod <- lm(ys ~ bs(xs, knots = knots, degree = degree))
      pred <- predict(mod)
    }, error = function(e) {
      mod <- lm(ys ~ xs)
      pred <- predict(mod)
    })
  }
  return(data.frame(x = xs, y_pred = pred))
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
# 4. DATA LOADING & PLOTTING (per SHAP directory)
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
  p1 <- file.path(shap_dir, "bootstrap_distributions_GII.parquet")
  if (file.exists(p1)) {
    m <- max(as.matrix(read_parquet(p1)), na.rm = TRUE)
    if (m > global_max) global_max <- m
  }
  p2 <- file.path(shap_dir, "stratified_noise_distributions_GII.parquet")
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
boot_path <- file.path(SHAP_DIR, "bootstrap_distributions_GII.parquet")
noise_path <- file.path(SHAP_DIR, "stratified_noise_distributions_GII.parquet")

if (!file.exists(micro_path) || !file.exists(boot_path) || !file.exists(noise_path)) {
  cat(sprintf("[WARNING] Missing parquet files in %s, skipping.\n", shap_label))
  next
}

df_micro <- read_parquet(micro_path)
df_boot <- read_parquet(boot_path)
df_noise <- read_parquet(noise_path)

# -----------------------------------------------------------------------------
# 5. PLOTTING LOOP
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
    labs(x = "GII Magnitude", y = "Density")

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

  # TRANSFORM SHAP
  if (NEGATE_SHAP){
    df_m$shap_value <- -df_m$shap_value
  }
  df_m$shap_value <- (df_m$shap_value / OUTCOME_RANGE) * 100

  p2 <- NULL
  legend_title <- "Feature Value"

  # --- CUSTOM Y-AXIS LABEL GROB ---
  # Transparent background implicit in textGrob
  y_grob_title <- textGrob(Y_AXIS_LABEL, rot = 90,
                           gp = gpar(fontsize = 5.5, fontface = "bold", col = "black"))
  y_grob_sub   <- textGrob(Y_AXIS_SUBLABEL, rot = 90,
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
      trend_data <- calc_v_spline_pred(df_m$x_plot, df_m$shap_value, SPLINE_K_KNOTS, SPLINE_DEGREE)
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
      if (m_type == "nominal" && nlevels(fac) > 5) {
        top <- names(sort(table(fac), decreasing = TRUE))[1:5]
        df_m <- df_m %>% filter(create_ordered_factor(main_feature_raw, feature_value) %in% top)
        fac <- create_ordered_factor(df_m$main_feature_raw, df_m$feature_value)
      }

      df_m$x_plot <- as.integer(fac)
      x_labels <- levels(fac)
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
