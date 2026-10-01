<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-10-01T10:00:00-04:00" />
  <input_reports>
    <report path="(inline user directive)" mode="orchestrator" key_items="2" />
  </input_reports>
  <resolved_directives>
    <directive>The per-individual path runs fully sequentially (no mclapply, no forking). The data-preparation cost for each individual is a filter, a sort, and sign flips; the rendering cost dominates. Parallel data preparation provides negligible wall-clock benefit and still forks the process, so the entire per-individual loop is sequential. This is option (b) from the plan-phase decision (user approved 2026-10-01).</directive>
    <directive>The make_y_grob helper is duplicated in both render_indiv_main_effects_plots and render_indiv_interactions_plots. Each function is self-contained. This duplication is unchanged from the pre-fix code.</directive>
    <directive>The y-axis grob is built once before the render loop (it is identical for all individuals) rather than rebuilt per individual.</directive>
  </resolved_directives>
  <changes>
    <change id="C1" priority="P1" source_item="user directive: extend fork-safety to render_indiv_main_effects_plots">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Convert render_indiv_main_effects_plots (lines 1280-1463) from parallel mclapply+ggsave to a fully sequential loop. Remove plot_one_individual, mclapply dispatch, and the post-dispatch message loop. Replace with a single sequential for-loop over individuals that prepares data and renders in one pass.</description>
      <spec>
**Structure after the change:**

1. Lines 1280-1311 (unchanged): function signature, parquet read, output directory creation, sig_GII filter, `ids <- unique(df_sig$id)`, info log.

2. Build the y-axis grob once (was inside per-individual worker):
```r
make_y_grob <- function(y_lbl, y_sub) { ... }  # definition unchanged
y_axis_grob <- make_y_grob(y_label, y_sublabel)
```

3. Detect multiclass schema (unchanged): `is_multiclass_main <- "class" %in% names(df_sig)`.

4. Log: `cat("[INFO] Rendering per-individual main-effects plots sequentially.\n")`.

5. Replace the `plot_one_individual` closure, the `mclapply`/`lapply` dispatch block (lines 1331-1457), and the post-dispatch message loop (lines 1459-1461) with a single sequential `for` loop:

```r
for (indiv_id in ids) {
  tryCatch({
    df_i <- df_sig %>% filter(id == indiv_id)
    if (nrow(df_i) == 0) {
      cat(sprintf("[INFO] [SKIP] %s: no sig_GII features\n", indiv_id))
      next
    }

    class_levels <- if (is_multiclass_main) unique(as.character(df_i[["class"]])) else NA_character_

    for (cl_val in class_levels) {
      if (is_multiclass_main) {
        df_c <- df_i %>% filter(as.character(.data[["class"]]) == cl_val)
      } else {
        df_c <- df_i
      }
      if (nrow(df_c) == 0) next

      below_floor <- FALSE
      if ("oob_count" %in% names(df_c)) {
        below_floor <- any(!is.na(df_c$oob_count) & df_c$oob_count < OOB_FLOOR_MIN)
      }

      # x-axis ordering, y_plot, ci_lo_plot, ci_hi_plot, color_raw
      # ... (verbatim from current lines 1360-1392)

      safe_id <- str_replace_all(as.character(indiv_id), "[^a-zA-Z0-9_\\-]", "_")
      if (is_multiclass_main) {
        safe_cl <- str_replace_all(as.character(cl_val), "[^a-zA-Z0-9_\\-]", "_")
        fname <- sprintf("%s_main_effects_%s.png", safe_id, safe_cl)
      } else {
        fname <- sprintf("%s_main_effects.png", safe_id)
      }
      fpath <- file.path(indiv_plot_dir, fname)

      p <- ggplot(df_c, aes(x = feature, y = y_plot, color = color_raw)) +
        # ... (verbatim ggplot code from current lines 1395-1427)

      p_with_axis <- arrangeGrob(p, left = y_axis_grob)
      ggsave(fpath, p_with_axis, width = 10, height = 5, dpi = 300, bg = "transparent")
      cat(sprintf("[INFO] Saved: %s\n", fname))
    }
  }, error = function(e) {
    cat(sprintf("[INFO] [ERROR] individual %s: %s\n", indiv_id, e$message))
  })
}
```

**Behavioral contracts:**
- The tryCatch wraps each individual, so one individual's rendering error does not abort the rest.
- Log formats are byte-identical to the current output: `[INFO] Saved: <fname>`, `[INFO] [SKIP] <id>: ...`, `[INFO] [ERROR] individual <id>: ...`.
- The ggplot code is verbatim from the current worker body.
- The `n_cores` parameter remains in the function signature but is no longer used. It is retained for API stability.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - removes forking entirely; data-preparation and rendering logic are unchanged</risk>
      <rollback>git checkout -- src/boost_shap_gii/scripts/plot.R</rollback>
    </change>
    <change id="C2" priority="P1" source_item="user directive: extend fork-safety to render_indiv_interactions_plots">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Convert render_indiv_interactions_plots (lines 1466-1640) from parallel mclapply+ggsave to a fully sequential loop. Same structural change as C1 but for the interactions function.</description>
      <spec>
**Structure after the change:**

Same as C1 with these differences: the x aesthetic is `pair_label` (not `feature`), and labs uses `x = "Feature Pair"`.

**Behavioral contracts:**
- Same as C1: per-individual tryCatch error isolation, byte-identical log formats, verbatim ggplot code from the current worker, `n_cores` parameter retained for API stability.
      </spec>
      <dependencies>none (C1 and C2 modify non-overlapping line ranges in the same file; applied sequentially)</dependencies>
      <risk>low - removes forking entirely; data-preparation and rendering logic are unchanged</risk>
      <rollback>git checkout -- src/boost_shap_gii/scripts/plot.R</rollback>
    </change>
  </changes>
  <execution_order>C1, C2</execution_order>
</implement_plan>
