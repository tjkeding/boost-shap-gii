<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-09-23T10:15:00-04:00" />
  <input_reports>
    <report path="inline (conversation-locked critique cycle 4, Topics T1-T8)" mode="brainstorm" key_items="8" />
  </input_reports>
  <changes>
    <change id="C1" priority="P1" source_item="T1: Performance panel vertical stacking">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Change performance panel layout from 2-column quadrant grid to single-column vertical stack. Each individual panel retains its current dimensions; the overall PNG grows taller and narrower.</description>
      <spec>
Three sites:

1. has_boot_perf branch (line 476): change `facet_wrap(~ metric, scales = "free", ncol = 2)` to `facet_wrap(~ metric, scales = "free", ncol = 1)`.

2. CI-band fallback branch (line 514): change `facet_wrap(~ metric, scales = "free", ncol = 2)` to `facet_wrap(~ metric, scales = "free", ncol = 1)`.

3. Dynamic sizing (lines 530-533): replace the block:
```r
n_metrics <- nrow(df_obs)
n_rows <- ceiling(n_metrics / 2)
fig_w <- 5.1
fig_h <- max(1.275, n_rows * 1.275)
```
with:
```r
n_metrics <- nrow(df_obs)
fig_w <- 2.75
fig_h <- max(1.275, n_metrics * 1.275)
```
fig_w = 2.75 is calibrated to maintain the per-panel plot-area width from the prior ncol=2 layout at fig_w=5.1 (each panel was approximately 2.3 inches of plot area; 2.75 provides 2.3 inches plus y-axis labels and margins for a single-column layout). This value may need visual tuning during /run-local.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - mechanical facet and sizing substitution; visual tuning via /run-local</risk>
      <rollback>Revert ncol to 2, restore original fig_w/fig_h computation</rollback>
    </change>

    <change id="C2" priority="P1" source_item="T2 + T6: Stat label placement above ticks and margin reduction">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Reposition stat labels from below the x-axis tick marks (vjust=3.5 at y=-Inf) to between the distribution base and the tick marks (vjust=1.0 at y=-Inf). Reduce bottom margins that were inflated to accommodate the now-superseded below-tick positioning. Applies to both the performance panel (both branches) and the M-panel.</description>
      <spec>
Five vjust sites, all change from 3.5 to 1.0:

1. Performance panel has_boot_perf branch, trained annotation (line 469):
   `vjust = 3.5, hjust = 0.5, size = 1.5, color = "black"` → `vjust = 1.0, hjust = 0.5, size = 1.5, color = "black"`

2. Performance panel has_boot_perf branch, null annotation (line 473):
   `vjust = 3.5, hjust = 0.5, size = 1.5, color = "black"` → `vjust = 1.0, hjust = 0.5, size = 1.5, color = "black"`

3. Performance panel fallback branch, null annotation (line 511):
   `vjust = 3.5, hjust = 0.5, size = 1.5, color = "black"` → `vjust = 1.0, hjust = 0.5, size = 1.5, color = "black"`

4. M-panel noise annotation (line 636):
   `vjust = 3.5, hjust = 0.5, size = 1.5, color = "black"` → `vjust = 1.0, hjust = 0.5, size = 1.5, color = "black"`

5. M-panel signal annotation (line 639):
   `vjust = 3.5, hjust = 0.5, size = 1.5, color = "black"` → `vjust = 1.0, hjust = 0.5, size = 1.5, color = "black"`

Three margin sites, reduced since labels no longer render below ticks:

6. Performance panel has_boot_perf branch (line 487):
   `plot.margin = margin(5.5, 5.5, 20, 5.5)` → `plot.margin = margin(5.5, 5.5, 8, 5.5)`

7. Performance panel fallback branch (line 525):
   `plot.margin = margin(5.5, 5.5, 20, 5.5)` → `plot.margin = margin(5.5, 5.5, 8, 5.5)`

8. M-panel (line 661):
   `plot.margin = unit(c(1, 0.5, 12, 1), "mm")` → `plot.margin = unit(c(1, 0.5, 4, 1), "mm")`

The vjust=1.0 at y=-Inf places the top edge of the text at the panel boundary: the text body extends into the margin, sitting between the distribution floor and the tick marks. The mean vlines do not extend into the margin area, so no occlusion occurs. Bottom margins are reduced to near their original values (pre-cycle-2) since the extra margin accommodation for below-tick text is no longer needed.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - value substitutions; visual tuning via /run-local if vjust=1.0 needs adjustment</risk>
      <rollback>Revert vjust to 3.5, margins to 20/20/12mm</rollback>
    </change>

    <change id="C3" priority="P1" source_item="T3: Remove Distribution legend title">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Remove the "Distribution" legend title from the performance panel's shared legend. The entries ("Permutation Null" / "Trained") are self-explanatory.</description>
      <spec>
Two sites in the has_boot_perf branch:

1. Line 458: `name = "Distribution"` → `name = NULL`
   In: `scale_fill_manual(values = c("Permutation Null" = "#CCCCCC", "Trained" = "#377eb8"), name = "Distribution")`

2. Line 459: `name = "Distribution"` → `name = NULL`
   In: `scale_color_manual(values = c("Permutation Null" = "#666666", "Trained" = "#08306b"), name = "Distribution")`

The CI-band fallback branch has no shared legend, so no change is needed there.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - string substitution</risk>
      <rollback>Revert name to "Distribution"</rollback>
    </change>

    <change id="C4" priority="P1" source_item="T4: Null mean line solid">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Change the permutation null distribution's mean vertical line from dashed to solid, matching the trained distribution's mean line style. Differentiated by color only (gray for null, blue for trained).</description>
      <spec>
Two sites:

1. has_boot_perf branch (line 465):
   `color = "#666666", linewidth = 0.5, linetype = "dashed"` → `color = "#666666", linewidth = 0.5`
   (removing linetype defaults to "solid")

2. CI-band fallback branch (line 507):
   `color = "#666666", linewidth = 0.5, linetype = "dashed"` → `color = "#666666", linewidth = 0.5`
      </spec>
      <dependencies>none</dependencies>
      <risk>low - parameter removal</risk>
      <rollback>Re-add linetype = "dashed"</rollback>
    </change>

    <change id="C5" priority="P1" source_item="T5: M-panel ggsave height increase">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Increase the ggsave height for GII plots (M+V combined) to compensate for the M-panel's internal whitespace (bottom legend, x-axis title, margins) that compresses the distribution area relative to the V-panel. Both the singleton and interaction save paths use height = 1.275; increase to 1.5.</description>
      <spec>
Two sites:

1. Singleton ggsave (line 1100):
   `ggsave(fpath, g, width = save_width, height = 1.275, dpi = 300, bg = "transparent")` →
   `ggsave(fpath, g, width = save_width, height = 1.5, dpi = 300, bg = "transparent")`

2. Interaction ggsave (line 957):
   `ggsave(fpath_ori, g, width = save_width_ori, height = 1.275, dpi = 300, bg = "transparent")` →
   `ggsave(fpath_ori, g, width = save_width_ori, height = 1.5, dpi = 300, bg = "transparent")`

The increase from 1.275 to 1.5 (0.225 inches, approximately 5.7mm) compensates for the M-panel's bottom horizontal legend and reduced bottom margin (4mm after C2). This value may need visual tuning via /run-local; the V-panel will also grow slightly taller, which should not degrade its appearance.
      </spec>
      <dependencies>C2 (margin reduction affects internal whitespace budget)</dependencies>
      <risk>low - single parameter change; visual tuning via /run-local</risk>
      <rollback>Revert height to 1.275</rollback>
    </change>

    <change id="C6" priority="P1" source_item="T7: NA sentinel and underscore-to-space">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Fix excessive whitespace around NA x-axis labels. Root cause: the "__NA__" sentinel (lines 680, 757) is processed by gsub("_", "\n", ...) (lines 840, 1021) which converts it to "\n\nNA\n\n" (four newlines around "NA"). Two-part fix: (a) change the NA sentinel from "__NA__" to "NA", and (b) change gsub("_", "\n", ...) to gsub("_", " ", ...) at all x-axis label construction sites, eliminating multi-line labels entirely and preventing inconsistent vertical allocation across labels.</description>
      <spec>
Four sites:

1. Singleton NA sentinel (line 680):
   `"__NA__", as.character(main_feature_raw)` → `"NA", as.character(main_feature_raw)`

2. Interaction NA sentinel (line 757):
   `"__NA__", focal_raw)` → `"NA", focal_raw)`

3. Interaction x-axis labels (line 840):
   `x_labels <- gsub("_", "\n", levels(fac))` → `x_labels <- gsub("_", " ", levels(fac))`

4. Singleton x-axis labels (line 1021):
   `x_labels <- gsub("_", "\n", levels(fac))` → `x_labels <- gsub("_", " ", levels(fac))`
      </spec>
      <dependencies>none</dependencies>
      <risk>low - string substitutions; the "NA" sentinel cannot collide with legitimate category labels because CatBoost encodes missing values as a distinct numeric value (separate from any category that happens to be named "NA")</risk>
      <rollback>Revert sentinels to "__NA__" and gsub back to "\n"</rollback>
    </change>

    <change id="C7" priority="P1" source_item="T8: Legend ordering low-to-high with NA at bottom">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Ensure all feature-value legends display low values at top and high values at bottom, with NA always at the absolute bottom. Three sub-changes: (a) remove reverse=TRUE from the discrete singleton legend, (b) add reverse=TRUE to the continuous singleton colorbar, (c) modify create_ordered_factor to guarantee the NA sentinel is always the last factor level regardless of CatBoost's numeric encoding.</description>
      <spec>
Three sites:

1. Discrete singleton legend (line 1052):
   `guide = guide_legend(reverse = TRUE, override.aes = list(alpha = 1))` →
   `guide = guide_legend(override.aes = list(alpha = 1))`

2. Continuous singleton gradient (line 991):
   `scale_color_gradient(low = "#b2182b", high = "#2166ac", name = legend_title)` →
   `scale_color_gradient(low = "#b2182b", high = "#2166ac", name = legend_title, guide = guide_colorbar(reverse = TRUE))`

3. create_ordered_factor function (lines 361-366): modify to move the NA sentinel to the last position in the factor levels, regardless of where CatBoost's encoding placed it numerically. Replace:
```r
create_ordered_factor <- function(raw_vec, enc_vec) {
  df_map <- data.frame(raw = as.character(raw_vec), enc = as.numeric(enc_vec)) %>%
    distinct() %>%
    arrange(enc)
  return(factor(as.character(raw_vec), levels = df_map$raw))
}
```
with:
```r
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
```

Rationale for each sub-change:
(a) create_ordered_factor sorts levels ascending by encoded value (low first, high last). Without reverse=TRUE, the default guide_legend follows this order: low at top, high at bottom. This matches the user's request.
(b) The default guide_colorbar renders low at bottom, high at top (thermometer orientation). Adding reverse=TRUE flips it to low at top, high at bottom. Colors are unaffected (low stays red, high stays blue).
(c) CatBoost assigns missing values a specific numeric encoding whose position in the sorted order is data-dependent. The explicit relocation of the NA sentinel to the end of the levels vector guarantees it always appears at the bottom of the legend, regardless of encoding.

The interaction legend (scale_color_manual at line 858) already displays low-to-high (no reverse=TRUE is present) and does not include NA strata (stratify_moderator filters NA at line 248), so no change is needed for interactions.
      </spec>
      <dependencies>C6 (NA sentinel must be "NA" before this change references it as the default na_sentinel)</dependencies>
      <risk>low - the discrete and colorbar changes are parameter additions/removals; the create_ordered_factor change adds a conditional reordering step that is backwards-compatible (no-op when no NA sentinel is present)</risk>
      <rollback>Re-add reverse=TRUE to discrete legend, remove reverse=TRUE from colorbar, revert create_ordered_factor to original</rollback>
    </change>
  </changes>
  <execution_order>C1, C3, C4, C2, C6, C7, C5</execution_order>
</implement_plan>
