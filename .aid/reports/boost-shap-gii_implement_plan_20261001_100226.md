<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-10-01T14:02:26Z" />
  <input_reports>
    <report path="(inline user directive, /implement invocation arguments)" mode="user" key_items="1" />
  </input_reports>
  <locked_decisions>
    <decision>An explicit YAML null (null or ~) for plot.bootstrap_ribbons.max_subsample_n disables subsampling: bootstrap_spline_sd() receives NULL and resamples the full data. A missing key (or a missing bootstrap_ribbons block) keeps the 5000 default. Discrimination uses key presence in names(cfg$plot$bootstrap_ribbons), not value nullness (user directive).</decision>
    <decision>Python-side validation of plot.bootstrap_ribbons.max_subsample_n is included in this cycle (user scope call). Logging the effective cap in plot.R is excluded (user scope call).</decision>
    <decision>Empty-sublabel drift is fixed in code (user scope call): Python validation accepts "" for gii_y_sublabel / indiv_y_sublabel, and the GII y-axis grob collapses to the title alone when the sublabel is empty (user scope call).</decision>
    <decision>Minimum accepted integer is 10, equal to MIN_BOOT_N in plot.R, because bootstrap_spline_sd() returns NULL below that count; a cap smaller than the floor would make every subsampled ribbon degenerate.</decision>
  </locked_decisions>
  <changes>
    <change id="C1" priority="P1" source_item="user directive">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Distinguish an explicit null from a missing key when resolving the bootstrap-ribbon subsample cap. The current `%||% 5000L` maps both to 5000, so an explicit null never disables subsampling as the example config documents.</description>
      <spec>
Replace the single line at plot.R:94:

    MAX_BOOT_SUBSAMPLE_N <- cfg$plot$bootstrap_ribbons$max_subsample_n %||% 5000L

with:

    # Explicit null disables subsampling (full-data bootstrap); a missing key defaults to 5000.
    # yaml::read_yaml keeps a null-valued key in names(), which %||% cannot distinguish.
    MAX_BOOT_SUBSAMPLE_N <- if ("max_subsample_n" %in% names(cfg$plot$bootstrap_ribbons)) {
      cfg$plot$bootstrap_ribbons$max_subsample_n
    } else {
      5000L
    }

Semantics (verified with yaml::yaml.load under R 4.3.1):
  - key present, value null or ~      -> NULL  (bootstrap_spline_sd skips its subsample branch via !is.null)
  - key present, integer value n      -> n
  - key absent                        -> 5000L
  - bootstrap_ribbons block absent    -> names(NULL) is NULL; %in% returns FALSE -> 5000L
No change to bootstrap_spline_sd() or its two call sites (lines ~855, ~920); they already pass MAX_BOOT_SUBSAMPLE_N through and treat NULL as no subsampling.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - single assignment; downstream consumer already handles NULL; default path for configs without the key is unchanged.</risk>
      <rollback>Restore the original one-line %||% assignment.</rollback>
    </change>
    <change id="C2" priority="P1" source_item="user scope call on related validation gap">
      <file path="src/boost_shap_gii/utils.py" action="modify" />
      <description>Validate plot.bootstrap_ribbons.max_subsample_n in validate_plot_config() so invalid values fail fast in Python before Rscript launches, instead of producing silently all-NA ribbons (0) or R errors (negative, non-integer, string).</description>
      <spec>
In validate_plot_config(config) (utils.py:866), append after the label-string loop (end of function, currently ~line 917):

    # -- plot.bootstrap_ribbons.max_subsample_n (optional) --
    # Absent: plot.R defaults to 5000. Explicit null: subsampling disabled.
    # Integer: must be >= 10 (plot.R MIN_BOOT_N; below this every resample is skipped).
    ribbons_cfg = plot_cfg.get("bootstrap_ribbons") or {}
    if "max_subsample_n" in ribbons_cfg:
        msn = ribbons_cfg["max_subsample_n"]
        if msn is not None and (
            isinstance(msn, bool) or not isinstance(msn, int) or msn < 10
        ):
            raise ValueError(
                "plot.bootstrap_ribbons.max_subsample_n must be null (disable subsampling) "
                f"or an integer >= 10, got {msn!r}."
            )

Docstring update: add one bullet to the "Raises ValueError" list:
      - plot.bootstrap_ribbons.max_subsample_n present but not null and not an integer >= 10

Notes:
  - `or {}` handles both a missing bootstrap_ribbons block and an explicit `bootstrap_ribbons: null`.
  - bool is excluded explicitly because bool is a subclass of int in Python (matches the existing negate_shap / compute_global_on_inference idiom).
  - Floats (including 5000.0) are rejected: plot.R passes the value to sample(), which truncates silently; strict typing keeps the config contract unambiguous.
  - If bootstrap_ribbons is present but not a mapping (e.g., a scalar), `"max_subsample_n" in ribbons_cfg` on a non-dict would misbehave; guard with `isinstance(ribbons_cfg, dict)` and raise ValueError("plot.bootstrap_ribbons must be a mapping, got ...") when it is not.
      </spec>
      <dependencies>none (independent of C1; both encode the same null-vs-missing contract)</dependencies>
      <risk>low - additive check on an optional key; configs that omit the key or use a valid integer are unaffected. Existing configs with max_subsample_n: 5000 pass.</risk>
      <rollback>Remove the appended block and the docstring bullet.</rollback>
    </change>
    <change id="C3" priority="P1" source_item="user scope call on empty-sublabel validation drift">
      <file path="src/boost_shap_gii/utils.py" action="modify" />
      <description>Allow empty-string sublabels in validate_plot_config(). INPUT_SPECIFICATION.md documents passing "" to suppress gii_y_sublabel / indiv_y_sublabel, and plot.R supports it, but the Python check currently rejects "" (and whitespace-only strings), so the documented suppression path fails validation.</description>
      <spec>
In validate_plot_config(), split the label-string loop so that:
  - gii_y_label and indiv_y_label: unchanged (required, must be a non-empty string after strip()).
  - gii_y_sublabel and indiv_y_sublabel: required (key must be present and not None), must be a str; empty or whitespace-only strings are accepted.
Error message for a sublabel of the wrong type:
    f"{dotted} must be a string (use \"\" to suppress), got {val!r}."
Missing-key message unchanged in wording for both groups.
Docstring: replace the bullet
      - plot.gii_y_label / plot.gii_y_sublabel / plot.indiv_y_label / plot.indiv_y_sublabel
        missing or empty string
with
      - plot.gii_y_label / plot.indiv_y_label missing or empty string
      - plot.gii_y_sublabel / plot.indiv_y_sublabel missing or not a string ("" suppresses)
      </spec>
      <dependencies>none (C2 appends after this loop; edit C3 first or anchor C2 at the function end)</dependencies>
      <risk>low - relaxes a check only for sublabels; all currently passing configs still pass.</risk>
      <rollback>Restore the single four-key loop with the non-empty check.</rollback>
    </change>
    <change id="C4" priority="P1" source_item="user scope call on GII y-axis grob asymmetry">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>When gii_y_sublabel is empty or whitespace-only, render the GII y-axis label as the title grob alone instead of a two-column arrangeGrob with a blank 2.0 mm column, matching the per-individual make_y_grob() behavior.</description>
      <spec>
In the Phase 2 sequential rendering loop, replace (plot.R ~lines 996-1001):

    y_grob_title <- textGrob(GII_Y_LABEL, rot = 90,
                             gp = gpar(fontsize = 5.5, fontface = "bold", col = "black"))
    y_grob_sub   <- textGrob(GII_Y_SUBLABEL, rot = 90,
                             gp = gpar(fontsize = 4.5, fontface = "plain", col = "black"))
    y_axis_grob <- arrangeGrob(y_grob_title, y_grob_sub, ncol = 2,
                               widths = unit(c(2.5, 2.0), "mm"))

with:

    y_grob_title <- textGrob(GII_Y_LABEL, rot = 90,
                             gp = gpar(fontsize = 5.5, fontface = "bold", col = "black"))
    if (nchar(trimws(GII_Y_SUBLABEL)) > 0) {
      y_grob_sub  <- textGrob(GII_Y_SUBLABEL, rot = 90,
                              gp = gpar(fontsize = 4.5, fontface = "plain", col = "black"))
      y_axis_grob <- arrangeGrob(y_grob_title, y_grob_sub, ncol = 2,
                                 widths = unit(c(2.5, 2.0), "mm"))
    } else {
      y_axis_grob <- y_grob_title
    }

Fonts, sizes, and widths for the non-empty case are unchanged. Downstream consumers of y_axis_grob must accept a bare textGrob; verify during build that every use (e.g., arrangeGrob/grid.arrange `left =` argument) accepts a grob of either class, as the per-individual path already does.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - layout-only change gated on an empty sublabel; non-empty sublabel output is byte-identical in construction.</risk>
      <rollback>Restore the unconditional two-column arrangeGrob.</rollback>
    </change>
  </changes>
  <execution_order>C1, C4, C3, C2</execution_order>
  <test_coverage_note>No existing tests reference max_subsample_n or MAX_BOOT_SUBSAMPLE_N. /test should add: (a) R-level resolution for null, ~, absent key, absent block, and integer; (b) validate_plot_config acceptance of absent/null/10/5000 and rejection of 9, 0, -1, True, 5000.0, "5000", and a non-mapping bootstrap_ribbons; (c) validate_plot_config acceptance of "" and "  " for both sublabels, rejection of None/non-str sublabels and of empty labels; (d) plot.R GII grob is the bare title grob when gii_y_sublabel is empty and a two-column arrangeGrob otherwise.</test_coverage_note>
</implement_plan>
