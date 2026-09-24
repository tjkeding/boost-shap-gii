<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-09-23T23:00:00-04:00" />
  <assumptions>
    <assumption>A 45-degree angle with hjust=1 is the standard ggplot2 convention for long category labels and provides a good balance between readability and space efficiency. The user will verify via /run-local.</assumption>
    <assumption>The conditional gate (is_focal_discrete / is_discrete) prevents angling numeric x-axis labels on continuous-feature V-panels, which are short and never overlap.</assumption>
    <assumption>No bottom-margin increase is applied initially. If ggsave clips the rotated labels at the current 1mm bottom margin, this can be adjusted after visual re-verification.</assumption>
  </assumptions>
  <input_reports>
    <report path="(direct user instruction, this session)" mode="user-feedback" key_items="1" />
  </input_reports>
  <changes>
    <change id="C1" priority="P1" source_item="user visual feedback: V-panel x-axis category labels overlap when feature names are long (e.g., 'domestic violence', 'unintentional injury', 'traumatic death')">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Add conditional 45-degree x-axis label rotation for discrete-focal V-panels. The V-panel renders categorical feature names (after underscore-to-space conversion via gsub) as horizontal x-axis labels. When names are long (5+ characters per category across 3-5 categories), horizontal labels overlap at the 3.25-inch panel width. A 45-degree rotation with right-aligned hjust eliminates overlap while maintaining readability. The rotation applies only to discrete x-axes (nominal, ordinal, binary, or low-cardinality continuous features), not to continuous numeric x-axes where labels are short numbers.</description>
      <spec>
Two edit sites within plot.R, both adding a conditional theme override after the base theme block:

(A) Interaction V-panel (after line 940, the `labs(y = NULL, x = focal_label)` call):
Add immediately after `labs(y = NULL, x = focal_label)`:
    if (is_focal_discrete) {
      p2 <- p2 + theme(axis.text.x = element_text(angle = 45, hjust = 1))
    }

The variable `is_focal_discrete` is already in scope (defined at line 766 for each orientation iteration) and correctly distinguishes discrete from continuous focal features. This conditional theme override appends to the base theme at lines 920-939 without modifying it; ggplot2's theme-layering semantics ensure only axis.text.x is affected.

(B) Singleton V-panel (after line 1080, the `labs(y = NULL, x = feat_name)` call in the common theme section):
Add immediately after `labs(y = NULL, x = feat_name)`:
    if (is_discrete) {
      p2 <- p2 + theme(axis.text.x = element_text(angle = 45, hjust = 1))
    }

The variable `is_discrete` is already in scope (defined at line 966) and correctly distinguishes discrete from continuous singleton features. Same layering semantics as site A.

Post-edit verification:
  - angle = 45 occurrences in V-panel blocks: 2 (one in interaction, one in singleton)
  - hjust = 1 in axis.text.x: 2 (same two sites)
  - is_focal_discrete conditional gate: 1 (interaction site)
  - is_discrete conditional gate: 1 (singleton site)
  - Interaction theme block at lines 920-939: UNCHANGED
  - Singleton common theme block at lines 1057-1079: UNCHANGED
  - M-panel theme: UNCHANGED
  - Performance panel: UNCHANGED
      </spec>
      <dependencies>none</dependencies>
      <risk>low - additive theme override; base theme blocks are untouched; conditional gate prevents unnecessary rotation on continuous x-axes; ggplot2 theme layering is well-defined</risk>
      <rollback>Remove both conditional theme(axis.text.x = ...) blocks</rollback>
    </change>
  </changes>
  <execution_order>C1</execution_order>
</implement_plan>
