<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-09-24T00:15:00-04:00" />
  <assumptions>
    <assumption>None beyond what is directly verified in the codebase: both NA-recoding sites (singleton line 679-684, interaction line 756-760) already normalize is.na()/"nan"/"NaN" raw values to the bare 2-character string "NA" before create_ordered_factor is called, so the literal "__NA__" sentinel structurally cannot reach the levels(fac) label-construction step. This was confirmed by direct inspection of the current plot.R source, not assumed.</assumption>
  </assumptions>
  <input_reports>
    <report path="(direct code verification this session, correcting the immediately prior implement cycle boost-shap-gii_implement_plan_20260924_000500.md)" mode="self-correction" key_items="1" />
  </input_reports>
  <changes>
    <change id="C1" priority="P1" source_item="self-correction: the __NA__ sentinel guard added in the immediately prior implement cycle is unnecessary dead code and violates an existing, deliberate test invariant (tests/test_build_20260923.py::TestNASentinelSimplified) that the deprecated __NA__ sentinel must not reappear anywhere in plot.R">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Revert the ifelse("__NA__", "NA", ...) guard added moments ago at both x_labels construction sites back to the plain underscore-to-newline transform. The guard was based on a mistaken belief that the "__NA__" sentinel could reach levels(fac); direct inspection of the upstream recoding sites (lines 679-684 and 756-760) confirms actual missing values are already normalized to bare "NA" before reaching this point, via a fix applied earlier in this same session (T7 critique cycle). The literal "__NA__" string never appears in main_feature_raw/focal_raw, so the newline transform never has an opportunity to act on it. Reintroducing the literal "__NA__" string into plot.R (as I did in the immediately prior cycle) is itself the regression: an existing test explicitly asserts this deprecated sentinel must not appear anywhere in the source.</description>
      <spec>
Two edit sites within plot.R, reverting the just-applied guard back to the plain form:

(A) Interaction V-panel (line 842):
Replace:
    x_labels <- ifelse(levels(fac) == "__NA__", "NA", gsub("_", "\n", levels(fac)))
with:
    x_labels <- gsub("_", "\n", levels(fac))

(B) Singleton V-panel (line 1024):
Replace:
    x_labels <- ifelse(levels(fac) == "__NA__", "NA", gsub("_", "\n", levels(fac)))
with:
    x_labels <- gsub("_", "\n", levels(fac))

Post-edit verification:
  - Literal "__NA__" string count in plot.R: 0
  - gsub("_", "\n", levels(fac)) (plain form) count: 2
  - ifelse(levels(fac) == "__NA__", ...) count: 0
      </spec>
      <dependencies>none</dependencies>
      <risk>low - reverts a change made in the immediately prior cycle back to its pre-change state; no other code depends on the removed guard</risk>
      <rollback>Re-add the ifelse("__NA__", "NA", ...) guard (not expected to be needed)</rollback>
    </change>
  </changes>
  <execution_order>C1</execution_order>
</implement_plan>
