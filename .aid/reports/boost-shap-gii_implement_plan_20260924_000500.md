<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-09-24T00:05:00-04:00" />
  <assumptions>
    <assumption>The "__NA__" sentinel is matched by exact string equality (post-gsub-would-be-applied form), not by regex pattern, since it is a fixed literal produced by the pipeline's nominal-missing-value fill logic (utils.py), not a user-supplied value that could vary in form.</assumption>
  </assumptions>
  <input_reports>
    <report path="(direct user instruction, this session)" mode="user-feedback" key_items="1" />
  </input_reports>
  <changes>
    <change id="C1" priority="P1" source_item="user-directed fix: the __NA__ sentinel must not receive the general underscore-to-newline word-wrap transform, since it produces two leading and two trailing blank lines around the letters NA; all other discrete category labels must continue to receive full underscore-to-newline wrapping">
      <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
      <description>Special-case the literal "__NA__" sentinel string to strip its leading/trailing underscores (producing the clean label "NA") instead of applying the general underscore-to-newline transform, which would otherwise convert the sentinel's four underscores into four newlines surrounding "NA". All other category labels are unaffected and continue to receive the underscore-to-newline transform in full.</description>
      <spec>
Two edit sites within plot.R, both replacing an unconditional gsub call with a vectorized conditional:

(A) Interaction V-panel (line 842):
Replace:
    x_labels <- gsub("_", "\n", levels(fac))
with:
    x_labels <- ifelse(levels(fac) == "__NA__", "NA", gsub("_", "\n", levels(fac)))

(B) Singleton V-panel (line 1024):
Replace:
    x_labels <- gsub("_", "\n", levels(fac))
with:
    x_labels <- ifelse(levels(fac) == "__NA__", "NA", gsub("_", "\n", levels(fac)))

Both sites use `ifelse` for exact, elementwise, vectorized string equality against the literal "__NA__" sentinel. Levels equal to "__NA__" resolve to the clean 2-character string "NA" (no underscores, no newlines). All other levels are passed through the existing gsub("_", "\n", ...) transform unchanged, preserving the underscore-to-newline word-wrap behavior for every other multi-word category label.

Post-edit verification:
  - ifelse(levels(fac) == "__NA__", "NA", gsub("_", "\n", levels(fac))) count: 2 (one per site)
  - Unconditional gsub("_", "\n", levels(fac)) (without the ifelse guard) count: 0 (both sites now guarded)
  - No other x_labels construction sites are affected (only these two discrete-focal sites construct x_labels via levels(fac); the continuous-focal branches use scale_x_continuous with numeric labels, not levels(fac)).
      </spec>
      <dependencies>none</dependencies>
      <risk>low - single vectorized string-comparison guard added ahead of the existing transform; does not change behavior for any label other than the exact literal "__NA__"; both sites are edited identically and symmetrically</risk>
      <rollback>Revert both sites to the unconditional x_labels <- gsub("_", "\n", levels(fac))</rollback>
    </change>
  </changes>
  <execution_order>C1</execution_order>
</implement_plan>
