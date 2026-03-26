<?xml version="1.0" encoding="UTF-8"?>
<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-03-16T04:00:00Z" />
  <input_reports>
    <report path="boost-shap-gii_brainstorm_20260316_040000.md" mode="brainstorm" key_items="2" />
  </input_reports>
  <changes>

    <change id="C1" priority="P0" source_item="T1/A1">
      <file path="check_env.py" action="modify" />
      <description>
        Line 59 embeds a backslash escape inside an f-string expression, which is a SyntaxError
        on Python 3.11+ (PEP 701). The entire check_env.py script is non-functional on modern
        Python versions. Fix by computing the quoted list outside the f-string, then interpolating
        the pre-computed variable.
      </description>
      <spec>
        Before (line 59):
          print(f"[HINT]  Install in R: install.packages(c({', '.join([f'"{m}"' for m in missing])}))")

        After:
          quoted = ', '.join(f'"{m}"' for m in missing)
          print(f"[HINT]  Install in R: install.packages(c({quoted}))")

        The new lines replace line 59 within the `if missing:` block in check_r().
        No change to logic — only syntax refactor for 3.11+ compatibility.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Pure syntactic equivalence; output is identical.</risk>
      <rollback>Revert to single-line f-string; acceptable only on Python &lt;3.11.</rollback>
    </change>

    <change id="C2" priority="P1" source_item="T2/A1">
      <file path="tests/test_multiclass_pipeline.py" action="modify" />
      <description>
        The shape assertion on line 27 expects (N, P+1, P+1, C) = (10, 2, 2, 3), but CatBoost 1.2.10
        returns (N, C, P+1, P+1) = (10, 3, 2, 2) for ShapInteractionValues on a multiclass model.
        The pipeline code in shap_utils.py already indexes raw[:, class_idx, :-1, :-1], which is
        correct for this layout. Only the test assertion and its comment are wrong.
      </description>
      <spec>
        Line 25 comment: change from
          "# type='ShapInteractionValues' for multiclass returns (N, P+1, P+1, C)"
        to
          "# type='ShapInteractionValues' for multiclass returns (N, C, P+1, P+1)"

        Line 27: change from
          assert phi.shape == (10, 2, 2, 3) # (N, features+bias, features+bias, classes)
        to
          assert phi.shape == (10, 3, 2, 2) # (N, classes, features+bias, features+bias)
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Test-only change; no pipeline logic altered.</risk>
      <rollback>Revert assertion tuple; test will fail on CatBoost 1.2.10.</rollback>
    </change>

  </changes>
  <execution_order>C1, C2</execution_order>
</implement_plan>
