<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-03-16T09:00:00Z" />
  <input_reports>
    <report path="boost-shap-gii_clean_20260316_080000.md" mode="clean" key_items="4" />
  </input_reports>
  <changes>
    <change id="C1" priority="P2" source_item="F1">
      <file path="shap_utils.py" action="modify" />
      <description>Replace two-line stale comment at line 750 that referenced removed NaN-to-zero behavior. The NaN replacement was removed in Session 3 (C7); the comment "Compute BEFORE replacing NaN with 0.0." no longer describes any existing behavior.</description>
      <spec>Lines 749-750: collapse the two-line comment block into a single accurate line: `# Track V spline failure rate before downstream aggregation.`</spec>
      <dependencies>none</dependencies>
      <risk>low - comment only, no logic change</risk>
      <rollback>Restore original two-line comment block</rollback>
    </change>
    <change id="C2" priority="P2" source_item="F2">
      <file path="predict.py" action="modify" />
      <description>Remove development marker `# NEW:` from line 67; convert to standard lowercase comment.</description>
      <spec>Line 67: `# NEW: Load Shadow Feature Names if available (for Noise Calibration)` → `# Load shadow feature names if available (for noise calibration)`</spec>
      <dependencies>none</dependencies>
      <risk>low - comment only</risk>
      <rollback>Restore original comment text</rollback>
    </change>
    <change id="C3" priority="P2" source_item="F3">
      <file path="train.py" action="modify" />
      <description>Remove trailing development annotation `# CHANGED: Uses correct path key` from line 348.</description>
      <spec>Line 348: `run_dir = config["paths"]["output_dir"] # CHANGED: Uses correct path key` → `run_dir = config["paths"]["output_dir"]`</spec>
      <dependencies>none</dependencies>
      <risk>low - inline comment removal only</risk>
      <rollback>Restore original inline comment</rollback>
    </change>
    <change id="C4" priority="P2" source_item="F4">
      <file path="shap_utils.py" action="modify" />
      <description>Remove trailing development annotation `# NEW: ID` from line 619 dict literal.</description>
      <spec>Line 619: `"id": id_vals, # NEW: ID` → `"id": id_vals,`</spec>
      <dependencies>none</dependencies>
      <risk>low - inline comment removal only</risk>
      <rollback>Restore original inline comment</rollback>
    </change>
  </changes>
  <execution_order>C1, C4, C2, C3</execution_order>
</implement_plan>
