<implement_report>
 <meta project="boost-shap-gii" mode="implement" submodule="build" timestamp="2026-03-27T12:59:14Z" />
 <spec_ref>Inline specification from user (P1 edge case fix for _to_numeric_matrix)</spec_ref>
 <changes_applied>
 <change id="change-1" status="done">
 <files_modified>
 <file path="src/boost_shap_gii/shap_utils.py" lines_changed="5" />
 </files_modified>
 <notes>Added elif branch for object-dtype columns (lines 108-114) after the existing is_string_dtype check. Uses identical category-code encoding logic. Includes inline comments explaining the pandas >= 3.0.1 edge case. No deviations from spec.</notes>
 </change>
 </changes_applied>
 <summary>
 <total_changes>1</total_changes>
 <completed>1</completed>
 </summary>
 <next_steps>Recommended: run /test to validate the fix against the existing test suite and the new edge case (object-dtype columns with None values).</next_steps>
</implement_report>
