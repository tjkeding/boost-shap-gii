<test_report>
 <meta project="boost-shap-gii" mode="test" timestamp="2026-03-27T13:21:37Z" />
 <pre_design_run>
 <total>424</total>
 <passed>423</passed>
 <failed>1</failed>
 <errors>0</errors>
 <coverage_pct>N/A</coverage_pct>
 <failures>
 <failure test="test_object_strings_with_none_current_behavior" file="tests/test_dtype_bugfix.py" line="136">
 <error_type>Failed: DID NOT RAISE</error_type>
 <message>pytest.raises(ValueError, match="could not convert string to float") did not raise because the new elif df_num[col].dtype == object: fallback branch (shap_utils.py lines 108-114) now handles object-dtype columns with None gracefully.</message>
 <traceback>tests/test_dtype_bugfix.py:136: Failed: DID NOT RAISE ValueError</traceback>
 </failure>
 </failures>
 </pre_design_run>
 <design_phase>
 <tests_created>18</tests_created>
 <tests_modified>1</tests_modified>
 <files_created>
 <file path="tests/test_dtype_bugfix.py" test_count="57" coverage_target="Object-dtype fallback branch in _to_numeric_matrix (shap_utils.py lines 108-114)" />
 </files_created>
 <design_rationale>
 1. UPDATED test_object_strings_with_none_current_behavior (renamed to test_object_strings_with_none_handled_by_fallback):
 The original test expected a ValueError because the object-dtype+None case was a known limitation.
 The new fallback branch resolves this, so the test now asserts correct encoding (x=0, y=1, None->sentinel=2).

 2. NEW TestToNumericMatrixObjectDtypeFallback class (18 tests) covering:
 - Routing verification: confirms is_string_dtype returns False for object+None (prerequisite for the fallback)
 - Correctness of sentinel encoding for various None patterns (single None, multiple Nones, all None)
 - Single-value-with-None edge case
 - __NA__ sentinel string coexisting with None
 - Multi-column DataFrames mixing object+None with float, category, and string[pyarrow] columns
 - Multiple object+None columns (independent encoding verification)
 - Heterogeneous Python types in object columns (int+str+None, bool+None)
 - Edge cases: single-row, empty-string-with-None, many-levels stress test (200 levels + None)
 - Integration: shadow permutation -> object+None -> _to_numeric_matrix roundtrip
 </design_rationale>
 </design_phase>
 <post_design_run>
 <total>442</total>
 <passed>442</passed>
 <failed>0</failed>
 <errors>0</errors>
 <coverage_pct>N/A</coverage_pct>
 <failures />
 </post_design_run>
 <summary>
 <all_passing>true</all_passing>
 <recommendation>proceed_to_document</recommendation>
 </summary>
 <action_items />
</test_report>
