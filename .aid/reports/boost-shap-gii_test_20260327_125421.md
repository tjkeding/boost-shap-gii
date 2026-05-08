<test_report>
 <meta project="boost-shap-gii" mode="test" timestamp="2026-03-27T12:54:21Z" />
 <pre_design_run>
 <total>385</total>
 <passed>385</passed>
 <failed>0</failed>
 <errors>0</errors>
 <coverage_pct>N/A</coverage_pct>
 <failures></failures>
 </pre_design_run>
 <design_phase>
 <tests_created>39</tests_created>
 <tests_modified>0</tests_modified>
 <files_created>
 <file path="tests/test_dtype_bugfix.py" test_count="39" coverage_target="_to_numeric_matrix dtype handling (string[pyarrow], object, category with __NA__ sentinel) and shadow permutation dtype preservation (train.py, shap_utils.py)" />
 </files_created>
 <design_rationale>
 Three code changes were made for the _to_numeric_matrix dtype bug fix:
 1. shap_utils.py line 102: `pd.api.types.is_string_dtype` replacing `dtype == object`
 2. shap_utils.py lines 1031-1034: dtype preservation in Boruta shadow permutation
 3. train.py lines 700-709: dtype preservation in training shadow permutation

 No existing tests covered string[pyarrow] dtype, the `__NA__` sentinel in category columns,
 or shadow permutation dtype preservation. Tests were designed in six classes:

 - TestToNumericMatrixStringPyArrow (7 tests): Validates the primary fix target. Confirms
 is_string_dtype correctly identifies string[pyarrow] (the old `dtype == object` check
 would return False). Covers: no NA, with NA, all NA, single level, many levels, mixed
 with continuous columns, and an explicit predicate assertion.

 - TestToNumericMatrixObjectDtype (4 tests): Regression tests for object dtype. Documents
 a discovered edge case: in pandas 3.0.1, is_string_dtype returns False for object
 columns containing None, causing a ValueError (see P1 action item below). Tests cover:
 object without None (passes), object with None (expected failure documented), numeric
 strings, and dtype distinction from pyarrow.

 - TestToNumericMatrixCategoryNASentinel (4 tests): Validates that the `__NA__` sentinel
 string used by the pipeline for missing nominal values is preserved as a distinct
 category code and does not interfere with NaN handling. Covers: own code assignment,
 coexistence with actual NaN, sentinel-only columns, and no spurious fillna.

 - TestShadowPermutationDtypePreservation (8 tests): Validates the dtype preservation
 fix in both train.py and shap_utils.py. Covers: baseline destruction of category dtype
 by rng.permutation, fix restoration, value preservation, ordered categories, __NA__
 sentinel, no-op on float/int columns, and NaN survival through permutation.

 - TestShadowPermutationMixedColumns (10 tests): Integration-level tests replicating the
 exact shadow permutation pattern from train.py (lines 700-709) and shap_utils.py
 (lines 1030-1034) on mixed DataFrames. Covers: dtype preservation, value permutation,
 shape preservation, column renaming, concat dtype preservation, determinism,
 seed-sensitivity, __NA__ sentinel columns, and the critical roundtrip test through
 _to_numeric_matrix.

 - TestToNumericMatrixEdgeCases (6 tests): Boundary conditions including empty DataFrames,
 single-row inputs, multiple string[pyarrow] columns, output type assertion, all-NaN
 categories, and a 500-level stress test.
 </design_rationale>
 </design_phase>
 <post_design_run>
 <total>424</total>
 <passed>424</passed>
 <failed>0</failed>
 <errors>0</errors>
 <coverage_pct>N/A</coverage_pct>
 <failures></failures>
 </post_design_run>
 <summary>
 <all_passing>true</all_passing>
 <recommendation>implement_fixes</recommendation>
 </summary>
 <action_items>
 <item priority="P1" target_mode="implement" description="[BUG DISCOVERED] _to_numeric_matrix fails on object dtype columns containing None in pandas 3.0.1. pd.api.types.is_string_dtype returns False for object columns with None values, causing neither the category nor string branch to trigger, resulting in a ValueError when astype(float) is called on string data. This is a residual gap in the dtype fix. Recommended fix: add a third branch `elif df_num[col].dtype == object` (the old check) as a fallback after the is_string_dtype check, or use `pd.api.types.is_object_dtype(df_num[col])` to catch mixed/None-containing object columns. The test test_object_strings_with_none_current_behavior documents this failure mode as a pytest.raises expectation; when fixed, update the test to assert correct conversion." />
 </action_items>
</test_report>
