<test_report>
  <meta project="boost-shap-gii" mode="test" submodule="run_suite" timestamp="2026-03-26T15:55:22Z" />
  <pre_design_run>
    <total>385</total>
    <passed>385</passed>
    <failed>0</failed>
    <errors>0</errors>
    <warnings>3</warnings>
    <coverage_pct>N/A</coverage_pct>
    <warning_details>
      <warning test="test_ordinal_empty_levels_list" file="tests/test_adversarial.py" line="484">
        <type>Pandas4Warning</type>
        <message>Constructing a Categorical with a dtype and values containing non-null entries not in that dtype's categories is deprecated and will raise in a future version.</message>
        <cause>Test uses `data.astype(cat_type).cat.codes` directly on out-of-category values without the `.where()` pre-filter that production code (train.py:488, predict.py:173, infer.py:191) employs. The existing `warnings.simplefilter("ignore", FutureWarning)` filter does not catch Pandas4Warning because its MRO is DeprecationWarning, not FutureWarning.</cause>
      </warning>
      <warning test="test_ordinal_unknown_values_become_nan" file="tests/test_infer.py" line="207">
        <type>Pandas4Warning</type>
        <message>Constructing a Categorical with a dtype and values containing non-null entries not in that dtype's categories is deprecated and will raise in a future version.</message>
        <cause>Same root cause as above: test deliberately constructs out-of-category scenario but used FutureWarning filter instead of DeprecationWarning.</cause>
      </warning>
      <warning test="test_single_sample_bootstrap" file="tests/test_adversarial.py" line="N/A">
        <type>UndefinedMetricWarning (sklearn)</type>
        <message>R^2 score is not well-defined with less than two samples.</message>
        <cause>Expected: adversarial test exercises single-sample R^2, which is mathematically undefined. Warning originates from sklearn internals, not project code. No action required.</cause>
      </warning>
    </warning_details>
    <failures />
  </pre_design_run>
  <design_phase>
    <tests_created>0</tests_created>
    <tests_modified>2</tests_modified>
    <files_modified>
      <file path="tests/test_adversarial.py" change="Changed warning filter from FutureWarning to DeprecationWarning in test_ordinal_empty_levels_list (line 483). Pandas4Warning inherits from DeprecationWarning, not FutureWarning." />
      <file path="tests/test_infer.py" change="Changed warning filter from FutureWarning to DeprecationWarning in test_ordinal_unknown_values_become_nan (line 206). Same root cause as above." />
    </files_modified>
    <design_rationale>
      The Pandas4Warning class (introduced in pandas 2.2+, present in 3.0.1) has the MRO:
        Pandas4Warning -> PandasDeprecationWarning -> PandasChangeWarning -> DeprecationWarning -> Warning
      It does NOT inherit from FutureWarning. The existing `warnings.simplefilter("ignore", FutureWarning)` filters were therefore ineffective. Changing to `DeprecationWarning` correctly suppresses both the current Pandas4Warning and any future pandas deprecation warnings in these known-intentional test scenarios.

      Production code (train.py, predict.py, infer.py) is NOT affected: all three files already use a `.where(X[c].isin(levels) | X[c].isna(), other=pd.NA)` pre-filter that replaces out-of-category values with pd.NA before the `.astype(cat_type)` call, preventing the warning from ever firing in production.
    </design_rationale>
  </design_phase>
  <post_design_run>
    <total>385</total>
    <passed>385</passed>
    <failed>0</failed>
    <errors>0</errors>
    <warnings>1</warnings>
    <coverage_pct>N/A</coverage_pct>
    <remaining_warnings>
      <warning test="test_single_sample_bootstrap" file="tests/test_adversarial.py">
        <type>UndefinedMetricWarning (sklearn)</type>
        <message>R^2 score is not well-defined with less than two samples.</message>
        <status>Expected/acceptable: adversarial test of mathematically undefined metric. Originates from sklearn, not project code.</status>
      </warning>
    </remaining_warnings>
    <failures />
  </post_design_run>
  <summary>
    <all_passing>true</all_passing>
    <pandas4warnings_eliminated>true</pandas4warnings_eliminated>
    <futurewarnings_eliminated>true</futurewarnings_eliminated>
    <tests_specifically_verified>
      - test_adversarial.py::TestOrdinalEncodingEdgeCases::test_ordinal_empty_levels_list: PASSED, no Pandas4Warning
      - test_infer.py::TestTypeCasting::test_ordinal_unknown_values_become_nan: PASSED, no Pandas4Warning
      - All ordinal feature casting tests in train.py, predict.py, infer.py test files: PASSED, no warnings
    </tests_specifically_verified>
    <recommendation>proceed_to_document</recommendation>
  </summary>
  <action_items />
</test_report>
