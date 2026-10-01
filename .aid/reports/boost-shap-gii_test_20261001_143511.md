<test_report>
  <meta project="boost-shap-gii" mode="test" timestamp="2026-10-01T14:35:11Z" />

  <input_reports>
    <report path="boost-shap-gii_implement_build_20261001_101141.md" mode="implement" key_items="4" />
    <report path="boost-shap-gii_implement_plan_20261001_100226.md" mode="implement" key_items="4" />
  </input_reports>

  <pre_design_run>
    <total>1079</total>
    <passed>1075</passed>
    <failed>4</failed>
    <errors>0</errors>
    <failing_tests>
      <test name="TestValidatePlotConfigErrors::test_label_empty_string[gii_y_sublabel]" file="tests/test_indiv_reports_validators.py" />
      <test name="TestValidatePlotConfigErrors::test_label_empty_string[indiv_y_sublabel]" file="tests/test_indiv_reports_validators.py" />
      <test name="TestValidatePlotConfigErrors::test_label_whitespace_only[gii_y_sublabel]" file="tests/test_indiv_reports_validators.py" />
      <test name="TestValidatePlotConfigErrors::test_label_whitespace_only[indiv_y_sublabel]" file="tests/test_indiv_reports_validators.py" />
    </failing_tests>
  </pre_design_run>

  <failing_test_dispositions>
    <entry test="test_label_empty_string[gii_y_sublabel]" disposition="obsolete-test">
      <intended_contract>Prior to change C3 (boost-shap-gii_implement_plan_20261001_100226.md), all four plot.*_label/*_sublabel keys were required non-empty, non-whitespace strings; an empty or whitespace-only sublabel raised ValueError identically to a label.</intended_contract>
      <current_test_claim>validate_plot_config(cfg) raises ValueError matching "plot.gii_y_sublabel must be a non-empty string" when gii_y_sublabel is "".</current_test_claim>
      <evidence>boost-shap-gii_implement_plan_20261001_100226.md change C3 (locked decision) relaxed validate_plot_config in utils.py so that gii_y_sublabel/indiv_y_sublabel accept any string, including "" and whitespace-only, to allow suppressing the y-axis subtitle; boost-shap-gii_implement_build_20261001_101141.md confirms C3 applied exactly as specified. utils.py:866-953 (read directly) shows the sublabel branch no longer calls the non-empty/non-whitespace check applied to labels.</evidence>
      <action>Re-expressed, not deleted: the sublabel parametrize cases were removed from test_label_empty_string's label_key list (now ["gii_y_label", "indiv_y_label"] only, preserving the original postcondition for the still-required label keys), and a new, stronger-in-aggregate acceptance test (test_sublabel_empty_string_accepted, class TestValidatePlotConfigSublabelRelaxation) was added asserting the opposite postcondition (no exception raised) for both sublabel keys.</action>
    </entry>
    <entry test="test_label_empty_string[indiv_y_sublabel]" disposition="obsolete-test">
      <intended_contract>Same as above, for indiv_y_sublabel.</intended_contract>
      <current_test_claim>validate_plot_config(cfg) raises ValueError matching "plot.indiv_y_sublabel must be a non-empty string" when indiv_y_sublabel is "".</current_test_claim>
      <evidence>Same as above.</evidence>
      <action>Same re-expression as above (single parametrized test covers both keys).</action>
    </entry>
    <entry test="test_label_whitespace_only[gii_y_sublabel]" disposition="obsolete-test">
      <intended_contract>Same contract as test_label_empty_string, applied to a whitespace-only ("   ") value rather than "".</intended_contract>
      <current_test_claim>validate_plot_config(cfg) raises ValueError matching "plot.gii_y_sublabel must be a non-empty string" when gii_y_sublabel is "   ".</current_test_claim>
      <evidence>Same as above; utils.py's relaxed sublabel branch does not call trimws/strip-based rejection for sublabels.</evidence>
      <action>Re-expressed: test_label_whitespace_only's label_key list narrowed to ["gii_y_label", "indiv_y_label"]; new test_sublabel_whitespace_only_accepted added asserting no exception for both sublabel keys under a whitespace-only value.</action>
    </entry>
    <entry test="test_label_whitespace_only[indiv_y_sublabel]" disposition="obsolete-test">
      <intended_contract>Same as above, for indiv_y_sublabel.</intended_contract>
      <current_test_claim>validate_plot_config(cfg) raises ValueError matching "plot.indiv_y_sublabel must be a non-empty string" when indiv_y_sublabel is "   ".</current_test_claim>
      <evidence>Same as above.</evidence>
      <action>Same re-expression as above (single parametrized test covers both keys).</action>
    </entry>
  </failing_test_dispositions>

  <design_phase>
    <coverage_gaps_addressed>
      <gap id="a" description="R-level MAX_BOOT_SUBSAMPLE_N resolution for null, tilde, absent key, absent block, and integer (plot.R lines 94-100)">
        <file path="tests/test_plot_r_bootstrap_subsample_cap.py" action="create" test_count="5" />
      </gap>
      <gap id="b" description="Python-side validate_plot_config accept/reject matrix for plot.bootstrap_ribbons.max_subsample_n (utils.py)">
        <file path="tests/test_plot_config_bootstrap_ribbons.py" action="create" test_count="13" />
      </gap>
      <gap id="c" description="Python-side sublabel relaxation acceptance (empty/whitespace accepted; non-string/explicit-None still rejected)">
        <file path="tests/test_indiv_reports_validators.py" action="modify" test_count_added="8" test_count_narrowed="4" />
      </gap>
      <gap id="d" description="R-level GII y-axis grob empty-sublabel collapse to bare textGrob vs. two-column arrangeGrob (plot.R lines 1002-1011)">
        <file path="tests/test_plot_r_gii_grob_sublabel.py" action="create" test_count="3" />
      </gap>
    </coverage_gaps_addressed>
    <extraction_patterns_reused>
      <pattern name="_extract_r_function (balanced-brace)" source="tests/test_build_20261001.py" used_in="none (both new R-level target blocks are flat top-level if/else statements, not function definitions)" />
      <pattern name="_extract_between_markers (literal-marker)" source="tests/test_build_20260923b.py" used_in="tests/test_plot_r_bootstrap_subsample_cap.py, tests/test_plot_r_gii_grob_sublabel.py" />
    </extraction_patterns_reused>
    <design_decisions>
      <decision>tests/test_plot_r_bootstrap_subsample_cap.py constructs its synthetic cfg via yaml::read_yaml() on a written YAML fixture, not a bare R list() literal, because R silently drops a NULL-valued key from names() on direct list construction while yaml::read_yaml() preserves a YAML null/tilde-valued key in names() -- the exact distinction the production code (plot.R:94-100) relies on. This makes the test exercise the real config-loading path rather than a non-representative R-native construction.</decision>
      <decision>tests/test_plot_config_bootstrap_ribbons.py deliberately omits a case for bootstrap_ribbons=[] (empty list). The ribbons_cfg = plot_cfg.get("bootstrap_ribbons") or {} idiom in utils.py would silently coerce an empty list to {} rather than raising the "must be a mapping" error, which is a separate, out-of-scope product-behavior question not called for by the implement plan's test_coverage_note; non-empty list/string values were used instead for the non-mapping-rejection cases.</decision>
      <decision>tests/test_plot_r_gii_grob_sublabel.py discriminates the two code paths via inherits(y_axis_grob, "gtable") rather than a visual/rendered comparison, after empirically confirming under the project's R environment that textGrob() returns class c("text","grob","gDesc") and arrangeGrob() returns class c("gtable","gTree","grob","gDesc").</decision>
    </design_decisions>
  </design_phase>

  <post_design_run>
    <total>1105</total>
    <passed>1105</passed>
    <failed>0</failed>
    <errors>0</errors>
    <warnings>28 (datetime.utcnow() deprecation in indiv_reports.py; expected cluster-bootstrap i.i.d.-fallback RuntimeWarnings in small-n test fixtures per Ukoumunne et al. 2003 -- both pre-existing and unrelated to this cycle's changes)</warnings>
    <wall_clock_s>106.04</wall_clock_s>
    <receipt_verified>true (independent on-disk cross-check of the tee'd receipt file against the agent-returned receipt; nonce, END marker, and "1105 passed" summary line all matched)</receipt_verified>
  </post_design_run>

  <summary>
    <assertions_preserved_or_strengthened>true</assertions_preserved_or_strengthened>
    <bugs_routed_to_implement>0</bugs_routed_to_implement>
    <net_new_tests>29</net_new_tests>
    <narrative>
      All 4 pre-design failures were caused by test_indiv_reports_validators.py encoding the pre-change (pre-C3) contract for plot.gii_y_sublabel/plot.indiv_y_sublabel; each was disposed as obsolete-test against the implement plan's locked C3 decision and the build report's confirmation that C3 was applied as specified. No product bugs were identified. The design phase closed all 4 test_coverage_note gaps from the implement plan (R-level null/tilde/absent-key/absent-block/integer resolution for MAX_BOOT_SUBSAMPLE_N; the paired Python-side validate_plot_config accept/reject matrix for the same key; sublabel-relaxation acceptance coverage; and the GII y-axis grob's empty-sublabel-to-bare-title-grob collapse), adding 29 net new tests across 3 new files and 1 modified file while narrowing (not deleting) 2 existing parametrized tests to the still-valid label-only cases. The post-design full suite run is 1105/1105 passing with zero failures and zero errors; the 28 warnings are pre-existing and orthogonal to this cycle (a datetime API deprecation and expected small-n cluster-bootstrap fallback notices).
    </narrative>
  </summary>

  <action_items>
    <item priority="P2" target_mode="document">
      Documentation gaps to address in the next /document invocation (identified during a prior preflight, not part of this test cycle's scope): missing max_subsample_n entry in INPUT_SPECIFICATION.md, stale bootstrap_group_mean_sd() reference (renamed to group_mean_sd() in Session 22), Phase 1/Phase 2 parallel-design description for plot.R's compute-parallel/render-sequential restructure, and the now-unused n_cores parameter note. Per standing user constraint, /document must NOT reference deployment-environment-specific content.
    </item>
    <item priority="P1" target_mode="run-local">
      Open from a prior run-local report: the fork-safety fix for plot.R's compute-parallel/render-sequential restructure is unit-tested (this cycle's 1105/1105 pass) but not yet validated end-to-end on a multi-core Linux server under multi-core foreach %dopar%, which is the only way to genuinely exercise the fork-unsafety the fix addresses.
    </item>
  </action_items>
</test_report>
