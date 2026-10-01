<document_report>
  <meta project="boost-shap-gii" mode="document" timestamp="2026-10-01T15:06:02Z" />
  <files_updated>
    <file path="README.md" changes="V-panel discrete error bars re-described as analytical standard errors of the level mean; plot.bootstrap_ribbons.n_boot row narrowed to spline SD ribbons; new plot.bootstrap_ribbons.max_subsample_n config-table row (integer or null, default 5000, minimum 10, sqrt(m/n) rescaling, null disables); sublabel empty-string suppression and pre-launch Python validation noted.">
      <type>readme</type>
    </file>
    <file path="INPUT_SPECIFICATION.md" changes="plot section: optional-key count corrected, Python-side validation (utils.validate_plot_config via cli.cmd_plot) documented, label/sublabel contracts (non-empty labels; string sublabels with empty/whitespace suppression and bare-title collapse) specified; n_boot row narrowed to ribbons; new bootstrap_ribbons.max_subsample_n row (type, default, floor matching MIN_BOOT_N, rejected types, m-out-of-n citations). Section 4: stale bootstrap group-mean paragraph replaced with spline SD ribbon and analytical group-mean SE description; new compute-parallel / render-sequential execution-model paragraph (worker return contract, parent-process rendering, sequential per-individual plots, fork-safety rationale, unseeded ribbon variability). Section 10.3 sublabel rows note empty-string suppression.">
      <type>input_spec</type>
    </file>
    <file path="AID_LOG.md" changes="Test-count references updated to 1105 tests across 49 test files; two new session entries (2026-09-29 fork-safe plot rendering and V-panel uncertainty-overlay corrections; 2026-10-01 sequential per-individual rendering and subsampling-cap / sublabel contract fixes) with scope, tool disclosure, key decisions, test metrics, and audit references. Version History section untouched.">
      <type>aid_log</type>
    </file>
    <file path="src/boost_shap_gii/scripts/plot.R" changes="Comment-only edits: n_cores marked as retained-but-unused on both per-individual render functions; Phase 1 worker return contract (plain structured lists, no graphics/Rcpp state post-fork) annotated above the foreach loop. No functional code changed; full suite 1105/1105 after the edits.">
      <type>inline_comment</type>
    </file>
  </files_updated>
  <aid_log>
    <status>updated</status>
    <sections_modified>Section 4 (Development Workflow test counts); Section 4 key properties test-count line; Section 7 (two new session entries)</sections_modified>
  </aid_log>
  <coverage>
    <public_functions_documented>not re-audited this invocation (scope limited to plot.R visualization changes)</public_functions_documented>
    <classes_documented>not re-audited this invocation</classes_documented>
    <modules_with_docstrings>not re-audited this invocation</modules_with_docstrings>
  </coverage>
  <security_gate>
    <agents>5</agents>
    <gate_type>combined</gate_type>
    <files_scanned>README.md; INPUT_SPECIFICATION.md; AID_LOG.md; src/boost_shap_gii/scripts/plot.R; .aid/project_claude.md</files_scanned>
    <violations_after_union_dedup>0</violations_after_union_dedup>
    <result>clean (5/5 agents returned empty violations arrays)</result>
  </security_gate>
  <summary>User-facing and LLM-facing documentation now reflects the compute-parallel / render-sequential plotting design, the m-out-of-n subsampling cap on spline SD ribbons, the analytical group-mean SE for discrete V-panel error bars, the max_subsample_n null-versus-absent contract, and the sublabel suppression contract. All additions are project-agnostic. .aid/project_claude.md verified current and left unchanged.</summary>
</document_report>
