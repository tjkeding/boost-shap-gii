<document_report>
  <meta project="boost-shap-gii" mode="document" timestamp="2026-08-26T19:50:17Z" />
  <files_updated>
    <file path="README.md" changes="Updated the SHAP Back-Transformation section to describe per-fold scaling (each fold's own constant scale factor alpha, applied only to the rows that fold produced) in place of the prior single-global-scale-factor description, and to note that bootstrap CI refits compute their own exact alpha.">
      <type>readme</type>
    </file>
    <file path="INPUT_SPECIFICATION.md" changes="Updated the transformations.back_transform_shap description (per-fold alpha, not a single scalar). Replaced the transform_config.json shap_scale_factor field (float) with fold_shap_scale_factors (list[float], one per CV fold) in the artifact schema table. Added bootstrap_alphas.npy to the bootstrap_refits/ artifact description and directory tree (B x K float64 array of exact per-refit alphas), and added bootstrap_alphas_saved to the bootstrap_metadata.json field list description.">
      <type>input_spec</type>
    </file>
    <file path="AID_LOG.md" changes="Added a new session entry (2026-08-26) covering the per-fold SHAP scaling P0 fix: problem statement, mathematical basis (SHAP linearity under affine transformation), key researcher-approved decisions (per-row fold-specific scaling, informational diagnostic over hard halt, exact per-bootstrap alpha, inference-mode data boundary), test metrics across the pre-design/post-design cycle, and audit trail references.">
      <type>aid_log</type>
    </file>
  </files_updated>
  <aid_log>
    <status>updated</status>
    <sections_modified>Session history (new entry added); Section 8 unchanged (no version bump this session)</sections_modified>
  </aid_log>
  <coverage>
    <public_functions_documented>n/a</public_functions_documented>
    <classes_documented>n/a</classes_documented>
    <modules_with_docstrings>n/a</modules_with_docstrings>
  </coverage>
  <summary>No functional code changes were made this session (that work was completed under /implement and /test in prior turns of this session); this /document pass brought README.md and INPUT_SPECIFICATION.md into alignment with the fold_shap_scale_factors / bootstrap_alphas.npy artifacts introduced by the per-fold SHAP scaling fix, replacing all stale references to the retired single-scalar shap_scale_factor field. AID_LOG.md received a new session entry. Eight session reports (brainstorm, 2 implement plans, 2 implement builds, 2 test reports, this document report) were synced to .aid/reports/. The mandatory PII/PHI and LLM-attribution security gate (5 parallel scans, union-with-dedup aggregation) found zero PII/PHI and zero Tier 1 LLM-attribution violations across all 11 scanned files. Five Tier 2 findings surfaced in AID_LOG.md's standing "Claude {model}: used for {tasks}" disclosure format (four pre-existing across prior sessions, one newly authored this session following the same convention); the researcher reviewed and confirmed this format acceptable as tool-disclosure framing, so no remediation was applied.</summary>
</document_report>
