<document_report>
 <meta project="boost-shap-gii" mode="document" timestamp="2026-05-08T14:53:34Z" />

 <files_updated>
 <file path="README.md" changes="Removed 'GII as a Cobb-Douglas Composite' subsection; relocated 'Per-individual SHAP reports' to bottom with minimal scope; added one-line per-individual note under Quick Start; removed use-case-specific/treatment/multi-arm experimental study/individuals/SHAP-tailored language and example YAML configuration block.">
 <type>readme</type>
 </file>
 <file path="INPUT_SPECIFICATION.md" changes="L678-684 reframed per-individual reports as use-case-agnostic inspection tools (removed use-case-specific-aid/treatment-decision/multi-arm multi-arm experimental study language); L982 'user-supplied anchor' to 'user-supplied anchor'; L1022 'applied settings' to use-case-agnostic; L229 'minimum meaningful difference (minimum-meaningful-difference threshold)' to 'minimum-meaningful-difference threshold'.">
 <type>input_spec</type>
 </file>
 <file path="AID_LOG.md" changes="Reframed entry as use-case-agnostic 'Per-individual inspection framing' (preserved methodological diagnostic terminology); removed internal cross-reference markers from Section 7 (,, parentheticals); preserved P0/P1/P2 severity terminology; preserved AID Framework LLM-tool disclosures.">
 <type>aid_log</type>
 </file>
 <file path="src/boost_shap_gii/indiv_reports.py" changes="L1 docstring reframed to 'individual-case model inspection' (removed 'hypothesis-generating inspection'); L3, L33, L36, L190, L235, L867 stripped internal change-code markers from comments and docstrings.">
 <type>docstring</type>
 </file>
 <file path="src/boost_shap_gii/train.py" changes="L950 stripped internal CR-finding marker from shadow-leakage comment.">
 <type>inline_comment</type>
 </file>
 <file path="src/boost_shap_gii/utils.py" changes="L434 'maximum, minimum-meaningful-difference threshold, or meaningful anchor' to use-case-agnostic 'maximum, minimum-meaningful-difference threshold, or any domain-specific anchor'; L781 stripped internal plan-resolution marker from docstring.">
 <type>inline_comment</type>
 </file>
 <file path="src/boost_shap_gii/infer.py" changes="L166 'pre- training run' to 'legacy training run' (stripped internal change-code marker).">
 <type>inline_comment</type>
 </file>
 <file path="src/boost_shap_gii/predict.py" changes="L147 'pre- training run' to 'legacy training run' (stripped internal change-code marker).">
 <type>inline_comment</type>
 </file>
 <file path="src/boost_shap_gii/scripts/plot.R" changes="L11, L439, L543, L682 stripped internal CR-finding and topic markers from comments.">
 <type>inline_comment</type>
 </file>
 <file path=".aid/reports/*.md (52 files)" changes="Sanitization passes 1-4 plus residual cleanup applied across all.aid/reports/. Pass 1: 38 files 724 substitutions (study-specific terms: outcome_a to outcome_a, outcome_b to outcome_b, dataset_v2 to dataset_v2, dataset_v1 to dataset_v1, discharge_total_* to target_*, intake_total_* to feature_*, plus initial internal-marker scrub). Pass 2: 15 files 282 substitutions (xml id/title attributes, T-A decision IDs, F-code ranges with em-dash, comment-form markers, residual multi-arm experimental study/intake-feature/treatment-tailored language). Pass 3: 38 files 764 substitutions (R-code/F-code/T-code/C-code/G-code/A-code parentheticals, slash-lists, hyphen-ranges, comma-lists, per-N references, source_item attributes, bare standalone codes excluding R2 metric). Pass 4: 5 files 64 substitutions (residual file-path patterns, minimum-meaningful-difference threshold, outcome-prediction, use-case-specific tailoring, precision-medicine, hypothesis-generating inspection, use-case-specific/users/clinically-X variants, individuals, multi-arm experimental study, control, baseline-assessment/feature/battery). Final residual cleanup: 24 files (empty parentheses, stray-open parens, double spaces, trailing whitespace).">
 <type>aid_artifact</type>
 </file>
 </files_updated>

 <aid_log>
 <status>updated</status>
 <sections_modified>Section 6 entry reframed; Section 7 internal-marker parentheticals removed (,,); Section 8 markers removed. Preserved: AID Framework disclosure structure, LLM-tool framing, P0/P1/P2 severity terminology, all version history entries, all literature citations.</sections_modified>
 </aid_log>

 <coverage>
 <public_functions_documented>n/a (pre-existing; this cycle restricted to comment/docstring sanitization)</public_functions_documented>
 <classes_documented>n/a (pre-existing)</classes_documented>
 <modules_with_docstrings>n/a (pre-existing)</modules_with_docstrings>
 </coverage>

 <gates>
 <pii_screening status="pass">
 <scan_scope>README.md, INPUT_SPECIFICATION.md, AID_LOG.md, src/boost_shap_gii/*.py, src/boost_shap_gii/scripts/*.R/*.sh.aid/project_claude.md.aid/reports/*.md (52 files).</scan_scope>
 <patterns_scanned>absolute filesystem paths (/Users/, /Volumes/, /home/), email addresses, hostnames (.local.internal, institutional HPC hostnames), IPv4 addresses, UUIDs, conda/miniconda paths, sandbox-path leakage, usernames outside public GitHub URLs.</patterns_scanned>
 <findings>None. Public GitHub URL (https://github.com/tjkeding/boost-shap-gii) is exempt per CONVENTIONS.md PII Screening Gate doctrine.</findings>
 </pii_screening>
 <llm_attribution status="pass">
 <scan_scope>All files created or modified this session.</scan_scope>
 <tier1_findings>None. Zero co-authorship trailers, zero "by Claude" "written by Claude" "authored by Claude" "refactored by Claude" "generated by Claude" patterns across all in-scope files.</tier1_findings>
 <tier2_findings>AID_LOG.md contains permitted AID Framework disclosures framing LLM use exclusively as tool-use (Sections 5, 6, 7, 8, Version History). All other files (README.md, INPUT_SPECIFICATION.md, src/.aid/reports/) contain zero LLM references.</tier2_findings>
 </llm_attribution>
 </gates>

 <summary>Documentation cycle completed under user directive (a) to apply use-case-agnostic remediation to v1.2.0 and (b) to remove all internal-flag markers (F#, C#, R#, T#, G#, A#, CR#, Items #) from publicly committed files. README.md restructured to remove Cobb-Douglas subsection and relocate per-individual reports to a minimal bottom section; INPUT_SPECIFICATION.md, AID_LOG.md, src/*, and 52.aid/reports/*.md files sanitized of use-case-specific/treatment/multi-arm experimental study/individuals/intake/minimum-meaningful-difference threshold/precision-medicine/SHAP-tailored language and internal cross-reference codes. PII Screening Gate and LLM-Attribution Scrub Gate both pass. Documentation state is ready for the corrected v1.2.0 force-push via /publish.</summary>
</document_report>
