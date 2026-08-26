<brainstorm_report>
  <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-08-26T18:15:50Z" />
  <context_files>
    <file path="boost-shap-gii_test_20260826_152600.md" relevance="P0 action item: cross-fold shap_scale_factor logic (train.py:1209-1219) mis-specified; 32 tests blocked" />
    <file path="src/boost_shap_gii/train.py" relevance="Defect sites: per-fold alpha computation (1028-1045), post-loop aggregation (1209-1219), transform_config.json emission (1221-1232)" />
    <file path="src/boost_shap_gii/predict.py" relevance="Consumption: shap_scale_factor passed to shap_utils (490-491) and indiv_reports (559)" />
    <file path="src/boost_shap_gii/infer.py" relevance="Consumption: shap_scale_factor read from transform_config.json (248)" />
    <file path="src/boost_shap_gii/shap_utils.py" relevance="Application site: SHAP_vals * shap_scale_factor at line 972; shadow scaling at 973-974" />
    <file path="src/boost_shap_gii/indiv_reports.py" relevance="Downstream consumption of shap_scale_factor in generate_indiv_reports" />
  </context_files>
  <topics>
    <topic id="T1" title="Per-fold SHAP scaling architecture">
      <summary>Replace the single-scalar shap_scale_factor with per-row, per-originating-fold scaling. Each row of the pooled OOF SHAP matrix is scaled by the alpha from the fold that produced it. In inference mode, each fold model's SHAP values are scaled by that fold's alpha before cross-fold averaging. The mathematical basis is SHAP linearity under affine transformation (Lundberg and Lee 2017): for output_transform g^{-1}(p) = alpha_k * p + beta_k, SHAP_{original} = alpha_k * SHAP_{transformed}. The bug class is data-dependent affine transforms (e.g., z-score standardization where sigma is estimated per fold from the training partition), which produce legitimately different alpha_k values across folds. Fixed-parameter affine transforms are unaffected (identical fold alphas). Non-affine transforms are unaffected (the alpha block is skipped entirely; fold_alpha defaults to 1.0).</summary>
      <research>No external research dispatched. The design resolves from first principles: SHAP linearity axiom (Lundberg and Lee 2017, Theorem 1) combined with the chain rule for affine compositions. The asymptotic SE of a sample standard deviation is sigma / sqrt(2*(n-1)), giving a coefficient of variation of ~11% at n=40, confirming that the rtol=1e-6 tolerance is impossibly tight for finite-sample estimates.</research>
      <approaches>
        <approach id="A1" label="Full-dataset scale parameter" feasibility="high" risk="high">
          <description>Compute the scale parameter once from the full dataset (not per fold) and use as the single global scalar. Decouples reporting-side scale from per-fold transforms.</description>
          <pros>Single scalar; no structural change to shap_utils consumption.</pros>
          <cons>Mathematically incorrect: full-dataset sigma differs from any fold's sigma. Fold k's SHAP values are in the sigma_k scale, not the sigma_full scale. Introduces systematic bias of (sigma_full / sigma_k - 1) per fold.</cons>
          <statistical_considerations>Bias grows with inter-fold variance of the scale parameter. For small n or heterogeneous outcomes, this is not negligible.</statistical_considerations>
        </approach>
        <approach id="A2" label="Per-row fold-specific scaling" feasibility="high" risk="low">
          <description>Construct a per-row alpha vector using fold_assignments.json (which fold held out each row) and fold_transform_metadata.json (each fold's _pipeline_alpha). Multiply SHAP matrix row-wise: SHAP_vals * alpha_vec[:, None]. In inference mode, scale each fold model's SHAP by that fold's alpha before averaging.</description>
          <pros>Mathematically exact. Required data structures already exist (fold_assignments.json, fold_transform_metadata.json with _pipeline_alpha). No additional estimation. Eliminates both the rtol=1e-6 hard halt and the single-scalar approximation.</pros>
          <cons>Requires structural change at shap_utils.py:972 (scalar multiplication becomes row-wise broadcasting). Downstream consumers (predict.py, infer.py, indiv_reports.py) must propagate per-fold alpha information instead of a single scalar.</cons>
          <statistical_considerations>Exact by construction; no approximation, no tolerance threshold to tune.</statistical_considerations>
        </approach>
        <approach id="A3" label="Mean of fold alphas" feasibility="high" risk="medium">
          <description>Use the arithmetic mean of fold alphas as the single global scalar. Pair with a materially looser consistency check.</description>
          <pros>Preserves single-scalar architecture. Bias is second-order: O(CV(alphas)^2).</pros>
          <cons>Still an approximation; strictly less correct than per-fold scaling with no compensating advantage. Threshold for the consistency check is arbitrary. For z-score with large n, bias is small but not zero.</cons>
          <statistical_considerations>Acceptable when fold alphas have low CV (say, less than 5%), but why approximate when the exact answer is equally available?</statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A2">Per-row fold-specific scaling is the only mathematically exact option, requires no additional estimation, and the data structures (fold_assignments.json, fold_transform_metadata.json) are already in place. Implementation scope: train.py (post-loop aggregation), predict.py (propagate fold_alphas array), infer.py (propagate fold_alphas array), shap_utils.py (row-wise multiplication at line 972). Individual reports handling (per-bootstrap alpha computation vs. fold-alpha approximation) deferred to /implement plan.</decision>
    </topic>
    <topic id="T2" title="Cross-fold consistency check disposition">
      <summary>The post-loop block at train.py:1209-1219 (rtol=1e-6 hard halt + fold_alphas[0] assignment) is fully replaced by T1's per-row scaling. The question is whether any diagnostic replaces it.</summary>
      <research>No external research dispatched. The design question is one of user-facing observability, not statistical methodology.</research>
      <approaches>
        <approach id="B1" label="Remove entirely" feasibility="high" risk="low">
          <description>Delete the cross-fold check. Per-row scaling handles correctness regardless of inter-fold spread.</description>
          <pros>Simplest implementation. No false halts.</pros>
          <cons>A buggy user transform producing wildly different alphas across folds would proceed silently with no diagnostic trail.</cons>
        </approach>
        <approach id="B2" label="Informational diagnostic" feasibility="high" risk="low">
          <description>Replace the hard halt with a print block reporting fold alphas and their coefficient of variation. No halt, no threshold.</description>
          <pros>Gives the user visibility into inter-fold variability. A high CV (~50%+) signals that the transform's scale parameter is poorly estimated or the transform is misspecified, even though per-row scaling will still apply correct per-fold values. Diagnostic trail for debugging.</pros>
          <cons>Adds one print statement; negligible complexity.</cons>
        </approach>
        <approach id="B3" label="Hard halt for extreme divergence" feasibility="high" risk="medium">
          <description>Halt only on sign flip or order-of-magnitude divergence in fold alphas.</description>
          <pros>Catches genuinely pathological transforms (e.g., a buggy output_transform that produces negative slope in some folds).</pros>
          <cons>Requires choosing a principled threshold. Any halt risks repeating the current bug pattern (legitimate variation misclassified as pathological).</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="B2">Informational diagnostic: log fold alphas and CV, no halt. Per-row scaling ensures correctness; the diagnostic provides observability without risk of false halts. Exact format deferred to /implement plan.</decision>
    </topic>
  </topics>
  <action_items>
    <item priority="P0" target_mode="implement" description="Replace train.py post-loop cross-fold aggregation (lines 1209-1219) with per-fold alpha array construction and informational CV diagnostic. Replace single-scalar shap_scale_factor in transform_config.json with per-fold alpha array (or equivalent). Update predict.py and infer.py to propagate per-fold alphas to shap_utils. Update shap_utils.py:972 from scalar multiplication to per-row fold-specific scaling using fold_assignments. Update indiv_reports.py shap_scale_factor consumption. Resolve per-bootstrap alpha handling (compute in worker vs. approximate with fold alpha) during /implement plan." />
  </action_items>
  <next_steps>Run /implement (plan then build) to resolve the P0 defect. Follow with /test to confirm the 32 blocked dry-run tests pass and no regressions are introduced.</next_steps>
</brainstorm_report>
