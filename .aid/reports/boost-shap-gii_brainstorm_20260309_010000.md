<brainstorm_report>
  <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-03-09T01:00:00Z" status="IN_PROGRESS" />
  <context_files>
    <file path="boost-shap-gii_cr_20260309_000000.md" relevance="CR report being discussed finding-by-finding" />
    <file path="train.py" relevance="Shadow model iteration count (F4), inner CV seed (F2), ordinal validation (F3)" />
    <file path="predict.py" relevance="Syntax error (F1), model count validation (F5)" />
    <file path="shap_utils.py" relevance="Statistical findings F14-F25" />
    <file path="utils.py" relevance="Config defaults (F6, F7), bootstrap CI (F10), check_env (F8)" />
    <file path="plot.R" relevance="Core cap (F11)" />
  </context_files>

  <!-- =================================================================
       SESSION 1 BRAINSTORM: CR FINDINGS TRIAGE
       Status: INCOMPLETE — reviewed F1-F10, stopped at F10 pending user input.
       Findings F11-F25 not yet discussed.
       ================================================================= -->

  <topics>

    <topic id="T1" title="F1: Syntax error in predict.py:4">
      <summary>Trailing ß character on `annotations` import. Trivial fix.</summary>
      <decision status="decided" chosen="fix">Remove trailing ß. No discussion needed.</decision>
    </topic>

    <topic id="T2" title="F2: Inner CV seed reuse">
      <summary>Inner and outer CV use identical random_state. Minimal practical impact but theoretically impure.</summary>
      <decision status="decided" chosen="fix">Offset inner CV seed: `random_state=seed + fold_idx + 1`. Trivial one-line fix.</decision>
    </topic>

    <topic id="T3" title="F3: Ordinal unknown-value threshold metric">
      <summary>Uses fraction of unique values, not fraction of observations. Can miss high-frequency unknowns.</summary>
      <approaches>
        <approach id="A1" label="Two-tier check" feasibility="high" risk="low">
          <description>Keep unique-value fraction check (hard error at >50% unknowns). Add observation-level fraction check (loud warning at >10%). Report both metrics in warning messages.</description>
          <pros>Catches both misconfiguration (unique-value) and data quality (observation-level) issues.</pros>
          <cons>Slightly more code. Two hardcoded thresholds.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Two-tier approach. Hardcoded thresholds are fine — these are safety guards, not tunable parameters.</decision>
    </topic>

    <topic id="T4" title="F4: Shadow model iteration count">
      <summary>Shadow model trains on 2p features with iteration count tuned for p features. Deep analysis showed this is ANTI-CONSERVATIVE (inflates false positive rate) because noise features are disproportionately attenuated by undertraining, lowering the noise baseline.</summary>
      <approaches>
        <approach id="A1" label="Early stopping with outer validation fold" feasibility="high" risk="low">
          <description>Use tuned_iters * 2 as iteration ceiling. Add early stopping using the outer validation fold (X_val_full, y_val). Same early_stopping_rounds patience as inner CV. No data leakage because the shadow model's predictions are never evaluated for performance — it only produces SHAP values for noise calibration.</description>
          <pros>Data-adaptive. Lets shadow model find its own optimal. No leakage. Uses existing validation data.</pros>
          <cons>Slightly more compute (up to 2x iterations). Minor code complexity.</cons>
          <statistical_considerations>The outer validation fold is safe because shadow model outputs are never used for predictive evaluation. Early stopping is standard regularization that makes the shadow model better calibrated for the data it will produce SHAP values on.</statistical_considerations>
        </approach>
        <approach id="A2" label="Fixed multiplier" feasibility="high" risk="med">
          <description>Multiply tuned_iters by 1.5x or 2x for the shadow model.</description>
          <pros>Simple.</pros>
          <cons>Arbitrary. Unvalidated assumption that iterations scale linearly with features.</cons>
        </approach>
        <approach id="A3" label="Leave as-is, document" feasibility="high" risk="med">
          <description>Document the assumption and its anti-conservative direction.</description>
          <pros>No code change.</pros>
          <cons>Anti-conservative direction is harder to defend in peer review.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Option (a): early stopping with outer validation fold, tuned_iters * 2 ceiling, same early_stopping_rounds patience.</decision>
    </topic>

    <topic id="T5" title="F5: predict.py model count validation">
      <summary>No check that discovered model files match expected CV fold count.</summary>
      <decision status="decided" chosen="fix">Add assertion: len(model_files) == splitter.get_n_splits().</decision>
    </topic>

    <topic id="T6" title="F6: Depth search space too restrictive for small n">
      <summary>Depth range collapses to [2,2] for n&lt;20.</summary>
      <decision status="decided" chosen="fix">Add floor: max(3, min(10, int(np.log2(max(n / 5, 4))))) to guarantee at least [2, 3].</decision>
    </topic>

    <topic id="T7" title="F7: one_hot_max_size bound uses feature count">
      <summary>Semantically incorrect — should relate to max categorical cardinality, not total features.</summary>
      <decision status="decided" chosen="fix">Use fixed range {"low": 2, "high": 25}, removing dependency on p.</decision>
    </topic>

    <topic id="T8" title="F8: check_env.py missing joblib and statsmodels">
      <summary>Both imported by shap_utils.py but not in pre-flight check.</summary>
      <decision status="decided" chosen="fix">Add joblib and statsmodels to PYTHON_DEPS.</decision>
    </topic>

    <topic id="T9" title="F9: __NA__ sentinel undocumented">
      <summary>Missing nominal values filled with __NA__ as a real category level. Reasonable default but not documented.</summary>
      <decision status="decided" chosen="document">Document in README and config comments. No code change.</decision>
    </topic>

    <topic id="T10" title="F10: Bootstrap CI drops resamples with &lt;2 unique y values">
      <summary>For imbalanced classification with small n, dropped iterations bias CIs upward.</summary>
      <approaches>
        <approach id="A1" label="Drop and warn" feasibility="high" risk="low">
          <description>Keep current dropping behavior (metrics cannot be computed with single class). Add tracking: warn if >5% of iterations are dropped.</description>
        </approach>
        <approach id="A2" label="Assign worst-case score" feasibility="med" risk="med">
          <description>Instead of dropping, assign a worst-case score (0.5 for AUC, majority-class proportion for accuracy).</description>
          <statistical_considerations>Worst-case assignment is more conservative but may be overly pessimistic.</statistical_considerations>
        </approach>
      </approaches>
      <decision status="open" chosen="none">User was asked to choose between drop-and-warn vs worst-case-score. Session ended before response.</decision>
    </topic>

    <!-- FINDINGS NOT YET DISCUSSED -->
    <!-- F11: plot.R N_CORES cap -->
    <!-- F12: Column ordering (note — no action needed) -->
    <!-- F13: Ensemble soft voting calibration (note — document) -->
    <!-- F14: Interaction SHAP 2x scale -->
    <!-- F15: 2D spline zero-return fallback -->
    <!-- F16: NaN-to-zero bootstrap bias -->
    <!-- F17: Boruta adaptive noise baseline -->
    <!-- F18: Missing +1 correction on exceedance p-values -->
    <!-- F19: 2D energy check heuristic limitation -->
    <!-- F20: Bootstrap method-switching at discrete threshold -->
    <!-- F21: Stability metric for zero-variance estimates -->
    <!-- F22: NaN mask alignment fragility -->
    <!-- F23: Permutation test effective count reporting -->
    <!-- F24: SHAP decomposition non-additivity (note) -->
    <!-- F25: GII insensitivity to constant-effect features (note) -->

  </topics>

  <action_items>
    <item priority="P0" target_mode="implement" description="F1: Remove trailing ß from predict.py:4" />
    <item priority="P1" target_mode="implement" description="F2: Offset inner CV seed by fold_idx + 1 in train.py" />
    <item priority="P1" target_mode="implement" description="F3: Add two-tier ordinal unknown check (unique-value + observation-level)" />
    <item priority="P1" target_mode="implement" description="F4: Shadow model early stopping with outer val fold, 2x iteration ceiling" />
    <item priority="P1" target_mode="implement" description="F5: Add model-count vs fold-count assertion in predict.py" />
    <item priority="P1" target_mode="implement" description="F8: Add joblib and statsmodels to check_env.py" />
    <item priority="P2" target_mode="implement" description="F6: Add floor to depth search space" />
    <item priority="P2" target_mode="implement" description="F7: Fix one_hot_max_size to fixed [2, 25] range" />
    <item priority="P2" target_mode="document" description="F9: Document __NA__ sentinel behavior" />
    <item priority="P2" target_mode="implement" description="F10: PENDING — add drop tracking + warning at minimum; worst-case-score TBD" />
  </action_items>

  <next_steps>
    Resume brainstorm at F10 (pending user decision on drop-and-warn vs worst-case-score), then continue through F11–F25. After all findings are triaged, proceed to /implement mode for the decided fixes.
  </next_steps>
</brainstorm_report>
