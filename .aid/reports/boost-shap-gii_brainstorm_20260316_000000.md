<brainstorm_report>
  <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-03-16T00:00:00Z" status="COMPLETE" />
  <context_files>
    <file path="boost-shap-gii_cr_20260309_000000.md" relevance="CR report — all 25 findings triaged" />
    <file path="train.py" relevance="F2 (inner CV seed), F3 (ordinal validation), F4 (shadow model iterations)" />
    <file path="predict.py" relevance="F1 (syntax error), F5 (model count validation)" />
    <file path="shap_utils.py" relevance="F14-F25 (statistical/algorithmic findings)" />
    <file path="utils.py" relevance="F6, F7 (config defaults), F10 (bootstrap CI), F23 (permutation test)" />
    <file path="plot.R" relevance="F11 (N_CORES cap), F25 (left panel M vs GII)" />
    <file path="check_env.py" relevance="F8 (missing deps)" />
  </context_files>

  <topics>

    <!-- ================================================================= -->
    <!--  SESSION 1: F1–F9 (decided 2026-03-09/10)                         -->
    <!-- ================================================================= -->

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
      <summary>Shadow model trains on 2p features with iteration count tuned for p features. Anti-conservative — inflates false positive rate because noise features are disproportionately attenuated by undertraining, lowering the noise baseline.</summary>
      <approaches>
        <approach id="A1" label="Early stopping with outer validation fold" feasibility="high" risk="low">
          <description>Use tuned_iters * 2 as iteration ceiling. Add early stopping using the outer validation fold (X_val_full, y_val). Same early_stopping_rounds patience as inner CV. No data leakage because the shadow model's predictions are never evaluated for performance — it only produces SHAP values for noise calibration.</description>
          <pros>Data-adaptive. Lets shadow model find its own optimal. No leakage. Uses existing validation data.</pros>
          <cons>Slightly more compute (up to 2x iterations). Minor code complexity.</cons>
          <statistical_considerations>The outer validation fold is safe because shadow model outputs are never used for predictive evaluation. Early stopping is standard regularization that makes the shadow model better calibrated for the data it will produce SHAP values on.</statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Early stopping with outer validation fold, tuned_iters * 2 ceiling, same early_stopping_rounds patience.</decision>
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

    <!-- ================================================================= -->
    <!--  SESSION 2: F10–F25 (decided 2026-03-16)                          -->
    <!-- ================================================================= -->

    <topic id="T10" title="F10: Bootstrap CI drops resamples with &lt;2 unique y values">
      <summary>For imbalanced classification with small n, dropped iterations bias CIs upward. Discussed drop-and-warn vs worst-case-score vs while-loop retry approaches.</summary>
      <approaches>
        <approach id="A1" label="Drop and warn with n_boot_effective" feasibility="high" risk="low">
          <description>Keep current dropping behavior (metrics cannot be computed with single class). Track and report n_boot_effective as a confidence proxy. Warn if >5% of iterations are dropped.</description>
          <pros>Transparent. Metrics are genuinely undefined for single-class resamples — no imputation of undefined quantities. n_boot_effective communicates confidence directly.</pros>
          <cons>Effective resolution reduced for severely imbalanced small-n datasets.</cons>
          <statistical_considerations>Worst-case-score imputation rejected: it assigns a value to an undefined quantity, introducing its own assumption. While-loop retry rejected for bootstraps: failure rate is diagnostic information about sample size adequacy that should be preserved, not masked.</statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Drop-and-warn with n_boot_effective reporting. n_boot_effective serves as a transparent confidence proxy (n_boot_effective = n_boot indicates maximum confidence).</decision>
    </topic>

    <topic id="T11" title="F11: plot.R N_CORES cap">
      <summary>N_CORES passed directly from config without capping at machine's actual core count.</summary>
      <decision status="decided" chosen="fix">Cap: N_CORES &lt;- min(cfg$execution$n_jobs, parallel::detectCores(logical = FALSE) %||% cfg$execution$n_jobs). Handles detectCores() returning NA.</decision>
    </topic>

    <topic id="T12" title="F12: Column ordering (note)">
      <summary>CR confirmed column ordering between train/predict/infer is consistent. No issue.</summary>
      <decision status="decided" chosen="no_action">No action needed.</decision>
    </topic>

    <topic id="T13" title="F13: Ensemble soft voting calibration (note)">
      <summary>Soft voting assumes calibrated probability outputs. Checked all currently available loss functions (RMSE, MultiRMSE, Logloss, MultiClass) — no calibration concern with any.</summary>
      <decision status="decided" chosen="no_action">No action. No calibration issue with current loss functions. Documenting hypothetical vulnerabilities from unsupported configurations is unnecessary.</decision>
    </topic>

    <topic id="T14" title="F14: Interaction SHAP 2× scale">
      <summary>Interaction SHAP computed as phi_inter[:, i, j] + phi_inter[:, j, i] = 2 * phi_inter[:, i, j] due to symmetric matrix. Deep discussion of SHAP interaction matrix structure, Shapley interaction index, and GII prediction-decomposition interpretation.</summary>
      <approaches>
        <approach id="A1" label="Keep summed convention (current code)" feasibility="high" risk="low">
          <description>The SHAP interaction matrix automatically divides phi(i,j) by 2 across both cells. The full matrix (both triangles) is required to recover the true Shapley interaction index I(i,j) = phi(i,j) + phi(j,i). Using one triangle would UNDERSTATE interaction importance by 2× relative to main effects, because main effects phi(i,i) are stored at full scale while each off-diagonal cell is half-scale.</description>
          <pros>Correct prediction-decomposition scale. Cross-type GII comparisons are valid. Consistent with the SHAP decomposition identity.</pros>
          <cons>None identified.</cons>
          <statistical_considerations>GII is a prediction-decomposition-based importance measure. M and V must reflect the actual prediction contribution of each effect. The summed convention achieves this: a main effect with GII=0.5 and an interaction with GII=0.5 both contribute comparably to the model. One-triangle would create a 0.5× artifact understating interactions.</statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">No code change. Current summed convention is correct for GII's prediction-decomposition interpretation. Document that SHAP interaction matrices divide phi(i,j) by 2 per cell, so the full matrix is required to recover the true interaction contribution.</decision>
    </topic>

    <topic id="T15" title="F15: 2D spline zero-return fallback">
      <summary>calculate_v_spline_2d returns 0.0 when either axis has &lt;2 interior knots, systematically suppressing interactions involving low-resolution features. Investigated why 0.0 was originally implemented — the 2D group-means fallback is pathological for continuous data (groupby exact pairs creates near-singleton cells, V degenerates to total variability).</summary>
      <approaches>
        <approach id="A1" label="Stacked-spline routing" feasibility="high" risk="low">
          <description>Route insufficient-knots case to calculate_v_stacked_spline: treat the low-resolution axis as a discrete grouping variable, fit 1D splines along the well-resolved axis within each group. If BOTH axes lack resolution, fall back to calculate_v_group_means_2d (appropriate when both axes are quasi-discrete, producing few cells with adequate n).</description>
          <pros>Principled handling of mixed-resolution interactions. Inherits full protection chain (density gate, adaptive knots, energy gate, group-means fallback). Already used for mixed continuous×discrete interactions.</pros>
          <cons>None identified.</cons>
          <statistical_considerations>Exploding V concern addressed: stacked-spline inherits the 1D energy gate (spline total variation ≤ data total variation) per group. M/V scale comparability confirmed: stacked-spline V is on the same scale as other V methods (units of outcome, bounded by data variability). No systematic M/V dominance introduced — same method already in use for mixed interactions.</statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Replace return 0.0 with stacked-spline routing based on which axis failed knot placement. Group-means only when both axes lack resolution.</decision>
    </topic>

    <topic id="T16" title="F16: NaN-to-zero bootstrap bias">
      <summary>NaN from spline failures replaced with 0.0, biasing GII estimates downward and inflating p-values. Conflates computational failure with genuinely zero effects.</summary>
      <decision status="decided" chosen="fix">Remove NaN-to-zero replacement (lines 709-711). Downstream nanmean/nanpercentile calls already handle NaN exclusion natively. Makes real and shadow NaN handling consistent.</decision>
    </topic>

    <topic id="T17" title="F17: Boruta adaptive noise baseline">
      <summary>Shadow SHAP values are conditioned on real features — the noise baseline is model-adaptive. This is standard Boruta behavior (Kursa &amp; Rudnicki, 2010), not a novel design choice.</summary>
      <decision status="decided" chosen="no_action">No action. No documentation needed — standard Boruta behavior does not require proactive justification.</decision>
    </topic>

    <topic id="T18" title="F18: Missing +1 correction on exceedance p-values">
      <summary>Exceedance p-values lack the Davison &amp; Hinkley (1997) / Phipson &amp; Smyth (2010) +1 correction. Minimum achievable p-value is 0 instead of 1/(n_boot+1). Inconsistent with the pipeline's own permutation test which has the correction.</summary>
      <decision status="decided" chosen="fix">Apply +1 correction to all exceedance p-values (M, V, GII): p = (sum(boot &lt;= noise) + 1) / (n_boot + 1).</decision>
    </topic>

    <topic id="T19" title="F19: 2D energy check heuristic limitation">
      <summary>Axis-wise total variation is a heuristic that may miss oblique oscillation. However, the stability gate provides downstream protection, and the F15 fix reduces the scope of 2D spline usage.</summary>
      <decision status="decided" chosen="no_action">No action. No documentation needed.</decision>
    </topic>

    <topic id="T20" title="F20: Bootstrap method-switching at discrete threshold">
      <summary>Features near discrete_threshold may switch between spline and group-means across bootstrap iterations. Current design is intentional (commented), conservative direction, stability gate mitigates.</summary>
      <decision status="decided" chosen="no_action">No action.</decision>
    </topic>

    <topic id="T21" title="F21: Stability metric for zero-variance estimates">
      <summary>stab=0 for CI_width≈0 penalizes genuinely stable near-zero effects. However, such effects are extremely rare and would clear significance via exceedance regardless.</summary>
      <decision status="decided" chosen="no_action">No action. Conservative, implausible edge case.</decision>
    </topic>

    <topic id="T22" title="F22: NaN mask alignment fragility">
      <summary>X_vals and nan_mask could theoretically become misaligned in future refactoring. No current bug. Shape assertion insufficient (doesn't catch index misalignment), encapsulation over-engineered.</summary>
      <decision status="decided" chosen="no_action">No action.</decision>
    </topic>

    <topic id="T23" title="F23: Permutation test effective count reporting">
      <summary>Permutation test filters NaN from null distribution without reporting effective count. Discussed while-loop vs for-loop: permutation tests preserve full y distribution (no class-loss issue), so failures are rare and non-diagnostic. While-loop retry is statistically valid and appropriate.</summary>
      <approaches>
        <approach id="A1" label="While-loop with cap" feasibility="high" risk="low">
          <description>Replace for-loop with while-loop that continues until n_perm successful iterations are collected. Cap at 2 * n_perm total attempts. If cap is reached, warn and proceed with effective count.</description>
          <pros>Guarantees full resolution. Statistically valid (permutation preserves y distribution, no selection bias). Loop will almost never run beyond n_perm + handful of extra iterations.</pros>
          <cons>Slightly more control flow complexity.</cons>
          <statistical_considerations>Unlike bootstraps (where failure rate is diagnostic of class imbalance × small n), permutation failures are rare numerical artifacts with no diagnostic value. Retry is appropriate. While-loop approach specifically NOT recommended for bootstraps — failure rate is informative there.</statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">While-loop with 2 * n_perm cap. Warn and proceed with effective count if cap reached.</decision>
    </topic>

    <topic id="T24" title="F24: SHAP decomposition non-additivity (note)">
      <summary>GII values cannot be summed to reconstruct marginal SHAP importance. Inherent property of the framework.</summary>
      <decision status="decided" chosen="no_action">No action. No documentation needed — same reasoning as F13 (do not document hypothetical misuse).</decision>
    </topic>

    <topic id="T25" title="F25: GII insensitivity to constant-effect features (note) + Plot panel change">
      <summary>Features with high M but V≈0 get GII≈0. Pipeline already reports M separately. Discussion led to identifying that plot.R left panel shows GII distribution instead of M distribution — changing to M provides proper two-panel decomposition (M on left, V pattern on right) and visually represents the magnitude component that would otherwise be invisible for high-M, low-V features.</summary>
      <approaches>
        <approach id="A1" label="Change left plot panel from GII to M" feasibility="high" risk="low">
          <description>In plot.R, load bootstrap_distributions_M.parquet and stratified_noise_distributions_M.parquet instead of GII versions for the left density panel. Update x-axis label to "Importance Magnitude (M)". Update get_global_x_limit_dir to use M parquets. No other plotting changes — dimensions, styling, right panel all unchanged.</description>
          <pros>Left panel shows fundamental M component. Right panel shows V pattern. Together they decompose GII visually. High-M low-V features now have visual representation.</pros>
          <cons>None.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Change left panel to M distribution. X-axis label: "Importance Magnitude (M)". No other plotting changes. Data files already exist (bootstrap_distributions_M.parquet, stratified_noise_distributions_M.parquet).</decision>
    </topic>

  </topics>

  <action_items>
    <!-- P0 -->
    <item priority="P0" target_mode="implement" description="F1: Remove trailing ß from predict.py:4" />

    <!-- P1 — Code fixes -->
    <item priority="P1" target_mode="implement" description="F2: Offset inner CV seed by fold_idx + 1 in train.py" />
    <item priority="P1" target_mode="implement" description="F3: Add two-tier ordinal unknown check (unique-value + observation-level fraction) in train.py, predict.py, infer.py" />
    <item priority="P1" target_mode="implement" description="F4: Shadow model early stopping with outer val fold, 2x iteration ceiling, same early_stopping_rounds patience in train.py" />
    <item priority="P1" target_mode="implement" description="F5: Add model-count vs fold-count assertion in predict.py" />
    <item priority="P1" target_mode="implement" description="F15: Replace return 0.0 in calculate_v_spline_2d with stacked-spline routing (low-resolution axis as grouping var); group-means only when both axes lack resolution" />
    <item priority="P1" target_mode="implement" description="F16: Remove NaN-to-zero replacement (shap_utils.py lines 709-711); let existing nanmean/nanpercentile handle exclusion natively" />
    <item priority="P1" target_mode="implement" description="F18: Apply +1 correction to all exceedance p-values (M, V, GII) in shap_utils.py: p = (sum + 1) / (n_boot + 1)" />

    <!-- P2 — Code fixes -->
    <item priority="P2" target_mode="implement" description="F6: Add floor to depth search space: max(3, ...) in utils.py" />
    <item priority="P2" target_mode="implement" description="F7: Fix one_hot_max_size to fixed [2, 25] range in utils.py" />
    <item priority="P2" target_mode="implement" description="F8: Add joblib and statsmodels to PYTHON_DEPS in check_env.py" />
    <item priority="P2" target_mode="implement" description="F10: Add n_boot_effective tracking and >5% drop warning to compute_bootstrap_ci in utils.py" />
    <item priority="P2" target_mode="implement" description="F11: Cap N_CORES at min(config, detectCores()) with NA fallback in plot.R" />
    <item priority="P2" target_mode="implement" description="F23: Convert permutation test to while-loop with 2*n_perm cap, warn if cap reached, in utils.py" />
    <item priority="P2" target_mode="implement" description="F25: Change left plot panel from GII to M distribution; x-axis label 'Importance Magnitude (M)'; update get_global_x_limit_dir to use M parquets in plot.R" />

    <!-- P2 — Documentation -->
    <item priority="P2" target_mode="document" description="F9: Document __NA__ sentinel behavior for nominal missingness in README and config comments" />
    <item priority="P2" target_mode="document" description="F14: Document that SHAP interaction matrices divide phi(i,j) by 2 per cell; full matrix required to recover true interaction contribution" />

    <!-- No action: F12, F13, F17, F19, F20, F21, F22, F24 -->
  </action_items>

  <next_steps>
    Proceed to /implement mode. 15 code changes (1 P0, 7 P1, 7 P2) and 2 documentation items. Recommended implementation order: P0 first (unblocks execution), then P1 fixes (statistical correctness), then P2 fixes (robustness and polish), then documentation. Design test suite before implementing to validate each fix.
  </next_steps>
</brainstorm_report>
