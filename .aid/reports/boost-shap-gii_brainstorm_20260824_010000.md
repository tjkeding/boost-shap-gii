<brainstorm_report>
  <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-08-24T01:00:00-04:00" />
  <context_files>
    <file path="&lt;external_project&gt;_implement_plan_20260821_200000.md" relevance="Primary input: harmonized implement plan for aggregate stratum + transformations API (C1-C6)" />
    <file path="&lt;external_project&gt;_boost-shap-gii_changes_20260817.md" relevance="Source document for C1 (aggregate noise stratum split)" />
    <file path="boost-shap-gii_implement_plan_20260819_130623.md" relevance="Source document for C2-C6 (transformations API)" />
    <file path="boost_shap_gii_techspec_cv_strategy_inner_repeats.md" relevance="Context: cv_strategy and n_inner_repeats already implemented in v1.4.0" />
    <file path="src/boost_shap_gii/shap_utils.py" relevance="Exceedance test stratum logic (lines 1088-1163), _get_effect_stratum (lines 142-160), aggregate effects (line 604)" />
    <file path="src/boost_shap_gii/utils.py" relevance="fill_config_defaults (lines 430-506), target for C2 additions" />
    <file path="src/boost_shap_gii/train.py" relevance="Outer CV loop (lines 931+), scaler block (lines 844-856), transform integration target (C3)" />
    <file path="src/boost_shap_gii/predict.py" relevance="OOF prediction loop (lines 249-271), scaler inverse-transform (lines 282-290), C4 target" />
    <file path="src/boost_shap_gii/infer.py" relevance="Ensemble prediction loop (lines 265-278), scaler inverse-transform (lines 327-332), C5 target" />
  </context_files>
  <topics>
    <topic id="T1" title="Aggregate stratum exceedance calibration with small feature counts">
      <summary>C1 creates a singleton_aggregate stratum in the Stratified Max Boruta Exceedance Test. With k aggregate groups, the shadow max distribution is max(k shadow aggregates) per bootstrap iteration. Under the null, the expected exceedance rate is 1/(k+1): 50% for k=1, 33% for k=2, 25% for k=3. Type-I error control is correct for all k, but statistical power is severely limited for small k. The status quo (pooling aggregates into continuous) inflates the continuous stratum's noise floor, so C1 fixes one problem at the cost of reduced power in small aggregate strata. This is an inherent property of having few hypotheses to test, not a defect.</summary>
      <research>No external research dispatched. Analysis from first principles using order-statistic theory: the probability that a specific draw is the maximum of k+1 i.i.d. values is 1/(k+1). The existing global-max fallback (shap_utils.py lines 1119-1150) handles zero-shadow strata; no analogous safeguard exists for small-but-nonzero strata. Option B (fallback to global max for small strata) was considered but rejected because it would be MORE conservative, making aggregate features harder to detect, which contradicts C1's original motivation.</research>
      <approaches>
        <approach id="A1" label="C1 as-is" feasibility="high" risk="low">
          <description>Dedicated aggregate stratum with no minimum size requirement.</description>
          <pros>Correct calibration for both continuous and aggregate strata.</pros>
          <cons>Low statistical power when few aggregate groups are configured (k=1-2).</cons>
          <statistical_considerations>Type-I error control valid for all k. Power limited for small k but this is inherent, not a defect.</statistical_considerations>
        </approach>
        <approach id="A2" label="Minimum-size fallback" feasibility="high" risk="low">
          <description>Strata with fewer than N features fall back to global max noise.</description>
          <pros>More stable threshold for small strata.</pros>
          <cons>Conservative: global max is typically higher than stratum max, making aggregate features even harder to detect. Works against C1's motivation.</cons>
        </approach>
        <approach id="A3" label="C1 + warning" feasibility="high" risk="low">
          <description>Dedicated stratum plus a log warning when any stratum has fewer than 3 shadow features, noting reduced statistical power.</description>
          <pros>Correct behavior, user is informed of power limitation, no distortion of calibration.</pros>
          <cons>None identified.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A3">C1 as proposed plus a small-stratum warning when n_shadow_in &lt; 3 in the stratum-count log block (shap_utils.py lines 1152-1156). The stratum separation fixes the continuous-stratum inflation documented in the external applied analysis. Limited power in small aggregate strata is inherent and does not warrant a behavioral change.</decision>
    </topic>

    <topic id="T2" title="Multi_regression scaler and transform ordering conflict">
      <summary>For multi_regression tasks, train.py z-scores all targets via StandardScaler (lines 845-856) before the fold loop. The transform's input_transform then overrides y from df_raw inside the fold loop, effectively bypassing the scaler. The scaler is still fitted, persisted to target_scaler.json, and its z-scored y is immediately overridden. The predict/infer guards (and transform_module is None) correctly prevent stale inverse-transformation, but train-time produces wasted computation and a misleading artifact.</summary>
      <research>No external research needed. Analysis from codebase: the transform module is loaded (plan's C3 step 2) before the scaler block (line 845), so the transform_module variable is available for a guard condition.</research>
      <approaches>
        <approach id="A1" label="Guard scaler block" feasibility="high" risk="low">
          <description>Add 'and transform_module is None' to the scaler block condition at train.py line 845. Skips scaler fitting, z-scoring, and target_scaler.json when transforms are active.</description>
          <pros>No wasted computation, no misleading artifact, consistent with predict/infer guards.</pros>
          <cons>Transforms and multi_regression scalers become mutually exclusive. A user who needs both must implement scaling inside their transform script.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Add 'and transform_module is None' guard to the multi_regression scaler block in train.py. Transforms take full ownership of the outcome space when configured. The transform API is general enough for the user to implement z-scoring inside input_transform/output_transform if both are needed.</decision>
    </topic>

    <topic id="T3" title="required_cols validation gap at inference time">
      <summary>The plan validates required_cols only in train.py via validate_transform_config. If the back-transform needs those columns from df_raw at inference time (e.g., baseline scores for residual back-transformation), missing columns in the inference dataset would produce a KeyError from the user's transform script rather than a clear pipeline-level ValueError. The fix has two parts: (1) persist required_cols in the inter-stage artifact, and (2) validate at both predict and infer time.</summary>
      <research>No external research needed. Codebase analysis: transform_config.json (C3 step 5) currently stores {active, file, params}. required_cols is available from config but not persisted in the artifact.</research>
      <approaches>
        <approach id="A1" label="Persist + validate" feasibility="high" risk="low">
          <description>Add required_cols to transform_config.json. Validate against df_raw in both predict.py and infer.py after loading tx_info.</description>
          <pros>Self-describing artifact (no dependency on original config YAML at inference time). Clear error messages at pipeline level. Consistent with validate_cv_config's group_column pattern.</pros>
          <cons>None identified.</cons>
        </approach>
        <approach id="A2" label="Re-read config YAML" feasibility="high" risk="medium">
          <description>Load required_cols from the resolved config at inference time.</description>
          <pros>No artifact change.</pros>
          <cons>Fragile if config is edited between training and inference. Violates the self-describing artifact principle.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Persist required_cols in transform_config.json (C3 step 5). Validate against df_raw in both predict.py (C4 step 2) and infer.py (C5 step 2) immediately after loading tx_info.</decision>
    </topic>

    <topic id="T4" title="SHAP back-transformation under outcome transformations">
      <summary>SHAP values always correctly decompose the model's prediction function regardless of the outcome transformation. SHAP computation never "breaks." However, for affine transforms (y = alpha * y_transformed + beta_i), SHAP values can be validly back-transformed to original-scale units by multiplying by alpha, preserving additivity. For nonlinear transforms, back-transformation breaks SHAP additivity and must not be attempted. The pipeline should support SHAP back-transformation as a config-driven opt-in when the transform is affine, with an upfront affinity check that halts if the user requests back-transformation on a non-affine transform.</summary>
      <research>No external research dispatched. Analysis from SHAP theory (Lundberg and Lee 2017): SHAP values are Shapley values of the model's prediction function. For any model f(X), the decomposition f(x) = E[f(X)] + sum(phi_j(x)) is always valid. If back_transform(pred) = alpha * pred + beta_i (affine), then SHAP_original_j = alpha * SHAP_transformed_j, preserving additivity. For nonlinear back_transform, SHAP_original_j cannot be defined as back_transform(SHAP_transformed_j) because the nonlinearity breaks the additive decomposition.</research>
      <approaches>
        <approach id="A1" label="Document only" feasibility="high" risk="low">
          <description>Document that SHAP values are in the transformed space. No back-transformation capability.</description>
          <pros>Simplest implementation.</pros>
          <cons>Users with alpha != 1 transforms (e.g., standardization) must manually scale SHAP outputs.</cons>
        </approach>
        <approach id="A2" label="Config-driven back_transform_shap" feasibility="high" risk="low">
          <description>New config key back_transform_shap (boolean, default false). When true: upfront affinity test validates the transform is affine and halts if not; first-fold alpha computation provides precise scale factor; SHAP values scaled by alpha in run_shap_pipeline before M/V/GII computation. When false or absent: SHAP in transformed space (current behavior).</description>
          <pros>User-facing option, transparent, correctly handles alpha=1 (no-op), alpha!=1 (scaling), and nonlinear (halt). All downstream artifacts (M, V, GII, microdata, individual reports, spline fits, noise distributions) automatically in original-scale units. Exceedance test invariant (both real and shadow scaled by same alpha).</pros>
          <cons>Adds one config key, one smoke-test extension, one first-fold computation, one parameter to run_shap_pipeline.</cons>
          <statistical_considerations>Scaling both real and shadow SHAP by the same constant alpha preserves the exceedance test: P(alpha * real > alpha * max_shadow) = P(real > max_shadow). Significance results are invariant. M, V, and GII values change by alpha, alpha, and alpha respectively, but their significance does not.</statistical_considerations>
        </approach>
        <approach id="A3" label="Always auto-scale" feasibility="high" risk="low">
          <description>Always compute alpha and scale SHAP when affine, without a config option.</description>
          <pros>No config decision for the user.</pros>
          <cons>Removes user control. A user who wants SHAP in transformed units (e.g., standardized units for cross-study comparison) cannot opt out.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A2">Config-driven back_transform_shap (boolean, default false). Implementation across six sites: (1) fill_config_defaults normalizes to boolean; (2) upfront smoke test extends with affinity check, halts if back_transform_shap=true and not affine; (3) first fold computes precise alpha from real fold metadata; (4) transform_config.json persists is_affine, back_transform_shap, and shap_scale_factor; (5) run_shap_pipeline accepts optional shap_scale_factor, scales both real and shadow SHAP matrices before M/V/GII; (6) predict.py and infer.py read shap_scale_factor from transform_config.json and pass to run_shap_pipeline.</decision>
    </topic>
  </topics>
  <action_items>
    <item priority="P0" target_mode="implement" description="Update the harmonized implement plan (&lt;external_project&gt;_implement_plan_20260821_200000.md) to incorporate T1-T4 decisions: (T1) add small-stratum warning to C1; (T2) add scaler guard to C3; (T3) persist required_cols in transform_config.json and add validation to C4/C5; (T4) add back_transform_shap config key to C2, affinity test + alpha computation to C3, shap_scale_factor plumbing to C4/C5, and scaling parameter to shap_utils.py run_shap_pipeline." />
    <item priority="P0" target_mode="implement" description="Add upfront transform smoke test to C3 (train.py): execution check, shape validation, finiteness, metadata JSON-serializability, output_transform round-trip, and affinity test. Runs once before the fold loop on a 20-row subset of df_raw." />
  </action_items>
  <next_steps>Update the harmonized implement plan with all T1-T4 amendments, then proceed to /implement plan + build. The plan's C1-C6 structure remains intact; T1-T4 add targeted modifications within C1, C2, C3, C4, C5, and a new shap_utils.py change for SHAP scaling. C6 (documentation) should reflect the back_transform_shap option and the smoke test behavior.</next_steps>
</brainstorm_report>
