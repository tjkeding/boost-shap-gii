<brainstorm_report>
  <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-08-11T21:46:42Z" />
  <context_files>
    <file path="tests/test_build_20260507.py" relevance="C18 TestCobbDouglasAnchorPresence class: 3 of 4 tests fail post-Cobb-Douglas removal (Session 12)" />
    <file path="TO_IMPLEMENT_IN_BOOST_SHAP_GII.md" relevance="Aggregate SHAP spec (Sections 1-10) and _CATBOOST_USER_PARAM_ALLOWLIST additions (Section 11)" />
    <file path="src/boost_shap_gii/shap_utils.py" relevance="Target for _aggregate_effects() insertion; current GII geometric-mean framing (no Cobb-Douglas)" />
    <file path="src/boost_shap_gii/train.py" relevance="Shadow generation architecture (lines 915-930): per-fold independent permutation; requires block-permutation modification for S1" />
    <file path="src/boost_shap_gii/predict.py" relevance="Calls run_shap_pipeline; needs aggregate_shap config passthrough" />
    <file path="src/boost_shap_gii/infer.py" relevance="Calls run_shap_pipeline (line 533); needs aggregate_shap config passthrough (spec gap: omitted from Section 9.1)" />
    <file path="src/boost_shap_gii/indiv_reports.py" relevance="_CATBOOST_USER_PARAM_ALLOWLIST (lines 74-94): 18-entry allowlist, missing model_size_reg and max_ctr_complexity" />
    <file path="INPUT_SPECIFICATION.md" relevance="Retains Hill (1910) and Goldstein citations in geometric-mean framing; no Cobb-Douglas" />
    <file path="README.md" relevance="GII Interpretation section: geometric mean only, no Hill/Goldstein citations" />
  </context_files>
  <topics>
    <topic id="T1" title="P2 C18 Test Re-expression Strategy">
      <summary>The TestCobbDouglasAnchorPresence class (4 tests in tests/test_build_20260507.py) was written in Session 9 to guard the Cobb-Douglas framing of GII. In Session 12, all Cobb-Douglas references were removed per user directive. Three tests now fail (aligned-classification: product change, not test defect). The fourth test (quarantine: no "calibration study" / "in prep" in public files) still passes. Re-expression updates the three failing tests to assert the current geometric-mean framing with file-specific assertion sets reflecting the intentional asymmetry (README lacks Hill/Goldstein citations; INPUT_SPECIFICATION and shap_utils.py retain them).</summary>
      <research>No external research required. Codebase inspection confirmed the current state of all three target files.</research>
      <approaches>
        <approach id="A1" label="Re-express to current framing" feasibility="high" risk="low">
          <description>Rename class to TestGIIFramingPresence. Re-express three failing tests to assert current geometric-mean framing with file-specific assertions: (1) shap_utils.py: assert "geometric mean", "Hill (1910)", "Goldstein"; (2) INPUT_SPECIFICATION.md: assert "Decision-theoretic interpretation", "geometric mean", "Hill (1910)", "Goldstein"; (3) README.md: assert "GII Interpretation", "geometric mean", "sqrt(M" (no Hill/Goldstein by design). Quarantine test unchanged. No addition of "Cobb-Douglas" to the forbidden-term list.</description>
          <pros>Tests assert what the codebase actually specifies; file-specific assertions reflect intentional asymmetry; quarantine guard preserved; strictly strengthened postconditions</pros>
          <cons>None identified</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Re-express the three failing C18 tests to assert the current geometric-mean framing. User explicitly approved the approach and declined adding "Cobb-Douglas" to the quarantine forbidden-term list.</decision>
    </topic>
    <topic id="T2" title="Aggregate SHAP: Statistical Validity of Additive Aggregation">
      <summary>The TO_IMPLEMENT spec proposes computing group-level M, V, and GII by summing individual SHAP values (main effects and pairwise interactions) within user-defined feature groups, then applying the existing Boruta exceedance pipeline to the aggregated values. Three research agents investigated the statistical foundations of this approach across three dimensions: main-effect summation validity, interaction value aggregation, and shadow-based null calibration under aggregation.</summary>
      <research>
        R1 (SHAP grouped aggregation): Jullum, Redelmeier and Aas (2021, arXiv:2106.12228) prove that post-hoc summation of individual SHAP values equals the formally computed group-level Shapley value (groupShapley) only under two jointly sufficient conditions: partial additive separability of f(x) across the group partition, and mutual statistical independence between groups. These conditions are rarely met in practice. Au et al. (2022, Data Mining and Knowledge Discovery 36:1401-1450) note that the formally correct group-level Shapley value requires redefining the coalition value function with groups as atomic players, which is not supported by TreeSHAP. Owen (1977) provides the classical game-theoretic alternative (Owen values) for predefined coalition structures.

        R2 (SHAP interaction aggregation): Grabisch and Roubens (1999, Int. J. Game Theory) define the axiomatically grounded interaction index for coalitions of any size via discrete-derivative/Mobius expansion; this is not equivalent to summing pairwise two-player interaction indices. Sundararajan, Dhamdhere and Agarwal (2020, ICML) show via the Shapley-Taylor interaction index that pairwise terms do not exhaust interaction structure when 3+ features jointly interact. However, the full SHAP interaction matrix satisfies exact efficiency (Lundberg et al. 2020), so block-partitioning the matrix into within-group and between-group sums is arithmetically valid, capturing all pairwise interaction mass.

        R3 (Shadow null calibration): Au et al. (2022) establish that group-level importance cannot be validly obtained by summing K independently-permuted univariate importance scores because the group total depends on within-group codependencies. Kursa and Rudnicki (2010) confirm that Boruta was designed for single-feature max-shadow comparison, not summed-shadow-group statistics. Rotari et al. (2024, Quality and Reliability Engineering International) document that high correlation among inputs causes Boruta-style shadow testing to overestimate importance. The consensus is that independently permuting and then summing produces a miscalibrated (anti-conservative) null when group constituents are correlated.
      </research>
      <approaches>
        <approach id="S1" label="Block-permute shadow features" feasibility="high" risk="low">
          <description>Modify shadow generation in train.py to apply a shared row-permutation index to all features within a defined aggregate group, preserving within-group correlation in the shadow. Ungrouped features retain independent permutation. Shadow SHAP values for the block-permuted group are then summed to produce the correctly calibrated aggregate null. Individual-feature Boruta tests for grouped features become slightly more conservative (shadow preserves correlation), which is correct behavior.</description>
          <pros>Theoretically correct null distribution; preserves within-group dependence structure; compatible with existing per-fold shadow architecture (lines 915-930 of train.py); individual-feature test conservatism is a feature</pros>
          <cons>Requires modifying shadow generation in train.py (currently "unchanged" in spec); aggregate_shap config must be available at training time; slight increase in shadow generation complexity</cons>
        </approach>
        <approach id="S2" label="Aggregates as descriptive statistics only" feasibility="high" risk="low">
          <description>Compute aggregate M, V, GII as descriptive summary statistics without Boruta exceedance testing. Significance determined only at the individual-feature level.</description>
          <pros>Simple; avoids calibration problem entirely</pros>
          <cons>No formal aggregate-level significance testing; users cannot directly test group importance</cons>
        </approach>
        <approach id="S3" label="Bootstrap CI only (no exceedance)" feasibility="med" risk="med">
          <description>Use bootstrap confidence intervals on aggregate GII as the inference mechanism instead of Boruta exceedance.</description>
          <pros>Uses existing bootstrap infrastructure</pros>
          <cons>Different inference framework from individual effects; bootstrap CI tests stability, not whether effect exceeds chance</cons>
        </approach>
        <approach id="S4" label="Defer aggregate significance" feasibility="high" risk="low">
          <description>Implement aggregate computation now; defer significance testing to the simulation study.</description>
          <pros>Gets descriptive statistics into pipeline immediately</pros>
          <cons>Two-phase rollout; pipeline in intermediate state</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="S1">
        Block-permute shadow features for aggregate groups (S1). Three sub-decisions locked:

        (1) Main-effect summation: defensible as descriptive statistic; document as "post-hoc additive decomposition of individual SHAP contributions within user-defined groups," not as coalition-consistent group Shapley value. Cite Jullum et al. (2021).

        (2) Interaction summation: arithmetically valid partition of the order-2 interaction matrix capturing all pairwise interaction mass within/between groups. Document as pairwise-only; higher-order (3-way+) interactions are not captured. Cite Grabisch and Roubens (1999).

        (3) Shadow calibration: block-permutation at shadow generation time in train.py. All features within an aggregate group receive the same row-permutation index, preserving within-group correlation. Summed shadow SHAP for the block-permuted group serves as the correctly calibrated null. Cite Au et al. (2022).

        Documentation framing: aggregate M, V, and GII are "post-hoc additive summaries of individual SHAP contributions within user-defined groups."
      </decision>
    </topic>
    <topic id="T3" title="Aggregate SHAP: Design Review">
      <summary>Evaluation of the TO_IMPLEMENT spec's architectural design (V axis convention, effect type taxonomy, output format, call-site insertion, backward compatibility) against the existing pipeline. Two design gaps identified and resolved: nominal features in groups and overlapping group membership.</summary>
      <research>No external research required. Design evaluation conducted against the existing pipeline architecture.</research>
      <approaches>
        <approach id="D1" label="Prohibit nominal features in aggregate groups" feasibility="high" risk="low">
          <description>Add a validation rule: error if any constituent feature in an aggregate group has feature_type == "nominal". Only continuous and ordinal features are permitted. The additive V axis (group total) is undefined for nominal codes, and the semantic interpretation ("total group severity/score") breaks down for nominal features.</description>
          <pros>Prevents meaningless V axis computation; prevents uninterpretable group totals; clean error at validation time</pros>
          <cons>Restricts group composition; users with nominal features cannot form groups</cons>
        </approach>
        <approach id="D2" label="Disjoint group membership" feasibility="high" risk="low">
          <description>Change spec Section 2.2 from "Feature appears in multiple groups: Allowed" to "Error." Each feature may belong to at most one aggregate group. Eliminates the overlapping-member problem in between-group interaction sums (where self-interaction terms would appear) and prevents SHAP contribution double-counting across aggregates.</description>
          <pros>Eliminates fi == gj edge case in between-group interactions; prevents double-counting; simpler implementation</pros>
          <cons>Prevents hierarchical groupings (e.g., subscale within full-scale); users must choose one level of grouping</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="D1,D2">
        Both design changes adopted:
        (1) Nominal features prohibited from aggregate groups (validation error). Only continuous and ordinal features permitted.
        (2) Disjoint group membership enforced (validation error if a feature appears in multiple groups). User rationale: "having items aggregate to more [than one] sum is incorrect in the context of interactions."
        Additional design elements confirmed without modification: output format (is_aggregate column, unconditional), call-site insertion (after fold-merge, before _run_bootstrap_pipeline), backward compatibility (aggregate_shap key optional; absence produces identical behavior).
        Spec gap identified: infer.py omitted from Section 9.1 changes table; needs identical aggregate_shap config passthrough as predict.py.
      </decision>
    </topic>
    <topic id="T4" title="_CATBOOST_USER_PARAM_ALLOWLIST Maintenance Strategy">
      <summary>The immediate need is adding model_size_reg and max_ctr_complexity to the allowlist in indiv_reports.py (currently 18 entries). The broader question is whether the allowlist approach should be replaced with a more systematic strategy to prevent recurrence as new projects use additional CatBoost parameters.</summary>
      <research>No external research required. Design evaluation based on CatBoost parameter taxonomy and failure-mode analysis.</research>
      <approaches>
        <approach id="B1" label="Strategy A: Ad-hoc allowlist (status quo)" feasibility="high" risk="med">
          <description>Add entries as projects discover gaps. Reviewed per addition.</description>
          <pros>Minimal code change; reviewed per addition</pros>
          <cons>Reactive; silent parameter-drop failures; each new project may hit the same issue</cons>
        </approach>
        <approach id="B2" label="Strategy B: Blocklist of dangerous keys" feasibility="high" risk="low">
          <description>Enumerate the small set of CatBoost internal/runtime keys that must NOT be passed to refit (cat_features, task_type, devices, thread_count, etc.) and allow everything else. Scoped narrowly per user directive ("be very selective about what is blocked").</description>
          <pros>Future-proof; new CatBoost parameters allowed by default; failure mode (visible runtime error from missed blocklist entry) strictly better than silent parameter drop</pros>
          <cons>Must identify dangerous keys correctly; missed dangerous key causes runtime conflict rather than silent drop</cons>
        </approach>
        <approach id="B3" label="Strategy C: Dynamic CatBoost introspection" feasibility="low" risk="high">
          <description>Query CatBoost API at runtime to build allowlist dynamically.</description>
          <pros>Always in sync with installed version</pros>
          <cons>Fragile across versions; some internal params appear in get_params(); adds runtime dependency</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="B2">Strategy B (blocklist) adopted. Blocklist will be narrowly scoped to only those CatBoost internal keys that cause runtime conflicts or silent corruption when passed to a refit Pool/model. User directive: "be very selective about what is blocked (we want maximum flexibility for those users that know what they're doing)." All user-facing hyperparameters, including unusual ones, pass through by default.</decision>
    </topic>
  </topics>
  <action_items>
    <item priority="P1" target_mode="implement" description="T1: Re-express 3 failing C18 tests in tests/test_build_20260507.py to assert current geometric-mean framing. Rename class to TestGIIFramingPresence. File-specific assertion sets per the decided approach." />
    <item priority="P1" target_mode="implement" description="T2/T3: Implement aggregate SHAP feature per TO_IMPLEMENT spec Sections 1-10 with the following modifications: (a) block-permutation for shadow generation of aggregate groups in train.py (S1); (b) nominal features prohibited from aggregate groups (validation error); (c) disjoint group membership enforced (validation error); (d) infer.py added to config passthrough alongside predict.py; (e) documentation frames aggregates as post-hoc additive summaries, not coalition-consistent group Shapley values, citing Jullum et al. (2021), Grabisch and Roubens (1999), and Au et al. (2022)." />
    <item priority="P1" target_mode="implement" description="T4: Replace _CATBOOST_USER_PARAM_ALLOWLIST in indiv_reports.py with a narrowly-scoped blocklist of dangerous internal CatBoost keys. All user-facing hyperparameters pass through by default." />
    <item priority="P2" target_mode="test" description="Design test suite covering: (a) re-expressed C18 tests; (b) aggregate SHAP validation rules (nominal prohibition, disjoint groups, empty groups, name collisions); (c) block-permutation shadow generation; (d) aggregate M/V/GII computation correctness; (e) blocklist behavior for refit parameter extraction." />
    <item priority="P2" target_mode="document" description="Update INPUT_SPECIFICATION.md and README.md with aggregate_shap config section documentation; update example_config_advanced.yaml with commented aggregate_shap example; document aggregate framing as post-hoc additive summaries with literature citations." />
  </action_items>
  <next_steps>The recommended sequence is /implement (plan phase covering all three action items: T1 re-expression, T2/T3 aggregate SHAP with spec modifications, T4 blocklist refactor), then /test (design phase for the new test suite), then /document, then /run-local to validate, then /publish.</next_steps>
</brainstorm_report>
