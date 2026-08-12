<brainstorm_report>
  <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-08-12T14:41:51Z" />
  <context_files>
    <file path="boost-shap-gii_cr_20260812_142729.md" relevance="Source CR report containing F1, F2, F3 findings and proposed fixes" />
    <file path="src/boost_shap_gii/shap_utils.py" relevance="Contains _nan_safe_fdr (F1, lines 1168-1196) and inference-mode X_stacked tiling (F3, lines 1402-1404)" />
    <file path="src/boost_shap_gii/indiv_reports.py" relevance="Contains _CATBOOST_REFIT_BLOCKLIST (F2, lines 74-86) and refit loop (lines 334-352)" />
  </context_files>
  <topics>
    <topic id="T1" title="F1: _nan_safe_fdr docstring vs behavior fix">
      <summary>
        The _nan_safe_fdr docstring incorrectly claims NaN p-values are "excluded from the BH
        denominator." The code replaces them with 1.0 placeholders included in the denominator,
        making the correction slightly more conservative than documented. Two fix options were
        evaluated: (A) docstring-only correction, (B) true denominator exclusion by subsetting
        p-values before multipletests. Option B would change statistical output in the
        less-conservative direction (reducing m from "all effects" to "testable effects only"),
        which, while more principled per Benjamini and Hochberg (1995), would alter behavior
        against which the simulation study baseline was run.
      </summary>
      <research>No research dispatch required; decision is deterministic from BH procedure specification.</research>
      <approaches>
        <approach id="A1" label="Docstring-only fix" feasibility="high" risk="low">
          <description>Correct the docstring to accurately describe the NaN-as-1.0 placeholder behavior. Zero behavioral change.</description>
          <pros>No statistical output change; no interaction risk; preserves simulation study baseline</pros>
          <cons>Leaves a marginally over-conservative correction in place (safe direction)</cons>
          <statistical_considerations>The conservative bias magnitude is proportional to (NaN count / total effects), typically less than 5%. Safe direction.</statistical_considerations>
        </approach>
        <approach id="A2" label="True denominator exclusion" feasibility="high" risk="low">
          <description>Pass only non-NaN p-values to multipletests, re-index results. Changes m to reflect testable effects only.</description>
          <pros>More statistically principled per BH (1995)</pros>
          <cons>Changes output in less-conservative direction; invalidates simulation study baseline comparison</cons>
          <statistical_considerations>Would reduce m, making q-values slightly smaller. Magnitude depends on NaN count, which is typically very small.</statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Docstring-only fix. Current behavior is defensible (conservative direction) and the simulation study baseline was run against it.</decision>
    </topic>

    <topic id="T2" title="F2: CatBoost refit blocklist defense">
      <summary>
        The _CATBOOST_REFIT_BLOCKLIST (12 entries) is empirically coupled to CatBoost 1.2.10.
        Future versions may add new internal-only params to get_all_params() that break the
        refit constructor in the B*K bootstrap loop. Three options evaluated: (A) try/except
        inside the inner loop with session-level cache, (B) pre-flight probe before the B-loop
        at the frozen_hps construction site, (C) pin CatBoost version. Option B concentrates
        discovery outside the hot loop, logs once, and handles multi-parameter discovery cleanly.
      </summary>
      <research>No research dispatch required; decision is deterministic from code structure analysis.</research>
      <approaches>
        <approach id="A1" label="Try/except in inner loop" feasibility="high" risk="low">
          <description>Wrap constructor call (lines 348-351) in try/except TypeError with module-level cache of discovered params.</description>
          <pros>Self-healing; zero overhead on clean runs</pros>
          <cons>Discovery happens inside hot B*K loop on first hit; TypeError message parsing is fragile</cons>
        </approach>
        <approach id="A2" label="Pre-flight probe" feasibility="high" risk="low">
          <description>At lines 996-998, after constructing frozen_hps, do a single trial CatBoost construction in a try/except loop (capped at 5 retries). Discover all offending params before the bootstrap loop starts, strip from all frozen_hps[k] entries, log a single RuntimeWarning.</description>
          <pros>All discovery outside hot loop; single warning; handles multi-parameter case; clean separation of concerns</pros>
          <cons>Trial construction creates a transient CatBoost object (minimal overhead, never fit)</cons>
        </approach>
        <approach id="A3" label="Pin CatBoost version" feasibility="high" risk="low">
          <description>Pin catboost==1.2.10 in environment.yaml.</description>
          <pros>Zero code change</pros>
          <cons>Blocks upgrades for bug fixes and performance; fragile long-term</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A2">Pre-flight probe at frozen_hps construction (lines 996-998). Helper function _probe_and_strip_refit_params(frozen_hps, task) does trial construction once, discovers unblocked internal params via try/except TypeError loop capped at 5 retries, strips from all frozen_hps[k] entries, logs a single RuntimeWarning listing discovered params and CatBoost version.</decision>
    </topic>

    <topic id="T3" title="F3: Inference-mode X_stacked tiling fix">
      <summary>
        In inference mode, X_stacked tiles fold 0's X_full K times, creating a mismatch between
        shadow feature values and shadow SHAP values for folds 1..K-1. The fix is a one-line
        change: replace pd.concat([chunks_X[0]] * n_folds, ignore_index=True) with
        pd.concat(chunks_X, ignore_index=True). All seven downstream consumers of X_stacked
        were traced and verified: (1) _aggregate_effects uses only real features (identical
        across folds), (2) nan_mask correctly reflects per-fold NaN patterns, (3) X_vals in
        bootstrap worker gets correct per-fold shadow values, (4) X_real_for_micro uses only
        real features, (5) X_display is passed separately, (6) column alignment is guaranteed
        by identical column names, (7) memory footprint is unchanged.
      </summary>
      <research>No research dispatch required; all interactions verified from direct code tracing.</research>
      <approaches>
        <approach id="A1" label="One-line concat fix" feasibility="high" risk="low">
          <description>Replace pd.concat([chunks_X[0]] * n_folds, ignore_index=True) with pd.concat(chunks_X, ignore_index=True) at shap_utils.py line 1404. Update the accompanying comment.</description>
          <pros>Eliminates anti-conservative bias in shadow V; zero interaction risk; mirrors OOF-mode pattern (line 1418); trivial implementation</pros>
          <cons>None identified</cons>
          <statistical_considerations>Shadow V estimates become correctly paired, eliminating the (negligible but anti-conservative) deflation of the noise threshold for the V exceedance test. Real effect V computation is unaffected.</statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">One-line fix. Replace the fold-0 tiling with pd.concat(chunks_X, ignore_index=True). Update comment.</decision>
    </topic>
  </topics>
  <action_items>
    <item priority="P2" target_mode="implement" description="T1/F1: Correct _nan_safe_fdr docstring (shap_utils.py:1168-1196) to accurately describe NaN-as-1.0 placeholder behavior included in BH denominator. No behavioral change." />
    <item priority="P2" target_mode="implement" description="T2/F2: Add _probe_and_strip_refit_params helper at indiv_reports.py lines 996-998 for pre-flight CatBoost constructor probe. Try/except TypeError loop capped at 5 retries, strips discovered params from all frozen_hps[k], logs single RuntimeWarning with param names and CatBoost version." />
    <item priority="P2" target_mode="implement" description="T3/F3: Replace X_stacked tiling at shap_utils.py line 1404 from pd.concat([chunks_X[0]] * n_folds, ignore_index=True) to pd.concat(chunks_X, ignore_index=True). Update comment." />
  </action_items>
  <next_steps>/implement to execute all three changes, then /test to validate.</next_steps>
</brainstorm_report>
