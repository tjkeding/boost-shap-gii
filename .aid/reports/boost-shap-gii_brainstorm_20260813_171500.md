<brainstorm_report>
  <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-08-13T17:15:00Z" />
  <context_files>
    <file path="boost_shap_gii_techspec_cv_strategy_inner_repeats.md" relevance="Source tech spec from CFTSI-behavioral brainstorm proposing cv_strategy and n_inner_repeats config options" />
    <file path="src/boost_shap_gii/utils.py" relevance="Contains get_cv_splitter() (L163-175), fill_config_defaults(), validation functions" />
    <file path="src/boost_shap_gii/train.py" relevance="Outer CV loop (L866-886), inner CV in Optuna (L525-564), shadow training" />
    <file path="src/boost_shap_gii/predict.py" relevance="Fold replication via get_cv_splitter (L237, L249)" />
    <file path="src/boost_shap_gii/shap_utils.py" relevance="OOF SHAP alignment via get_cv_splitter (L1511-1512)" />
    <file path="src/boost_shap_gii/indiv_reports.py" relevance="Fold assignment reconstruction (L234-237), bootstrap-of-CV inner loop (L365-369)" />
    <file path="src/boost_shap_gii/infer.py" relevance="Inference mode: no fold assignment dependency; SHAP bypasses splitter; indiv_reports loads from train_dir" />
    <file path="example_config_advanced.yaml" relevance="Full config template; needs new keys and comment cleanup" />
    <file path="example_config_minimal.yaml" relevance="Minimal config template; new keys omitted (defaults apply)" />
  </context_files>
  <topics>
    <topic id="T1" title="cv_strategy naming and semantics">
      <summary>Each cv_strategy value maps to exactly one splitter type with no task-dependent logic. "uniform" = KFold always, "stratified" = StratifiedKFold always (with quantile binning for regression, class labels for classification), "group" = GroupKFold always. The existing auto-stratification for classification (y.nunique() &lt; 20) is removed. This is a backward-compatibility break: existing classification configs that omit cv_strategy will get KFold instead of StratifiedKFold. Document in release notes.</summary>
      <research>No external research required; design decision based on codebase analysis of current get_cv_splitter() behavior (utils.py:163-175).</research>
      <approaches>
        <approach id="A1" label="Literal splitter selectors (chosen)" feasibility="high" risk="low">
          <description>Each cv_strategy value maps 1:1 to a splitter class. No conditional logic based on task type or y distribution.</description>
          <pros>Explicit, predictable, no hidden behavior. Config says exactly what it does.</pros>
          <cons>Backward-compatibility break for classification users who relied on auto-stratification.</cons>
        </approach>
        <approach id="A2" label="Auto-detect with override (rejected)" feasibility="high" risk="low">
          <description>Rename "uniform" to "auto" to preserve task-dependent behavior; "stratified" and "group" override.</description>
          <pros>Backward compatible.</pros>
          <cons>Hidden behavior under "auto"; user explicitly rejected this.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">User directive: "We don't want this option to do anything automatically." Each value is a literal splitter selector regardless of task type. Classification users who want stratification must explicitly set cv_strategy: "stratified".</decision>
    </topic>

    <topic id="T2" title="StratifiedKFold for regression: binning mechanics and encapsulation">
      <summary>When cv_strategy="stratified" and the task is regression, the continuous target is discretized into K quantile bins (K = cv_folds) via pd.qcut with pd.cut fallback for tied values. Binning is encapsulated inside a _StratifiedRegressionKFold wrapper class returned by get_cv_splitter(), so callers never handle binning directly. For classification, StratifiedKFold uses class labels directly (no binning).</summary>
      <research>R1 (Boehmke &amp; Greenwell 2020): quantile binning is the standard applied-ML approach. rsample defaults to 4 bins (quartiles) independent of fold count. No theoretical optimum for bins-to-folds ratio exists. pd.qcut edge case (tied values) handled by duplicates='drop' + pd.cut fallback (pandas GitHub #16328). scikit-learn PR #14560: declined to merge dedicated BinnedStratifiedKFold; no consensus on approach.</research>
      <approaches>
        <approach id="A1" label="K bins with encapsulated wrapper (chosen)" feasibility="high" risk="low">
          <description>Number of bins equals number of CV folds. _StratifiedRegressionKFold wrapper intercepts split() and bins y internally via stratify_labels_for_regression(). Callers pass original continuous y to split(); wrapper handles discretization.</description>
          <pros>Most granular stratification for the given fold count. Wrapper eliminates call-site binning errors. pd.qcut with pd.cut fallback handles tied values.</pros>
          <cons>With large K and small N, bins may be sparse (mitigated by duplicates='drop').</cons>
          <statistical_considerations>K bins ensures each fold draws proportionally from each quantile slice of the target distribution, directly addressing the fold-composition variance that motivated this feature (CFTSI childchild: one fold with SD(y)=7.58 vs. overall 13.0). The pd.cut fallback produces equal-width bins when quantile boundaries coincide, which is less optimal for skewed distributions but still reduces extreme fold imbalance.</statistical_considerations>
        </approach>
        <approach id="A2" label="Fixed 4 bins (rsample convention, rejected)" feasibility="high" risk="low">
          <description>Always use 4 quartile bins regardless of fold count.</description>
          <pros>Robust, citable convention (rsample).</pros>
          <cons>Coarser stratification when K &gt; 4; does not scale with fold count.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">K bins with encapsulated wrapper. stratify_labels_for_regression() helper function in utils.py. _StratifiedRegressionKFold wrapper returned by get_cv_splitter() when cv_strategy="stratified" and task is regression.</decision>
    </topic>

    <topic id="T3" title="GroupKFold design: group column, wrappers, fold balance, full call-site inventory">
      <summary>GroupKFold uses config["modeling"]["group_column"] to partition folds. The group column is excluded from feature candidates before FeatureSelector.fit(). A _GroupKFoldWrapper encapsulates groups at construction, forwarding them in split(). Unbalanced folds (max/min ratio &gt; 2.0) trigger a WARNING. All 6 CV call sites across the pipeline are addressed: 4 via get_cv_splitter() wrappers, 1 refactored to use get_cv_splitter() (train.py inner CV), 1 refactored with GroupKFold-to-KFold fallback (indiv_reports.py bootstrap-of-CV, since bootstrap resampling breaks group structure). n_inner_repeats &gt; 1 with cv_strategy="group" is supported via custom _RepeatedGroupKFold (revised from initial hard-error proposal).</summary>
      <research>GroupKFold in sklearn has no shuffle or random_state parameters; fold assignments are deterministic based on group membership. No RepeatedGroupKFold exists in sklearn. Custom implementation shuffles group-to-fold mapping per repeat while preserving the constraint that all members of a group stay in the same fold.</research>
      <approaches>
        <approach id="A1" label="Wrapper + custom RepeatedGroupKFold (chosen)" feasibility="high" risk="low">
          <description>_GroupKFoldWrapper stores groups at construction. _RepeatedGroupKFold permutes group-to-fold mapping per repeat with seeded RNG. Bootstrap-of-CV falls back to KFold for group CV.</description>
          <pros>Composes with n_inner_repeats across all three strategies uniformly. Wrapper keeps callers unchanged. Bootstrap fallback is methodologically sound (resampling breaks group structure).</pros>
          <cons>Custom _RepeatedGroupKFold requires implementation and testing (no sklearn equivalent).</cons>
        </approach>
        <approach id="A2" label="Hard error for group + repeats (rejected)" feasibility="high" risk="low">
          <description>Raise ValueError when cv_strategy="group" and n_inner_repeats &gt; 1.</description>
          <pros>Simple; avoids custom implementation.</pros>
          <cons>User explicitly required n_inner_repeats to work across all three strategies equally.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">_GroupKFoldWrapper encapsulates groups. _RepeatedGroupKFold shuffles group-to-fold mapping per repeat. Bootstrap-of-CV (indiv_reports.py site 6) falls back to KFold for group CV. Unbalanced fold WARNING at ratio &gt; 2.0. group_column excluded from feature candidates in train.py.</decision>
    </topic>

    <topic id="T3_callsites" title="Complete CV call-site inventory (6 sites)">
      <summary>The tech spec identified 3 call sites; brainstorm identified 6 total. All require updates.</summary>
      <research>Grep across src/boost_shap_gii/ for get_cv_splitter, KFold, StratifiedKFold, and split() calls.</research>
      <approaches>
        <approach id="A1" label="Full-pipeline update (chosen)" feasibility="high" risk="medium">
          <description>Site 1: train.py outer CV (get_cv_splitter). Site 2: train.py inner CV (refactor from direct KFold/StratifiedKFold to get_cv_splitter). Site 3: predict.py (load fold_assignments.json). Site 4: shap_utils.py (load fold_assignments.json). Site 5: indiv_reports.py _reconstruct_fold_assignments (load fold_assignments.json). Site 6: indiv_reports.py bootstrap-of-CV (refactor to use get_cv_splitter with GroupKFold-to-KFold fallback).</description>
          <pros>Complete coverage; no silent alignment bugs.</pros>
          <cons>Larger implementation scope than tech spec anticipated.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">All 6 sites updated. Sites 3-5 switch from splitter reconstruction to fold_assignments.json artifact loading.</decision>
    </topic>

    <topic id="T4" title="n_inner_repeats: implementation, cost, bounds">
      <summary>When n_inner_repeats &gt; 1, the inner CV uses sklearn's RepeatedKFold (uniform), RepeatedStratifiedKFold (stratified), or custom _RepeatedGroupKFold (group). The Optuna objective iterates over all K*R splits and averages scores. Default=1 (no repeats). WARNING when n_inner_repeats &gt; 10 (diminishing returns per Vanwinckelen &amp; Blockeel 2012). WARNING when total inner fits per outer fold exceeds 5000.</summary>
      <research>R2 (Krstajic et al. 2014): repeated CV is an established variance-reduction strategy for HPO; no closed-form optimal repeat count. Kim (2009): repeated 10-fold outperforms .632+ bootstrap for adaptive classifiers. Vanwinckelen &amp; Blockeel (2012): repeated CV is pseudo-replication (reduces estimator variance, not bias). sklearn RepeatedKFold/RepeatedStratifiedKFold: default n_repeats=10. No TPE-specific literature found.</research>
      <approaches>
        <approach id="A1" label="sklearn Repeated* + custom _RepeatedGroupKFold (chosen)" feasibility="high" risk="low">
          <description>Use idiomatic sklearn classes for uniform/stratified; custom class for group. Single iterator yields K*R splits. Optuna objective averages all K*R fold scores.</description>
          <pros>Idiomatic; seed management handled by sklearn; composes with all three cv_strategy values; mathematically equivalent to manual repeat loop.</pros>
          <cons>_RepeatedGroupKFold is custom (no sklearn equivalent).</cons>
          <statistical_considerations>Repeated CV reduces the variance of the cross-validation estimate but does not increase the effective independent information (Vanwinckelen &amp; Blockeel 2012). For HPO, this translates to a smoother Optuna objective landscape (fewer pathological fold-composition artifacts) at R-fold computational cost. Diminishing returns beyond R~5-10 in practice.</statistical_considerations>
        </approach>
        <approach id="A2" label="Manual repeat loop with seed offsetting (tech spec, rejected)" feasibility="high" risk="low">
          <description>Wrap inner CV in a for-loop over repeats with seed = seed + fold_idx + 1 + rep * 1000.</description>
          <pros>Explicit control; per-repeat scores visible.</pros>
          <cons>Custom seed management; more code; does not compose as cleanly with wrapper approach.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">RepeatedKFold/RepeatedStratifiedKFold (sklearn) for uniform/stratified. Custom _RepeatedGroupKFold for group. Default n_inner_repeats=1. WARNING at n_inner_repeats &gt; 10. WARNING when total inner fits per outer fold &gt; 5000.</decision>
    </topic>

    <topic id="T5" title="Cross-module fold alignment: persist fold assignments">
      <summary>train.py persists fold_assignments.json (array of length N, element i = fold index for sample i as validation). predict.py, shap_utils.py, and indiv_reports.py load this artifact instead of re-splitting. Eliminates alignment dependencies on data identity, sklearn determinism, and stratification replication. infer.py is unaffected (inference dataset is not partitioned; fold assignments are only needed for training data OOB accounting, loaded from train_dir).</summary>
      <research>No external research needed; architectural decision based on analysis of fragility in the reconstruction approach when adding data-dependent binning (stratified regression) and group column extraction (GroupKFold).</research>
      <approaches>
        <approach id="A1" label="Persist fold assignments (chosen)" feasibility="high" risk="low">
          <description>train.py saves fold_assignments.json after outer CV loop. Downstream modules load it. _reconstruct_fold_assignments() refactored to read artifact.</description>
          <pros>Robust; immune to data/version drift; eliminates alignment bug class.</pros>
          <cons>New artifact dependency; minor disk overhead.</cons>
        </approach>
        <approach id="A2" label="Extend reconstruction approach (rejected)" feasibility="high" risk="medium">
          <description>Keep re-splitting via get_cv_splitter(); ensure wrappers produce identical results.</description>
          <pros>No new artifact.</pros>
          <cons>Fragile; depends on identical data loading, sklearn determinism, and identical quantile boundaries.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Persist fold_assignments.json. Consumers: predict.py, shap_utils.py, indiv_reports.py (all from run_dir or train_dir). infer.py unaffected.</decision>
    </topic>

    <topic id="T6" title="Config validation and fill_config_defaults integration">
      <summary>Two new defaults in fill_config_defaults(): modeling.cv_strategy="uniform" and modeling.tuning.n_inner_repeats=1. New validate_cv_config() function in utils.py validates cv_strategy enum, group_column existence (when cv_strategy="group"), n_inner_repeats bounds, and interaction constraints. Example configs simplified: one-line inline comments only, theory/derivations moved to INPUT_SPECIFICATION.md, no commented-out blocks.</summary>
      <research>No external research needed.</research>
      <approaches>
        <approach id="A1" label="Defaults + validator + config cleanup (chosen)" feasibility="high" risk="low">
          <description>_setdefault_nested for both keys. validate_cv_config() called after data loading in train.py. Configs simplified per one-line-comment rule.</description>
          <pros>Clean validation surface. Configs become scannable. Theory in the right place (docs).</pros>
          <cons>None significant.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Defaults via _setdefault_nested. validate_cv_config() in utils.py. Example configs simplified: one-line inline comments, no multi-paragraph explanations, no commented-out blocks.</decision>
    </topic>
  </topics>
  <action_items>
    <item priority="P0" target_mode="implement" description="Add cv_strategy config option: 'uniform' (KFold), 'stratified' (StratifiedKFold with _StratifiedRegressionKFold wrapper for regression), 'group' (GroupKFold with _GroupKFoldWrapper). Refactor get_cv_splitter() signature to (config, y, seed_override=None, groups=None). Add stratify_labels_for_regression() helper. Add wrapper classes. Update all 6 call sites (train.py outer/inner CV, predict.py, shap_utils.py, indiv_reports.py fold reconstruction, indiv_reports.py bootstrap-of-CV). Remove auto-stratification for classification." />
    <item priority="P0" target_mode="implement" description="Add n_inner_repeats config option: use RepeatedKFold (uniform), RepeatedStratifiedKFold (stratified), or custom _RepeatedGroupKFold (group) when n_inner_repeats > 1. Refactor Optuna objective to iterate over K*R splits. Add cost/diminishing-returns WARNINGs." />
    <item priority="P0" target_mode="implement" description="Persist fold_assignments.json from train.py. Refactor predict.py, shap_utils.py, and indiv_reports.py to load fold assignments from artifact instead of re-splitting." />
    <item priority="P0" target_mode="implement" description="Add validate_cv_config() in utils.py. Add cv_strategy and n_inner_repeats defaults to fill_config_defaults(). Add group_column exclusion logic in train.py." />
    <item priority="P1" target_mode="implement" description="Simplify example_config_advanced.yaml and example_config_minimal.yaml: one-line inline comments only, no multi-paragraph explanations, no commented-out blocks. Add cv_strategy and n_inner_repeats entries to advanced config." />
    <item priority="P1" target_mode="document" description="Update INPUT_SPECIFICATION.md and README.md with cv_strategy and n_inner_repeats documentation. Note backward-compatibility break for classification auto-stratification removal in release notes." />
    <item priority="P1" target_mode="test" description="Unit tests for get_cv_splitter() with all three strategies (regression + classification). Tests for _StratifiedRegressionKFold, _GroupKFoldWrapper, _RepeatedGroupKFold. Tests for stratify_labels_for_regression() including tied-value fallback. Tests for validate_cv_config(). Integration test for n_inner_repeats > 1. Backward-compatibility test: omitted keys produce identical KFold behavior. fold_assignments.json round-trip test." />
  </action_items>
  <next_steps>Proceed to /implement (plan + build) to execute the P0 and P1 action items across the full pipeline. The implementation scope spans utils.py, train.py, predict.py, shap_utils.py, indiv_reports.py, both example configs, and documentation. Follow with /test to validate all new code paths and backward compatibility.</next_steps>
</brainstorm_report>
