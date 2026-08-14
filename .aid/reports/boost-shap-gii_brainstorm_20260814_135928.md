<brainstorm_report>
  <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-08-14T13:59:28Z" />
  <context_files>
    <file path="boost-shap-gii_cr_20260814_133000.md" relevance="Input CR report with 8 findings (F1-F8); F1, F4, F7 triaged to brainstorm discussion" />
    <file path="src/boost_shap_gii/shap_utils.py" relevance="SHAP bootstrap (_run_bootstrap_pipeline lines 986-1014), cluster-bootstrap machinery (989-1012), _nan_safe_fdr (1168-1198)" />
    <file path="src/boost_shap_gii/utils.py" relevance="_GroupKFoldWrapper (191-202), _RepeatedGroupKFold (204-226), get_cv_splitter (229-244)" />
    <file path="src/boost_shap_gii/train.py" relevance="Uncommitted cv_strategy changes; GroupKFold balance check (892-900); run_optuna_tuning groups threading" />
  </context_files>
  <topics>
    <topic id="T1" title="Cluster-bootstrap for group CV SHAP bootstrap">
      <summary>
        When cv_strategy="group", OOF SHAP values for same-group members are computed by the same
        fold model, inducing within-group correlation. The non-inference SHAP bootstrap resamples
        rows i.i.d. (shap_utils.py:1014), treating all observations as exchangeable. This produces
        anti-conservative significance calls: bootstrap variance is underestimated for real features
        (within-group correlation preserved) while shadow features (column-permuted, breaking group
        correlation) are correctly estimated.

        The existing cluster-bootstrap machinery (shap_utils.py:989-1012) handles inference mode
        but requires equal cluster sizes (assertion at line 999). For predict-mode group CV, groups
        have unequal sizes, so the implementation must be generalized to handle ragged cluster
        resamples. This is an /implement specification, not a design decision.

        A fallback to i.i.d. bootstrap with a warning is specified when n_unique_groups is below 20,
        based on the practical minimum for reliable cluster-bootstrap finite-sample performance
        (Ukoumunne et al. 2003 recommend ~24 clusters per arm). The threshold is hardcoded, not
        configurable, because it is a statistical guardrail, not a user preference.
      </summary>
      <research>
        R1 (Cameron, Gelbach & Miller 2008, REStat 90(3):414-427): Confirmed that ignoring
        clustering produces severely anti-conservative inference (rejection rates 0.43-0.50 vs.
        nominal 0.05). Severity scales with ICC strength (0.054 at rho_x=0 to 0.770 at rho_x=1)
        and cluster size (0.106 at size 2 to 0.679 at size 100). Correction: the CR's claim that
        severity is "most severe when number of clusters is moderate (10-50)" is not supported;
        the uncorrected method's over-rejection is roughly invariant to G. The moderate-G problem
        applies to CORRECTED methods (CRVE/cluster-bootstrap), not the uncorrected method. Caveat:
        CGM 2008 tests conventional Wald inference, not bootstrap vs. bootstrap specifically; the
        underlying principle (variance underestimation under clustering) is the same.

        R2 (Field & Welsh 2007, JRSS-B 69(3):369-390): Confirmed cluster bootstrap consistency
        under both transformation and random-effect models. Requires only correct specification of
        the grouping structure (weakest assumption among clustered-data bootstraps). Consistency is
        asymptotic (G to infinity); Ukoumunne et al. (2003, Statistics in Medicine) recommend ~24
        clusters per arm as a practical minimum for adequate finite-sample coverage.
      </research>
      <approaches>
        <approach id="A1" label="Implement cluster bootstrap with fallback" feasibility="high" risk="low">
          <description>
            Thread group_column values through predict.py SHAP context. Activate cluster_ids in
            _run_bootstrap_pipeline for non-inference mode when groups are present. Generalize
            the existing cluster-bootstrap code (lines 989-1012) to handle unequal cluster sizes
            (ragged resamples). When n_unique_groups less than 20, fall back to i.i.d. bootstrap with
            a RuntimeWarning documenting the limitation.
          </description>
          <pros>
            Correct inference under group CV with sufficient clusters. Reuses existing
            cluster-bootstrap logic (generalized). Defensible during peer review. Explicit
            degradation path for low-cluster-count datasets.
          </pros>
          <cons>
            Implementation requires generalizing the equal-size cluster-bootstrap to ragged
            resamples. Data plumbing through predict.py context dict.
          </cons>
          <statistical_considerations>
            Cluster bootstrap consistency requires G to infinity (Field & Welsh 2007).
            Practical minimum ~24 clusters/arm (Ukoumunne et al. 2003). Below 20 groups,
            cluster-bootstrap variance estimates are themselves unreliable. The i.i.d. fallback
            is anti-conservative but is the best available option at low G; the warning ensures
            the user is informed.
          </statistical_considerations>
        </approach>
        <approach id="A2" label="Document as known limitation" feasibility="high" risk="medium">
          <description>
            Document in INPUT_SPECIFICATION.md that group CV SHAP bootstrap uses i.i.d. resampling.
            Provide quantitative guidance for when this matters.
          </description>
          <pros>Zero implementation cost.</pros>
          <cons>
            Users may miss the warning and publish anti-conservative results. Pipeline silently
            produces biased inference under a supported configuration. Reviewers will ask why
            the pipeline does not use cluster bootstrap when it already implements it for
            inference mode.
          </cons>
          <statistical_considerations>
            Anti-conservative bias scales with ICC and cluster size (CGM 2008). Not mitigated
            by stability threshold or BH-FDR correction.
          </statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">
        Implement cluster bootstrap with i.i.d. fallback at n_unique_groups less than 20. The
        fallback threshold is hardcoded (statistical guardrail, not user preference). The existing
        cluster-bootstrap code must be generalized to handle unequal cluster sizes.
      </decision>
    </topic>

    <topic id="T2" title="RepeatedGroupKFold fold-assignment algorithm">
      <summary>
        Single-repeat group CV (_GroupKFoldWrapper) uses sklearn's greedy LPT algorithm (Graham
        1969), which sorts groups by descending sample count and assigns each to the lightest fold.
        Multi-repeat group CV (_RepeatedGroupKFold) used round-robin assignment on a random
        permutation, producing different (and generally less balanced) fold-size distributions.

        The original CR recommendation (match sklearn's greedy LPT) was invalidated during the
        pre-report audit: greedy LPT produces a deterministic partition given group sizes, so
        every repeat yields identical folds, making n_repeats greater than 1 entirely redundant.

        After empirical simulation and literature verification, the decision was revised to use
        Graham's (1966) list scheduling with randomized group ordering: process groups in a
        random permutation order, assign each to the fold with the fewest total samples. This
        provides full repeat diversity with significantly better balance than round-robin.
      </summary>
      <research>
        R4 (sklearn source, direct verification): GroupKFold(shuffle=False) uses greedy LPT.
        _GroupKFoldWrapper delegates to this default. Confirmed.

        R5 (Vanwinckelen & Blockeel 2012, BNAIC): CR's citation was mischaracterized. The paper
        shows variance reduction from repeated CV is sub-1/n even with distinct fold compositions
        (because repeats share the same data). Does NOT address group CV or limited unique
        fold compositions.

        R8 (Graham 1966/1969, Albers & Janke 2020): Graham's list scheduling with random arrival
        order has worst-case makespan ratio 2 minus 1/m (same as adversarial order; randomization
        provides no formal worst-case improvement). LPT tightens this to 4/3 but eliminates repeat
        diversity. The specific combination of "repeated group CV with randomized-order greedy" is
        not a named technique in the CV literature, but the algorithm itself is a textbook primitive
        (Graham 1966, ~1500+ citations). sklearn GroupKFold(shuffle=True, v1.6+) uses a related
        randomized group-ordering approach.

        Empirical simulation (4 scenarios, 50 repeats each): A4 (greedy-random) produces full
        repeat diversity (50/50 unique assignments) with mean fold-size ratios 2-5x better than
        round-robin for heterogeneous group sizes. Under high heterogeneity (sizes 5-500): A4
        mean ratio 2.97, max 4.88 vs. round-robin mean 14.01, max 36.00.
      </research>
      <approaches>
        <approach id="A1-original" label="Greedy LPT per repeat" feasibility="high" risk="high">
          <description>Match sklearn's greedy algorithm per repeat.</description>
          <pros>Optimal balance per repeat.</pros>
          <cons>Produces identical folds across all repeats, making n_repeats greater than 1
          entirely redundant. Invalidated.</cons>
          <statistical_considerations>Zero repeat diversity defeats the purpose of repeated CV.</statistical_considerations>
        </approach>
        <approach id="A2" label="Keep round-robin + diagnostic" feasibility="high" risk="low">
          <description>Keep current round-robin. Add fold-size diagnostic warning.</description>
          <pros>Simple, transparent, full repeat diversity.</pros>
          <cons>Poor balance for heterogeneous group sizes (max ratio up to 36x in simulation).
          Ignores group sizes entirely.</cons>
          <statistical_considerations>Unbalanced folds produce high-variance per-fold score estimates,
          degrading Optuna tuning quality.</statistical_considerations>
        </approach>
        <approach id="A4" label="Greedy in random order (Graham 1966 list scheduling)" feasibility="high" risk="low">
          <description>
            Process groups in a random permutation order, assign each to the fold with the fewest
            total samples (Graham 1966 list scheduling). Different random orderings produce
            different assignments. Include fold-size diagnostic warning when max/min ratio
            exceeds 2.0.
          </description>
          <pros>
            Full repeat diversity (50/50 in simulation). Significantly better balance than
            round-robin (2-5x better mean ratio). Uses a well-established algorithm (Graham 1966).
            Consistent assignment logic with single-repeat path (both use "assign to lightest").
          </pros>
          <cons>
            Worse balance than greedy LPT (expected cost of repeat diversity). Worst-case
            bound (2 minus 1/m) same as adversarial-order greedy. Not a named technique in the
            CV literature (but neither is round-robin in this context).
          </cons>
          <statistical_considerations>
            Empirically verified: balance improvement over round-robin is consistent across
            homogeneous, moderate, high heterogeneity, and few-group scenarios. The approach
            trades optimal single-repeat balance (LPT) for repeat diversity, which is the
            purpose of n_repeats greater than 1.
          </statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A4">
        Replace round-robin in _RepeatedGroupKFold with Graham's (1966) list scheduling in
        randomized group order. Include fold-size diagnostic warning when max/min fold-size
        ratio exceeds 2.0 within any repeat. Describable as: "Groups are processed in a random
        permutation order and each is assigned to the fold with the fewest total samples
        (Graham 1966 list scheduling)."
      </decision>
    </topic>

    <topic id="T3" title="FDR method scope and PRDS documentation">
      <summary>
        The pipeline uses BH-FDR (method='fdr_bh') for all three independent FDR calls. BH
        controls FDR at the nominal level under independence or PRDS (Benjamini & Yekutieli 2001).
        Under arbitrary (non-PRDS) dependence, actual FDR can inflate by up to c(m) = H_m,
        approximately ln(m) + gamma. For the largest family (~1275 tests), this is ~7.7x nominal.
        The PRDS assumption is plausible but undocumented. The conservative BY alternative
        (method='fdr_by') is not exposed to the user.
      </summary>
      <research>
        R7 (Benjamini & Yekutieli 2001, Annals of Statistics 29(4):1165-1188): All claims
        confirmed. PRDS condition defined; BH controls FDR under PRDS at nominal level; BY
        correction uses c(m) = sum(1/i for i in 1..m) for arbitrary dependence control;
        c(m) approximately equals ln(m) + 0.5772. The CR's ~7x estimate for m=1275 is
        consistent (c(1275) approximately equals 7.7).
      </research>
      <approaches>
        <approach id="A1" label="Document PRDS assumption only" feasibility="high" risk="low">
          <description>Add PRDS assumption documentation to INPUT_SPECIFICATION.md Section 3.</description>
          <pros>Transparency improvement. Zero code changes.</pros>
          <cons>Users needing conservative FDR control must modify source code.</cons>
          <statistical_considerations>Documentation-only; no change to inference properties.</statistical_considerations>
        </approach>
        <approach id="A2" label="Document + expose fdr_method config key" feasibility="high" risk="low">
          <description>
            Add PRDS documentation to INPUT_SPECIFICATION.md. Add shap.bootstrapping.fdr_method
            config key accepting "bh" (default) or "by". Thread through to _nan_safe_fdr.
          </description>
          <pros>
            Config-driven flexibility. Users can switch to conservative FDR control without code
            modification. BH default preserves current behavior.
          </pros>
          <cons>Modest implementation scope (config schema, both example configs, _nan_safe_fdr
          parameterization, INPUT_SPECIFICATION.md).</cons>
          <statistical_considerations>
            BY is strictly more conservative than BH. Defaulting to BH is appropriate because
            PRDS is plausible for the pipeline's exceedance statistics. BY is available for
            users whose reviewers require arbitrary-dependence control.
          </statistical_considerations>
        </approach>
      </approaches>
      <decision status="decided" chosen="A2">
        Expose shap.bootstrapping.fdr_method config key ("bh" default, "by" alternative).
        Document the PRDS assumption in INPUT_SPECIFICATION.md Section 3. BH remains the default.
      </decision>
    </topic>
  </topics>
  <action_items>
    <item priority="P0" target_mode="implement" description="T1: Implement cluster-bootstrap path in _run_bootstrap_pipeline for non-inference mode when cv_strategy='group'. Thread group_column through predict.py SHAP context. Generalize existing cluster-bootstrap code (shap_utils.py:989-1012) to handle unequal cluster sizes. Add i.i.d. fallback with RuntimeWarning when n_unique_groups less than 20." />
    <item priority="P1" target_mode="implement" description="T2: Replace round-robin in _RepeatedGroupKFold (utils.py:204-226) with Graham (1966) list scheduling in randomized group order. Add fold-size diagnostic warning when max/min fold-size ratio exceeds 2.0 within any repeat." />
    <item priority="P1" target_mode="implement" description="F2 (CR direct): Add n_unique_groups >= cv_folds and n_unique_groups >= inner_cv_folds validation to validate_cv_config (utils.py:514-546)." />
    <item priority="P1" target_mode="implement" description="F3 (CR direct): Remove redundant splitter recreation at train.py:900." />
    <item priority="P2" target_mode="implement" description="T3: Add shap.bootstrapping.fdr_method config key ('bh' default, 'by' alternative). Parameterize _nan_safe_fdr (shap_utils.py:1168-1198). Update both example configs. Document PRDS assumption in INPUT_SPECIFICATION.md Section 3." />
    <item priority="P2" target_mode="implement" description="F5 (CR direct): Add n_unique_inner_groups diagnostic in run_optuna_tuning. Warn when n_unique_inner less than 2 * inner_cv_folds." />
    <item priority="P2" target_mode="implement" description="F6 (CR direct): Add bin-count diagnostic to stratify_labels_for_regression (utils.py:161-165). Warn when actual bin count diverges from n_bins." />
  </action_items>
  <next_steps>
    Proceed to /implement with this report as input. All action items target implement mode.
    Sequencing: P0 (T1 cluster bootstrap) first, as it blocks cv_strategy release; then P1
    (T2 algorithm + F2 validation + F3 cleanup); then P2 (T3 fdr_method + F5/F6 diagnostics).
    Follow with /test to verify all changes, then /document and /publish for the cv_strategy
    feature release.
  </next_steps>
</brainstorm_report>
