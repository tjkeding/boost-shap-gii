<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-14T14:15:00Z" />
  <input_reports>
    <report path="boost-shap-gii_brainstorm_20260814_135928.md" mode="brainstorm" key_items="7" />
  </input_reports>
  <changes>
    <change id="C1" priority="P0" source_item="T1: Cluster bootstrap for group CV SHAP bootstrap">
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>
        Thread group-membership information from predict.py through the SHAP pipeline so that
        _run_bootstrap_pipeline can apply cluster-aware resampling when cv_strategy="group".
        Generalize the existing cluster-bootstrap code (shap_utils.py:989-1012) to handle
        unequal cluster sizes. Add i.i.d. fallback with RuntimeWarning when n_unique_groups
        is below 20.

        Three sub-tasks:

        (a) predict.py (lines 441-455): Extract group_column from config["modeling"] and,
            when cv_strategy="group" and group_column is present in df_raw, extract group
            values and add "groups" (np.ndarray of per-row group labels) and "cv_strategy"
            (str) to the shap_ctx dict. predict.py already loads df_raw from config["paths"]["input_data"]
            (line 76-85) and has access to config (line 44). Group extraction mirrors
            train.py:726-736 logic (extract group_column from config, pull groups from df_raw).
            The group_column itself is NOT a feature candidate in predict.py (train.py already
            excludes it from trained_features.json), so no feature-list surgery is needed.

        (b) shap_utils.py, run_shap_pipeline (lines 1459-1551) and _run_shap_for_slice
            (lines 1307-1365, post-fold merge lines 1392-1456):
            - run_shap_pipeline: extract ctx.get("groups", None) and ctx.get("cv_strategy", "uniform").
              Pass both to _run_shap_for_slice as new keyword arguments.
            - _run_shap_for_slice: accept groups and cv_strategy keyword arguments.
              In the non-inference post-fold merge (lines 1417-1420), when cv_strategy == "group"
              and groups is not None, construct cluster_ids from fold_assignments and groups.
              Specifically: each unique group label becomes a cluster. cluster_ids is a 1-D array
              of length n_samples mapping each row to its group label. This is semantically
              identical to inference-mode cluster_ids (which maps rows to observation indices)
              but uses group labels instead. The _run_bootstrap_pipeline call at lines 1443-1456
              already passes cluster_ids=cluster_ids; no change needed there.

        (c) shap_utils.py, _run_bootstrap_pipeline (lines 989-1012): Generalize the
            cluster bootstrap to handle unequal cluster sizes.
            Current code: asserts equal cluster sizes (line 999), uses a fixed
            rows_per_cluster to pre-allocate all_indices as shape (n_boot, n_clusters * rows_per_cluster).
            Generalized code: remove the equal-size assertion. For each bootstrap iteration,
            sample n_clusters cluster indices with replacement. Concatenate the row indices
            for each sampled cluster (which may have different lengths). The resulting
            all_indices rows have variable lengths, so use a list-of-arrays representation
            instead of a fixed 2-D array.
            This changes the downstream indices_split (line 1016) from np.array_split on a
            2-D array to splitting a list of 1-D arrays across workers. Each worker's
            _bootstrap_chunk function already iterates over rows of its indices block, so
            the worker code needs to handle variable-length index arrays.

            Add fallback guard: when n_unique_groups is below 20, emit a RuntimeWarning
            (warnings.warn(..., RuntimeWarning)) documenting the limitation and fall through
            to the i.i.d. bootstrap (existing line 1014 path). The threshold is hardcoded
            at 20, not configurable.
      </description>
      <spec>
        ## predict.py changes

        After shap_ctx construction (line 455), before run_shap_pipeline call (line 457):

        ```python
        # Thread group info for cluster-aware SHAP bootstrap
        cv_strategy = config["modeling"].get("cv_strategy", "uniform")
        group_column = config["modeling"].get("group_column")
        if cv_strategy == "group" and group_column is not None and group_column in df_raw.columns:
            shap_ctx["groups"] = df_raw[group_column].values
            shap_ctx["cv_strategy"] = cv_strategy
        ```

        ## shap_utils.py changes

        ### run_shap_pipeline (lines 1459-1551)

        After extracting existing ctx fields (line 1497), add:
        ```python
        groups = ctx.get("groups", None)
        cv_strategy = ctx.get("cv_strategy", "uniform")
        ```

        Pass to _run_shap_for_slice call (line 1544-1548):
        ```python
        _run_shap_for_slice(
            ctx, shap_dir, shadow_paths, splits,
            all_feature_types, slice_idx, slice_label,
            inference_mode=inference_mode,
            groups=groups,
            cv_strategy=cv_strategy,
        )
        ```

        ### _run_shap_for_slice (lines 1307-1365)

        Add parameters to signature:
        ```python
        def _run_shap_for_slice(
            ...,
            inference_mode: bool = False,
            groups: Optional[np.ndarray] = None,
            cv_strategy: str = "uniform",
        ) -> None:
        ```

        In the non-inference post-merge block (after line 1420), before _aggregate_effects:
        ```python
        if not inference_mode and cv_strategy == "group" and groups is not None:
            cluster_ids = groups
        ```

        This sets cluster_ids to the group labels array. Each unique group label
        maps to all rows belonging to that group. _run_bootstrap_pipeline will
        use np.unique(cluster_ids) and np.where(cluster_ids == cid) to build
        cluster_to_rows, exactly as the inference path does (lines 993-997).

        ### _run_bootstrap_pipeline cluster bootstrap generalization (lines 989-1012)

        Replace lines 989-1012 with:
        ```python
        if cluster_ids is not None:
            unique_clusters = np.unique(cluster_ids)
            n_clusters = len(unique_clusters)

            _CLUSTER_BOOTSTRAP_MIN_GROUPS = 20

            if n_clusters < _CLUSTER_BOOTSTRAP_MIN_GROUPS:
                warnings.warn(
                    f"Only {n_clusters} unique groups detected (minimum {_CLUSTER_BOOTSTRAP_MIN_GROUPS} "
                    f"recommended for reliable cluster bootstrap; Ukoumunne et al. 2003). "
                    f"Falling back to i.i.d. bootstrap. SHAP significance calls may be "
                    f"anti-conservative under within-group correlation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                cluster_ids = None  # fall through to i.i.d. path below

        if cluster_ids is not None:
            unique_clusters = np.unique(cluster_ids)
            n_clusters = len(unique_clusters)
            cluster_to_rows: Dict[Any, np.ndarray] = {}
            for cid in unique_clusters:
                cluster_to_rows[cid] = np.where(cluster_ids == cid)[0]

            all_indices = []
            sampled_cluster_idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
            for b in range(n_boot):
                expanded = np.concatenate(
                    [cluster_to_rows[unique_clusters[c]] for c in sampled_cluster_idx[b]]
                )
                all_indices.append(expanded)
        else:
            all_indices = rng.integers(0, n_samples, size=(n_boot, n_samples))
        ```

        The downstream indices_split (line 1016) must handle both cases:
        - When all_indices is a 2-D ndarray (i.i.d. path): np.array_split works as before.
        - When all_indices is a list of 1-D arrays (cluster path): use Python list slicing
          to split into n_jobs chunks.

        Replace line 1016:
        ```python
        n_split = max(1, n_jobs)
        if isinstance(all_indices, list):
            chunk_size = (len(all_indices) + n_split - 1) // n_split
            indices_split = [all_indices[i*chunk_size:(i+1)*chunk_size] for i in range(n_split)]
            indices_split = [c for c in indices_split if c]
        else:
            indices_split = np.array_split(all_indices, n_split)
        ```

        The _bootstrap_chunk worker function (which processes each indices_split element)
        must handle list-of-arrays input. Currently it iterates `for b_local in range(block.shape[0])`
        and indexes `block[b_local]`. With a list input, iterate `for b_local in range(len(block))`
        and index `block[b_local]`. Both produce a 1-D index array per iteration, so downstream
        logic (SHAP_vals[idx], X_vals[idx]) is unchanged.

        Add `import warnings` at top of shap_utils.py (verify it is not already present).
      </spec>
      <dependencies>none</dependencies>
      <risk>medium - Generalizing the bootstrap index structure from fixed-shape 2-D array to
        variable-length list-of-arrays requires careful handling in _bootstrap_chunk and
        indices_split. Downstream workers use integer indexing into SHAP_vals and X_vals,
        which is shape-agnostic, so risk is contained to the index-generation and splitting
        logic. The fallback guard at n_groups &lt; 20 mitigates small-sample risk.</risk>
      <rollback>Revert predict.py shap_ctx additions and shap_utils.py changes to
        _run_shap_for_slice, run_shap_pipeline, and _run_bootstrap_pipeline.
        The cluster_ids pathway for inference mode is unchanged, so inference
        functionality is not affected by a rollback.</rollback>
    </change>

    <change id="C2" priority="P1" source_item="T2: RepeatedGroupKFold fold-assignment algorithm">
      <file path="src/boost_shap_gii/utils.py" action="modify" />
      <description>
        Replace round-robin assignment in _RepeatedGroupKFold.split() (lines 204-226) with
        Graham's (1966) list scheduling in randomized group order. Add fold-size diagnostic
        warning when max/min fold-size ratio exceeds 2.0 within any repeat.
      </description>
      <spec>
        Replace the body of _RepeatedGroupKFold.split() (lines 211-226).

        Current implementation (round-robin on random permutation):
        ```python
        for rep in range(self.n_repeats):
            rng = np.random.default_rng(self.random_state + rep)
            unique_groups = np.unique(groups)
            perm = rng.permutation(unique_groups)
            group_to_fold = {}
            for i, g in enumerate(perm):
                group_to_fold[g] = i % self.n_splits
            fold_labels = np.array([group_to_fold[g] for g in groups])
            for k in range(self.n_splits):
                ...
        ```

        New implementation (greedy-in-random-order):
        ```python
        for rep in range(self.n_repeats):
            rng = np.random.default_rng(self.random_state + rep)
            unique_groups = np.unique(groups)
            perm = rng.permutation(unique_groups)

            # Graham (1966) list scheduling: assign each group (in random
            # arrival order) to the fold with the fewest total samples.
            group_sizes = {g: int(np.sum(groups == g)) for g in unique_groups}
            fold_loads = np.zeros(self.n_splits, dtype=np.int64)
            group_to_fold = {}
            for g in perm:
                lightest = int(np.argmin(fold_loads))
                group_to_fold[g] = lightest
                fold_loads[lightest] += group_sizes[g]

            # Fold-size diagnostic
            if fold_loads.min() > 0:
                ratio = fold_loads.max() / fold_loads.min()
                if ratio > 2.0:
                    print(
                        f"[WARNING] _RepeatedGroupKFold repeat {rep}: fold sizes "
                        f"{fold_loads.tolist()} are unbalanced (max/min ratio = "
                        f"{ratio:.2f}, threshold: 2.0)."
                    )

            fold_labels = np.array([group_to_fold[g] for g in groups])
            for k in range(self.n_splits):
                train_idx = np.where(fold_labels != k)[0]
                val_idx = np.where(fold_labels == k)[0]
                yield train_idx, val_idx
        ```

        group_sizes computation uses np.sum(groups == g) for each group.
        This is O(G * N) overall; for the expected G and N ranges in this
        pipeline (&lt;100 groups, &lt;100K rows), this is negligible. A dict
        comprehension over unique_groups makes the intent clear.

        The fold-size diagnostic matches the existing check in train.py:894-899
        but is embedded per-repeat rather than post-hoc, since each repeat
        produces a different partition.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Replaces a simple assignment algorithm with another simple assignment
        algorithm. The greedy-in-random-order approach was empirically verified in the
        brainstorm session (4 scenarios, 50 repeats, full diversity with 2-5x better
        balance than round-robin). No change to the split() yield contract (train_idx,
        val_idx pairs). get_n_splits() is unchanged.</risk>
      <rollback>Restore the round-robin body of _RepeatedGroupKFold.split().</rollback>
    </change>

    <change id="C3" priority="P1" source_item="F2: n_unique_groups >= cv_folds validation">
      <file path="src/boost_shap_gii/utils.py" action="modify" />
      <description>
        Add validation in validate_cv_config (lines 517-541) that n_unique_groups is at
        least cv_folds (outer) and at least inner_cv_folds (inner) when cv_strategy="group".
        Without this check, GroupKFold silently produces empty folds or raises an opaque
        sklearn error.
      </description>
      <spec>
        Within the `if cv_strategy == "group":` block (after line 535), add:

        ```python
        if df is not None:
            n_unique_groups = df[group_column].nunique()
            cv_folds = config["modeling"].get("cv_folds")
            if cv_folds is not None and n_unique_groups < cv_folds:
                raise ValueError(
                    f"cv_strategy='group' requires at least cv_folds={cv_folds} unique "
                    f"groups, but group_column='{group_column}' has only "
                    f"{n_unique_groups} unique values."
                )
            inner_cv_folds = config.get("modeling", {}).get("tuning", {}).get("inner_cv_folds")
            if inner_cv_folds is not None and n_unique_groups < inner_cv_folds:
                raise ValueError(
                    f"cv_strategy='group' requires at least inner_cv_folds={inner_cv_folds} "
                    f"unique groups, but group_column='{group_column}' has only "
                    f"{n_unique_groups} unique values."
                )
        ```

        Note: cv_folds and inner_cv_folds may be None at the validate_cv_config call site
        if fill_config_defaults has not yet run. The `is not None` guard handles this.
        The inner_cv_folds check validates against the FULL dataset's unique groups, not the
        per-fold training subset's groups. This is an upper-bound check (the training subset
        for inner CV will have fewer unique groups). F5/C6 adds a tighter per-fold diagnostic.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Adds a validation check that raises ValueError on an invalid configuration.
        No change to any execution path when configuration is valid.</risk>
      <rollback>Remove the added validation block from validate_cv_config.</rollback>
    </change>

    <change id="C4" priority="P1" source_item="F3: Remove redundant splitter recreation">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <description>
        Remove the redundant `splitter = get_cv_splitter(config, y_for_split, groups=groups)`
        at line 900. This line recreates the splitter that was already created at line 892.
        The intervening code (lines 894-899) only reads from the splitter via .split() for
        the fold-size diagnostic; it does not mutate or exhaust the splitter's state in a
        way that requires recreation (sklearn splitters are reusable).
      </description>
      <spec>
        Delete line 900:
        ```python
        splitter = get_cv_splitter(config, y_for_split, groups=groups)
        ```

        The splitter created at line 892 is a _GroupKFoldWrapper, _StratifiedRegressionKFold,
        or _RepeatedGroupKFold instance. All three implement split() as a generator that can
        be called multiple times (each call restarts the iteration). The fold_sizes list
        comprehension at line 895 consumes one iteration, and the CV loop at line 921
        consumes another. No recreation is needed.

        However: verify that _GroupKFoldWrapper.split() is indeed re-entrant. It delegates
        to sklearn GroupKFold.split(), which returns a fresh generator on each call. Confirmed:
        sklearn KFold/GroupKFold/StratifiedKFold.split() is re-entrant by design.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Removes a redundant line. The splitter is re-entrant. If the splitter were
        NOT re-entrant (which would be a sklearn contract violation), the fold_sizes diagnostic
        at line 895 would have already consumed the only iteration, and the existing code
        would be masking that bug by recreating. In that hypothetical case, removing the
        recreation would surface the bug, which is the correct behavior.</risk>
      <rollback>Re-add the splitter recreation line at the original location.</rollback>
    </change>

    <change id="C5" priority="P2" source_item="T3: FDR method config key">
      <file path="src/boost_shap_gii/utils.py" action="modify" />
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <file path="example_config_advanced.yaml" action="modify" />
      <description>
        Add shap.bootstrapping.fdr_method config key ("bh" default, "by" alternative).
        Register the default in fill_config_defaults. Validate the value. Parameterize
        _nan_safe_fdr to accept the method string. Update example_config_advanced.yaml.
      </description>
      <spec>
        ## utils.py: fill_config_defaults (lines 458-464)

        After `_set(["shap", "bootstrapping", "fdr_correct"], True)` (line 462), add:
        ```python
        _set(["shap", "bootstrapping", "fdr_method"], "bh")
        ```

        ## utils.py: validation

        Add a new validation function or extend an existing one. The simplest approach
        is to add validation inline in fill_config_defaults or in a dedicated validator
        called from train.py after fill_config_defaults. Since validate_cv_config is the
        pattern for config validation, add a new function:

        ```python
        def validate_bootstrap_config(config: dict) -> None:
            fdr_method = config.get("shap", {}).get("bootstrapping", {}).get("fdr_method", "bh")
            valid_methods = {"bh", "by"}
            if fdr_method not in valid_methods:
                raise ValueError(
                    f"shap.bootstrapping.fdr_method must be one of {sorted(valid_methods)}, "
                    f"got '{fdr_method}'."
                )
        ```

        Call site: validate_bootstrap_config must be called after fill_config_defaults.
        The cleanest insertion point is in train.py after the existing validate_cv_config
        call, and similarly in predict.py and infer.py if they call validators. However,
        since _run_bootstrap_pipeline reads the config value directly, validation at the
        point of use is also acceptable. The build agent should place the call alongside
        validate_cv_config in train.py's main flow.

        ## shap_utils.py: _run_bootstrap_pipeline (lines 971-977)

        After extracting fdr_correct (line 976), add:
        ```python
        fdr_method_key = config["shap"]["bootstrapping"].get("fdr_method", "bh")
        fdr_method_scipy = "fdr_bh" if fdr_method_key == "bh" else "fdr_by"
        ```

        ## shap_utils.py: _nan_safe_fdr (lines 1168-1198)

        Change the function to accept and use the method parameter. The function is a
        closure inside _run_bootstrap_pipeline, so it can capture fdr_method_scipy from
        the enclosing scope. Replace the hardcoded 'fdr_bh' at lines 1195 and 1198:

        ```python
        def _nan_safe_fdr(p_vals, alpha_val):
            nan_mask_p = np.isnan(p_vals)
            if nan_mask_p.any():
                p_clean = p_vals.copy()
                p_clean[nan_mask_p] = 1.0
                q_vals = multipletests(p_clean, alpha=alpha_val, method=fdr_method_scipy)[1]
                q_vals[nan_mask_p] = np.nan
                return q_vals
            return multipletests(p_vals, alpha=alpha_val, method=fdr_method_scipy)[1]
        ```

        The docstring should be updated to reflect the parameterization (replace "BH-FDR"
        with "BH-FDR or BY-FDR" and note the method is set by config).

        ## example_config_advanced.yaml (lines 113-118)

        Add fdr_method after fdr_correct (line 116):
        ```yaml
          bootstrapping:
            n_boot: 10000
            alpha: 0.05
            fdr_correct: true
            fdr_method: "bh"    # FDR correction method: "bh" (Benjamini-Hochberg) or "by" (Benjamini-Yekutieli)
            stab_thresh: 2
            output_boots_n: 25
        ```
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Adds a new config key with a default that preserves existing behavior.
        The only functional change is when a user explicitly sets fdr_method: "by", which
        produces strictly more conservative significance calls (expected and desired).
        statsmodels multipletests supports both 'fdr_bh' and 'fdr_by' natively.</risk>
      <rollback>Remove fdr_method from fill_config_defaults, validate_bootstrap_config,
        _run_bootstrap_pipeline config extraction, _nan_safe_fdr parameterization, and
        example_config_advanced.yaml.</rollback>
    </change>

    <change id="C6" priority="P2" source_item="F5: Inner groups diagnostic">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <description>
        Add a diagnostic warning in run_optuna_tuning (lines 520-548) when the number of
        unique groups in the inner training subset is less than 2 * inner_cv_folds. This
        condition means some inner folds may contain very few groups, producing unreliable
        per-fold estimates for Optuna tuning.
      </description>
      <spec>
        After inner_cv creation (line 538), within the `if groups is not None:` context
        (groups is already available as a parameter, passed as inner_groups from line 933-934):

        ```python
        if groups is not None and cv_strategy == "group":
            n_unique_inner = len(np.unique(groups))
            if n_unique_inner < 2 * inner_cv_folds:
                print(
                    f"[WARNING] Inner CV has only {n_unique_inner} unique groups for "
                    f"{inner_cv_folds} folds. Some folds may have very few groups, "
                    f"producing unreliable tuning estimates."
                )
        ```

        cv_strategy is available in run_optuna_tuning's scope because config is passed as
        a parameter (line 483). Extract it:
        ```python
        cv_strategy = config["modeling"].get("cv_strategy", "uniform")
        ```

        Place the extraction near the existing config reads at lines 520-528. Place the
        diagnostic after inner_cv construction (line 538).

        Note: groups here is inner_groups (the outer-fold training subset's groups), so
        n_unique_inner reflects the actual inner-CV group count, not the full dataset's.
        This is the correct quantity to check.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Diagnostic warning only. No change to execution path.</risk>
      <rollback>Remove the diagnostic block from run_optuna_tuning.</rollback>
    </change>

    <change id="C7" priority="P2" source_item="F6: Bin-count diagnostic for stratify_labels_for_regression">
      <file path="src/boost_shap_gii/utils.py" action="modify" />
      <description>
        Add a diagnostic warning to stratify_labels_for_regression (lines 163-167) when
        the actual number of bins produced by pd.qcut (with duplicates='drop') diverges
        from the requested n_bins. This occurs with heavily tied outcome distributions and
        means stratification is less effective than intended.
      </description>
      <spec>
        Modify stratify_labels_for_regression to capture and report bin count:

        ```python
        def stratify_labels_for_regression(y: pd.Series, n_bins: int) -> pd.Series:
            try:
                result = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
            except ValueError:
                result = pd.cut(y, bins=n_bins, labels=False)
            actual_bins = int(result.nunique())
            if actual_bins < n_bins:
                print(
                    f"[WARNING] Stratification produced {actual_bins} bins instead of "
                    f"the requested {n_bins} (likely due to tied outcome values)."
                )
            return result
        ```

        The diagnostic fires when duplicates='drop' reduces the number of bins (common
        with discrete or heavily rounded outcomes) or when pd.cut produces fewer bins
        than expected (rare but possible with extreme distributions).
      </spec>
      <dependencies>none</dependencies>
      <risk>low - Diagnostic warning only. No change to the returned bin labels or any
        downstream logic. The actual bin count is computed from the result that is already
        being returned.</risk>
      <rollback>Restore the original two-line function body without the diagnostic.</rollback>
    </change>
  </changes>
  <execution_order>C4, C2, C3, C7, C6, C5, C1</execution_order>
  <execution_order_rationale>
    C4 (remove redundant splitter) is a one-line deletion with zero dependencies; executing
    first reduces noise in train.py for subsequent changes.

    C2 (_RepeatedGroupKFold algorithm) and C3 (validation) are independent P1 changes in
    utils.py; C2 first because it modifies the class that C3's validation protects.

    C7 (bin-count diagnostic) is a self-contained utils.py change with no dependencies.

    C6 (inner groups diagnostic) modifies train.py and reads cv_strategy from config;
    it has no code dependency on other changes but is logically related to C3's validation.

    C5 (fdr_method) touches utils.py and shap_utils.py but in non-overlapping regions from
    C1. Executing before C1 reduces the number of shap_utils.py changes in flight during
    the highest-risk change.

    C1 (cluster bootstrap threading) is last because it is the highest-risk change, touches
    the most files (predict.py + shap_utils.py), and benefits from all other changes being
    stable before it modifies the bootstrap pipeline.
  </execution_order_rationale>
</implement_plan>
