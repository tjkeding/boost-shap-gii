<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-12T14:44:10Z" />
  <input_reports>
    <report path="boost-shap-gii_brainstorm_20260812_144151.md" mode="brainstorm" key_items="3" />
  </input_reports>
  <changes>
    <change id="C1" priority="P2" source_item="T1/F1: docstring-only fix">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Correct the _nan_safe_fdr docstring to accurately describe the NaN handling behavior. The current docstring claims NaN p-values are "excluded from the BH denominator," but the code replaces them with 1.0 conservative placeholders that remain in the denominator. No code logic changes.</description>
      <spec>
        Replace lines 1169-1175 (the docstring body before the Parameters section) with:

        """Apply BH-FDR correction to a pooled p-value vector with NaN pass-through.

        Applied to the pooled set of all effects (singletons + interactions across
        all noise strata). NaN p-values (from failed calculations) are replaced with
        1.0 conservative placeholders that remain in the BH denominator, inflating m
        and making the correction slightly more conservative for non-NaN effects.
        After correction, NaN entries are restored as NaN q-values.
        No cross-family correction is applied between sig_M, sig_V, and sig_GII;
        each family receives an independent BH call.

        The docstring wording matches the following line-level semantics:
        - Line 1192: p_clean[nan_mask_p] = 1.0 replaces NaN with conservative placeholder.
        - Line 1193: multipletests receives the full-length vector (m = total effects including placeholders).
        - Line 1194: q_vals[nan_mask_p] = np.nan restores NaN for failed effects.
      </spec>
      <dependencies>None</dependencies>
      <risk>low - docstring text change only, zero behavioral impact</risk>
      <rollback>Revert the docstring text</rollback>
    </change>

    <change id="C2" priority="P2" source_item="T2/F2: pre-flight probe for CatBoost refit">
      <file path="src/boost_shap_gii/indiv_reports.py" action="modify" />
      <description>Add a _probe_and_strip_refit_params helper that does a single trial CatBoost construction to discover any unblocked internal-only params not in the static blocklist. Called once at frozen_hps construction time (line 996-999), before the B*K bootstrap loop. Logs a single RuntimeWarning if extra params are discovered.</description>
      <spec>
        1. Add `import warnings` to the imports block (after line 53, grouped with stdlib imports).

        2. Add new helper function after _extract_user_level_params (after line 96):

        def _probe_and_strip_refit_params(frozen_hps: List[dict], task: str) -> List[dict]:
            """Trial-construct a CatBoost model to discover params rejected by the constructor.

            Iteratively attempts construction, stripping offending params on each TypeError,
            up to 5 retries. Discovered params are removed from ALL frozen_hps entries.
            A single RuntimeWarning is emitted listing discovered params and CatBoost version.
            """
            import re
            probe_params = dict(frozen_hps[0])
            probe_params["verbose"] = False
            probe_params["allow_writing_files"] = False
            cls = CatBoostRegressor if task in ("regression", "multi_regression") else CatBoostClassifier
            discovered: list = []
            for _ in range(5):
                try:
                    cls(**probe_params)
                    break
                except TypeError as exc:
                    match = re.search(r"unexpected keyword argument '(\w+)'", str(exc))
                    if not match:
                        break
                    bad_key = match.group(1)
                    discovered.append(bad_key)
                    probe_params.pop(bad_key, None)
            if discovered:
                cb_ver = getattr(__import__("catboost"), "__version__", "unknown")
                warnings.warn(
                    f"CatBoost {cb_ver} returned internal params not in the static blocklist: "
                    f"{discovered}. These were stripped from refit hyperparameters.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                stripped = []
                for hp in frozen_hps:
                    stripped.append({k: v for k, v in hp.items() if k not in discovered})
                return stripped
            return frozen_hps

        3. Replace lines 996-999 (the frozen_hps construction) with:

            frozen_hps: List[dict] = [
                _extract_user_level_params(m.get_all_params()) for m in orig_models
            ]
            frozen_hps = _probe_and_strip_refit_params(frozen_hps, task)
      </spec>
      <dependencies>None</dependencies>
      <risk>low - pre-flight probe creates a transient CatBoost object (never fit); try/except is bounded at 5 retries; regex parse is specific to CatBoost's standard TypeError format</risk>
      <rollback>Remove the helper function, remove the frozen_hps = _probe_and_strip... line, remove the warnings import</rollback>
    </change>

    <change id="C3" priority="P2" source_item="T3/F3: inference-mode X_stacked tiling fix">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Replace inference-mode X_stacked construction from fold-0 tiling to full concatenation, preserving per-fold shadow feature values for correct shadow V computation.</description>
      <spec>
        Replace lines 1402-1404:

            # X_stacked: tile the single X matrix K times to match K*N SHAP rows
            n_folds = len(chunks_X)
            X_stacked = pd.concat([chunks_X[0]] * n_folds, ignore_index=True)

        With:

            # X_stacked: concatenate all K folds' X matrices to preserve per-fold shadow features
            n_folds = len(chunks_X)
            X_stacked = pd.concat(chunks_X, ignore_index=True)
      </spec>
      <dependencies>None</dependencies>
      <risk>low - one-line change; all seven downstream consumers verified in brainstorm; real feature columns are identical across folds so only shadow columns are affected</risk>
      <rollback>Revert to pd.concat([chunks_X[0]] * n_folds, ignore_index=True)</rollback>
    </change>
  </changes>
  <execution_order>C1, C2, C3 (no dependencies; can be executed in any order; C1 and C3 target different sections of shap_utils.py with no overlap)</execution_order>
</implement_plan>
