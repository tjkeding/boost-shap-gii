<brainstorm_report>
 <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-04-23T12:35:47Z" />
 <context_files>
 <file path="src/boost_shap_gii/train.py" relevance="Contains the failing line at main nominal-encoding loop; locally at line 595 (with uncommitted Session 6 diagnostic), at line 449 in committed HEAD ec4398b installed on Milgram HPC." />
 <file path="src/boost_shap_gii/predict.py" relevance="Contains the same fillna pattern at line 135; must be patched in lockstep or predict will fail on Categorical-dtype parquet after training succeeds." />
 <file path="src/boost_shap_gii/infer.py" relevance="Contains the same fillna pattern at line 154; same reasoning as predict." />
 <file path="memory/session_history.md" relevance="Session 5 (commit c1a6dd7) addressed a related-but-distinct pandas 3.0 Categorical issue in _to_numeric_matrix, not in main's nominal-encoding loop." />
 </context_files>
 <topics>
 <topic id="topic" title="Version reconciliation: HPC install vs GitHub HEAD vs local working tree">
 <summary>The user suspected a version drift when the HPC traceback reported train.py:449 while the local working copy shows the same statement at line 595. Reconciliation confirms the installed HPC version IS the current GitHub HEAD (ec4398b). The local-vs-HPC line-number delta is explained entirely by uncommitted Session 6 changes (outcome distribution diagnostic) in the local working tree, which insert ~146 lines above the failing statement. No fix for this specific bug exists in any commit, branch, or stash.</summary>
 <research>Skipped: the question was purely a git-archaeology check; no literature relevant.</research>
 <approaches>
 <approach label="Verify HPC version by line-number match" feasibility="high" risk="low">
 <description>Run `git show ec4398b:src/boost_shap_gii/train.py` and confirm line 449 exactly matches the HPC traceback statement. Search `git log --all -S "add_categories"` and `git log --all -S "astype(object).fillna"` to rule out any prior fix on any branch. Inspect `git stash list` for pending patches.</description>
 <pros>Decisive evidence; eliminates the alternative hypothesis that user is running a stale version.</pros>
 <cons>None.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Verified: line 449 at ec4398b is the exact failing statement. No prior fix exists in history. The bug is real in the published code.</decision>
 </topic>
 <topic id="topic" title="Root cause: pandas 3.0 Categorical fillna tightening">
 <summary>In pandas 3.0, `Series.fillna(value)` on a CategoricalDtype validates `value` against the existing category index before filling, raising TypeError if the fill scalar is not an existing category. In pandas &lt;3.0 this auto-promoted silently. The boost-shap-gii nominal-encoding loop assumed the old behavior: `X[c].fillna("__NA__").astype(str).astype("category")` on a parquet-sourced Categorical column now fails because "__NA__" is not a pre-existing level. The traceback confirms the column dispatches through `pandas/core/arrays/categorical.py::_validate_setitem_value`, which is only reachable for Categorical arrays.</summary>
 <research>Skipped: the pandas error message is self-documenting and the behavioral change is explicit in pandas 3.0. Literature review would not alter the diagnosis.</research>
 <approaches>
 <approach label="Insert astype(object) before fillna" feasibility="high" risk="low">
 <description>Change `X[c].fillna("__NA__").astype(str).astype("category")` to `X[c].astype(object).fillna("__NA__").astype(str).astype("category")`. The object-dtype cast bypasses Categorical's setitem validation; downstream astype chain rebuilds a fresh CategoricalDtype including "__NA__". Applied identically at train.py:449, predict.py:135, infer.py:154 (line numbers at HEAD ec4398b).</description>
 <pros>Minimal diff (one token per site); dtype-agnostic across Categorical, object, pyarrow-string, numeric-with-NaN inputs; mirrors the Session 5 object-fallback pattern; no dtype branching; no regression on existing paths.</pros>
 <cons>Transient object-dtype copy of the column (negligible vs CatBoost Pool memory footprint).</cons>
 <statistical_considerations>None. This is a dtype-handling fix with no effect on model fit, SHAP computation, or GII statistics. The NaN→"__NA__" informativeness convention is preserved byte-for-byte.</statistical_considerations>
 </approach>
 <approach label="Branch on dtype and use cat.add_categories" feasibility="high" risk="low">
 <description>Check `isinstance(X[c].dtype, pd.CategoricalDtype)`; if true, call `X[c].cat.add_categories(["__NA__"])` before fillna.</description>
 <pros>Preserves pre-existing Categorical metadata (ordering, dtype identity) through the fill.</pros>
 <cons>Adds dtype branching to three call sites; the subsequent `astype(str).astype("category")` chain discards the preserved metadata anyway, nullifying the advantage; larger test surface.</cons>
 </approach>
 <approach label="Cast to nullable string dtype first" feasibility="med" risk="med">
 <description>`X[c].astype("string").fillna("__NA__").astype("category")`.</description>
 <pros>Pandas-native NA semantics; avoids object-dtype intermediate.</pros>
 <cons>Silent numeric-to-string coercion for numeric-coded nominals; pyarrow-string vs python-string backend divergence requires pin; behavioral change relative to current code.</cons>
 </approach>
 <approach label="Mask-based fill" feasibility="med" risk="med">
 <description>Capture `notna` mask first; `astype(str)`; `.where(mask, "__NA__")`; `astype("category")`.</description>
 <pros>Avoids object-dtype copy.</pros>
 <cons>Easy to introduce a bug by ordering the mask capture after the astype(str) (NaN becomes literal "nan"); more lines; lower readability.</cons>
 </approach>
 </approaches>
 <decision status="decided" chosen="">Smallest defensible change consistent with the pandas 3.0 contract; dtype-agnostic; matches the object-fallback pattern already used in Session 5's `_to_numeric_matrix` fix; zero statistical impact.</decision>
 </topic>
 </topics>
 <action_items>
 <item priority="P0" target_mode="implement" description="Apply patch at three sites: train.py:449, predict.py:135, infer.py:154 (line numbers at commit ec4398b; local working tree lines are 595, 135, 154 respectively — predict/infer are unaffected by local Session 6 insertions). Ensure all three ship in the same commit to avoid train/predict dtype divergence." />
 <item priority="P0" target_mode="test" description="Add regression tests exercising CategoricalDtype columns with NaN at each of train, predict, infer entry points. Tests must construct a pd.Series with CategoricalDtype whose categories exclude '__NA__' and contain NaN, apply the patched encoding, and assert (a) no TypeError, (b) result dtype is CategoricalDtype, (c) '__NA__' is present in the resulting category index, (d) NaN positions are correctly mapped to '__NA__'." />
 <item priority="P1" target_mode="implement" description="Verify pandas version constraint in pyproject.toml. If pandas 3.0+ is now a supported target (HPC uses pandas 3.0.1), this patch must be present in any pandas-3.0-capable release." />
 </action_items>
 <next_steps>/implement (plan phase) to scope the patch and its regression test suite; then /implement (build phase); then /test to validate; then /publish to commit and push so the user's HPC install picks up the fix via pip reinstall from GitHub.</next_steps>
</brainstorm_report>
