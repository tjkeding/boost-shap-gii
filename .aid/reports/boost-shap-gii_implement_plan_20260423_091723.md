<implement_plan>
 <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-04-23T13:17:23Z" />
 <input_reports>
 <report path="boost-shap-gii_brainstorm_20260423_083547.md" mode="brainstorm" key_items="3" />
 </input_reports>
 <changes>
 <change id="change-1" priority="P0" source_item=" patch site 1 of 3)">
 <file path="src/boost_shap_gii/train.py" action="modify" />
 <description>Insert `.astype(object)` before `.fillna("__NA__")` in the nominal-encoding loop so that pandas 3.0's CategoricalDtype setitem validation is bypassed. The object-dtype cast makes the subsequent fill dtype-agnostic across Categorical, object, pyarrow-string, and numeric-with-NaN inputs. Downstream `.astype(str).astype("category")` rebuilds a fresh CategoricalDtype that includes "__NA__" by inclusion (not by mutation of an existing category index).</description>
 <spec>
At local working-tree line 595 (within the `for c in nom_feats:` block), replace:

 X[c] = X[c].fillna("__NA__").astype(str).astype("category")

with:

 X[c] = X[c].astype(object).fillna("__NA__").astype(str).astype("category")

No other edits to this file. Surrounding comments (lines 591–593 explaining the "__NA__" informativeness convention) remain unchanged because the semantic behavior is preserved byte-for-byte. Do not add new comments — the one-token insertion is self-explanatory in context and adding a comment would violate the "no narrating WHAT the code does" rule.
 </spec>
 <dependencies>none</dependencies>
 <risk>low - One-token insertion in a loop body; the object-dtype cast is idempotent on object inputs, trivial on pyarrow-string inputs, and correct on Categorical inputs. No effect on model fit, SHAP computation, or GII statistics. The NaN→"__NA__" informativeness convention is preserved exactly.</risk>
 <rollback>`git checkout HEAD -- src/boost_shap_gii/train.py` (before commit) or a single-line revert edit (after commit).</rollback>
 </change>
 <change id="change-2" priority="P0" source_item=" patch site 2 of 3)">
 <file path="src/boost_shap_gii/predict.py" action="modify" />
 <description>Apply the identical patch to the nominal-encoding loop in predict.py. This MUST ship with because a model trained on Categorical-dtype inputs (with applied) would fail at predict time without this change.</description>
 <spec>
At local working-tree line 135 (within the `for c in nom_feats:` block), replace:

 X[c] = df_raw[c].fillna("__NA__").astype(str).astype("category")

with:

 X[c] = df_raw[c].astype(object).fillna("__NA__").astype(str).astype("category")

No other edits to this file.
 </spec>
 <dependencies>none (independent of at the file level; co-required for behavioral consistency)</dependencies>
 <risk>low - Same rationale as.</risk>
 <rollback>`git checkout HEAD -- src/boost_shap_gii/predict.py`.</rollback>
 </change>
 <change id="change-3" priority="P0" source_item=" patch site 3 of 3)">
 <file path="src/boost_shap_gii/infer.py" action="modify" />
 <description>Apply the identical patch to the nominal-encoding loop in infer.py. MUST ship with because new-data inference uses the same nominal-encoding contract as train/predict.</description>
 <spec>
At local working-tree line 154 (within the `for c in nom_feats:` block), replace:

 X[c] = df_raw[c].fillna("__NA__").astype(str).astype("category")

with:

 X[c] = df_raw[c].astype(object).fillna("__NA__").astype(str).astype("category")

No other edits to this file.
 </spec>
 <dependencies>none (independent of at the file level; co-required for behavioral consistency)</dependencies>
 <risk>low - Same rationale as.</risk>
 <rollback>`git checkout HEAD -- src/boost_shap_gii/infer.py`.</rollback>
 </change>
 <change id="V1" priority="P1" source_item="brainstorm action_items (pyproject.toml verification)">
 <file path="pyproject.toml" action="verify" />
 <description>VERIFICATION-ONLY (no code change). Confirm that `pyproject.toml:17` declares `pandas` unpinned (no version floor or ceiling). An unpinned dependency permits pandas 3.0+, which is the HPC runtime. The patches are therefore required in any release targeting this pyproject.toml, which is satisfied by this plan. Resolution per user decision: verification-only; no pyproject.toml edit. Record the verification outcome in the build report.</description>
 <spec>
Read `pyproject.toml`, confirm line 17 contains `"pandas",` (unpinned). No file modification. Build agent must surface the verified state in its return payload under `deviations` as "V1 verified: pandas is unpinned in pyproject.toml; patches are required and applied."
 </spec>
 <dependencies>none</dependencies>
 <risk>low - verification only; no filesystem mutation.</risk>
 <rollback>n/a</rollback>
 </change>
 </changes>
 <execution_order> may execute in parallel (independent files, non-overlapping edits). V1 is a read-only verification and runs alongside or after. Recommended dispatch: single execution agent handling all four items (three modifications + one verification) because the total edit surface is minimal and the verification is local to the same worktree.</execution_order>
</implement_plan>
