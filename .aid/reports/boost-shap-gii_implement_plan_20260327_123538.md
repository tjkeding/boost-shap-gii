<implement_plan>
 <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-03-27T12:35:38Z" />
 <input_reports>
 <report path="inline (user-provided brainstorm analysis)" mode="brainstorm" key_items="3" />
 </input_reports>
 <changes>
 <change id="change-1" priority="P0" source_item="Fix A (Primary)">
 <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
 <description>Replace `elif df_num[col].dtype == object:` with `elif pd.api.types.is_string_dtype(df_num[col]):` in `_to_numeric_matrix` (~line 102). This catches `object`, `string`, `string[pyarrow]`, and `StringDtype` uniformly, ensuring all string-like columns are encoded as integer category codes rather than falling through to `.astype(float)` and crashing on string values.</description>
 <spec>Single predicate substitution. No signature or logic changes.</spec>
 <dependencies>None</dependencies>
 <risk>Low - `pd.api.types.is_string_dtype` is a stable pandas API, strictly more permissive than `== object`; no regression for existing `object`-typed columns.</risk>
 <rollback>Revert the single-line change via `git checkout -- src/boost_shap_gii/shap_utils.py`</rollback>
 </change>
 <change id="change-2" priority="P0" source_item="Fix B (Defensive) - shap_utils.py">
 <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
 <description>In `_run_shap_for_slice` (~lines 1029-1031), save `orig_dtype` before `rng.permutation` and re-cast `category` columns afterward. This prevents `rng.permutation` from converting `category` dtype to `string[pyarrow]` in modern pandas.</description>
 <spec>Add `orig_dtype = X_val_shadow[c].dtype` before permutation, and `if orig_dtype.name == 'category': X_val_shadow[c] = X_val_shadow[c].astype(orig_dtype)` after.</spec>
 <dependencies>None</dependencies>
 <risk>Low - the `if` guard ensures re-casting is a no-op for non-category columns.</risk>
 <rollback>Revert via `git checkout -- src/boost_shap_gii/shap_utils.py`</rollback>
 </change>
 <change id="change-3" priority="P1" source_item="Fix B (Defensive) - train.py consistency">
 <file path="src/boost_shap_gii/train.py" action="modify" />
 <description>Apply the same dtype-preservation pattern to both `X_train_shadow` and `X_val_shadow` permutation loops (~lines 700-703) for consistency. Although `train.py` does not call `_to_numeric_matrix`, CatBoost Pool construction expects consistent `category` dtypes for `cat_features`.</description>
 <spec>Same pattern as Change 2: save `orig_dtype`, permute, conditionally re-cast if `orig_dtype.name == 'category'`.</spec>
 <dependencies>None</dependencies>
 <risk>Low - identical pattern to, applied to train.py for defensive consistency.</risk>
 <rollback>Revert via `git checkout -- src/boost_shap_gii/train.py`</rollback>
 </change>
 </changes>
 <execution_order></execution_order>
</implement_plan>
