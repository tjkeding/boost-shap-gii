<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-03-26T15:47:19Z" />
  <input_reports>
    <report path="(inline specification)" mode="user-provided" key_items="9" />
  </input_reports>
  <changes>
    <change id="C1" priority="P0" source_item="Items 1-3: ordinal CategoricalDtype casting in train.py, predict.py, infer.py">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>
        The ordinal CategoricalDtype casting pattern `X[c].astype(cat_type).cat.codes` relies on
        pandas' deprecated silent NaN coercion when data values are not in `categories`. In pandas 2.0+
        this triggers FutureWarning/DeprecationWarning. The fix: explicitly mask out-of-category values
        to NaN BEFORE the `.astype(cat_type)` call, so CategoricalDtype never encounters values outside
        its category set.

        Pattern (all three files are structurally identical):
        BEFORE:
          cat_type = pd.CategoricalDtype(categories=levels, ordered=True)
          X[c] = X[c].astype(cat_type).cat.codes.astype("Int64")   # (or df_raw[c] in predict/infer)
          X.loc[X[c] == -1, c] = pd.NA

        AFTER:
          cat_type = pd.CategoricalDtype(categories=levels, ordered=True)
          source = X[c]  # (or df_raw[c] in predict/infer)
          # Explicitly set out-of-category values to NaN before casting to avoid deprecated coercion
          source = source.where(source.isin(levels) | source.isna(), other=pd.NA)
          X[c] = source.astype(cat_type).cat.codes.astype("Int64")
          X.loc[X[c] == -1, c] = pd.NA

        The `.where()` condition keeps values that are either (a) in the levels list, or (b) already
        NaN/NA. All other values are replaced with pd.NA before the cast, so CategoricalDtype never
        performs the deprecated coercion.
      </description>
      <spec>
        train.py ~line 484-488: Replace the 3-line casting block.
        predict.py ~line 170-174: Replace the 3-line casting block (source is df_raw[c]).
        infer.py ~line 188-190: Replace the 3-line casting block (source is df_raw[c]).
      </spec>
      <dependencies>none</dependencies>
      <risk>low - semantically equivalent; unknown values still become NaN, just via explicit masking rather than implicit coercion</risk>
      <rollback>Revert the 4 lines back to the original 3-line pattern in each file</rollback>
    </change>

    <change id="C2" priority="P1" source_item="Items 4-6: nominal .astype('category') in train.py, predict.py, infer.py">
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>
        The nominal casting pattern `.fillna("__NA__").astype(str).astype("category")` uses untyped
        `"category"` (not CategoricalDtype with an explicit categories list). Because `.fillna("__NA__")`
        ensures no NaN survives into the cast, and `astype("category")` infers categories from the data
        (no fixed category set to violate), this pattern CANNOT trigger the deprecation warning.

        Analysis: The FutureWarning fires only when a value is NOT in the fixed `categories` argument
        of a CategoricalDtype. Bare `.astype("category")` has no fixed set, so the warning is
        impossible. No code change is needed for these three lines.
      </description>
      <spec>No change required. Document rationale in plan only.</spec>
      <dependencies>none</dependencies>
      <risk>none - these lines are confirmed safe from the deprecation</risk>
      <rollback>N/A</rollback>
    </change>

    <change id="C3" priority="P1" source_item="Item 7: shap_utils.py _to_numeric_matrix">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>
        `_to_numeric_matrix` uses `.astype('category').cat.codes` on columns that are already
        category-typed or object-typed. Similar to C2, bare `.astype('category')` infers categories
        from data and cannot trigger the deprecation. However, the re-cast is unnecessary for columns
        already of dtype `category` and can be simplified to avoid any future ambiguity.

        Fix: For columns already of dtype `category`, call `.cat.codes` directly without redundant
        `.astype('category')`. For object columns, the bare `.astype('category')` remains safe.
      </description>
      <spec>
        shap_utils.py ~line 95-96:
        BEFORE:
          if df_num[col].dtype.name == 'category' or df_num[col].dtype == object:
              codes = df_num[col].astype('category').cat.codes

        AFTER:
          if df_num[col].dtype.name == 'category':
              codes = df_num[col].cat.codes
          elif df_num[col].dtype == object:
              codes = df_num[col].astype('category').cat.codes
      </spec>
      <dependencies>none</dependencies>
      <risk>low - eliminates a redundant re-cast; object columns still use the safe bare astype</risk>
      <rollback>Revert to the original single-branch conditional</rollback>
    </change>

    <change id="C4" priority="P1" source_item="Item 8: test_adversarial.py test_ordinal_empty_levels_list">
      <file path="tests/test_adversarial.py" action="modify" />
      <description>
        `test_ordinal_empty_levels_list` calls `data.astype(cat_type)` where `cat_type` has
        `categories=[]`, meaning every value is out-of-category, triggering the deprecation warning.
        Fix: wrap the `.astype()` call in `warnings.catch_warnings` to suppress the expected warning
        without changing test logic or assertions.
      </description>
      <spec>
        test_adversarial.py ~line 478-482:
        BEFORE:
          cat_type = pd.CategoricalDtype(categories=levels, ordered=True)
          data = pd.Series(["a", "b", "c"])
          coded = data.astype(cat_type).cat.codes

        AFTER:
          cat_type = pd.CategoricalDtype(categories=levels, ordered=True)
          data = pd.Series(["a", "b", "c"])
          with warnings.catch_warnings():
              warnings.simplefilter("ignore", FutureWarning)
              coded = data.astype(cat_type).cat.codes

        Requires `import warnings` at top of file (if not already present).
      </spec>
      <dependencies>none</dependencies>
      <risk>low - no logic change, only warning suppression for a known-deprecated pattern</risk>
      <rollback>Remove the context manager wrapper</rollback>
    </change>

    <change id="C5" priority="P1" source_item="Item 9: test_infer.py test_ordinal_unknown_values_become_nan">
      <file path="tests/test_infer.py" action="modify" />
      <description>
        `test_ordinal_unknown_values_become_nan` deliberately casts a Series containing "UNKNOWN"
        against a CategoricalDtype that does not include it. This triggers the deprecation warning.
        Fix: wrap the `.astype(cat_type)` call in `warnings.catch_warnings` to suppress the expected
        warning. Test logic and assertions remain unchanged.
      </description>
      <spec>
        test_infer.py ~line 202-203:
        BEFORE:
          s = pd.Series(["low", "UNKNOWN", "high"])
          coded = s.astype(cat_type).cat.codes.astype("Int64")

        AFTER:
          s = pd.Series(["low", "UNKNOWN", "high"])
          with warnings.catch_warnings():
              warnings.simplefilter("ignore", FutureWarning)
              coded = s.astype(cat_type).cat.codes.astype("Int64")

        Requires `import warnings` at top of file (if not already present).
      </spec>
      <dependencies>none</dependencies>
      <risk>low - no logic change, only warning suppression for a known-deprecated pattern</risk>
      <rollback>Remove the context manager wrapper</rollback>
    </change>
  </changes>
  <execution_order>C1, C3, C4, C5</execution_order>
  <notes>
    C2 (nominal .astype("category") in train/predict/infer) requires no code change; the bare
    .astype("category") pattern is confirmed safe from the Pandas 4 deprecation warning because it
    infers categories from data rather than specifying a fixed set.
  </notes>
</implement_plan>
