<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-25T14:02:00Z" />
  <input_reports>
    <report path="boost-shap-gii_clean_20260825_135559.md" mode="clean" key_items="7" />
  </input_reports>
  <changes>
    <change id="C1" priority="P1" source_item="F1">
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <file path="src/boost_shap_gii/utils.py" action="modify" />
      <file path="src/boost_shap_gii/indiv_reports.py" action="modify" />
      <description>Remove 14 verified unused imports across 5 production modules.</description>
      <spec>
predict.py (line 11, lines 17-32):
  - Remove `Dict, Any, List` from line 11: `from typing import Dict, Any, List` → delete entire line (no typing imports remain in predict.py).
  - Remove `save_json_atomic,` from the utils import block (line 23).
  - Remove `is_classification,` from the utils import block (line 25).

infer.py (lines 16, 20-35):
  - Remove `import yaml` (line 16).
  - Remove `is_classification,` from the utils import block (line 27).

shap_utils.py (lines 44, 47, 52, 53, 57, 61):
  - Remove `import sys` (line 44).
  - Remove `Union` from the typing import (line 47): `from typing import Dict, Any, List, Optional, Tuple, Union` → `from typing import Dict, Any, List, Optional, Tuple`.
  - Remove `import yaml` (line 52).
  - Remove `import joblib` (line 53). Keep line 59 `from joblib import Parallel, delayed`.
  - Remove `from scipy import stats` (line 57). Keep line 58 `from statsmodels.stats.multitest import multipletests`.
  - Remove `detect_task` from the utils import (line 61): `from .utils import detect_task, is_regression, _block_permute_shadow` → `from .utils import is_regression, _block_permute_shadow`.

indiv_reports.py (lines 59, 63):
  - Remove `Any, Callable` from the typing import (line 59): `from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union` → `from typing import Dict, List, Literal, Optional, Tuple, Union`.
  - Remove `CatBoost` from the catboost import (line 63): `from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, Pool` → `from catboost import CatBoostClassifier, CatBoostRegressor, Pool`.

utils.py (line 5):
  - Remove `import copy` (line 5).
      </spec>
      <dependencies>none</dependencies>
      <risk>low - AST-verified dead imports; no call sites, no re-exports, no __all__</risk>
      <rollback>Restore removed import lines.</rollback>
    </change>

    <change id="C2" priority="P1" source_item="F2">
      <file path="src/boost_shap_gii/cli.py" action="modify" />
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <description>Wire validate_plot_config into cmd_plot and validate_bootstrap_config into train.py early validation.</description>
      <spec>
cli.py cmd_plot (lines 64-93):
  After line 72 (`run_preflight()`), add YAML config loading and validation:
    ```
    import yaml
    from .utils import validate_plot_config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    validate_plot_config(config)
    ```
  Note: use local imports (inside cmd_plot) to match the existing pattern in cli.py where all module imports are deferred inside each cmd_* function. The yaml and validate_plot_config imports go inside cmd_plot, not at module level.

train.py main() (around line 726):
  After `validate_cv_config(config, df=df_raw)` (line 726), add:
    `validate_bootstrap_config(config)`
  Add `validate_bootstrap_config` to the existing utils import block (lines 27-41). Insert after `validate_cv_config,` (line 38):
    `validate_bootstrap_config,`
      </spec>
      <dependencies>C1 (import cleanup must complete first so line numbers are not shifted by unrelated edits in the same files; however C2 only touches train.py imports and cli.py, neither of which are touched by C1's import removals in train.py's import block. C1 does not touch train.py at all. Therefore no true dependency; C1 and C2 can proceed independently on non-overlapping files. The shared file concern is cli.py (not touched by C1) and train.py (C1 does not touch train.py). No dependency.)</dependencies>
      <risk>low - validators already exist and are tested; wiring only adds call sites</risk>
      <rollback>Remove the added call sites and import additions.</rollback>
    </change>

    <change id="C3" priority="P2" source_item="F3">
      <file path="src/boost_shap_gii/indiv_reports.py" action="modify" />
      <description>Replace inline model-loading block in orchestrate_bootstrap_cache with _load_one_model call.</description>
      <spec>
indiv_reports.py orchestrate_bootstrap_cache (lines 694-701):
  Replace:
    ```python
    for k, mpath in enumerate(model_files):
        if task in ("regression", "multi_regression"):
            m = CatBoostRegressor()
        else:
            m = CatBoostClassifier()
        m.load_model(mpath)
        params.append(_extract_user_level_params(m.get_all_params()))
    ```
  With:
    ```python
    for k, mpath in enumerate(model_files):
        m = _load_one_model(mpath, task)
        params.append(_extract_user_level_params(m.get_all_params()))
    ```
  _load_one_model is defined at line 142 in the same file. No new imports needed.
      </spec>
      <dependencies>C1 (C1 modifies indiv_reports.py import line; C3 modifies body. No overlap, but sequence for clarity.)</dependencies>
      <risk>low - _load_one_model already encapsulates the exact same logic</risk>
      <rollback>Restore the inline if/else block.</rollback>
    </change>

    <change id="C4" priority="P1" source_item="F4">
      <file path="src/boost_shap_gii/utils.py" action="modify" />
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>Extract load_dataframe to utils.py; replace 3 duplicated data-loading blocks.</description>
      <spec>
utils.py: Add new function after save_json_atomic (after line 127):
    ```python
    def load_dataframe(data_path: str) -> pd.DataFrame:
        """Load a CSV or Parquet file and sanitize whitespace-only cells to NaN."""
        if data_path.endswith('.csv'):
            try:
                df = pd.read_csv(data_path)
            except (pd.errors.ParserError, ValueError, Exception):
                print("[WARNING] Standard CSV parsing failed. Attempting auto-detection (sep=None, engine='python')...")
                df = pd.read_csv(data_path, sep=None, engine='python')
        else:
            df = pd.read_parquet(data_path)
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        return df
    ```

train.py main() (lines 650-664):
  Replace the 12-line block:
    ```python
    data_path = config["paths"]["input_data"]
    print(f"[INFO] Loading data from {data_path}")

    # Check extension
    if data_path.endswith('.csv'):
        ...
    df_raw = df_raw.replace(r'^\s*$', pd.NA, regex=True)
    ```
  With:
    ```python
    data_path = config["paths"]["input_data"]
    print(f"[INFO] Loading data from {data_path}")
    df_raw = load_dataframe(data_path)
    ```
  Add `load_dataframe` to the existing utils import block (lines 27-41).

predict.py main() (lines 77-89):
  Replace the 11-line block with:
    ```python
    data_path = config["paths"]["input_data"]
    print(f"[INFO] Loading data from {data_path}")
    df_raw = load_dataframe(data_path)
    ```
  Add `load_dataframe` to the existing utils import block (lines 17-32).

infer.py main() (lines 95-107):
  Replace the 11-line block with:
    ```python
    data_path = args.data
    print(f"[INFO] Loading inference data from {data_path}")
    df_raw = load_dataframe(data_path)
    ```
  Add `load_dataframe` to the existing utils import block (lines 20-35).
      </spec>
      <dependencies>C1 (C1 modifies import blocks in predict.py, infer.py; C4 also modifies import blocks and body. Must apply C1 first so C4 edits against the post-C1 import state.)</dependencies>
      <risk>low - function body is verbatim copy of existing logic; callers retain their own log messages</risk>
      <rollback>Remove load_dataframe from utils.py; restore inline blocks in all 3 callers.</rollback>
    </change>

    <change id="C5" priority="P2" source_item="F5">
      <file path="src/boost_shap_gii/indiv_reports.py" action="modify" />
      <description>Collapse identical if/else branches in shared_indices reconstruction.</description>
      <spec>
indiv_reports.py generate_indiv_reports (lines 877-881):
  Replace:
    ```python
    # Reconstruct list (handles both ragged and rectangular)
    if shared_indices_arr.dtype == object:
        shared_indices_list = [shared_indices_arr[b] for b in range(B)]
    else:
        shared_indices_list = [shared_indices_arr[b] for b in range(B)]
    ```
  With:
    ```python
    # Reconstruct list (handles both ragged and rectangular)
    shared_indices_list = [shared_indices_arr[b] for b in range(B)]
    ```
      </spec>
      <dependencies>C3 (both modify indiv_reports.py body; sequence for clarity)</dependencies>
      <risk>low - both branches are identical; collapsing is a no-op behaviorally</risk>
      <rollback>Restore the if/else block.</rollback>
    </change>

    <change id="C6" priority="P2" source_item="F6">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Consolidate _to_numeric_matrix sentinel logic by restructuring dtype dispatch to produce codes, then applying shared sentinel block.</description>
      <spec>
shap_utils.py _to_numeric_matrix (lines 93-120):
  Replace the for-loop body (lines 94-118):
    ```python
    for col in df_num.columns:
        if df_num[col].dtype.name == 'category':
            codes = df_num[col].cat.codes
            max_code = codes.max()
            codes = codes.where(codes != -1, max_code + 1)
            df_num[col] = codes
        elif pd.api.types.is_string_dtype(df_num[col]):
            codes = df_num[col].astype('category').cat.codes
            max_code = codes.max()
            codes = codes.where(codes != -1, max_code + 1)
            df_num[col] = codes
        elif df_num[col].dtype == object:
            codes = df_num[col].astype('category').cat.codes
            max_code = codes.max()
            codes = codes.where(codes != -1, max_code + 1)
            df_num[col] = codes
        df_num[col] = df_num[col].astype(float)

        if df_num[col].isnull().any():
            df_num[col] = df_num[col].fillna(0.0)
    ```
  With:
    ```python
    for col in df_num.columns:
        if df_num[col].dtype.name == 'category':
            codes = df_num[col].cat.codes
        elif pd.api.types.is_string_dtype(df_num[col]):
            codes = df_num[col].astype('category').cat.codes
        elif df_num[col].dtype == object:
            codes = df_num[col].astype('category').cat.codes
        else:
            df_num[col] = df_num[col].astype(float)
            if df_num[col].isnull().any():
                df_num[col] = df_num[col].fillna(0.0)
            continue
        max_code = codes.max()
        codes = codes.where(codes != -1, max_code + 1)
        df_num[col] = codes.astype(float)
        if df_num[col].isnull().any():
            df_num[col] = df_num[col].fillna(0.0)
    ```
  Key structural change: the dtype branches now produce `codes` only. The shared sentinel logic (max_code + 1, assignment, float cast, NaN fill) runs once after the branches. An explicit `else: ... continue` handles non-categorical/non-string/non-object columns (the existing code fell through to the astype(float)/fillna block for these, which is preserved).
      </spec>
      <dependencies>C1 (C1 modifies shap_utils.py imports; C6 modifies body. No overlap.)</dependencies>
      <risk>low - sentinel logic is identical across all 3 branches; structural refactor only</risk>
      <rollback>Restore the three-branch inline sentinel pattern.</rollback>
    </change>

    <change id="C7" priority="P1" source_item="F7">
      <file path="src/boost_shap_gii/utils.py" action="modify" />
      <file path="src/boost_shap_gii/train.py" action="modify" />
      <file path="src/boost_shap_gii/predict.py" action="modify" />
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>Extract coerce_ordinal_column to utils.py; replace 3 duplicated ordinal coercion blocks.</description>
      <spec>
utils.py: Add new function. Place after load_dataframe (from C4) or after save_json_atomic if C4 has not yet been applied. The function must be placed in the file's utility function region, before the validator functions (validate_cv_config starts at ~line 600):
    ```python
    def coerce_ordinal_column(series: pd.Series, levels: list, column_name: str) -> pd.Series:
        """Coerce an ordinal column to integer codes with two-tier unknown-value validation.

        Applies quote normalization, validates against known levels (hard error if >50%
        of unique values are unknown; warning if >10% of observations are unknown),
        and converts to ordered CategoricalDtype integer codes with NaN preservation.
        """
        levels = [_normalize_quotes(l) for l in levels]
        series = series.map(lambda v: _normalize_quotes(v) if isinstance(v, str) else v)

        unique_vals = series.dropna().unique()
        unknowns = [v for v in unique_vals if v not in levels]
        if unknowns:
            unknown_frac = len(unknowns) / len(unique_vals)
            if unknown_frac > 0.5:
                raise ValueError(
                    f"Feature '{column_name}': {unknown_frac:.0%} of unique values not in YAML levels "
                    f"{levels}. Check for case mismatches or missing level definitions."
                )
            print(f"[WARNING] Feature '{column_name}': {len(unknowns)} unique value(s) not in YAML levels: {unknowns}")
            obs_vals = series.dropna()
            n_unknown_obs = sum(v not in levels for v in obs_vals)
            obs_frac = n_unknown_obs / len(obs_vals) if len(obs_vals) > 0 else 0.0
            if obs_frac > 0.10:
                print(
                    f"[WARNING] Feature '{column_name}': {obs_frac:.1%} of non-missing observations "
                    f"({n_unknown_obs}/{len(obs_vals)}) have values not in YAML levels. "
                    f"This may indicate systematic data quality issues."
                )

        cat_type = pd.CategoricalDtype(categories=levels, ordered=True)
        src = series.where(series.isin(levels) | series.isna(), other=pd.NA)
        codes = src.astype(cat_type).cat.codes.astype("Int64")
        codes[codes == -1] = pd.NA
        return codes
    ```

train.py (lines 907-941):
  Replace the ordinal for-loop body with:
    ```python
    for c in ord_feats:
        levels = selector.feature_metadata[c]['levels']
        X[c] = coerce_ordinal_column(X[c], levels, c)
    ```
  Add `coerce_ordinal_column` to the existing utils import block.

predict.py (lines 181-216):
  Replace the ordinal for-loop body with:
    ```python
    for c in ord_feats:
        levels = feature_meta[c]['levels']
        X[c] = coerce_ordinal_column(df_raw[c], levels, c)
    ```
  Add `coerce_ordinal_column` to the existing utils import block.

infer.py (lines 175-207):
  Replace the ordinal for-loop body with:
    ```python
    for c in ord_feats:
        levels = feature_meta[c]['levels']
        X[c] = coerce_ordinal_column(df_raw[c], levels, c)
    ```
  Add `coerce_ordinal_column` to the existing utils import block.

  Note: predict.py and infer.py pass df_raw[c] (not X[c]) because the raw data column contains the original ordinal strings. train.py passes X[c] which is a copy of df_raw[final_cols] (line 772: `X = df_raw[final_cols].copy()`). The quote normalization in coerce_ordinal_column operates on the passed Series, not df_raw directly, so this is safe: predict/infer callers do not mutate df_raw because the .map() inside coerce_ordinal_column returns a new Series. However, predict.py line 183 currently mutates df_raw[c] in place (`df_raw[c] = df_raw[c].map(...)`). The extracted function avoids this mutation by operating on the passed series copy. This is a behavioral improvement: the current predict.py/infer.py code mutates df_raw as a side effect, which is not needed since only X[c] is used downstream.
      </spec>
      <dependencies>C1 (import block edits in predict.py and infer.py), C4 (import block edits in all 3 callers and utils.py body)</dependencies>
      <risk>medium - largest refactor; the extracted function must reproduce the exact same validation thresholds, error messages, and coercion behavior. The df_raw mutation elimination is a deliberate behavioral improvement (no downstream code reads the ordinal column from df_raw after this point).</risk>
      <rollback>Remove coerce_ordinal_column from utils.py; restore inline blocks in all 3 callers.</rollback>
    </change>
  </changes>
  <execution_order>C1, C2, C3, C4, C5, C6, C7</execution_order>
</implement_plan>
