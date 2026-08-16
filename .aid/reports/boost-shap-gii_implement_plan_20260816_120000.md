<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-16T12:00:00-04:00" />
  <input_reports>
    <report path="(conversation)" mode="user_report" key_items="2" />
  </input_reports>
  <changes>
    <change id="C1" priority="P1" source_item="conversation: spline warning verbosity">
      <file path="src/boost_shap_gii/shap_utils.py" action="modify" />
      <description>Replace per-call spline downgrade print with a single upfront diagnostic emitted once at the start of SHAP analysis. The diagnostic inspects each non-nominal feature's unique interior value count in the full dataset and reports a summary table of features whose spline degree will be downgraded from cubic to linear.</description>
      <spec>
1. Remove the print statement at line 180-181 inside `_get_adaptive_knots_and_degree`. The function continues to return `(knots, 1)` silently.

2. Add a new private function `_diagnose_spline_downgrades(X: pd.DataFrame, feature_types: Dict[str, str], spline_cfg: Dict[str, int]) -> None` in the "IO and Helpers" section (after `_get_effect_stratum`, before `_get_adaptive_knots_and_degree`). Logic:

   a. Extract `n_knots = spline_cfg["n_knots"]`, `degree = spline_cfg["degree"]`.
   b. If `degree <= 1`, return immediately (no downgrade possible).
   c. For each feature name in X.columns where `feature_types.get(name, "continuous")` is not `"nominal"` and name does not start with `"shadow_"`:
      - Compute `vals = X[name].values` (the full-dataset column).
      - Apply the same knot-generation logic as `_get_adaptive_knots_and_degree`:
        `quantiles = np.linspace(0, 100, n_knots + 2)[1:-1]`
        `knots = np.percentile(vals, quantiles)`
        `knots = np.unique(knots)`
        `knots = knots[(knots > np.min(vals)) & (knots < np.max(vals))]`
      - If `len(knots) < 4`: append to a `downgraded` list as `(name, len(knots))`.
   d. If `downgraded` is empty, return (no diagnostic emitted).
   e. Print a single summary block:
      `[SHAP] Spline degree will be downgraded from {degree} to 1 for {len(downgraded)} feature(s):`
      `[SHAP]   (fewer than 4 unique interior knots in the full dataset)`
      For each `(name, n_knots_actual)` in `downgraded`:
        `[SHAP]     {name} ({n_knots_actual} interior knot(s))`
      `[SHAP]   Interactions involving these features will also use degree-1 splines.`
      `[SHAP]   This is expected for low-cardinality ordinal/continuous features and does not affect correctness.`

3. Call `_diagnose_spline_downgrades` in `run_shap_pipeline`, immediately after the shadow feature type map is built (after line 1557, before the "Determine slices" block at line 1559). Pass `X_aligned`, `all_feature_types`, and `config["shap"]["splines"]`.

Rationale: Placing the call in `run_shap_pipeline` ensures the diagnostic fires exactly once per pipeline invocation (for both predict and infer modes), before any fold processing begins. Using the full-dataset column values provides a deterministic lower bound on knot count (bootstrap resamples can only reduce unique value count, never increase it), so any feature flagged here will be downgraded in every bootstrap iteration.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - removes a print, adds a diagnostic function with no effect on SHAP computation logic</risk>
      <rollback>Revert shap_utils.py to prior state (restore print in _get_adaptive_knots_and_degree, remove _diagnose_spline_downgrades and its call site).</rollback>
    </change>
    <change id="C2" priority="P0" source_item="conversation: psutil missing from dependencies">
      <file path="pyproject.toml" action="modify" />
      <file path="environment.yaml" action="modify" />
      <file path="src/boost_shap_gii/check_env.py" action="modify" />
      <description>Add psutil as a hard dependency. It is required by the indiv_reports memory guard (introduced in Session 9, F1 remediation) but was never declared in any dependency manifest. Also add it to check_env.py's PYTHON_DEPS list so `boost-shap-gii check-env` catches its absence before the pipeline crashes at the indiv_reports stage.</description>
      <spec>
1. pyproject.toml: Add `"psutil"` to the `dependencies` list (after `"catboost"`, before `"optuna"`; alphabetical by convention is not enforced, but grouping near other utility deps is fine).

2. environment.yaml: Add `- psutil` to the `dependencies` list (after `pyarrow`, before `pip`). psutil is available on conda-forge (23 versions, latest 7.2.2) and installs cleanly.

3. check_env.py: Add `"psutil"` to the `PYTHON_DEPS` list (line 8-10). Insert alphabetically or at end; the list is not ordered.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - additive dependency declaration; no code logic changes</risk>
      <rollback>Remove psutil from all three files.</rollback>
    </change>
  </changes>
  <execution_order>C2, C1</execution_order>
</implement_plan>
