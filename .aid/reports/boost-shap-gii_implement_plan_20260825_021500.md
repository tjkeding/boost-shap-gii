<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-08-25T02:15:00-04:00" />
  <input_reports>
    <report path="boost-shap-gii_test_20260825_001500.md" mode="test" key_items="2" />
  </input_reports>
  <resolved_decisions>
    <decision id="D1" source="P0 action item, two candidate approaches">
      <question>Uniform finite-prediction mask (approach a) versus per-branch per-column masks (approach b)?</question>
      <resolution>Approach (a): single uniform mask. output_transform receives the full prediction vector; NaN in required_cols propagates to all outcomes for a row (baseline is per-row, not per-column). Per-column masks would produce identical results in practice, and approach (a) is simpler to verify.</resolution>
    </decision>
  </resolved_decisions>
  <changes>
    <change id="C1" priority="P0" source_item="test_report action_items[0]">
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>Fix three supervised_mask-gated blocks to exclude rows with non-finite predictions before calling sklearn scoring functions. Currently, supervised_mask (line 132) filters only by outcome non-nullity. When output_transform produces NaN predictions (correctly, per the C3 warn-and-proceed contract), those NaN values reach sklearn metric functions unfiltered and crash with "Input contains NaN". Fix adds a finite-prediction intersection mask at each of the three affected sites.</description>
      <spec>
Three insertion sites, all within infer.py's main() function:

**Site 1: Per-model metrics (lines 323-365, inside the per-model loop)**

After line 328 (after fold_preds is finalized, including the conditional scaler inverse-transform), insert a per-model finite mask computation:

```python
            if fold_preds.ndim > 1:
                _pm_finite = np.all(np.isfinite(fold_preds), axis=1)
            else:
                _pm_finite = np.isfinite(fold_preds)
            _pm_mask = supervised_mask & _pm_finite
            if int(_pm_mask.sum()) == 0:
                continue
```

Then replace every `supervised_mask` reference in the per-model metrics branches (lines 331, 335, 341, 342, 350, 351) with `_pm_mask`. Six replacements total:
- Line 331: `df_raw[outcome_cols].values[supervised_mask]` -> `[_pm_mask]`
- Line 335: `fold_preds[supervised_mask][:, t_idx]` -> `fold_preds[_pm_mask][:, t_idx]`
- Line 341: `df_raw[outcome_cols[0]].values[supervised_mask]` -> `[_pm_mask]`
- Line 342: `fold_preds[supervised_mask]` -> `fold_preds[_pm_mask]`
- Line 350: `df_raw[outcome_cols[0]].values[supervised_mask]` -> `[_pm_mask]`
- Line 351: `fold_preds[supervised_mask]` -> `fold_preds[_pm_mask]`

**Site 2: Ensemble metrics (lines 377-454)**

Insert a new block BEFORE the existing `if has_outcomes and n_supervised > 0:` guard at line 378. This block computes the scorable mask and prints diagnostics:

```python
    n_scorable = 0
    _scorable_mask = supervised_mask
    if has_outcomes and n_supervised > 0:
        if ensemble_preds.ndim > 1:
            _ens_finite = np.all(np.isfinite(ensemble_preds), axis=1)
        else:
            _ens_finite = np.isfinite(ensemble_preds)
        _scorable_mask = supervised_mask & _ens_finite
        n_scorable = int(_scorable_mask.sum())
        _n_excl = n_supervised - n_scorable
        if _n_excl > 0:
            print(f"[WARNING] {_n_excl} supervised row(s) excluded from "
                  f"performance metrics due to non-finite predictions "
                  f"(e.g. NaN from missing transformations.required_cols).")
        if n_scorable == 0:
            print("[WARNING] No rows with both ground-truth outcome and "
                  "finite prediction. Performance metrics skipped.")
```

Then change the existing guard at line 378 from:
```python
    if has_outcomes and n_supervised > 0:
```
to:
```python
    if has_outcomes and n_scorable > 0:
```

Within this block:
- Replace `n_supervised` in the print at line 379 with `n_scorable`.
- Replace all `supervised_mask` references with `_scorable_mask` at lines 396, 397, 415, 416, 430, 431. Six replacements total.

**Site 3: Permutation test (lines 473-505+, inside the same `if has_outcomes and n_scorable > 0:` block)**

Replace `supervised_mask` with `_scorable_mask` at lines 491 and 493. Two replacements total.

Summary: 14 `supervised_mask` -> `_pm_mask`/`_scorable_mask` replacements; 1 `n_supervised` -> `n_scorable` replacement in the print; 1 guard condition change; 1 new mask-computation block; 1 new per-model mask block.

No changes to `supervised_mask` or `n_supervised` variable definitions (lines 132-133, 141-142); they retain their original outcome-only semantics for use in metadata (line 642) and any future downstream consumers.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - surgical insertion of finite-prediction guards at well-defined sites; no structural changes to control flow; supervised_mask and n_supervised originals preserved for metadata</risk>
      <rollback>Remove the inserted mask computation blocks and revert the 14 supervised_mask replacements and the guard condition change.</rollback>
    </change>
    <change id="C2" priority="P1" source_item="test_report action_items[1]">
      <file path="tests/test_required_cols_nan_handling.py" action="modify" />
      <description>Fix test_c4_internal_assertion_fires_when_drop_bypassed: the NaN row indices collide with the smoke test's random sample (seed=42), causing the pre-existing smoke-test finite-value check at train.py:829 to fire before the belt-and-suspenders assertion is reached. Fix computes the smoke sample indices dynamically and places NaN rows outside that set.</description>
      <spec>
In function `test_c4_internal_assertion_fires_when_drop_bypassed` (line 207), replace the hardcoded `nan_idx = [0, 1]` (line 222) with a dynamic computation that mirrors train.py's smoke test sampling:

```python
    n = 30
    seed = 42  # matches config["execution"]["random_seed"]
    _smoke_rng = np.random.RandomState(seed)
    _n_smoke = min(20, n)
    _smoke_sample = set(_smoke_rng.choice(n, _n_smoke, replace=False))
    _non_smoke = sorted(set(range(n)) - _smoke_sample)
    nan_idx = _non_smoke[:2]
```

This ensures:
1. The NaN row indices are deterministic (derived from the same seed and n used by train.py's smoke test).
2. The NaN rows are guaranteed to fall outside the smoke test's random 20-of-30 sample.
3. Execution reaches the belt-and-suspenders assertion (train.py:884-889) instead of tripping the smoke test's finite-value check (train.py:827-832).
4. The test is robust to changes in the smoke test's sample size (n_smoke = min(20, len(df_raw))).

The rest of the function remains unchanged: the monkeypatch bypass, the config construction, and the pytest.raises assertion all work identically with the new nan_idx values.
      </spec>
      <dependencies>none</dependencies>
      <risk>low - test-only change; deterministic index computation mirrors the exact RNG path in train.py</risk>
      <rollback>Revert nan_idx to [0, 1] (restoring the known-failing test).</rollback>
    </change>
  </changes>
  <execution_order>C1, C2 (independent; ordered by priority)</execution_order>
</implement_plan>
