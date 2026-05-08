<implement_plan>
 <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-03-30T18:48:53Z" />
 <input_reports>
 <report path="./boost-shap-gii_brainstorm_20260330_165306.md" mode="brainstorm" key_items="2 ( — zero-inflation diagnostic need, Huber loss consideration)" />
 <report path="./boost-shap-gii_brainstorm_20260330_172704.md" mode="brainstorm" key_items="5 (topics 1-5 — diagnostic battery, auto-loss literature, MAD-based delta, auto-mode architecture, asymmetric risk framework)" />
 </input_reports>

 <scope_note>
 The user explicitly scoped the implementation to TWO P1 changes:
 1. Diagnostic warning in train.py (advisory only; no auto-switching; regression/multi_regression only).
 2. Documentation updates to example_config_advanced.yaml and INPUT_SPECIFICATION.md.

 Items from the brainstorm reports that are OUT OF SCOPE for this implementation:
 - The full "auto" loss_function mode (no changes to _TASK_LOSS_SCORING, no "auto" option, no outcome_diagnostics.json output artifact).
 - The diptest dependency addition.
 - Changes to predict.py infer.py.
 - Changes to example_config_minimal.yaml defaults.
 - Changes to README.md.
 - The config_outcome_b.yaml fdr_correct fix (separate P0 item from the first brainstorm, not specified by user).
 - Fold-level diagnostic CSV (P1 from first brainstorm, not specified by user).
 - Tweedie loss support (P2).
 </scope_note>

 <changes>
 <change id="change-1" priority="P1" source_item="Brainstorm 172704 — diagnostic battery + MAD-based delta formula">
 <file path="src/boost_shap_gii/train.py" action="modify" />
 <description>
 Add a diagnostic warning block in train.py after y is constructed (line ~442) but before
 CV splitting (line ~569). For regression and multi_regression tasks only, compute three
 outcome distribution diagnostics per target column:

 1. Zero-inflation rate: fraction of y == 0 (threshold: >= 0.15)
 2. Skewness: scipy.stats.skew (threshold: |skewness| >= 2.0)
 3. Excess kurtosis: scipy.stats.kurtosis (Fisher=True) (threshold: >= 5.0)

 Emit WARNING-level print messages (matching existing [WARNING] convention) if any
 threshold is met. The warning message distinguishes:
 - Right-skew + zero-inflation: recommends Huber loss with MAD-based delta formula.
 - Left-skew or symmetric heavy tails: notes Huber may still help, but the pattern is
 less common and warrants manual inspection.
 - Pure heavy tails (kurtosis only): notes potential outlier influence on RMSE gradients.

 For multi_regression tasks, each target column is diagnosed independently.

 The warning includes:
 - The specific diagnostic values that triggered it.
 - The MAD-based delta formula: delta = 1.345 * 1.4826 * MAD(y).
 - The computed delta value for the user's data.
 - A reference to INPUT_SPECIFICATION.md Section 9 for the full derivation.

 The warning is ADVISORY ONLY. The pipeline continues with whatever loss the user
 specified. No config modification at runtime.
 </description>
 <spec>
 Location: train.py, after line 442 (y = df_raw[outcome_cols[0]].copy) and before
 line 444 (# A. Force Nominal to String -> Category).

 New import at top of train.py:
 from scipy import stats as sp_stats

 New function (module-level, in the "1. Helper Utilities" section after report_missingness):
 def _diagnose_outcome_distribution(y_series: pd.Series, col_name: str) -> None:
 """Emit advisory warnings for outcome distributions that may degrade RMSE loss.

 Computes zero-inflation rate, skewness, and excess kurtosis. If any threshold
 is exceeded, prints a [WARNING] with the diagnostic values, the recommended
 MAD-based Huber delta, and a reference to INPUT_SPECIFICATION.md.

 Parameters
 ----------
 y_series: pd.Series
 Outcome values (NaN-dropped).
 col_name: str
 Outcome column name for log messages.
 """
 y_vals = y_series.dropna.to_numpy(dtype=np.float64)
 n = len(y_vals)
 if n < 10:
 return # Too few observations for meaningful diagnostics

 # Compute diagnostics
 zero_frac = np.mean(y_vals == 0)
 skewness = sp_stats.skew(y_vals, bias=False)
 excess_kurt = sp_stats.kurtosis(y_vals, fisher=True, bias=False)

 # Compute MAD-based Huber delta (always, for inclusion in warning message)
 mad = np.median(np.abs(y_vals - np.median(y_vals)))
 delta = 1.345 * 1.4826 * mad

 # Check thresholds
 zero_flag = zero_frac >= 0.15
 skew_flag = abs(skewness) >= 2.0
 kurt_flag = excess_kurt >= 5.0

 if not (zero_flag or skew_flag or kurt_flag):
 return # No concerns

 # Build warning message
 header = (f"[WARNING] Outcome distribution diagnostic for '{col_name}' "
 f"(n={n}):"
)
 details =
 if zero_flag:
 details.append(f" Zero-inflation: {zero_frac:.1%} of observations are zero (threshold: 15%)")
 if skew_flag:
 direction = "right" if skewness > 0 else "left"
 details.append(f" Skewness: {skewness:.3f} ({direction}-skewed; threshold: |skew| >= 2.0)")
 if kurt_flag:
 details.append(f" Excess kurtosis: {excess_kurt:.3f} (heavy-tailed; threshold: >= 5.0)")

 # Determine message type
 if skewness > 0 and zero_flag:
 recommendation = (" This combination of right-skew and zero-inflation typically degrades\n"
 " RMSE loss by inflating gradients for extreme residuals. Consider using\n"
 f" Huber loss: loss_function: \"Huber:delta={delta:.4f}\"\n"
 f" (delta = 1.345 * 1.4826 * MAD(y) = {delta:.4f}; "
 f"MAD = {mad:.4f})"
)
 elif skewness < 0 and (zero_flag or skew_flag):
 recommendation = (" Left-skewed outcome distribution detected. Huber loss may still reduce\n"
 " outlier influence on RMSE gradients, but this pattern is less common\n"
 " and warrants manual inspection of the outcome distribution.\n"
 f" If using Huber: loss_function: \"Huber:delta={delta:.4f}\"\n"
 f" (delta = 1.345 * 1.4826 * MAD(y) = {delta:.4f}; "
 f"MAD = {mad:.4f})"
)
 else:
 recommendation = (" Heavy tails or skewness may cause outlier-driven gradient inflation\n"
 " under RMSE loss. Consider using Huber loss to cap residual influence.\n"
 f" If using Huber: loss_function: \"Huber:delta={delta:.4f}\"\n"
 f" (delta = 1.345 * 1.4826 * MAD(y) = {delta:.4f}; "
 f"MAD = {mad:.4f})"
)

 recommendation += ("\n See INPUT_SPECIFICATION.md Section 9 for the full derivation and literature."
)

 print(header)
 for d in details:
 print(d)
 print(recommendation)

 Insertion in main flow (after y is constructed, before type enforcement):
 # Outcome Distribution Diagnostics (regression/multi_regression only)
 task_prelim = detect_task(config)
 if is_regression(task_prelim):
 if isinstance(y, pd.DataFrame):
 for col in y.columns:
 _diagnose_outcome_distribution(y[col], col)
 else:
 _diagnose_outcome_distribution(y, outcome_cols[0])
 </spec>
 <dependencies>None (first change)</dependencies>
 <risk>Low — advisory print statements only; no control flow or data modification; uses existing scipy dependency; follows existing [WARNING] print convention.</risk>
 <rollback>Remove the _diagnose_outcome_distribution function and the call site in main. Remove the scipy import.</rollback>
 </change>

 <change id="change-2" priority="P1" source_item="User instruction #2 — example_config_advanced.yaml loss_function comment expansion">
 <file path="example_config_advanced.yaml" action="modify" />
 <description>
 Expand the loss_function comment block (currently lines 67-71) with guidance on when
 to use Huber loss and the MAD-based delta formula. The expanded comments should cover:
 - When Huber is recommended (zero-inflated, right-skewed, heavy-tailed outcomes).
 - The MAD-based delta formula: delta = 1.345 * 1.4826 * MAD(y).
 - The interpretation: k=1.345 yields 95% asymptotic relative efficiency at the normal
 model (Huber, 1981); 1.4826 is the consistency factor for MAD (Maronna et al., 2006).
 - Example CatBoost syntax: "Huber:delta=VALUE".
 - Note that the pipeline emits a diagnostic warning when the outcome distribution
 suggests Huber may be beneficial.
 </description>
 <spec>
 Replace lines 67-71 (the comment block above loss_function: "RMSE") with an expanded
 block. The loss_function value itself remains "RMSE" (this is the example config;
 users change it themselves).

 Current (lines 66-71):
 # Optimization metric for TRAINING
 # Regression options: "RMSE", "MAE", "Huber", "Quantile"
 # Multi-regression option: "MultiRMSE"
 # Binary classification options: "Logloss", "CrossEntropy"
 # Multiclass classification options: "MultiClass", "MultiClassOneVsAll"
 loss_function: "RMSE"

 Replace with expanded comment block (see build submodule for exact text).
 </spec>
 <dependencies>None</dependencies>
 <risk>Low — documentation-only change to a YAML comment block; no functional impact.</risk>
 <rollback>Restore original 5-line comment block.</rollback>
 </change>

 <change id="change-3" priority="P1" source_item="User instruction #2 — INPUT_SPECIFICATION.md outcome distribution section">
 <file path="INPUT_SPECIFICATION.md" action="modify" />
 <description>
 Add a new Section 9 ("Outcome Distribution Considerations") after the current Section 8
 ("Edge Cases and Known Limitations"). The section covers:

 1. Diagnostic criteria and thresholds with literature sources:
 - Zero-inflation >= 15% (Olsen and Schafer, 2001; Tooze et al., 2002)
 - |Skewness| >= 2.0 (Kim, 2013)
 - Excess kurtosis >= 5.0 (Kim, 2013, conservative from 7.0 threshold)
 2. Why formal normality tests (Shapiro-Wilk, etc.) are inappropriate as automated
 triggers (Razali and Wah, 2011: asymptotic consistency issue).
 3. Huber loss theory and the MAD-based delta derivation:
 - delta = k * sigma_hat = 1.345 * 1.4826 * MAD(y)
 - k = 1.345: yields 95% ARE at the normal (Huber, 1981)
 - 1.4826 = 1 Phi^{-1}(3/4): MAD consistency factor (Maronna et al., 2006)
 - CatBoost syntax: "Huber:delta=VALUE"
 4. Asymmetric risk framework: false trigger costs 5% efficiency loss vs. missed
 detection costs potentially catastrophic RMSE gradient bias.
 5. Classification exclusion: classification tasks are not warned because Logloss/
 CrossEntropy are inherently robust to moderate class imbalance through
 probability-based gradient computation.
 </description>
 <spec>
 Append after the final line of Section 8 (line 447: " (column `slice_idx` of the 4D
 interaction tensor).").

 New section with ### 9 heading, full subsection structure. See build submodule for
 exact content.
 </spec>
 <dependencies> (the diagnostic references "Section 9" in its warning message)</dependencies>
 <risk>Low — documentation-only addition; no functional impact. Risk of section number collision if future sections are added, but this follows the existing sequential numbering scheme.</risk>
 <rollback>Remove Section 9 from INPUT_SPECIFICATION.md.</rollback>
 </change>
 </changes>

 <execution_order> (and are independent of each other but 's section number is referenced by 's warning message, so 's code should be finalized first to confirm the reference)</execution_order>
</implement_plan>
