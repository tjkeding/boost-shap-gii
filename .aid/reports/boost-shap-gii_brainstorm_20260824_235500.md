<brainstorm_report>
  <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-08-24T23:55:00-04:00" />
  <context_files>
    <file path="boost-shap-gii_implement_plan_20260824_230000.md" relevance="Handoff implement plan from CFTSI-behavioral specifying H1-H4 changes for required_cols NaN handling" />
    <file path="src/boost_shap_gii/train.py" relevance="H1/H4 target: required_cols row-drop insertion point (lines 680-686) and belt-and-suspenders assertion (lines 861-862)" />
    <file path="src/boost_shap_gii/predict.py" relevance="H2 target: mirrored required_cols row-drop (lines 110-114) and transform_config.json early read" />
    <file path="src/boost_shap_gii/infer.py" relevance="H3 target: NaN-baseline warning before prediction loop (lines 275-291)" />
    <file path="src/boost_shap_gii/utils.py" relevance="validate_transform_config() (line 575) validates column existence only, not per-row completeness; this is the gap H1-H3 address" />
  </context_files>
  <topics>
    <topic id="T1" title="H1/H2 implementation review">
      <summary>Validated the plan's core approach (drop NaN required_cols rows with diagnostic message, mirroring the outcome-missing drop pattern) and identified three code quality refinements to the plan's implementation details.</summary>
      <research>No external research required. All findings verified against current codebase (v1.5.0 at commit 640cc62).</research>
      <approaches>
        <approach id="A1" label="Plan as-is" feasibility="high" risk="low">
          <description>Execute H1 and H2 exactly as specified in the handoff plan.</description>
          <pros>Directly executable; minimal deviation from the handoff document.</pros>
          <cons>Three code quality issues: (a) tx_cfg variable shadowing in H1 collides with line 771's tx_cfg definition; (b) H2 omits the active flag check that predict.py's later transform_config.json read performs (lines 254-257), creating an implicit assumption that file existence implies active=True; (c) H2 omits the empty-data guard present in H1, creating an asymmetry in the train/predict mirroring contract.</cons>
        </approach>
        <approach id="A2" label="Plan with refinements" feasibility="high" risk="low">
          <description>Execute H1 and H2 with three refinements: (1a) inline the config access in H1 as _tx_required_cols = config.get("transformations", {}).get("required_cols", []) to eliminate the tx_cfg shadowing; (1b) add if _tx_info_early.get("active", False) guard in H2's early transform_config.json read; (1c) add if len(df_raw) == 0: raise ValueError(...) guard in H2 after the row-drop. Also confirmed H2's source-of-truth (transform_config.json rather than config) is architecturally correct.</description>
          <pros>Eliminates variable shadowing; aligns H2 with predict.py's existing active-check pattern; completes the train/predict mirroring contract; no change to approach or architecture.</pros>
          <cons>Marginal additional code (2-3 lines per refinement).</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A2">All three refinements approved by user. H2 source-of-truth (transform_config.json) confirmed correct: predict.py should derive transform behavior from training-time artifacts, not the current YAML config.</decision>
    </topic>
    <topic id="T2" title="H3/H4 scope review">
      <summary>Confirmed H3 (infer.py warning-only) is correct for infer.py's "predict all samples" contract, and H4 (belt-and-suspenders assertion) provides genuine, low-cost regression protection.</summary>
      <research>No external research required. Verified from codebase architecture.</research>
      <approaches>
        <approach id="A1" label="H3 warning-only" feasibility="high" risk="low">
          <description>Warning when required_cols contain NaN; no row drop. NaN in back-transformed predictions correctly signals that back-transformation is undefined for those rows.</description>
          <pros>Preserves infer.py's contract (one prediction per input row); NaN semantics are correct (back-transformation undefined without baseline values); warning gives user full information to handle externally.</pros>
          <cons>Raw predictions in transformed space are overwritten by NaN back-transformed predictions. The transformed-space prediction is lost. This is an acceptable trade-off for the immediate fix; preserving both would be a separate enhancement.</cons>
        </approach>
        <approach id="A2" label="H4 belt-and-suspenders assertion" feasibility="high" risk="low">
          <description>Internal assertion after smoke test verifying no NaN remains in required_cols after H1's drop. Fires only on pipeline logic bugs.</description>
          <pros>5 lines, zero runtime cost, self-documenting with [INTERNAL] prefix; guards against future regressions where pipeline reordering bypasses H1's drop; converts opaque LAPACK errors into clear pipeline diagnostics.</pros>
          <cons>Redundant by construction if H1 is well-tested; trivial maintenance coupling (reads same tx_cfg and columns as H1).</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1,A2">Both H3 and H4 kept as specified in the handoff plan. No modifications needed. H4 uses tx_cfg from line 771 (correct scope after T1's 1a refinement eliminates the shadowing).</decision>
    </topic>
  </topics>
  <action_items>
    <item priority="P0" target_mode="implement" description="Execute H1-H4 from the handoff plan (boost-shap-gii_implement_plan_20260824_230000.md) with three T1 refinements: (1a) inline H1's config access as _tx_required_cols to avoid tx_cfg shadowing; (1b) add active flag check in H2's early transform_config.json read; (1c) add empty-data guard in H2 after row-drop. H3 and H4 unchanged." />
    <item priority="P0" target_mode="test" description="Test coverage for: train.py with NaN in required_cols (drop and proceed); predict.py with NaN in required_cols (drop and proceed); infer.py with NaN in required_cols (warn and produce NaN); no-transform config (H1/H2 are no-ops); H4 internal assertion (should not fire under normal conditions)." />
  </action_items>
  <next_steps>Proceed to /implement to execute H1-H4 with the three T1 refinements, consuming both the handoff plan and this brainstorm report as input. Then /test to cover the new row-drop paths.</next_steps>
</brainstorm_report>
