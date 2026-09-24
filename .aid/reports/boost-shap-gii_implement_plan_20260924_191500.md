<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-09-24T19:15:00-04:00" />
  <input_reports>
    <report path="boost-shap-gii_test_20260924_181500.md" mode="test" key_items="1" />
  </input_reports>
  <changes>
    <change id="C1" priority="P2" source_item="TestInferDataPathGuard action item">
      <file path="src/boost_shap_gii/infer.py" action="modify" />
      <description>Normalize the data_path checkpoint guard to compare absolute paths rather than raw strings. Without normalization, a relative path like "../data/file.csv" and its absolute equivalent "/full/path/data/file.csv" would trigger a spurious checkpoint invalidation even though they resolve to the same file.</description>
      <spec>Two one-line edits in infer.py, both replacing `args.data` with `os.path.abspath(args.data)`:

1. Line 89 (comparison site): change `existing_checkpoint.get("data_path") != args.data` to `existing_checkpoint.get("data_path") != os.path.abspath(args.data)`.

2. Line 117 (storage site): change `"data_path": args.data` to `"data_path": os.path.abspath(args.data)`.

No new imports needed (`os` is already imported and used at line 86 for `os.remove`). No other code paths reference the stored `data_path` value, so no downstream consumers need updating.</spec>
      <dependencies>none</dependencies>
      <risk>low - two token-level substitutions on adjacent lines in a single function; no logic change, no new imports, no behavioral change beyond path normalization</risk>
      <rollback>Revert both `os.path.abspath(args.data)` back to `args.data`.</rollback>
    </change>
  </changes>
  <execution_order>C1</execution_order>
</implement_plan>
