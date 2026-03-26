<implement_plan>
  <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-03-26T16:44:34Z" />
  <input_reports>
    <report path="(inline specification from user)" mode="user-directive" key_items="5" />
  </input_reports>

  <assumptions>
    1. The .gitignore uses an ignore-everything-then-whitelist pattern; new whitelist entries must be appended.
    2. No writing-* reports exist in the history directories (confirmed by ls).
    3. Session UUID mapping: session-1-prefix -> session-1, session-2-prefix -> session-2, session-3-prefix -> session-3, session-4-prefix -> session-4 (chronological order based on session_history.md).
    4. The GitHub URL https://github.com/tjkeding/boost-shap-gii is explicitly excluded from username sanitization.
    5. The sanitization of the project CLAUDE.md preserves the GitHub URL intact.
  </assumptions>

  <changes>
    <change id="C1" priority="P0" source_item="Task 1: Create directory structure">
      <file path=".ai_development/reports/" action="create" />
      <description>Create the .ai_development/reports/ directory tree in the project root.</description>
      <spec>mkdir -p ".ai_development/reports/"</spec>
      <dependencies>none</dependencies>
      <risk>low - directory creation only</risk>
      <rollback>rm -rf .ai_development/</rollback>
    </change>

    <change id="C2" priority="P0" source_item="Task 2: Copy and sanitize structured reports">
      <file path=".ai_development/reports/*.md" action="create" />
      <description>
        Copy 15 report files from Claude project history directories plus 8 files from today's working directory into .ai_development/reports/. Apply sanitization sed replacements to each file.

        Source files (15 from history):
          brainstorm_history/boost-shap-gii_brainstorm_20260309_010000.md
          brainstorm_history/boost-shap-gii_brainstorm_20260316_000000.md
          brainstorm_history/boost-shap-gii_brainstorm_20260316_040000.md
          cr_history/boost-shap-gii_cr_20260309_000000.md
          clean_history/boost-shap-gii_clean_20260316_080000.md
          document_history/boost-shap-gii_document_20260316_060000.md
          implement_history/boost-shap-gii_implement_build_20260316_020000.md
          implement_history/boost-shap-gii_implement_build_20260316_040000.md
          implement_history/boost-shap-gii_implement_build_20260316_090100.md
          implement_history/boost-shap-gii_implement_plan_20260316_020000.md
          implement_history/boost-shap-gii_implement_plan_20260316_040000.md
          implement_history/boost-shap-gii_implement_plan_20260316_090000.md
          test_history/boost-shap-gii_test_20260316_030000.md
          test_history/boost-shap-gii_test_20260316_040000.md
          session_memories/boost-shap-gii_MEMORY_20260316_100000.md

        Source files (8 from today's working directory):
          boost-shap-gii_implement_plan_20260326_104459.md
          boost-shap-gii_implement_build_20260326_104459.md
          boost-shap-gii_implement_plan_20260326_154719.md
          boost-shap-gii_implement_build_20260326_154719.md
          boost-shap-gii_implement_plan_20260326_155902.md
          boost-shap-gii_implement_build_20260326_160311.md
          boost-shap-gii_test_20260326_110733.md
          boost-shap-gii_test_20260326_155522.md

        Sanitization rules (applied via sed in order):
          1. s|<conda_env>|<conda_env>|g
          2. s|<claude_project>/|<claude_project>/|g
          3. s|<claude_config>/|<claude_config>/|g
          4. s|<sandbox>|<sandbox>|g
          5. s|<project_root>/|<project_root>/|g
          6. s|<user>/|<user>/|g
          7. s|session-1|session-1|g
          8. s|session-2|session-2|g
          9. s|session-3|session-3|g
          10. s|session-4|session-4|g
          11. Protect GitHub URL then sanitize remaining <user>, then restore:
              s|github.com/tjkeding/|github.com/tjkeding/|g
              s|<user>|<user>|g
              s|github.com/tjkeding/|github.com/tjkeding/|g

        Order matters: longer paths must be replaced before shorter ones to avoid partial matches.
      </description>
      <spec>Shell script using cp + sed pipeline for each file.</spec>
      <dependencies>C1</dependencies>
      <risk>low - file copy with text substitution, originals untouched</risk>
      <rollback>rm .ai_development/reports/*.md</rollback>
    </change>

    <change id="C3" priority="P0" source_item="Task 3: Copy project CLAUDE.md">
      <file path=".ai_development/project_claude.md" action="create" />
      <description>Copy CLAUDE.md from project root to .ai_development/project_claude.md with same sanitization rules applied.</description>
      <spec>cp + sed pipeline, same substitutions as C2.</spec>
      <dependencies>C1</dependencies>
      <risk>low - file copy with text substitution</risk>
      <rollback>rm .ai_development/project_claude.md</rollback>
    </change>

    <change id="C4" priority="P0" source_item="Task 4: Create AI_DEVELOPMENT_LOG.md">
      <file path="AI_DEVELOPMENT_LOG.md" action="create" />
      <description>Create the top-level AI transparency disclosure document with 7 sections per the AID Framework specification provided by the user. Professional, third-person, academic tone. No emojis. No user identity references.</description>
      <spec>Write file directly with structured markdown content covering: Purpose, Scope, Tools Used, Development Workflow, Human Oversight, Audit Trail, References.</spec>
      <dependencies>none</dependencies>
      <risk>low - new file creation</risk>
      <rollback>rm AI_DEVELOPMENT_LOG.md</rollback>
    </change>

    <change id="C5" priority="P0" source_item="Task 5: Update .gitignore">
      <file path=".gitignore" action="modify" />
      <description>Append whitelist entries for AI_DEVELOPMENT_LOG.md and the .ai_development/ directory tree.</description>
      <spec>
        Append to .gitignore:
          # AI development transparency artifacts
          !AI_DEVELOPMENT_LOG.md
          !.ai_development/
          !.ai_development/reports/
          !.ai_development/reports/*.md
          !.ai_development/project_claude.md
      </spec>
      <dependencies>none</dependencies>
      <risk>low - append-only modification</risk>
      <rollback>Remove the appended lines</rollback>
    </change>
  </changes>

  <execution_order>C1, C2 (depends on C1), C3 (depends on C1), C4 (independent), C5 (independent)</execution_order>
  <parallel_groups>
    <group step="1">C1, C4, C5</group>
    <group step="2">C2, C3</group>
  </parallel_groups>
</implement_plan>
