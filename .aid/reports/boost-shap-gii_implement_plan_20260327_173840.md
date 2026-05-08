<implement_plan>
 <meta project="boost-shap-gii" mode="implement" submodule="plan" timestamp="2026-03-27T17:38:40Z" />
 <input_reports>
 <report path="inline-user-specification" mode="user-request" key_items="1" />
 </input_reports>
 <changes>
 <change id="change-1" priority="P0" source_item="user-request: replace arrow with nanoparquet in plot.R">
 <file path="src/boost_shap_gii/scripts/plot.R" action="modify" />
 <description>Replace `arrow` with `nanoparquet` in the dependency comment (line 6) and the `library` call (line 43). The six `read_parquet` call sites (lines 199, 257, 262, 288, 289, 290) are unchanged because `nanoparquet::read_parquet` exports the same function name and returns a standard `data.frame`, which is fully compatible with downstream `dplyr`/`ggplot2` operations.</description>
 <spec>
 Line 6: "# Dependencies: ggplot2, dplyr, arrow, tidyr..." -> "# Dependencies: ggplot2, dplyr, nanoparquet, tidyr..."
 Line 43: "library(arrow)" -> "library(nanoparquet)"
 </spec>
 <dependencies>none</dependencies>
 <risk>low - drop-in replacement; same function name and compatible return type (data.frame)</risk>
 <rollback>Revert line 6 comment and line 43 library call back to `arrow`.</rollback>
 </change>
 <change id="change-2" priority="P0" source_item="consistency: update check_env.py R dependency list">
 <file path="src/boost_shap_gii/check_env.py" action="modify" />
 <description>Replace `"arrow"` with `"nanoparquet"` in the `R_DEPS` list (line 14) so that the pre-flight environment check validates the correct package.</description>
 <spec>Line 14: `"arrow"` -> `"nanoparquet"` in R_DEPS list</spec>
 <dependencies></dependencies>
 <risk>low - single string replacement in a list literal</risk>
 <rollback>Revert `"nanoparquet"` back to `"arrow"` in R_DEPS.</rollback>
 </change>
 <change id="change-3" priority="P1" source_item="consistency: update environment.yaml R package comments">
 <file path="environment.yaml" action="modify" />
 <description>Replace all three occurrences of `arrow` in the R dependencies comment block (lines 36, 40) with `nanoparquet`.</description>
 <spec>
 Line 36: "ggplot2, dplyr, arrow, tidyr..." -> "ggplot2, dplyr, nanoparquet, tidyr..."
 Line 40: `"arrow"` -> `"nanoparquet"` in install.packages example
 </spec>
 <dependencies></dependencies>
 <risk>low - comments only; no runtime impact</risk>
 <rollback>Revert `nanoparquet` back to `arrow` in comment block.</rollback>
 </change>
 <change id="change-4" priority="P1" source_item="consistency: update INPUT_SPECIFICATION.md">
 <file path="INPUT_SPECIFICATION.md" action="modify" />
 <description>Replace `arrow` with `nanoparquet` in the R verification description (line 61).</description>
 <spec>Line 61: "`arrow`" -> "`nanoparquet`"</spec>
 <dependencies></dependencies>
 <risk>low - documentation only</risk>
 <rollback>Revert `nanoparquet` back to `arrow` in line 61.</rollback>
 </change>
 </changes>
 <execution_order></execution_order>
</implement_plan>
