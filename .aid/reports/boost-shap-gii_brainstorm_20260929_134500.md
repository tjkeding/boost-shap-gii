<brainstorm_report>
  <meta project="boost-shap-gii" mode="brainstorm" timestamp="2026-09-29T13:45:00-04:00" />
  <context_files>
    <file path="src/boost_shap_gii/scripts/plot.R" relevance="Primary file: contains the foreach %dopar% loop, bootstrap_spline_sd, bootstrap_group_mean_sd, and ggsave calls that cause the hang" />
    <file path="src/boost_shap_gii/check_env.py" relevance="Verified ggtext is in the R dependency list (line 14) and passes preflight" />
  </context_files>
  <topics>
    <topic id="T1" title="Root cause: plotting hang under multi-core execution after v1.7.0">
      <summary>The v1.7.0 plotting code hangs indefinitely inside the foreach %dopar% loop (line 593) under multi-core execution. htop shows 1 of 48 cores active, load average ~2, and the Python CLI process sleeping at 0% CPU. With N approximately 300, bootstrap computation is trivially fast (seconds per feature), ruling out computation time. The root cause is fork-unsafe ggsave() combined with ggtext's Rcpp-based rendering in forked child processes.</summary>
      <research>
        <finding src="R parallel::mclapply documentation (https://stat.ethz.ch/R-manual/R-devel/library/parallel/html/mclapply.html)">R's official mclapply documentation warns: "Child processes should never use on-screen graphics devices." ggsave() opens a PNG device in each forked worker, falling into this category.</finding>
        <finding src="RcppParallel documentation (https://rdrr.io/cran/RcppParallel/man/isProcessForkedChild.html)">Rcpp packages in forked children are explicitly flagged as unstable: "packages invoked from within parallel::mclapply() should call isProcessForkedChild() and fall back to serial code."</finding>
        <finding src="Community reports (rstudio/rstudio#2732, rstudio/rstudio#2597, Posit Community)">Multiple reports document ggsave() inside mclapply() producing hangs or NULL returns.</finding>
        <finding src="gridtext DESCRIPTION (github.com/wilkelab/gridtext)">gridtext (ggtext's rendering engine) uses Rcpp for markdown parsing and text layout. It does NOT depend on systemfonts/fontconfig; the deadlock is from Rcpp execution in forked children during ggsave rendering, not from system font library initialization.</finding>
        <finding src="Git diff v1.6.0..v1.7.0">ggsave() was inside the foreach %dopar% body in v1.6.0 as well (confirmed at lines 649, 827, 1005 of the v1.6.0 plot.R). It worked because the render path used only standard ggplot2 elements (pure R, no Rcpp). The v1.7.0 addition of ggtext::element_markdown() at line 648 introduced Rcpp execution during rendering, destabilizing the previously marginal fork configuration.</finding>
        <finding src="Henrik Bengtsson, Wishlist-for-R#94 (github.com/HenrikBengtsson/Wishlist-for-R/issues/94)">PSOCK clusters (fresh R processes, not forks) are documented as the safe alternative for parallel workloads involving graphics devices or Rcpp packages.</finding>
      </research>
      <approaches>
        <approach id="A1" label="Compute parallel, render sequential" feasibility="high" risk="low">
          <description>Restructure the foreach %dopar% body into two phases. Phase 1 (%dopar%): each worker computes all data for its feature (filtered dataframes, bootstrap SDs, spline trends, stat-label strings, stratification results) and returns a structured list. No ggplot construction, no ggsave, no Rcpp execution. Phase 2 (sequential): the parent iterates over returned data, constructs ggplot objects (including element_markdown()), assembles grobs via arrangeGrob(), and saves via ggsave(). Per-feature progress output via cat() works naturally in the sequential phase.</description>
          <pros>Permanently eliminates all fork-safety concerns (no graphics devices, no Rcpp rendering in forked children). Retains parallel execution for computationally expensive work (bootstrap, spline fitting). Naturally enables per-feature progress reporting. Works for any dataset size (fork copy-on-write preserved for large microdata).</pros>
          <cons>Requires medium-scope refactor of the ~500-line foreach body to separate data computation from plot construction. Return objects from workers must carry all intermediate data needed for plot construction.</cons>
        </approach>
        <approach id="A2" label="Switch to PSOCK clusters" feasibility="high" risk="low">
          <description>Replace registerDoParallel(cores = N_CORES) with a PSOCK-based cluster: cl = makeCluster(N_CORES, type = "PSOCK"); registerDoParallel(cl). PSOCK workers are fresh R processes (not forks), eliminating all fork-safety issues.</description>
          <pros>Near-one-line change. Avoids all fork-safety issues. ggsave and ggtext work in PSOCK workers.</pros>
          <cons>Data must be serialized to each worker (slow for large microdata). Workers use more memory (each gets a full copy). Startup overhead (spawning N R processes). Does not address per-feature progress reporting.</cons>
        </approach>
        <approach id="A3" label="Sequential loop (%do%)" feasibility="high" risk="low">
          <description>Change %dopar% to %do%. Entirely sequential execution.</description>
          <pros>Minimal change. No fork-safety issues. Per-feature progress output works via cat().</pros>
          <cons>No parallelism. Total time = sum of all features, not max. Scales poorly for large datasets with many features.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="A1">Approach A1 approved by user. Permanently separates data computation (fork-safe, parallel) from plot rendering (sequential, fork-unsafe). Most robust long-term solution.</decision>
    </topic>
    <topic id="T2" title="Bootstrap subsampling for large-dataset efficiency (Fix 2)">
      <summary>Independent of the fork-safety fix, bootstrap_spline_sd (line 317) fits a spline to the full stratum data (potentially 100K+ rows) for each of B=2000 iterations. For large datasets, subsampling to n_sub = min(n, 5000) before each bootstrap iteration reduces per-iteration cost from O(n log n) to O(n_sub log n_sub), with an m-out-of-n correction factor of sqrt(n_sub / n) applied to the final SD. Not needed for the current N approximately 300 dataset but becomes critical at larger sample sizes.</summary>
      <research>
        <finding src="Efron (1987), Better Bootstrap Confidence Intervals, JASA 82:171-185">B=25-50 is sufficient for standard error estimation. B=200 is more than adequate (Efron and Tibshirani, 1993, Ch. 6). B=2000 is appropriate for percentile CIs (as used in infer.py) but excessive for SE estimation by a factor of 10.</finding>
        <finding src="Politis, Romano, and Wolf (1999), Subsampling, Springer">The m-out-of-n bootstrap with correction factor sqrt(m/n) provides asymptotically correct variance estimates when fitting smooth functionals (such as spline predictions) to subsampled data.</finding>
        <finding src="Bickel and Sakov (2008), Bernoulli 14:1001-1012">Formal treatment of the m-out-of-n bootstrap correction. For m >= 2000 with approximately 8 spline basis functions, the approximation error is negligible.</finding>
      </research>
      <approaches>
        <approach id="B1" label="Subsample within bootstrap_spline_sd" feasibility="high" risk="low">
          <description>Before the bootstrap loop, subsample the data to n_sub = min(n, 5000). The reference spline (fitted once, outside the loop) continues to use all data. Only the bootstrap iterations use subsampled data. Apply sqrt(n_sub / n) correction to the final SD. The per-feature time becomes approximately constant (~2.6s with B=2000) regardless of original data size. Configuration via plot.bootstrap_ribbons.max_subsample_n with default 5000.</description>
          <pros>10-130x speedup for n=50K-500K per stratum. Approximately constant runtime regardless of data size. Correction factor ensures statistical validity. Reference spline unchanged.</pros>
          <cons>Adds one config parameter. Slight approximation in ribbon width (negligible for n_sub >= 2000).</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="B1">Included as a P1 action item for future large-dataset support. Not needed for the current N approximately 300 case.</decision>
    </topic>
    <topic id="T3" title="Analytical SE for discrete features (Fix 3)">
      <summary>bootstrap_group_mean_sd (line 347) computes sd(replicate(B, mean(sample(subset_y, replace=TRUE)))), which converges to the analytical result sd(subset_y) / sqrt(length(subset_y)) (Efron 1979, Annals of Statistics 7:1-26). The analytical formula is exact (the limit the bootstrap estimates), has zero computational cost, and is equally valid as the bootstrap implementation since both assume i.i.d. observations within each level.</summary>
      <approaches>
        <approach id="C1" label="Replace discrete bootstrap with analytical SE" feasibility="high" risk="low">
          <description>Replace the for/replicate loop in bootstrap_group_mean_sd with the closed-form sd(subset_y) / sqrt(length(subset_y)). Eliminates B=2000 iterations entirely for discrete features.</description>
          <pros>Exact result (not approximate). Zero computational cost. Simpler code. No new parameters.</pros>
          <cons>Assumes CLT validity (adequate for n >= 10, the existing min_boot_n threshold). Minor: removes the non-parametric robustness of the bootstrap for small samples, though the existing implementation also assumes i.i.d.</cons>
        </approach>
      </approaches>
      <decision status="decided" chosen="C1">Included as a P2 action item. Minor optimization; discrete features are already fast.</decision>
    </topic>
  </topics>
  <action_items>
    <item priority="P0" target_mode="implement" description="Restructure plot.R foreach %dopar% loop into two phases: Phase 1 (%dopar%) computes all data per feature (filtered dataframes, bootstrap SDs, spline trends, stat-label strings, stratification results) and returns structured lists; Phase 2 (sequential) constructs ggplot objects, assembles grobs via arrangeGrob(), and saves via ggsave(). Eliminates fork-unsafe ggsave() and Rcpp (ggtext/gridtext) execution in forked children. Add per-feature progress output in the sequential render phase." />
    <item priority="P1" target_mode="implement" description="Add bootstrap subsampling to bootstrap_spline_sd: subsample to n_sub = min(n, max_subsample_n) before the bootstrap loop, apply sqrt(n_sub / n) correction to final SD. Reference spline uses full data. Config key: plot.bootstrap_ribbons.max_subsample_n (default 5000). Include in example_config_advanced.yaml." />
    <item priority="P2" target_mode="implement" description="Replace bootstrap_group_mean_sd's replicate loop with analytical SE: sd(subset_y) / sqrt(length(subset_y)). Retain the min_boot_n guard (return NA for n less than min_boot_n)." />
  </action_items>
  <next_steps>Proceed to /implement for the P0 fix (fork-safety restructure). P1 and P2 can be bundled into the same implementation cycle. Follow with /test to verify the plotting works correctly under multi-core execution (requires actual parallel execution, not just unit tests). The existing test suite validates plot output file existence and structure, but the fork-safety fix requires end-to-end validation with parallel execution on Linux.</next_steps>
</brainstorm_report>
