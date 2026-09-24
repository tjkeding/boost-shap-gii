#!/usr/bin/env python3
"""CLI entry point for boost-shap-gii.

Provides independently callable subcommands:
    boost-shap-gii train    --config CONFIG
    boost-shap-gii predict  --config CONFIG
    boost-shap-gii infer    --config CONFIG --data DATA --output-subdir SUBDIR
    boost-shap-gii plot     --config CONFIG [--run-dir DIR]
    boost-shap-gii check-env
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib import resources


def _find_plot_r() -> str:
    """Locate plot.R bundled as package data.

    Returns the filesystem path to the plot.R script included in the
    ``boost_shap_gii.scripts`` package data directory.
    """
    ref = resources.files("boost_shap_gii") / "scripts" / "plot.R"
    # resources.as_file gives a context-managed path; for a real file on
    # disk (non-zip install), the path is stable, so we can use it directly.
    return str(ref)


def cmd_train(args: argparse.Namespace) -> None:
    """Dispatch to the train module."""
    from .check_env import run_preflight
    run_preflight()
    sys.argv = ["boost-shap-gii train", "--config", args.config]
    if args.force_restart:
        sys.argv.append("--force-restart")
    from .train import main
    main()


def cmd_predict(args: argparse.Namespace) -> None:
    """Dispatch to the predict module."""
    from .check_env import run_preflight
    run_preflight()
    sys.argv = ["boost-shap-gii predict", "--config", args.config]
    if args.force_restart:
        sys.argv.append("--force-restart")
    from .predict import main
    main()


def cmd_infer(args: argparse.Namespace) -> None:
    """Dispatch to the infer module."""
    from .check_env import run_preflight
    run_preflight()
    sys.argv = [
        "boost-shap-gii infer",
        "--config", args.config,
        "--data", args.data,
        "--output-subdir", args.output_subdir,
    ]
    if args.force_restart:
        sys.argv.append("--force-restart")
    from .infer import main
    main()


def _backfill_perf_bootstrap(run_dir: str, config: dict) -> None:
    """Generate bootstrap_distributions_perf.parquet from existing OOF predictions."""
    import os
    import numpy as np
    import pandas as pd
    import yaml

    oof_path = os.path.join(run_dir, "predictions_oof.csv")
    if not os.path.exists(oof_path):
        print("[WARN] Cannot backfill performance bootstraps: predictions_oof.csv not found.")
        return

    resolved_cfg_path = os.path.join(run_dir, "resolved_config.yaml")
    if os.path.exists(resolved_cfg_path):
        with open(resolved_cfg_path) as f:
            run_config = yaml.safe_load(f)
    else:
        run_config = config

    from .utils import compute_bootstrap_ci, get_scoring_function
    task = run_config["modeling"]["task_type"]
    n_boot = run_config.get("shap", {}).get("bootstrapping", {}).get("n_boot", 2000)
    boot_alpha = run_config.get("shap", {}).get("bootstrapping", {}).get("alpha", 0.05)

    oof_df = pd.read_csv(oof_path)

    if task == "regression":
        metrics_to_calc = ["neg_rmse", "neg_mae", "r2"]
    elif task == "multi_regression":
        metrics_to_calc = ["neg_rmse", "neg_mae", "r2"]
    elif task == "multiclass_classification":
        metrics_to_calc = ["balanced_accuracy", "f1_weighted"]
    else:
        metrics_to_calc = ["roc_auc", "accuracy"]

    boot_distributions = {}

    if task == "multi_regression":
        outcome_cols = [c.replace("y_true_", "") for c in oof_df.columns if c.startswith("y_true_")]
        for col in outcome_cols:
            y_true = oof_df[f"y_true_{col}"].values
            y_pred = oof_df[f"y_pred_{col}"].values
            for m_name in metrics_to_calc:
                fn = get_scoring_function(m_name)
                _, _, _, dist = compute_bootstrap_ci(
                    y_true, y_pred, fn, n_boot=n_boot, alpha=boot_alpha,
                    return_distribution=True
                )
                disp_name = f"{m_name.replace('neg_', '').upper()}_{col}"
                boot_distributions[disp_name] = -dist if m_name.startswith("neg_") else dist
    elif task == "multiclass_classification":
        y_true = oof_df["y_true"].values
        prob_cols = [c for c in oof_df.columns if c.startswith("prob_")]
        preds_labels = np.argmax(oof_df[prob_cols].values, axis=1) if prob_cols else oof_df["y_pred"].values
        for m_name in metrics_to_calc:
            fn = get_scoring_function(m_name)
            _, _, _, dist = compute_bootstrap_ci(
                y_true, preds_labels, fn, n_boot=n_boot, alpha=boot_alpha,
                return_distribution=True
            )
            boot_distributions[m_name.upper()] = dist
    else:
        y_true = oof_df["y_true"].values
        y_pred = oof_df["y_pred"].values
        for m_name in metrics_to_calc:
            fn = get_scoring_function(m_name)
            if task == "binary_classification" and m_name in ["accuracy", "f1"]:
                fn = lambda yt, yp, _fn=fn: _fn(yt, (yp > 0.5).astype(int))
            _, _, _, dist = compute_bootstrap_ci(
                y_true, y_pred, fn, n_boot=n_boot, alpha=boot_alpha,
                return_distribution=True
            )
            disp_name = m_name.replace("neg_", "").upper()
            boot_distributions[disp_name] = -dist if m_name.startswith("neg_") else dist

    if boot_distributions:
        max_len = max(len(v) for v in boot_distributions.values())
        boot_df = pd.DataFrame({
            k: np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
            for k, v in boot_distributions.items()
        })
        boot_df.to_parquet(os.path.join(run_dir, "bootstrap_distributions_perf.parquet"), index=False)
        print(f"[INFO] Backfilled bootstrap_distributions_perf.parquet ({len(boot_distributions)} metrics)")
    else:
        print("[WARN] No metrics computed for performance bootstrap backfill.")


def cmd_plot(args: argparse.Namespace) -> None:
    """Dispatch to Rscript plot.R with graceful degradation if R is absent.

    All plot configuration (outcome_max, negate_shap, y-axis labels) is read
    from config.plot.* keys inside the YAML config. No positional plot flags
    are passed on the command line.
    """
    from .check_env import run_preflight
    run_preflight()

    import yaml
    from .utils import validate_plot_config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    validate_plot_config(config)

    import os
    run_dir = args.run_dir if args.run_dir else config["paths"]["output_dir"]
    boot_perf_path = os.path.join(run_dir, "bootstrap_distributions_perf.parquet")
    if not os.path.exists(boot_perf_path):
        try:
            _backfill_perf_bootstrap(run_dir, config)
        except Exception as e:
            print(f"[WARN] Performance bootstrap backfill failed: {e}", file=sys.stderr)

    plot_r_path = _find_plot_r()

    cmd = [
        "Rscript", plot_r_path,
        args.config,
    ]
    if args.run_dir:
        cmd.append(args.run_dir)

    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(
            "[ERROR] Rscript not found on PATH. Install R to use the plot "
            "subcommand.\n"
            "[HINT]  On macOS: brew install r\n"
            "[HINT]  On Ubuntu: sudo apt-get install r-base",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_check_env(args: argparse.Namespace) -> None:
    """Dispatch to the check_env module."""
    from .check_env import main
    main()


def main() -> None:
    """Main CLI entry point with subcommand dispatch."""
    parser = argparse.ArgumentParser(
        prog="boost-shap-gii",
        description=(
            "Config-driven gradient boosting with SHAP-based global "
            "importance indices (GII)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = subparsers.add_parser(
        "train",
        help="Tune hyperparameters and train gradient boosting models.",
    )
    p_train.add_argument("--config", required=True, help="Path to config YAML.")
    p_train.add_argument("--force-restart", action="store_true", help="Delete this stage's checkpoint and restart from scratch")
    p_train.set_defaults(func=cmd_train)

    # --- predict ---
    p_predict = subparsers.add_parser(
        "predict",
        help="Evaluate trained models and compute SHAP-based GII.",
    )
    p_predict.add_argument("--config", required=True, help="Path to config YAML.")
    p_predict.add_argument("--force-restart", action="store_true", help="Delete this stage's checkpoint and restart from scratch")
    p_predict.set_defaults(func=cmd_predict)

    # --- infer ---
    p_infer = subparsers.add_parser(
        "infer",
        help="Apply trained models to an independent dataset.",
    )
    p_infer.add_argument("--config", required=True, help="Path to resolved config YAML.")
    p_infer.add_argument("--data", required=True, help="Path to inference dataset (CSV/Parquet).")
    p_infer.add_argument("--output-subdir", required=True, help="Subdirectory name for inference outputs.")
    p_infer.add_argument("--force-restart", action="store_true", help="Delete this stage's checkpoint and restart from scratch")
    p_infer.set_defaults(func=cmd_infer)

    # --- plot ---
    p_plot = subparsers.add_parser(
        "plot",
        help="Generate SHAP/GII and per-individual visualizations via Rscript.",
    )
    p_plot.add_argument("--config", required=True, help="Path to config YAML.")
    p_plot.add_argument("--run-dir", required=False, default=None, help="Override run directory (for inference plots).")
    p_plot.set_defaults(func=cmd_plot)

    # --- check-env ---
    p_check = subparsers.add_parser(
        "check-env",
        help="Verify Python and R dependencies.",
    )
    p_check.set_defaults(func=cmd_check_env)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
