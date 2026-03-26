#!/usr/bin/env python3
"""CLI entry point for boost-shap-gii.

Provides independently callable subcommands:
    boost-shap-gii train    --config CONFIG
    boost-shap-gii predict  --config CONFIG
    boost-shap-gii infer    --config CONFIG --data DATA --output-subdir SUBDIR
    boost-shap-gii plot     --config CONFIG --outcome-range RANGE --negate-shap BOOL --y-axis-label LABEL [--run-dir DIR]
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
    sys.argv = ["boost-shap-gii train", "--config", args.config]
    from .train import main
    main()


def cmd_predict(args: argparse.Namespace) -> None:
    """Dispatch to the predict module."""
    sys.argv = ["boost-shap-gii predict", "--config", args.config]
    from .predict import main
    main()


def cmd_infer(args: argparse.Namespace) -> None:
    """Dispatch to the infer module."""
    sys.argv = [
        "boost-shap-gii infer",
        "--config", args.config,
        "--data", args.data,
        "--output-subdir", args.output_subdir,
    ]
    from .infer import main
    main()


def cmd_plot(args: argparse.Namespace) -> None:
    """Dispatch to Rscript plot.R with graceful degradation if R is absent."""
    plot_r_path = _find_plot_r()

    cmd = [
        "Rscript", plot_r_path,
        args.config,
        args.outcome_range,
        args.negate_shap,
        args.y_axis_label,
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
    p_train.set_defaults(func=cmd_train)

    # --- predict ---
    p_predict = subparsers.add_parser(
        "predict",
        help="Evaluate trained models and compute SHAP-based GII.",
    )
    p_predict.add_argument("--config", required=True, help="Path to config YAML.")
    p_predict.set_defaults(func=cmd_predict)

    # --- infer ---
    p_infer = subparsers.add_parser(
        "infer",
        help="Apply trained models to an independent dataset.",
    )
    p_infer.add_argument("--config", required=True, help="Path to resolved config YAML.")
    p_infer.add_argument("--data", required=True, help="Path to inference dataset (CSV/Parquet).")
    p_infer.add_argument("--output-subdir", required=True, help="Subdirectory name for inference outputs.")
    p_infer.set_defaults(func=cmd_infer)

    # --- plot ---
    p_plot = subparsers.add_parser(
        "plot",
        help="Generate SHAP/GII visualizations via Rscript.",
    )
    p_plot.add_argument("--config", required=True, help="Path to config YAML.")
    p_plot.add_argument("--outcome-range", required=True, help="Theoretical maximum of outcome measure.")
    p_plot.add_argument("--negate-shap", required=True, help="'true' or 'false' to reverse SHAP y-axis.")
    p_plot.add_argument("--y-axis-label", required=True, help="Y-axis label for SHAP plots.")
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
