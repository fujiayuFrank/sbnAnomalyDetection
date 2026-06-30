#!/usr/bin/env python3
"""
Simplified GNN sweep plotting script.

This version uses internal paths/settings. You do NOT need to pass --input.

Optional command-line method filters:

    python plot_sweep_metrics_simplified.py --scores-only
    python plot_sweep_metrics_simplified.py --scores-max-only

Optional CSV summary output:

    python plot_sweep_metrics_simplified.py --plot-summary

By default, the script saves plots only.
CSV summary files are saved only when --plot-summary is given.

Expected CSV columns:

    model_name, method, threshold, normalization_mode,
    TP, TN, FP, FN,
    precision, recall, F1, accuracy

Optional CSV columns for training-setting plots:

    batch_size, learning_rate

If batch_size and learning_rate are not CSV columns, the script tries to parse
them from model_name fragments like:

    bs64_lr0p003
    bs128_lr0.001

Expected model_name format:

    0001_win20_stride10_hist1
    0002_win50_stride20_hist6
    0121_win1000_stride1000_hist10_bs32_lr0p003

The script extracts:

    window_size
    window_stride
    history
    batch_size
    learning_rate

Then it creates:

    plots/
      01_metric_histograms/
        accuracy_histogram.png
        precision_histogram.png
        recall_histogram.png
        F1_histogram.png

      02_metric_relations/
        accuracy/
          vary_window_size/
          vary_window_stride/
          vary_history/
        precision/
        recall/
        F1/

      03_training_setting_relations/
        accuracy/
          vary_batch_size_fixed_learning_rate/
          vary_learning_rate_fixed_batch_size/
        precision/
        recall/
        F1/

For normal relation plots, the script fixes two of:

    window_size, window_stride, history

and plots the metric against the remaining third variable.

For training-setting relation plots, the script ignores:

    window_size, window_stride, history

because these are assumed fixed during batch-size / learning-rate sweeps.

It creates:

    metric vs batch_size, fixed learning_rate
    metric vs learning_rate, fixed batch_size
"""

from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-GUI backend; safe for SSH/Plink sessions

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# User settings
# ============================================================

# Put either a CSV file path here:
INPUT_PATH = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/threshold_evaluation/threshold_metrics_summary.csv"
)

# Or use a directory containing CSV files:
# INPUT_PATH = Path(
#     "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/threshold_evaluation/"
# )

# Output directory for plots.
# If None, the script creates "plots" next to the CSV file
# or inside the input directory.
OUTPUT_DIR = None

# Optional filters.
# Use None to keep all values.
METHOD = None
# METHOD = "both_or"
# METHOD = "scores_only"
# METHOD = "scores_max_only"

THRESHOLD = None
# THRESHOLD = 0.5

NORMALIZATION_MODE = None
# NORMALIZATION_MODE = "tanh"

# Metrics to plot. TP/TN/FP/FN are intentionally ignored.
PLOT_METRICS = ["accuracy", "precision", "recall", "F1"]

# Histogram bin control.
# Examples:
#   HIST_BINS = 20
#   HIST_BINS = 50
#   HIST_BINS = [0.0, 0.1, 0.2, ..., 1.0]
HIST_BINS = 30

# Plot relation line markers.
MARKER = "o"

# Y-axis behavior for relation plots.
# True:
#   Zoom y-axis to the actual data range.
#   This is better when all metric values are close together.
# False:
#   Use full [0, 1] range for accuracy/precision/recall/F1.
AUTO_YLIM = True

# Padding fraction for automatic y-axis zoom.
Y_PAD_FRACTION = 0.10

# Minimum padding when all y-values are identical.
Y_MIN_PAD = 1e-4

# Default CSV behavior.
# Keep this False so the script does not save CSV files by default.
# Use --plot-summary to save CSV outputs.
SAVE_RELATION_CSV = False


# ============================================================
# Script internals
# ============================================================

REQUIRED_COLUMNS = {
    "model_name",
    "method",
    "threshold",
    "normalization_mode",
    "TP",
    "TN",
    "FP",
    "FN",
    "precision",
    "recall",
    "F1",
    "accuracy",
}

METRIC_COLUMNS = ["accuracy", "precision", "recall", "F1"]

# Original sweep variables.
MODEL_VARIABLES = ["window_size", "window_stride", "history"]

# New training-setting variables.
TRAINING_VARIABLES = ["batch_size", "learning_rate"]

CONTEXT_COLUMNS = ["method", "normalization_mode", "threshold"]


def find_csv_files(input_path: Path) -> list[Path]:
    """Return CSV file paths from either a file or directory."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Input file is not a CSV file: {input_path}")
        return [input_path]

    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {input_path}")
        return csv_files

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def load_csv_files(csv_files: list[Path]) -> pd.DataFrame:
    """Load and combine one or more CSV files."""
    frames = []

    for path in csv_files:
        df = pd.read_csv(path)
        df["source_csv"] = path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    missing = REQUIRED_COLUMNS - set(combined.columns)
    if missing:
        raise ValueError(
            "CSV is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in sorted(missing))
        )

    return combined


def parse_model_name(df: pd.DataFrame) -> pd.DataFrame:
    """Extract window_size, window_stride, and history from model_name."""
    parsed = df["model_name"].str.extract(
        r"win(?P<window_size>\d+)_stride(?P<window_stride>\d+)_hist(?P<history>\d+)"
    )

    bad_rows = parsed.isna().any(axis=1)

    if bad_rows.any():
        bad_names = df.loc[bad_rows, "model_name"].drop_duplicates().tolist()
        raise ValueError(
            "Some model_name values do not match expected pattern "
            "'win<size>_stride<stride>_hist<history>'. Examples:\n"
            + "\n".join(f"  - {name}" for name in bad_names[:20])
        )

    parsed = parsed.astype(int)

    out = df.copy()
    out["window_size"] = parsed["window_size"]
    out["window_stride"] = parsed["window_stride"]
    out["history"] = parsed["history"]

    return out


def parse_training_settings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure batch_size and learning_rate columns exist.

    Preferred:
        Use existing CSV columns:
            batch_size
            learning_rate

    Fallback:
        Parse from model_name if the name contains fragments like:
            bs64_lr0p003
            bs128_lr0.001
    """
    out = df.copy()

    if "batch_size" not in out.columns or "learning_rate" not in out.columns:
        parsed = out["model_name"].str.extract(
            r"bs(?P<batch_size>\d+)_lr(?P<learning_rate>[0-9p\.eE\-]+)"
        )

        if "batch_size" not in out.columns:
            out["batch_size"] = parsed["batch_size"]

        if "learning_rate" not in out.columns:
            out["learning_rate"] = parsed["learning_rate"]

    out["batch_size"] = pd.to_numeric(out["batch_size"], errors="coerce")

    # Allow learning-rate strings like 0p003 from filenames.
    out["learning_rate"] = (
        out["learning_rate"]
        .astype(str)
        .str.replace("p", ".", regex=False)
    )
    out["learning_rate"] = pd.to_numeric(out["learning_rate"], errors="coerce")

    missing_rows = out[["batch_size", "learning_rate"]].isna().any(axis=1)

    if missing_rows.any():
        bad_names = out.loc[missing_rows, "model_name"].drop_duplicates().tolist()
        raise ValueError(
            "Could not determine batch_size and learning_rate for some rows. "
            "Make sure the CSV has columns 'batch_size' and 'learning_rate', "
            "or model_name contains fragments like 'bs64_lr0p003'. Examples:\n"
            + "\n".join(f"  - {name}" for name in bad_names[:20])
        )

    return out


def filter_dataframe(
    df: pd.DataFrame,
    method: str | None,
    threshold: float | None,
    normalization_mode: str | None,
) -> pd.DataFrame:
    """Apply optional filters."""
    out = df.copy()

    if method is not None:
        out = out[out["method"] == method]

    if threshold is not None:
        out = out[out["threshold"].astype(float) == float(threshold)]

    if normalization_mode is not None:
        out = out[out["normalization_mode"] == normalization_mode]

    if out.empty:
        raise ValueError("No rows remain after filtering.")

    return out


def make_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert metric and parameter columns to numeric values."""
    out = df.copy()

    for col in METRIC_COLUMNS + MODEL_VARIABLES + TRAINING_VARIABLES + ["threshold"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=METRIC_COLUMNS + MODEL_VARIABLES + TRAINING_VARIABLES)

    return out


def safe_name(value: object) -> str:
    """Make a filesystem-safe name fragment."""
    text = str(value)
    text = text.replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text.strip("_")


def metric_label(metric: str) -> str:
    """Human-friendly metric label."""
    if metric == "F1":
        return "F1 score"
    return metric


def context_name(method: object, norm: object, threshold: object) -> str:
    """Compact label for method/norm/threshold."""
    return f"method={method}, norm={norm}, threshold={threshold}"


def context_filename(method: object, norm: object, threshold: object) -> str:
    """Filesystem-safe context name."""
    return (
        f"method-{safe_name(method)}"
        f"_norm-{safe_name(norm)}"
        f"_thr-{safe_name(threshold)}"
    )


def apply_y_limits(ax: plt.Axes, metric: str, y_values: pd.Series) -> None:
    """Apply y-axis limits for relation plots."""
    y_values = y_values.dropna()

    if y_values.empty:
        return

    if AUTO_YLIM:
        y_min = float(y_values.min())
        y_max = float(y_values.max())

        if y_min == y_max:
            pad = max(abs(y_min) * Y_PAD_FRACTION, Y_MIN_PAD)
        else:
            pad = (y_max - y_min) * Y_PAD_FRACTION

        ax.set_ylim(y_min - pad, y_max + pad)
    else:
        if metric in {"accuracy", "precision", "recall", "F1"}:
            ax.set_ylim(0.0, 1.0)


def plot_metric_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot one histogram per metric over all filtered rows/models.

    These are saved directly inside the first plot directory:

        output_dir / "01_metric_histograms"
    """
    hist_dir = output_dir / "01_metric_histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    for metric in PLOT_METRICS:
        values = df[metric].dropna()
        if values.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=HIST_BINS, edgecolor="black", alpha=0.8)

        ax.set_xlabel(metric_label(metric))
        ax.set_ylabel("Number of rows/models")
        ax.set_title(f"Distribution of {metric_label(metric)} over all filtered models")
        ax.grid(True, alpha=0.3)

        stats_text = (
            f"n = {len(values)}\n"
            f"mean = {values.mean():.4g}\n"
            f"median = {values.median():.4g}\n"
            f"min = {values.min():.4g}\n"
            f"max = {values.max():.4g}"
        )
        ax.text(
            0.98,
            0.95,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            fontsize=9,
        )

        fig.tight_layout()
        fig.savefig(hist_dir / f"{safe_name(metric)}_histogram.png", dpi=200)
        plt.close(fig)


def plot_relation_for_fixed_pair(
    df: pd.DataFrame,
    metric: str,
    fixed_vars: tuple[str, str],
    vary_var: str,
    output_dir: Path,
    save_csv: bool,
) -> None:
    """
    For one metric and one choice of fixed variables, plot metric vs vary_var.

    Example:
        fixed_vars = ("window_stride", "history")
        vary_var = "window_size"

    This creates one plot for every combination of:
        method, normalization_mode, threshold, fixed_var_1 value, fixed_var_2 value
    """
    metric_dir = output_dir / "02_metric_relations" / safe_name(metric) / f"vary_{vary_var}"
    metric_dir.mkdir(parents=True, exist_ok=True)

    group_cols = CONTEXT_COLUMNS + list(fixed_vars)

    for keys, part in df.groupby(group_cols, dropna=False):
        method, norm, threshold, fixed_value_1, fixed_value_2 = keys

        summary = (
            part.groupby(vary_var, as_index=False)[metric]
            .agg(mean="mean", std="std", count="count")
            .sort_values(vary_var)
        )

        if summary.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            summary[vary_var],
            summary["mean"],
            marker=MARKER,
            label=metric_label(metric),
        )

        # If there are duplicate rows for the same x value, show standard deviation.
        if (summary["count"] > 1).any() and summary["std"].notna().any():
            ax.errorbar(
                summary[vary_var],
                summary["mean"],
                yerr=summary["std"].fillna(0.0),
                fmt="none",
                capsize=3,
            )

        ax.set_xlabel(vary_var)
        ax.set_ylabel(metric_label(metric))
        ax.set_title(
            f"{metric_label(metric)} vs {vary_var}\n"
            f"fixed {fixed_vars[0]}={fixed_value_1}, {fixed_vars[1]}={fixed_value_2}\n"
            f"{context_name(method, norm, threshold)}"
        )
        ax.grid(True, alpha=0.3)

        apply_y_limits(ax, metric, summary["mean"])

        fig.tight_layout()

        filename = (
            f"{safe_name(metric)}_vs_{safe_name(vary_var)}"
            f"_fixed-{safe_name(fixed_vars[0])}-{safe_name(fixed_value_1)}"
            f"_{safe_name(fixed_vars[1])}-{safe_name(fixed_value_2)}"
            f"_{context_filename(method, norm, threshold)}.png"
        )

        fig.savefig(metric_dir / filename, dpi=200)
        plt.close(fig)

        if save_csv:
            csv_name = filename.replace(".png", ".csv")
            summary.to_csv(metric_dir / csv_name, index=False)


def plot_all_metric_relations(
    df: pd.DataFrame,
    output_dir: Path,
    save_csv: bool,
) -> None:
    """
    For each metric, fix two model variables and vary the third.

    Model variables:
        window_size, window_stride, history

    For three variables, this automatically creates:
        metric vs window_size, fixed window_stride/history
        metric vs window_stride, fixed window_size/history
        metric vs history, fixed window_size/window_stride
    """
    for metric in PLOT_METRICS:
        for fixed_vars in combinations(MODEL_VARIABLES, 2):
            vary_candidates = [v for v in MODEL_VARIABLES if v not in fixed_vars]
            if len(vary_candidates) != 1:
                raise RuntimeError("Expected exactly one varying variable.")

            vary_var = vary_candidates[0]

            plot_relation_for_fixed_pair(
                df=df,
                metric=metric,
                fixed_vars=fixed_vars,
                vary_var=vary_var,
                output_dir=output_dir,
                save_csv=save_csv,
            )


def plot_training_setting_relations(
    df: pd.DataFrame,
    output_dir: Path,
    save_csv: bool,
) -> None:
    """
    Plot metric behavior for training settings only.

    This ignores:
        window_size, window_stride, history

    It creates two plot types for each metric:

        1. fixed learning_rate, vary batch_size
        2. fixed batch_size, vary learning_rate
    """
    plot_specs = [
        {
            "vary_var": "batch_size",
            "fixed_var": "learning_rate",
            "subdir": "vary_batch_size_fixed_learning_rate",
        },
        {
            "vary_var": "learning_rate",
            "fixed_var": "batch_size",
            "subdir": "vary_learning_rate_fixed_batch_size",
        },
    ]

    for metric in PLOT_METRICS:
        for spec in plot_specs:
            vary_var = spec["vary_var"]
            fixed_var = spec["fixed_var"]

            metric_dir = (
                output_dir
                / "03_training_setting_relations"
                / safe_name(metric)
                / spec["subdir"]
            )
            metric_dir.mkdir(parents=True, exist_ok=True)

            group_cols = CONTEXT_COLUMNS + [fixed_var]

            for keys, part in df.groupby(group_cols, dropna=False):
                method, norm, threshold, fixed_value = keys

                summary = (
                    part.groupby(vary_var, as_index=False)[metric]
                    .agg(mean="mean", std="std", count="count")
                    .sort_values(vary_var)
                )

                if summary.empty:
                    continue

                fig, ax = plt.subplots(figsize=(8, 5))

                ax.plot(
                    summary[vary_var],
                    summary["mean"],
                    marker=MARKER,
                    label=metric_label(metric),
                )

                # If there are duplicate rows for the same x value,
                # show standard deviation.
                if (summary["count"] > 1).any() and summary["std"].notna().any():
                    ax.errorbar(
                        summary[vary_var],
                        summary["mean"],
                        yerr=summary["std"].fillna(0.0),
                        fmt="none",
                        capsize=3,
                    )

                ax.set_xlabel(vary_var)
                ax.set_ylabel(metric_label(metric))
                ax.set_title(
                    f"{metric_label(metric)} vs {vary_var}\n"
                    f"fixed {fixed_var}={fixed_value}\n"
                    f"{context_name(method, norm, threshold)}"
                )
                ax.grid(True, alpha=0.3)

                apply_y_limits(ax, metric, summary["mean"])

                fig.tight_layout()

                filename = (
                    f"{safe_name(metric)}_vs_{safe_name(vary_var)}"
                    f"_fixed-{safe_name(fixed_var)}-{safe_name(fixed_value)}"
                    f"_{context_filename(method, norm, threshold)}.png"
                )

                fig.savefig(metric_dir / filename, dpi=200)
                plt.close(fig)

                if save_csv:
                    csv_name = filename.replace(".png", ".csv")
                    summary.to_csv(metric_dir / csv_name, index=False)


def save_clean_dataframe(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save the parsed and filtered dataframe for checking.

    This function is only called when --plot-summary is given.
    """
    table_dir = output_dir / "03_tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        "model_name",
        "method",
        "threshold",
        "normalization_mode",
        "window_size",
        "window_stride",
        "history",
        "batch_size",
        "learning_rate",
        "accuracy",
        "precision",
        "recall",
        "F1",
        "source_csv",
    ]

    df[cols].to_csv(table_dir / "parsed_filtered_metrics.csv", index=False)

    top = df.sort_values(["F1", "recall", "precision", "accuracy"], ascending=False)
    top[cols].head(50).to_csv(table_dir / "top_50_by_F1.csv", index=False)


def print_summary(
    df: pd.DataFrame,
    output_dir: Path,
    effective_method: str | None,
    plot_summary: bool,
) -> None:
    """Print a compact terminal summary."""
    print()
    print("=" * 80)
    print("Loaded rows after filtering:", len(df))
    print("Output directory:", output_dir)
    print("Active method filter:", effective_method)
    print("Histogram bins:", HIST_BINS)
    print("Auto y-axis limits:", AUTO_YLIM)
    print("Full CSV summary:", plot_summary)
    print("=" * 80)

    print()
    print("Available methods in filtered data:")
    for value in sorted(df["method"].dropna().unique()):
        print(f"  {value}")

    print()
    print("Available normalization models in filtered data:")
    for value in sorted(df["normalization_mode"].dropna().unique()):
        print(f"  {value}")

    print()
    print("Available thresholds in filtered data:")
    for value in sorted(df["threshold"].dropna().unique()):
        print(f"  {value}")

    print()
    print("Available batch sizes in filtered data:")
    for value in sorted(df["batch_size"].dropna().unique()):
        print(f"  {value}")

    print()
    print("Available learning rates in filtered data:")
    for value in sorted(df["learning_rate"].dropna().unique()):
        print(f"  {value}")

    print()
    print("Metric summary:")
    print(df[PLOT_METRICS].describe().to_string())

    print()
    print("Top 10 by F1:")
    cols = [
        "model_name",
        "method",
        "threshold",
        "normalization_mode",
        "window_size",
        "window_stride",
        "history",
        "batch_size",
        "learning_rate",
        "accuracy",
        "precision",
        "recall",
        "F1",
    ]
    print(df.sort_values("F1", ascending=False)[cols].head(10).to_string(index=False))


def parse_cli_args() -> argparse.Namespace:
    """Parse optional method-filter and output flags.

    The script still uses internal paths/settings. These flags only override
    selected behavior when explicitly requested.
    """
    parser = argparse.ArgumentParser(
        description="Plot simplified GNN sweep metrics using internal paths/settings."
    )

    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--scores-only",
        "--scores_only",
        action="store_true",
        help="Only plot rows with method == 'scores_only'.",
    )
    method_group.add_argument(
        "--scores-max-only",
        "--scores_max_only",
        "--scores-max_only",
        "--scores_max-only",
        action="store_true",
        help="Only plot rows with method == 'scores_max_only'.",
    )

    parser.add_argument(
        "--plot-summary",
        "--plot_summary",
        action="store_true",
        help=(
            "Save CSV summary files in addition to plots. "
            "By default, no CSV files are written."
        ),
    )

    return parser.parse_args()


def get_effective_method(args: argparse.Namespace) -> str | None:
    """Return the method filter after applying command-line overrides."""
    if args.scores_only:
        return "scores_only"

    if args.scores_max_only:
        return "scores_max_only"

    return METHOD


def main() -> int:
    args = parse_cli_args()
    effective_method = get_effective_method(args)

    input_path = INPUT_PATH.resolve()

    if OUTPUT_DIR is not None:
        output_dir = Path(OUTPUT_DIR).resolve()
    elif input_path.is_dir():
        output_dir = input_path / "plots"
    else:
        output_dir = input_path.parent / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = find_csv_files(input_path)
    df = load_csv_files(csv_files)

    df = parse_model_name(df)
    df = parse_training_settings(df)

    df = filter_dataframe(
        df,
        method=effective_method,
        threshold=THRESHOLD,
        normalization_mode=NORMALIZATION_MODE,
    )

    df = make_numeric(df)

    if df.empty:
        raise ValueError("No valid numeric rows remain after parsing/filtering.")

    # Always save plots.
    plot_metric_histograms(df, output_dir)

    # Save relation CSVs only with --plot-summary, unless SAVE_RELATION_CSV
    # is manually set to True in the user settings.
    save_relation_csv = SAVE_RELATION_CSV or args.plot_summary

    # Original relation plots:
    #   vary window_size / window_stride / history.
    plot_all_metric_relations(
        df=df,
        output_dir=output_dir,
        save_csv=save_relation_csv,
    )

    # New relation plots:
    #   vary batch_size while fixing learning_rate
    #   vary learning_rate while fixing batch_size
    #
    # These ignore window_size, window_stride, and history.
    plot_training_setting_relations(
        df=df,
        output_dir=output_dir,
        save_csv=save_relation_csv,
    )

    # Save table CSVs only with --plot-summary.
    if args.plot_summary:
        save_clean_dataframe(df, output_dir)

    print_summary(
        df=df,
        output_dir=output_dir,
        effective_method=effective_method,
        plot_summary=args.plot_summary,
    )

    print()
    print("Done.")
    print(f"Histogram plots saved to: {output_dir / '01_metric_histograms'}")
    print(f"Relation plots saved to: {output_dir / '02_metric_relations'}")
    print(
        "Training-setting relation plots saved to:",
        output_dir / "03_training_setting_relations",
    )

    if args.plot_summary:
        print(f"CSV summaries saved to: {output_dir / '03_tables'}")
    else:
        print("CSV summaries were not saved. Use --plot-summary to save them.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())