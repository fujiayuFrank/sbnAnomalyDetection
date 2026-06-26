#!/usr/bin/env python3
"""
Plot GNN sweep summary CSV files.

This version uses internal paths/settings.
You do NOT need to pass --input from the command line.

Expected CSV columns:

    model_name, method, threshold, normalization_model,
    TP, TN, FP, FN,
    precision, recall, F1, accuracy

Expected model_name format:

    0001_win20_stride10_hist1
    0002_win50_stride20_hist6

The script extracts:

    window_size
    window_stride
    history

Then it creates:

    heatmaps/
    metric_vs_history/
    metric_vs_window_size/
    precision_recall/
    tables/
"""

from __future__ import annotations

import re
from pathlib import Path

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
#     "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/"
# )

# Output directory for plots and tables.
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

NORMALIZATION_MODEL = None
# NORMALIZATION_MODEL = "tanh"

# Which metric to plot.
# Options:
#   "all", "precision", "recall", "F1", "accuracy", "TP", "TN", "FP", "FN"
METRIC = "all"

# Number of top models saved in summary tables.
TOP_N = 10


# ============================================================
# Script internals
# ============================================================

REQUIRED_COLUMNS = {
    "model_name",
    "method",
    "threshold",
    "normalization_model",
    "TP",
    "TN",
    "FP",
    "FN",
    "precision",
    "recall",
    "F1",
    "accuracy",
}


METRIC_COLUMNS = [
    "precision",
    "recall",
    "F1",
    "accuracy",
    "TP",
    "TN",
    "FP",
    "FN",
]


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


def filter_dataframe(
    df: pd.DataFrame,
    method: str | None,
    threshold: float | None,
    normalization_model: str | None,
) -> pd.DataFrame:
    """Apply optional filters."""
    out = df.copy()

    if method is not None:
        out = out[out["method"] == method]

    if threshold is not None:
        out = out[out["threshold"].astype(float) == float(threshold)]

    if normalization_model is not None:
        out = out[out["normalization_model"] == normalization_model]

    if out.empty:
        raise ValueError("No rows remain after filtering.")

    return out


def safe_name(value: object) -> str:
    """Make a filesystem-safe name fragment."""
    text = str(value)
    text = text.replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text.strip("_")


def format_cell_value(metric: str, value: float) -> str:
    """Format heatmap cell labels."""
    if pd.isna(value):
        return ""

    if metric in {"TP", "TN", "FP", "FN"}:
        return f"{value:.0f}"

    return f"{value:.3g}"


def plot_heatmaps(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    """
    Make heatmaps.

    One heatmap per:

        method
        normalization_model
        threshold
        history

    Axes:

        x-axis: window_stride
        y-axis: window_size
        color: metric
    """
    heatmap_dir = output_dir / "heatmaps" / metric
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    group_cols = ["method", "normalization_model", "threshold", "history"]

    for keys, part in df.groupby(group_cols):
        method, norm, threshold, hist = keys

        pivot = part.pivot_table(
            index="window_size",
            columns="window_stride",
            values=metric,
            aggfunc="mean",
        )

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        im = ax.imshow(pivot.values, aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)

        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("window_stride")
        ax.set_ylabel("window_size")

        ax.set_title(
            f"{metric} heatmap\n"
            f"method={method}, norm={norm}, threshold={threshold}, history={hist}"
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(metric)

        # Add numbers inside cells.
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                label = format_cell_value(metric, value)
                if label:
                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        fig.tight_layout()

        filename = (
            f"heatmap_{metric}"
            f"_method-{safe_name(method)}"
            f"_norm-{safe_name(norm)}"
            f"_thr-{safe_name(threshold)}"
            f"_hist-{safe_name(hist)}.png"
        )

        fig.savefig(heatmap_dir / filename, dpi=200)
        plt.close(fig)


def plot_metric_vs_history(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    """
    Plot average metric vs history.

    One line per window_size.
    Separate plot per method, normalization_model, threshold.
    """
    line_dir = output_dir / "metric_vs_history" / metric
    line_dir.mkdir(parents=True, exist_ok=True)

    group_cols = ["method", "normalization_model", "threshold"]

    for keys, part in df.groupby(group_cols):
        method, norm, threshold = keys

        summary = (
            part.groupby(["history", "window_size"], as_index=False)[metric]
            .mean()
            .sort_values(["window_size", "history"])
        )

        if summary.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for window_size, p in summary.groupby("window_size"):
            ax.plot(
                p["history"],
                p[metric],
                marker="o",
                label=f"win={window_size}",
            )

        ax.set_xlabel("history")
        ax.set_ylabel(metric)
        ax.set_title(
            f"{metric} vs history\n"
            f"method={method}, norm={norm}, threshold={threshold}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        filename = (
            f"history_{metric}"
            f"_method-{safe_name(method)}"
            f"_norm-{safe_name(norm)}"
            f"_thr-{safe_name(threshold)}.png"
        )

        fig.savefig(line_dir / filename, dpi=200)
        plt.close(fig)


def plot_metric_vs_window_size(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    """
    Plot average metric vs window_size.

    One line per history.
    Separate plot per method, normalization_model, threshold.
    """
    line_dir = output_dir / "metric_vs_window_size" / metric
    line_dir.mkdir(parents=True, exist_ok=True)

    group_cols = ["method", "normalization_model", "threshold"]

    for keys, part in df.groupby(group_cols):
        method, norm, threshold = keys

        summary = (
            part.groupby(["window_size", "history"], as_index=False)[metric]
            .mean()
            .sort_values(["history", "window_size"])
        )

        if summary.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for hist, p in summary.groupby("history"):
            ax.plot(
                p["window_size"],
                p[metric],
                marker="o",
                label=f"hist={hist}",
            )

        ax.set_xlabel("window_size")
        ax.set_ylabel(metric)
        ax.set_title(
            f"{metric} vs window_size\n"
            f"method={method}, norm={norm}, threshold={threshold}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        filename = (
            f"window_size_{metric}"
            f"_method-{safe_name(method)}"
            f"_norm-{safe_name(norm)}"
            f"_thr-{safe_name(threshold)}.png"
        )

        fig.savefig(line_dir / filename, dpi=200)
        plt.close(fig)


def plot_metric_vs_stride(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    """
    Plot average metric vs window_stride.

    One line per history.
    Separate plot per method, normalization_model, threshold.
    """
    line_dir = output_dir / "metric_vs_stride" / metric
    line_dir.mkdir(parents=True, exist_ok=True)

    group_cols = ["method", "normalization_model", "threshold"]

    for keys, part in df.groupby(group_cols):
        method, norm, threshold = keys

        summary = (
            part.groupby(["window_stride", "history"], as_index=False)[metric]
            .mean()
            .sort_values(["history", "window_stride"])
        )

        if summary.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for hist, p in summary.groupby("history"):
            ax.plot(
                p["window_stride"],
                p[metric],
                marker="o",
                label=f"hist={hist}",
            )

        ax.set_xlabel("window_stride")
        ax.set_ylabel(metric)
        ax.set_title(
            f"{metric} vs window_stride\n"
            f"method={method}, norm={norm}, threshold={threshold}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        filename = (
            f"stride_{metric}"
            f"_method-{safe_name(method)}"
            f"_norm-{safe_name(norm)}"
            f"_thr-{safe_name(threshold)}.png"
        )

        fig.savefig(line_dir / filename, dpi=200)
        plt.close(fig)


def plot_precision_recall_top_models(
    df: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> None:
    """
    Plot precision-recall points for top models.

    This is most useful when multiple thresholds exist.
    Top models are selected by best F1 over all thresholds.
    """
    pr_dir = output_dir / "precision_recall"
    pr_dir.mkdir(parents=True, exist_ok=True)

    ranking = (
        df.sort_values("F1", ascending=False)
        .drop_duplicates(["model_name", "method", "normalization_model"])
        .head(top_n)
    )

    selected_keys = set(
        zip(
            ranking["model_name"],
            ranking["method"],
            ranking["normalization_model"],
        )
    )

    selected = df[
        df.apply(
            lambda row: (
                row["model_name"],
                row["method"],
                row["normalization_model"],
            )
            in selected_keys,
            axis=1,
        )
    ].copy()

    if selected.empty:
        return

    for keys, part in selected.groupby(["method", "normalization_model"]):
        method, norm = keys

        fig, ax = plt.subplots(figsize=(8, 6))

        for model_name, p in part.groupby("model_name"):
            p = p.sort_values("threshold")
            ax.plot(
                p["recall"],
                p["precision"],
                marker="o",
                label=model_name,
            )

            # Label threshold near each point.
            for _, row in p.iterrows():
                ax.text(
                    row["recall"],
                    row["precision"],
                    str(row["threshold"]),
                    fontsize=7,
                )

        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title(
            f"Precision-recall for top {top_n} models\n"
            f"method={method}, norm={norm}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

        fig.tight_layout()

        filename = (
            f"precision_recall"
            f"_method-{safe_name(method)}"
            f"_norm-{safe_name(norm)}.png"
        )

        fig.savefig(pr_dir / filename, dpi=200)
        plt.close(fig)


def save_top_tables(
    df: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> None:
    """Save top-N model tables ranked by F1."""
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    sort_cols = [
        "F1",
        "recall",
        "precision",
        "accuracy",
    ]

    base_cols = [
        "model_name",
        "method",
        "threshold",
        "normalization_model",
        "window_size",
        "window_stride",
        "history",
        "TP",
        "TN",
        "FP",
        "FN",
        "precision",
        "recall",
        "F1",
        "accuracy",
        "source_csv",
    ]

    top_all = df.sort_values(sort_cols, ascending=False).head(top_n)
    top_all[base_cols].to_csv(table_dir / f"top_{top_n}_overall.csv", index=False)

    group_cols = ["method", "normalization_model", "threshold"]

    top_by_group = (
        df.sort_values(sort_cols, ascending=False)
        .groupby(group_cols, as_index=False)
        .head(top_n)
    )

    top_by_group[base_cols].to_csv(table_dir / f"top_{top_n}_by_group.csv", index=False)

    best_per_history = (
        df.sort_values(sort_cols, ascending=False)
        .groupby(
            ["method", "normalization_model", "threshold", "history"],
            as_index=False,
        )
        .head(1)
    )

    best_per_history[base_cols].to_csv(table_dir / "best_per_history.csv", index=False)


def print_summary(df: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    """Print a compact terminal summary."""
    print()
    print("=" * 80)
    print("Loaded rows:", len(df))
    print("Output directory:", output_dir)
    print("=" * 80)

    print()
    print("Available methods:")
    for value in sorted(df["method"].dropna().unique()):
        print(f"  {value}")

    print()
    print("Available normalization models:")
    for value in sorted(df["normalization_model"].dropna().unique()):
        print(f"  {value}")

    print()
    print("Available thresholds:")
    for value in sorted(df["threshold"].dropna().unique()):
        print(f"  {value}")

    print()
    print(f"Top {top_n} rows by F1:")

    cols = [
        "model_name",
        "method",
        "threshold",
        "normalization_model",
        "window_size",
        "window_stride",
        "history",
        "precision",
        "recall",
        "F1",
        "accuracy",
        "TP",
        "FP",
        "FN",
    ]

    print(
        df.sort_values("F1", ascending=False)[cols]
        .head(top_n)
        .to_string(index=False)
    )


def main() -> int:
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

    df = filter_dataframe(
        df,
        method=METHOD,
        threshold=THRESHOLD,
        normalization_model=NORMALIZATION_MODEL,
    )

    # Make numeric columns safe.
    for col in METRIC_COLUMNS + ["threshold"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if METRIC == "all":
        metrics = ["F1", "recall", "precision", "accuracy", "TP", "FP", "FN"]
    else:
        if METRIC not in METRIC_COLUMNS:
            raise ValueError(
                f"Invalid METRIC={METRIC!r}. "
                f"Allowed values are: all, {', '.join(METRIC_COLUMNS)}"
            )
        metrics = [METRIC]

    for metric in metrics:
        plot_heatmaps(df, metric, output_dir)
        plot_metric_vs_history(df, metric, output_dir)
        plot_metric_vs_window_size(df, metric, output_dir)
        plot_metric_vs_stride(df, metric, output_dir)

    plot_precision_recall_top_models(df, output_dir, TOP_N)
    save_top_tables(df, output_dir, TOP_N)
    print_summary(df, output_dir, TOP_N)

    print()
    print("Done.")
    print(f"Plots saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())