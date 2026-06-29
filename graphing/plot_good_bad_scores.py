#!/usr/bin/env python3
"""Plot good/bad score distributions from one or many inference_scores.npz files.

This script can read either:

1. A direct path to one inference_scores.npz file
2. A model directory containing inference_scores.npz
3. A parent directory containing many model subdirectories, each with inference_scores.npz

Default expected sweep layout:

    checkpoints/gnn/
      0001_win20_stride10_hist1/
        inference_scores.npz
      0002_win20_stride10_hist2/
        inference_scores.npz

Each inference_scores.npz should contain at least:

    scores
    scores_max
    first_run

By default, histograms use:
  - log-spaced bins
  - log x-axis
  - log y-axis
  - fraction-per-bin normalization for good and bad separately

Use --counts to plot raw window counts instead of normalized fractions.

Examples:

    python3 plot_good_bad_scores_distribution.py checkpoints/gnn

    python3 plot_good_bad_scores_distribution.py checkpoints/gnn --scores-only

    python3 plot_good_bad_scores_distribution.py checkpoints/gnn --scores-max-only

    python3 plot_good_bad_scores_distribution.py checkpoints/gnn --start 81

    python3 plot_good_bad_scores_distribution.py checkpoints/gnn --counts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Force non-GUI backend so this works on remote machines without X11/display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


GOOD_RUNS = {
    18445,
    19724,
    20141,
    20142,
    20144,
}

BAD_RUNS = {
    19627,
    19946,
    20104,
}

DEFAULT_OUTPUT_DIR = Path(
    "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/"
    "threshold_evaluation/plots/04_good_bad_distribution"
)

NPZ_NAME = "inference_scores.npz"

# ============================================================
# Optional majority-class downsampling
# ============================================================
#
# If None:
#   use all good and bad windows, same as before.
#
# If a float:
#   keep the smaller class unchanged.
#   randomly sample the larger class to:
#
#       round(BALANCED_SAMPLE_FRACTION * min(n_good, n_bad))
#
# Examples:
#   n_good=15, n_bad=145, BALANCED_SAMPLE_FRACTION=1.0
#   -> plot 15 good windows and 15 randomly selected bad windows
#
#   n_good=100, n_bad=300, BALANCED_SAMPLE_FRACTION=0.5
#   -> plot 100 good windows and 50 randomly selected bad windows
#
#   n_good=300, n_bad=100, BALANCED_SAMPLE_FRACTION=0.5
#   -> plot 50 randomly selected good windows and 100 bad windows
#
BALANCED_SAMPLE_FRACTION: float | None = 1

# Fixed random seed for reproducible random sampling.
RANDOM_SEED = 12345


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot good/bad score histograms from one or many inference_scores.npz files."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Path to inference_scores.npz, a model directory containing "
            "inference_scores.npz, or a parent directory containing model subdirectories."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where plots will be saved. Default: {DEFAULT_OUTPUT_DIR}",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of log-spaced histogram bin edges. Default: 100",
    )

    parser.add_argument(
        "--bin-min",
        type=float,
        default=1e-1,
        help="Lower fallback limit for log-space bins. Default: 1e-1",
    )

    parser.add_argument(
        "--bin-max",
        type=float,
        default=1e10,
        help="Upper fallback limit for log-space bins. Default: 1e10",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help=(
            "Only plot model directories whose leading numeric prefix is greater "
            "than or equal to this value. Example: --start 81 uses 0081_..., 0082_..., etc."
        ),
    )

    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--scores-only",
        "--scores_only",
        action="store_true",
        help="Plot only the scores array.",
    )
    method_group.add_argument(
        "--scores-max-only",
        "--scores_max_only",
        action="store_true",
        help="Plot only the scores_max array.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Show plots interactively in addition to saving them. This is usually "
            "not needed on the cluster because the Agg backend is used."
        ),
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Search recursively for inference_scores.npz files. By default, only "
            "the input directory and its immediate subdirectories are checked."
        ),
    )

    parser.add_argument(
        "--counts",
        action="store_true",
        help="Use raw window counts instead of normalized fraction-per-bin histograms.",
    )

    return parser.parse_args()


def get_model_index(model_dir_name: str) -> int | None:
    """Extract leading numeric model index from a model directory name.

    Examples:
        0001_win20_stride10_hist1 -> 1
        0081_win100_stride10_hist1 -> 81
        test_model -> None
    """
    prefix = model_dir_name.split("_", 1)[0]
    if not prefix.isdigit():
        return None
    return int(prefix)


def passes_start_filter(score_file: Path, start: int | None) -> bool:
    """Return whether a score file passes the optional model-start filter."""
    if start is None:
        return True

    model_name = safe_model_name(score_file)
    model_index = get_model_index(model_name)

    if model_index is None:
        print(
            f"Skipping {model_name}: cannot read numeric prefix while --start={start}",
            flush=True,
        )
        return False

    return model_index >= start


def find_inference_score_files(
    input_path: Path,
    *,
    recursive: bool = False,
    start: int | None = None,
) -> list[Path]:
    """Find inference_scores.npz files from a file, model directory, or parent directory.

    Default behavior:
      - If input_path is a file, use it directly if it passes --start.
      - If input_path/inference_scores.npz exists, use it if it passes --start.
      - Then check each immediate subdirectory for subdir/inference_scores.npz.

    Recursive behavior:
      - If --recursive is given, use rglob to find all inference_scores.npz files.
    """
    input_path = input_path.expanduser().resolve()

    if start is not None and start < 0:
        raise ValueError(f"--start must be >= 0; got {start}")

    if input_path.is_file():
        if input_path.name != NPZ_NAME:
            print(
                f"Warning: input file is named {input_path.name}, not {NPZ_NAME}. "
                "Trying to read it anyway.",
                flush=True,
            )
        return [input_path] if passes_start_filter(input_path, start) else []

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not input_path.is_dir():
        raise ValueError(f"Input path is neither a file nor a directory: {input_path}")

    if recursive:
        files = sorted(input_path.rglob(NPZ_NAME))
        return [f for f in files if passes_start_filter(f, start)]

    files: list[Path] = []

    direct_file = input_path / NPZ_NAME
    if direct_file.exists() and passes_start_filter(direct_file, start):
        files.append(direct_file)

    for subdir in sorted(input_path.iterdir()):
        if not subdir.is_dir():
            continue

        score_file = subdir / NPZ_NAME
        if score_file.exists() and passes_start_filter(score_file, start):
            files.append(score_file)

    return files


def choose_score_arrays(args: argparse.Namespace) -> list[str]:
    """Decide whether to plot scores, scores_max, or both."""
    if args.scores_only:
        return ["scores"]

    if args.scores_max_only:
        return ["scores_max"]

    return ["scores", "scores_max"]


def load_required_array(data: np.lib.npyio.NpzFile, key: str, file_path: Path) -> np.ndarray:
    """Load one required key from the npz file with a clearer error message."""
    if key not in data:
        available_keys = list(data.keys())
        raise KeyError(
            f"Required key {key!r} not found in {file_path}. "
            f"Available keys: {available_keys}"
        )

    return np.asarray(data[key])


def maybe_sample_majority_class(
    good_values: np.ndarray,
    bad_values: np.ndarray,
    *,
    sample_fraction: float | None,
    rng: np.random.Generator,
    model_name: str,
    score_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally downsample only the majority class.

    If sample_fraction is None, return all values unchanged.

    If sample_fraction is a float:
      - find min_count = min(n_good, n_bad)
      - keep the smaller class unchanged
      - randomly sample the larger class to round(sample_fraction * min_count)

    This means:
      n_good=100, n_bad=300, sample_fraction=0.5
      -> good stays 100, bad becomes 50
    """

    if sample_fraction is None:
        return good_values, bad_values

    if sample_fraction <= 0:
        raise ValueError(
            f"BALANCED_SAMPLE_FRACTION must be positive or None; got {sample_fraction}"
        )

    n_good = len(good_values)
    n_bad = len(bad_values)

    if n_good == 0 or n_bad == 0:
        print(
            f"Sampling skipped for {model_name} {score_name}: "
            f"n_good={n_good}, n_bad={n_bad}",
            flush=True,
        )
        return good_values, bad_values

    if n_good == n_bad:
        print(
            f"Sampling skipped for {model_name} {score_name}: "
            f"already symmetric with n_good=n_bad={n_good}",
            flush=True,
        )
        return good_values, bad_values

    min_count = min(n_good, n_bad)
    target_majority_count = int(round(sample_fraction * min_count))

    # For very small positive fractions, avoid accidentally sampling zero.
    target_majority_count = max(1, target_majority_count)

    if n_good > n_bad:
        # Good is the majority class. Keep all bad, sample good.
        target_majority_count = min(target_majority_count, n_good)
        selected_idx = rng.choice(n_good, size=target_majority_count, replace=False)
        sampled_good_values = good_values[selected_idx]
        sampled_bad_values = bad_values

    else:
        # Bad is the majority class. Keep all good, sample bad.
        target_majority_count = min(target_majority_count, n_bad)
        selected_idx = rng.choice(n_bad, size=target_majority_count, replace=False)
        sampled_good_values = good_values
        sampled_bad_values = bad_values[selected_idx]

    print(
        f"Sampling {model_name} {score_name}: "
        f"original good={n_good}, original bad={n_bad}; "
        f"plotted good={len(sampled_good_values)}, plotted bad={len(sampled_bad_values)}; "
        f"BALANCED_SAMPLE_FRACTION={sample_fraction}",
        flush=True,
    )

    return sampled_good_values, sampled_bad_values


def split_good_bad(
    values: np.ndarray,
    first_run: np.ndarray,
    score_name: str,
    file_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a score array into good, bad, and unknown values using first_run."""
    values = np.asarray(values).reshape(-1)
    first_run = np.asarray(first_run).astype(int).reshape(-1)

    if values.shape != first_run.shape:
        raise ValueError(
            f"In {file_path}, {score_name} and first_run must have the same shape. "
            f"Got {score_name}.shape={values.shape}, first_run.shape={first_run.shape}."
        )

    good_mask = np.isin(first_run, list(GOOD_RUNS))
    bad_mask = np.isin(first_run, list(BAD_RUNS))
    known_mask = good_mask | bad_mask
    unknown_mask = ~known_mask

    good_values = values[good_mask]
    bad_values = values[bad_mask]
    unknown_values = values[unknown_mask]

    return good_values, bad_values, unknown_values


def maybe_balance_good_bad_windows(
    good_values: np.ndarray,
    bad_values: np.ndarray,
    *,
    balance_fraction: float | None,
    rng: np.random.Generator,
    model_name: str,
    score_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally balance good/bad samples before plotting.

    If balance_fraction is None, return all values unchanged.

    If balance_fraction is positive, first compute:

        target_n = int(min(n_good, n_bad) * balance_fraction)

    Then randomly select target_n values from each class without replacement.
    This avoids plots being visually dominated by the larger class.
    """
    good_values = np.asarray(good_values).reshape(-1)
    bad_values = np.asarray(bad_values).reshape(-1)

    if balance_fraction is None:
        return good_values, bad_values

    if not np.isfinite(balance_fraction) or balance_fraction <= 0:
        raise ValueError(
            "BALANCED_SAMPLE_FRACTION must be None or a positive finite number; "
            f"got {balance_fraction}"
        )

    n_good = len(good_values)
    n_bad = len(bad_values)
    min_n = min(n_good, n_bad)

    if min_n == 0:
        print(
            f"Skipping balancing for {model_name} {score_name}: "
            f"n_good={n_good}, n_bad={n_bad}.",
            flush=True,
        )
        return good_values, bad_values

    target_n = int(min_n * balance_fraction)
    target_n = max(1, target_n)
    target_n = min(target_n, n_good, n_bad)

    if target_n == n_good and target_n == n_bad:
        print(
            f"Balanced sampling for {model_name} {score_name}: "
            f"already balanced with n_good={n_good}, n_bad={n_bad}.",
            flush=True,
        )
        return good_values, bad_values

    good_indices = rng.choice(n_good, size=target_n, replace=False)
    bad_indices = rng.choice(n_bad, size=target_n, replace=False)

    print(
        f"Balanced sampling for {model_name} {score_name}: "
        f"good {n_good} -> {target_n}, bad {n_bad} -> {target_n} "
        f"using BALANCED_SAMPLE_FRACTION={balance_fraction}.",
        flush=True,
    )

    return good_values[good_indices], bad_values[bad_indices]


def safe_model_name(score_file: Path) -> str:
    """Use the parent directory name as the model name."""
    if score_file.name == NPZ_NAME:
        return score_file.parent.name
    return score_file.stem


def make_log_bins(
    good_values: np.ndarray,
    bad_values: np.ndarray,
    n_bins: int,
    default_min: float,
    default_max: float,
) -> np.ndarray:
    """Make log-spaced bins for positive score values."""
    if n_bins < 2:
        raise ValueError(f"--bins must be >= 2; got {n_bins}")

    if default_min <= 0:
        raise ValueError(f"--bin-min must be positive for log bins; got {default_min}")

    if default_max <= 0:
        raise ValueError(f"--bin-max must be positive for log bins; got {default_max}")

    if default_min >= default_max:
        raise ValueError(
            f"--bin-min must be smaller than --bin-max; got {default_min} >= {default_max}"
        )

    all_values = np.concatenate([good_values, bad_values])
    positive_values = all_values[np.isfinite(all_values) & (all_values > 0)]

    if len(positive_values) == 0:
        raise ValueError("No positive finite values found. Cannot make log-space bins.")

    # Use the data range, but keep the old notebook-style fallback limits.
    vmin = max(default_min, float(np.min(positive_values)))
    vmax = min(default_max, float(np.max(positive_values)))

    if vmin >= vmax:
        vmax = vmin * 10.0

    return np.logspace(np.log10(vmin), np.log10(vmax), n_bins)


def plot_good_bad_histogram(
    *,
    good_values: np.ndarray,
    bad_values: np.ndarray,
    score_name: str,
    model_name: str,
    output_path: Path,
    bins: int,
    bin_min: float,
    bin_max: float,
    normalize_bins: bool,
    show: bool,
) -> None:
    """Plot one good/bad histogram using log-spaced bins and log axes.

    If normalize_bins=True, each class is normalized separately so the sum over
    bins is 1 for good and 1 for bad. This gives fraction-of-windows per bin.
    """
    good_values = np.asarray(good_values).reshape(-1)
    bad_values = np.asarray(bad_values).reshape(-1)

    # Log scale cannot use zero, negative, nan, or inf values.
    good_values = good_values[np.isfinite(good_values) & (good_values > 0)]
    bad_values = bad_values[np.isfinite(bad_values) & (bad_values > 0)]

    if len(good_values) == 0 and len(bad_values) == 0:
        print(f"Skipping {model_name} {score_name}: no positive finite scores.", flush=True)
        return

    log_bins = make_log_bins(
        good_values=good_values,
        bad_values=bad_values,
        n_bins=bins,
        default_min=bin_min,
        default_max=bin_max,
    )

    plt.figure(figsize=(9, 6))

    if len(good_values) > 0:
        if normalize_bins:
            good_weights = np.ones_like(good_values, dtype=float) / len(good_values)
        else:
            good_weights = None

        plt.hist(
            good_values,
            bins=log_bins,
            weights=good_weights,
            label=f"Good, n={len(good_values)}",
            alpha=0.4,
        )

    if len(bad_values) > 0:
        if normalize_bins:
            bad_weights = np.ones_like(bad_values, dtype=float) / len(bad_values)
        else:
            bad_weights = None

        plt.hist(
            bad_values,
            bins=log_bins,
            weights=bad_weights,
            label=f"Bad, n={len(bad_values)}",
            alpha=0.4,
        )

    plt.legend()
    plt.xlabel("Normalized Score" if score_name in {"scores", "scores_max"} else score_name)
    plt.ylabel("Fraction of windows" if normalize_bins else "Windows")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"{model_name}: good vs bad distribution of {score_name}")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving: {output_path}", flush=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved:  {output_path}", flush=True)

    if show:
        plt.show()

    plt.close()


def print_file_summary(
    *,
    score_file: Path,
    first_run: np.ndarray,
    score_arrays_to_plot: list[str],
) -> None:
    """Print basic count information for one inference_scores.npz file."""
    first_run = np.asarray(first_run).astype(int).reshape(-1)

    good_mask = np.isin(first_run, list(GOOD_RUNS))
    bad_mask = np.isin(first_run, list(BAD_RUNS))
    known_mask = good_mask | bad_mask

    model_name = safe_model_name(score_file)

    print("=" * 80, flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"File:  {score_file}", flush=True)
    print(f"Arrays to plot: {', '.join(score_arrays_to_plot)}", flush=True)
    print(f"Total windows:   {len(first_run)}", flush=True)
    print(f"Good windows:    {int(good_mask.sum())}", flush=True)
    print(f"Bad windows:     {int(bad_mask.sum())}", flush=True)
    print(f"Unknown windows: {int((~known_mask).sum())}", flush=True)

    unique_runs, counts = np.unique(first_run, return_counts=True)

    print("Run counts:", flush=True)
    for run, count in zip(unique_runs, counts):
        run_int = int(run)

        if run_int in GOOD_RUNS:
            label = "good"
        elif run_int in BAD_RUNS:
            label = "bad"
        else:
            label = "unknown"

        print(f"  run {run_int}: {int(count)} windows ({label})", flush=True)


def process_one_file(
    *,
    score_file: Path,
    output_dir: Path,
    score_arrays_to_plot: list[str],
    bins: int,
    bin_min: float,
    bin_max: float,
    normalize_bins: bool,
    show: bool,
    rng: np.random.Generator,
) -> None:
    """Read one inference_scores.npz and save requested plots."""
    model_name = safe_model_name(score_file)
    rng = np.random.default_rng(RANDOM_SEED)

    with np.load(score_file, allow_pickle=True) as data:
        first_run = load_required_array(data, "first_run", score_file)

        print_file_summary(
            score_file=score_file,
            first_run=first_run,
            score_arrays_to_plot=score_arrays_to_plot,
        )

        for score_name in score_arrays_to_plot:
            values = load_required_array(data, score_name, score_file)

            good_values, bad_values, unknown_values = split_good_bad(
                values=values,
                first_run=first_run,
                score_name=score_name,
                file_path=score_file,
            )

            good_values, bad_values = maybe_sample_majority_class(
                good_values,
                bad_values,
                sample_fraction=BALANCED_SAMPLE_FRACTION,
                rng=rng,
                model_name=model_name,
                score_name=score_name,
            )

            if len(good_values) == 0 and len(bad_values) == 0:
                print(
                    f"Skipping {model_name} {score_name}: "
                    "no entries matched GOOD_RUNS or BAD_RUNS.",
                    flush=True,
                )
                continue

            if len(unknown_values) > 0:
                print(
                    f"Note: {model_name} {score_name} has "
                    f"{len(unknown_values)} unknown windows that will not be plotted.",
                    flush=True,
                )

            good_values, bad_values = maybe_balance_good_bad_windows(
                good_values,
                bad_values,
                balance_fraction=BALANCED_SAMPLE_FRACTION,
                rng=rng,
                model_name=model_name,
                score_name=score_name,
            )

            output_path = output_dir / f"{model_name}_good_bad_{score_name}_hist.png"

            print(f"Plotting {model_name} {score_name}...", flush=True)
            plot_good_bad_histogram(
                good_values=good_values,
                bad_values=bad_values,
                score_name=score_name,
                model_name=model_name,
                output_path=output_path,
                bins=bins,
                bin_min=bin_min,
                bin_max=bin_max,
                normalize_bins=normalize_bins,
                show=show,
            )


def main() -> int:
    args = parse_args()

    score_arrays_to_plot = choose_score_arrays(args)

    score_files = find_inference_score_files(
        input_path=args.input,
        recursive=args.recursive,
        start=args.start,
    )

    if not score_files:
        raise FileNotFoundError(
            f"No {NPZ_NAME} files found in {args.input}. Checked the input directory "
            "and its immediate subdirectories. Use --recursive if your files are deeper. "
            "If --start was used, no model directories may have passed the filter."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(score_files)} {NPZ_NAME} file(s).", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(f"Start filter: {args.start if args.start is not None else 'none'}", flush=True)
    print(f"Bin normalization: {'raw counts' if args.counts else 'fraction per bin'}", flush=True)
    print(
        "Balanced sampling: "
        f"{BALANCED_SAMPLE_FRACTION if BALANCED_SAMPLE_FRACTION is not None else 'disabled'}",
        flush=True,
    )

    rng = np.random.default_rng(RANDOM_SEED)

    for score_file in score_files:
        process_one_file(
            score_file=score_file,
            output_dir=args.output_dir,
            score_arrays_to_plot=score_arrays_to_plot,
            bins=args.bins,
            bin_min=args.bin_min,
            bin_max=args.bin_max,
            normalize_bins=not args.counts,
            show=args.show,
            rng=rng,
        )

    print("=" * 80, flush=True)
    print("Done.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
