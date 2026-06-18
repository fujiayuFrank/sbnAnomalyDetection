#!/usr/bin/env python3
"""
plot_channel_ranges_simple.py

Python/uproot version of the channel histogram plotter.

Compared with the previous version, this simplified version:
  - plots only the top channel histogram
  - removes reduced chi-square printing
  - removes chi-square table output
  - removes the bottom fractional-difference plot
  - removes error bars and shaded error bands

Run one range:
    python plot_channel_ranges_simple.py --channel-min 3800 --channel-max 6000

Run all predefined ranges:
    python plot_channel_ranges_simple.py --all-ranges

Fast tests:
    python plot_channel_ranges_simple.py --channel-min 3800 --channel-max 6000 --max-files 5
    python plot_channel_ranges_simple.py --all-ranges --max-files 5
"""

import argparse
import glob
import os
import re

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot


# ------------------------------------------------------------
# Good/bad classification Switch
# True  = good runs one color palette, bad runs another color palette
# False = each run gets its own color from matplotlib default cycle
# ------------------------------------------------------------

COLOR_BY_GOOD_BAD = True


# ------------------------------------------------------------
# Channel selection settings
# Use [channel_min, channel_max), meaning:
# channel_min included, channel_max excluded
# ------------------------------------------------------------

USE_CHANNEL_CUT = True

CHANNEL_RANGES = [
    # (0, 11276),  # full range
    (3900, 5700),
    (9600, 11276),
]


# ------------------------------------------------------------
# List of run directories
# First one is the reference run
# ------------------------------------------------------------

RUN_DIRS = [
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_19305/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_19308/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_19315/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_829/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20769/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20782/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20768/reco/",

    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20614/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20615/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20620/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20621/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20173/reco/",
    # "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_830/reco/",
]


# ------------------------------------------------------------
# Good/bad classification sets
# ------------------------------------------------------------

GOOD_RUNS = {19305, 19308, 19315, 829, 20769, 20782, 20768}
BAD_RUNS = {20614, 20615, 20620, 20621, 20173, 830}


# ------------------------------------------------------------
# Color palettes
# Good runs use cold colors.
# Bad runs use hot colors.
# ------------------------------------------------------------

GOOD_COLORS = [
    "#2483c8",
    "#063d6b",
    "#17becf",
    "#00a087",
    "#2ca02c",
    "#4daf4a",
    "#66c2a5",
]

BAD_COLORS = [
    "#d62728",
    "#e41a1c",
    "#b2182b",
    "#ff7f0e",
    "#a65628",
    "#613807",
]

UNKNOWN_COLOR = "black"


# ------------------------------------------------------------
# ROOT tree / branch settings
# ------------------------------------------------------------

TREE_PATH = "caloskim/TrackCaloSkim"
BRANCH = "hits2.h.channel"


def extract_run_number(path):
    match = re.search(r"CI_build_lar_ci_([0-9]+)", path)
    if match:
        return int(match.group(1))
    return -1


def get_run_status(run):
    if run in GOOD_RUNS:
        return "good"
    if run in BAD_RUNS:
        return "bad"
    return "unknown"


def get_run_color(run):
    if not COLOR_BY_GOOD_BAD:
        return None

    if run in GOOD_RUNS:
        index = 0
        for directory in RUN_DIRS:
            r = extract_run_number(directory)
            if r not in GOOD_RUNS:
                continue
            if r == run:
                return GOOD_COLORS[index % len(GOOD_COLORS)]
            index += 1

    if run in BAD_RUNS:
        index = 0
        for directory in RUN_DIRS:
            r = extract_run_number(directory)
            if r not in BAD_RUNS:
                continue
            if r == run:
                return BAD_COLORS[index % len(BAD_COLORS)]
            index += 1

    return UNKNOWN_COLOR


def make_channel_bins(channel_min, channel_max):
    """
    Make one bin per channel in [channel_min, channel_max).
    Example: [3800, 6000) gives 2200 bins with edges 3800..6000.
    """
    if channel_max <= channel_min:
        raise ValueError(f"Invalid channel range: [{channel_min}, {channel_max})")

    nbins = int(channel_max - channel_min)
    bin_edges = np.linspace(channel_min, channel_max, nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return nbins, bin_edges, bin_centers


def find_root_files(directory, max_files=None):
    pattern = os.path.join(directory, "DQMValidationTrees_*.root")
    files = sorted(glob.glob(pattern))

    if max_files is not None and max_files > 0:
        files = files[:max_files]

    return files


def read_channel_histogram_for_file(
    filename,
    channel_min,
    channel_max,
    bin_edges,
    max_entries=None,
):
    """
    Read one ROOT file with uproot.

    Returns:
        hist, n_entries

    If file is bad/unreadable:
        None, 0
    """
    try:
        with uproot.open(filename) as root_file:
            if TREE_PATH not in root_file:
                print(f"  Skipping file without tree {TREE_PATH}: {filename}")
                return None, 0

            tree = root_file[TREE_PATH]
            n_entries = tree.num_entries

            entry_stop = None
            if max_entries is not None and max_entries > 0:
                entry_stop = min(max_entries, n_entries)

            arr = tree[BRANCH].array(entry_stop=entry_stop)

            # hits2.h.channel is jagged: one list of hit channels per event.
            flat = ak.to_numpy(ak.flatten(arr, axis=None))
            flat = flat[np.isfinite(flat)]

            # Channel cut, matching ROOT:
            # hits2.h.channel >= channel_min && hits2.h.channel < channel_max
            if USE_CHANNEL_CUT:
                flat = flat[(flat >= channel_min) & (flat < channel_max)]

            hist, _ = np.histogram(flat, bins=bin_edges)
            return hist.astype(float), n_entries

    except Exception as exc:
        print(f"  Skipping bad/unreadable file: {filename}")
        print(f"    Reason: {exc}")
        return None, 0


def read_channel_histogram_for_run(
    directory,
    channel_min,
    channel_max,
    bin_edges,
    max_files=None,
    max_entries_per_file=None,
):
    files = find_root_files(directory, max_files=max_files)

    if not files:
        raise RuntimeError(f"No DQMValidationTrees_*.root files found in {directory}")

    nbins = len(bin_edges) - 1
    total_hist = np.zeros(nbins, dtype=float)
    n_good_files = 0
    total_entries = 0

    for file_index, filename in enumerate(files, start=1):
        print(f"  [{file_index}/{len(files)}] reading {filename}", flush=True)

        hist, n_entries = read_channel_histogram_for_file(
            filename,
            channel_min=channel_min,
            channel_max=channel_max,
            bin_edges=bin_edges,
            max_entries=max_entries_per_file,
        )

        if hist is None:
            continue

        total_hist += hist
        total_entries += n_entries
        n_good_files += 1

    return total_hist, n_good_files, total_entries


def plot_channel_histograms(
    hists,
    run_numbers,
    channel_min,
    channel_max,
    bin_centers,
    normalize=True,
):
    run_ref = run_numbers[0]

    fig, ax = plt.subplots(figsize=(14, 7))

    ymax = 0.0

    for hist, run in zip(hists, run_numbers):
        color = get_run_color(run) if COLOR_BY_GOOD_BAD else None
        status = get_run_status(run)

        if run == run_ref:
            label = f"Data ID {run} reference ({status})"
        else:
            if normalize:
                label = f"Data ID {run} normalized ({status})"
            else:
                label = f"Data ID {run} ({status})"

        # No error bars and no shaded uncertainty band.
        ax.step(
            bin_centers,
            hist,
            where="mid",
            linewidth=1.2,
            color=color,
            label=label,
        )

        ymax = max(ymax, float(np.max(hist)))

    ax.set_title(
        f"Hit Channel Comparison, channels [{channel_min}, {channel_max})"
    )
    ax.set_xlabel("Channel")
    ax.set_ylabel("Hits")
    ax.set_xlim(channel_min, channel_max)
    ax.set_ylim(bottom=0, top=1.15 * ymax if ymax > 0 else 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    run_tag = "_".join(str(run) for run in run_numbers)
    mode_tag = "good_bad_colors" if COLOR_BY_GOOD_BAD else "multi_colors"

    png_name = (
        f"channel_comparison_many_runs_{mode_tag}"
        f"_ch_{channel_min}_{channel_max}_{run_tag}.png"
    )
    pdf_name = (
        f"channel_comparison_many_runs_{mode_tag}"
        f"_ch_{channel_min}_{channel_max}_{run_tag}.pdf"
    )

    fig.savefig(png_name, dpi=200)
    fig.savefig(pdf_name)
    plt.close(fig)

    print(f"Saved {png_name} and {pdf_name}")


def plot_channel(
    channel_min=0,
    channel_max=500,
    max_files=None,
    max_entries_per_file=None,
    normalize=True,
):
    """
    Plot hits2.h.channel for one channel range [channel_min, channel_max).

    Example:
        plot_channel(3800, 6000)
        plot_channel(9500, 11276)
    """
    if len(RUN_DIRS) < 2:
        raise RuntimeError("Need at least two run directories.")

    nbins, bin_edges, bin_centers = make_channel_bins(channel_min, channel_max)

    print(f"Using channel cut: {BRANCH} >= {channel_min} && {BRANCH} < {channel_max}")
    print(f"Histogram bins: {nbins}, range [{channel_min}, {channel_max})")

    run_numbers = []
    raw_hists = []

    for directory in RUN_DIRS:
        run = extract_run_number(directory)
        run_numbers.append(run)

        print()
        print("=" * 70)
        print(f"Run {run}: {directory}")
        print("=" * 70)

        hist, n_good_files, total_entries = read_channel_histogram_for_run(
            directory,
            channel_min=channel_min,
            channel_max=channel_max,
            bin_edges=bin_edges,
            max_files=max_files,
            max_entries_per_file=max_entries_per_file,
        )

        print(f"Run {run}: added {n_good_files} readable ROOT files")
        print(f"Run {run} entries = {total_entries}")
        print(
            f"Raw number of hits in channel histogram for run {run} "
            f"channels [{channel_min}, {channel_max}) = {hist.sum()}"
        )

        raw_hists.append(hist)

    hists = [hist.copy() for hist in raw_hists]

    if normalize:
        ref_integral = hists[0].sum()

        for i in range(1, len(hists)):
            integral = hists[i].sum()

            if integral > 0:
                scale = ref_integral / integral
                hists[i] *= scale
                print(f"Scale factor applied to run {run_numbers[i]} = {scale}")

    plot_channel_histograms(
        hists,
        run_numbers,
        channel_min=channel_min,
        channel_max=channel_max,
        bin_centers=bin_centers,
        normalize=normalize,
    )


def plot_all_channel_ranges(max_files=None, max_entries_per_file=None, normalize=True):
    """
    Plot all predefined channel ranges in CHANNEL_RANGES.
    """
    for channel_min, channel_max in CHANNEL_RANGES:
        print()
        print("=" * 70)
        print(f"Plotting channel range [{channel_min}, {channel_max})")
        print("=" * 70)

        plot_channel(
            channel_min=channel_min,
            channel_max=channel_max,
            max_files=max_files,
            max_entries_per_file=max_entries_per_file,
            normalize=normalize,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot hits2.h.channel histograms across many DQM ROOT files "
            "for one channel range or all predefined ranges. "
            "No chi-square, no bottom ratio plot, and no error bars."
        )
    )

    parser.add_argument(
        "--channel-min",
        type=int,
        default=0,
        help="Minimum channel, included. Default: 0.",
    )

    parser.add_argument(
        "--channel-max",
        type=int,
        default=500,
        help="Maximum channel, excluded. Default: 500.",
    )

    parser.add_argument(
        "--all-ranges",
        action="store_true",
        help="Plot all predefined ranges in CHANNEL_RANGES.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Use only the first N ROOT files per run. Useful for testing.",
    )

    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Read only the first N entries per file. Useful for testing.",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not normalize runs to the reference run.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.all_ranges:
        plot_all_channel_ranges(
            max_files=args.max_files,
            max_entries_per_file=args.max_entries,
            normalize=not args.no_normalize,
        )
    else:
        plot_channel(
            channel_min=args.channel_min,
            channel_max=args.channel_max,
            max_files=args.max_files,
            max_entries_per_file=args.max_entries,
            normalize=not args.no_normalize,
        )
