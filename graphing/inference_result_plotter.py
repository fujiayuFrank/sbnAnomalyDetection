import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Input / output
# ============================================================

npz_path = "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/checkpoints/gnn/June-22-test/inference_scores.npz"

out_dir = "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/inference_result_plots/June-22-scores"
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# Plot settings
# ============================================================

bins = 60

# x-axis cutoff. Only show up to this percentile of the selected runs.
percentile = 75

# Do NOT use density for this mode.
# Instead, each bin is normalized by the total number of windows in that run.
normalize_to_windows = True

# Variable binning: more bins near low scores, fewer bins in the high-score tail.
binning_mode = "variable"   # options: "linear", "variable"

# For variable binning:
# [0, tail_start_percentile] gets many bins;
# [tail_start_percentile, percentile] gets fewer bins.
tail_start_percentile = 60
low_score_bin_fraction = 0.80   # the percentage of bins used in the low-score peak region

# ============================================================
# Good / bad classification
# Same as C++ ROOT plotter
# ============================================================

good_runs = {
    18445, 19724, 20141, 20142, 20144
}

bad_runs = {
    19627, 19946, 20104
}

# ============================================================
# Optional run selection
# If None or empty, plot all runs found in the NPZ.
# If a list is given, only plot these runs.
# Missing runs are skipped with a warning.
# ============================================================

runs_to_plot = [
    18445,
    19627,
    19724,
    19946,
    # 20104,
    20141,
    20142,
    20144,
]

# Use this instead if you want all runs:
# runs_to_plot = None


# ============================================================
# Color palettes
# Good runs use cold colors.
# Bad runs use hot colors.
# Unknown runs use sequential colors.
# ============================================================

good_colors = [
    "#2483c8",  # blue
    "#063d6b",  # medium blue
    "#17becf",  # cyan
    "#00a087",  # teal
    "#2ca02c",  # green
    "#4daf4a",  # medium green
    "#66c2a5",  # pale teal
]

bad_colors = [
    "#d62728",  # red
    "#e41a1c",  # bright red
    "#b2182b",  # dark red
    "#ff7f0e",  # orange
    "#a65628",  # brown-orange
    "#613807",  # dark brown-orange
]

unknown_colors = [
    "#7f7f7f",  # gray
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#bcbd22",  # olive
    "#1f77b4",  # blue
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
    "#c49c94",  # light brown
    "#dbdb8d",  # light olive
]


# ============================================================
# Helpers
# ============================================================

def get_run_color(run, good_idx, bad_idx, unknown_idx):
    if run in good_runs:
        color = good_colors[good_idx % len(good_colors)]
        good_idx += 1
        return color, good_idx, bad_idx, unknown_idx
    elif run in bad_runs:
        color = bad_colors[bad_idx % len(bad_colors)]
        bad_idx += 1
        return color, good_idx, bad_idx, unknown_idx
    else:
        color = unknown_colors[unknown_idx % len(unknown_colors)]
        unknown_idx += 1
        return color, good_idx, bad_idx, unknown_idx


def get_run_label(run):
    if run in good_runs:
        return f"Run {run} (Good)"
    elif run in bad_runs:
        return f"Run {run} (Bad)"
    else:
        return f"Run {run} (Unknown)"


def resolve_runs_to_plot(all_runs, requested_runs):
    """
    Decide which runs to plot.

    Parameters
    ----------
    all_runs : array-like
        Runs actually present in the NPZ file.
    requested_runs : list[int] or None
        User-selected runs. If None or empty, plot all runs.

    Returns
    -------
    selected_runs : list[int]
        Runs that exist and should be plotted.
    """
    all_runs_set = set(int(r) for r in all_runs)

    if requested_runs is None or len(requested_runs) == 0:
        return sorted(all_runs_set)

    selected_runs = []
    for run in requested_runs:
        run = int(run)
        if run in all_runs_set:
            selected_runs.append(run)
        else:
            print(f"WARNING: requested run {run} does not exist in this NPZ file; skipping.")

    if len(selected_runs) == 0:
        print("WARNING: none of the requested runs exist in this NPZ file.")

    return selected_runs


def make_bin_edges(
    selected_values,
    bins=80,
    percentile=99.5,
    binning_mode="linear",
    tail_start_percentile=70,
    low_score_bin_fraction=0.75,
):
    """
    Make common bin edges for all runs.

    linear:
        Uniform bin width from 0 to x_max.

    variable:
        More bins in the low-score region, fewer bins in the tail.
        This helps resolve the peak without using too many total bins.
    """

    selected_values = np.asarray(selected_values)
    selected_values = selected_values[np.isfinite(selected_values)]

    if selected_values.size == 0:
        return None, None, None

    x_min = 0.0
    x_max = np.percentile(selected_values, percentile)

    if not np.isfinite(x_max) or x_max <= x_min:
        print(f"WARNING: bad x_max={x_max}; using max selected value instead.")
        x_max = np.nanmax(selected_values)

    if not np.isfinite(x_max) or x_max <= x_min:
        return None, None, None

    if binning_mode == "linear":
        bin_edges = np.linspace(x_min, x_max, bins + 1)
        return bin_edges, x_min, x_max

    if binning_mode != "variable":
        raise ValueError(f"Unknown binning_mode={binning_mode!r}. Use 'linear' or 'variable'.")

    # Boundary between dense low-score bins and coarse tail bins
    x_split = np.percentile(selected_values, tail_start_percentile)

    # Make sure x_split is useful
    if not np.isfinite(x_split) or x_split <= x_min or x_split >= x_max:
        print(
            f"WARNING: bad x_split={x_split}; falling back to linear binning."
        )
        bin_edges = np.linspace(x_min, x_max, bins + 1)
        return bin_edges, x_min, x_max

    low_bins = int(round(bins * low_score_bin_fraction))
    tail_bins = bins - low_bins

    # Avoid degenerate cases
    low_bins = max(low_bins, 1)
    tail_bins = max(tail_bins, 1)

    low_edges = np.linspace(x_min, x_split, low_bins + 1)
    tail_edges = np.linspace(x_split, x_max, tail_bins + 1)

    # Combine, removing duplicate x_split
    bin_edges = np.concatenate([low_edges, tail_edges[1:]])

    # Remove accidental duplicate edges
    bin_edges = np.unique(bin_edges)

    if bin_edges.size < 2:
        return None, None, None

    return bin_edges, x_min, x_max


def plot_hist_by_run(
    values,
    runs,
    xlabel,
    title,
    output_path,
    bins=80,
    percentile=99.5,
    runs_to_plot=None,
    normalize_to_windows=True,
    binning_mode="variable",
    tail_start_percentile=70,
    low_score_bin_fraction=0.75,
):
    """
    Plot per-run histograms as ROOT-style line histograms.

    If normalize_to_windows=True:
        y = bin_count / total_number_of_windows_for_that_run

    This makes runs with many windows and few windows comparable in shape.
    """

    # Keep only finite entries
    finite_mask = np.isfinite(values) & np.isfinite(runs)
    values = np.asarray(values)[finite_mask]
    runs = np.asarray(runs)[finite_mask].astype(int)

    if values.size == 0:
        print(f"No valid values found for {title}")
        return

    all_runs = sorted(np.unique(runs))
    selected_runs = resolve_runs_to_plot(all_runs, runs_to_plot)

    if len(selected_runs) == 0:
        print(f"No runs to plot for {title}")
        return

    # Use selected runs only when computing percentile x range
    selected_mask = np.isin(runs, selected_runs)
    selected_values = values[selected_mask]

    if selected_values.size == 0:
        print(f"No values found for selected runs in {title}")
        return

    bin_edges, x_min, x_max = make_bin_edges(
        selected_values=selected_values,
        bins=bins,
        percentile=percentile,
        binning_mode=binning_mode,
        tail_start_percentile=tail_start_percentile,
        low_score_bin_fraction=low_score_bin_fraction,
    )

    if bin_edges is None:
        print(f"WARNING: cannot determine valid bin edges for {title}; skipping.")
        return

    plt.figure(figsize=(12, 7))

    good_idx = 0
    bad_idx = 0
    unknown_idx = 0

    plotted_any = False

    for run in selected_runs:
        mask = (runs == run)
        run_values_all = values[mask]
        n_total_windows = run_values_all.size

        if n_total_windows == 0:
            continue

        # Only histogram the visible range.
        run_values_visible = run_values_all[
            (run_values_all >= x_min) & (run_values_all <= x_max)
        ]

        if run_values_visible.size == 0:
            print(
                f"WARNING: run {run} exists, but has no entries within "
                f"[{x_min}, {x_max}] for {title}; skipping."
            )
            continue

        color, good_idx, bad_idx, unknown_idx = get_run_color(
            run, good_idx, bad_idx, unknown_idx
        )

        raw_counts, edges = np.histogram(
            run_values_visible,
            bins=bin_edges,
            density=False,
        )

        if normalize_to_windows:
            # Normalize by total number of windows in this run, not just visible windows.
            # This means bins above x_max are simply not shown, but the normalization
            # still represents fraction of the full run.
            y_values = raw_counts / n_total_windows
            ylabel = "Fraction of windows"
        else:
            y_values = raw_counts
            ylabel = "Count"

        label = (
            f"{get_run_label(run)}  "
            f"(N={n_total_windows}, shown={run_values_visible.size})"
        )

        # For a true step histogram, x has length nbins+1 and y is extended by one.
        y_step = np.r_[y_values, y_values[-1]]

        plt.step(
            edges,
            y_step,
            where="post",
            color=color,
            linewidth=2,
            label=label,
        )

        plotted_any = True

    if not plotted_any:
        print(f"WARNING: no runs were actually plotted for {title}.")
        plt.close()
        return

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if binning_mode == "variable":
        title_extra = (
            f"Shown up to {percentile}th percentile; "
            f"variable bins, split at {tail_start_percentile}th percentile"
        )
    else:
        title_extra = f"Shown up to {percentile}th percentile; linear bins"

    plt.title(f"{title}\n{title_extra} (x_max = {x_max:.4g})")
    plt.xlim(x_min, x_max)

    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    print(f"  plotted runs = {selected_runs}")
    print(f"  x_max ({percentile}th percentile of selected runs) = {x_max}")
    print(f"  binning_mode = {binning_mode}")
    print(f"  number of bins = {len(bin_edges) - 1}")


# ============================================================
# Main
# ============================================================

data = np.load(npz_path, allow_pickle=True)

print("=" * 80)
print(f"Reading NPZ file: {npz_path}")
print("=" * 80)
print("\nKeys:")
for key in data.keys():
    print(f"  - {key}")

scores = data["scores"]
scores_max = data["scores_max"]
first_run = data["first_run"]

print("\nArray shapes:")
print(f"  scores.shape     = {scores.shape}")
print(f"  scores_max.shape = {scores_max.shape}")
print(f"  first_run.shape  = {first_run.shape}")

if not (scores.shape == scores_max.shape == first_run.shape):
    raise ValueError("scores, scores_max, and first_run must have the same shape")

available_runs = sorted(np.unique(first_run[np.isfinite(first_run)]).astype(int))
print("\nAvailable runs in NPZ:")
print(available_runs)

# ============================================================
# Histogram 1: scores
# ============================================================

plot_hist_by_run(
    values=scores,
    runs=first_run,
    xlabel="Score",
    title="Histogram of window scores by run",
    output_path=os.path.join(out_dir, "scores_hist_by_run.png"),
    bins=bins,
    percentile=percentile,
    runs_to_plot=runs_to_plot,
    normalize_to_windows=normalize_to_windows,
    binning_mode=binning_mode,
    tail_start_percentile=tail_start_percentile,
    low_score_bin_fraction=low_score_bin_fraction,
)

# ============================================================
# Histogram 2: scores_max
# ============================================================

plot_hist_by_run(
    values=scores_max,
    runs=first_run,
    xlabel="Score max",
    title="Histogram of window max scores by run",
    output_path=os.path.join(out_dir, "scores_max_hist_by_run.png"),
    bins=bins,
    percentile=percentile,
    runs_to_plot=runs_to_plot,
    normalize_to_windows=normalize_to_windows,
    binning_mode=binning_mode,
    tail_start_percentile=tail_start_percentile,
    low_score_bin_fraction=low_score_bin_fraction,
)