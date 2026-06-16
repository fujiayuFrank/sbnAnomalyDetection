import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Input / output
# ============================================================

npz_path = "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/good_bad.npz"

out_dir = "/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/integral_comparison_plots"
os.makedirs(out_dir, exist_ok=True)


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

# Sequential colors for runs that are neither in good_runs nor bad_runs.
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

marker_styles = [
    "o",
    "^",
    "s",
    "D",
    "v",
    "P",
    "X",
    "*",
    "h",
]


def get_run_status(run):
    if run in good_runs:
        return "good"
    if run in bad_runs:
        return "bad"
    return "unknown"


def get_run_color(run, all_runs):
    """
    Good runs get colors from good_colors in good-run order.
    Bad runs get colors from bad_colors in bad-run order.
    Unknown runs get colors from unknown_colors in unknown-run order.
    """

    good_order = [r for r in all_runs if r in good_runs]
    bad_order = [r for r in all_runs if r in bad_runs]
    unknown_order = [
        r for r in all_runs
        if r not in good_runs and r not in bad_runs
    ]

    if run in good_runs:
        idx = good_order.index(run)
        return good_colors[idx % len(good_colors)]

    if run in bad_runs:
        idx = bad_order.index(run)
        return bad_colors[idx % len(bad_colors)]

    idx = unknown_order.index(run)
    return unknown_colors[idx % len(unknown_colors)]


# ============================================================
# Channel selection settings
# Same convention as ROOT:
# [channel_min, channel_max)
# ============================================================

channel_ranges = [
    # (0, 11276),   # full range if you want it
    (0, 500),
]


# ============================================================
# Histogram settings
# Same as C++ ROOT plotter
# ============================================================

nbins = 100
xmin = 0
xmax = 4000

bins = np.linspace(xmin, xmax, nbins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])


# ============================================================
# Load NPZ
# ============================================================

data = np.load(npz_path, allow_pickle=True)

integrals_flat = data["integrals_flat"]
channels_flat = data["channels_flat"]
offsets = data["offsets"]
evt_run = data["evt_run"]

print("Loaded:", npz_path)
print("integrals_flat shape:", integrals_flat.shape)
print("channels_flat shape: ", channels_flat.shape)
print("offsets shape:       ", offsets.shape)
print("evt_run shape:       ", evt_run.shape)

assert len(offsets) == len(evt_run) + 1, (
    "Expected offsets length to equal number of events + 1"
)


# ============================================================
# Decide run order
# Same order as your C++ ROOT plotter
# First run is the reference run
# ============================================================

valid_evt_mask = evt_run >= 0
runs_in_file = list(np.unique(evt_run[valid_evt_mask]))

preferred_order = [
    19305, 19308, 19315, 829, 20769, 20782, 20768,
    20614, 20615, 20620, 20621, 20173, 830
]

runs = [r for r in preferred_order if r in runs_in_file]
runs += [r for r in runs_in_file if r not in preferred_order]

print("Runs found in NPZ:")
print(runs)

if len(runs) < 2:
    raise RuntimeError("Need at least two runs to make a comparison plot.")


# ============================================================
# Plot one channel range
# ============================================================

def plot_integral_for_channel_range(channel_min, channel_max):
    print()
    print("========================================")
    print(f"Plotting channel range [{channel_min}, {channel_max})")
    print("========================================")

    channel_tag = f"ch_{channel_min}_{channel_max}"

    hist_counts = {}
    hist_errors = {}
    raw_integrals = {}
    n_events_by_run = {}
    n_hits_by_run = {}

    # --------------------------------------------------------
    # Build histogram for each run
    # --------------------------------------------------------

    for run in runs:
        evt_indices = np.where(evt_run == run)[0]
        n_events_by_run[run] = len(evt_indices)

        pieces = []

        for i_evt in evt_indices:
            start = int(offsets[i_evt])
            end = int(offsets[i_evt + 1])

            vals = integrals_flat[start:end]
            chs = channels_flat[start:end]

            channel_mask = (chs >= channel_min) & (chs < channel_max)
            vals = vals[channel_mask]

            if len(vals) > 0:
                pieces.append(vals)

        if len(pieces) == 0:
            run_integrals = np.array([], dtype=integrals_flat.dtype)
        else:
            run_integrals = np.concatenate(pieces)

        n_hits_by_run[run] = len(run_integrals)

        counts, _ = np.histogram(run_integrals, bins=bins)

        # Poisson uncertainty before scaling
        errors = np.sqrt(counts)

        hist_counts[run] = counts.astype(float)
        hist_errors[run] = errors.astype(float)
        raw_integrals[run] = float(np.sum(counts))

        print(
            f"Run {run}: "
            f"{n_events_by_run[run]} events, "
            f"{n_hits_by_run[run]} selected hits, "
            f"hist integral = {raw_integrals[run]}"
        )

    # --------------------------------------------------------
    # Normalize all non-reference runs to reference run
    # Same as C++ ROOT plotter
    # --------------------------------------------------------

    ref_run = runs[0]
    ref_integral = raw_integrals[ref_run]

    if ref_integral <= 0:
        print(
            f"Reference run {ref_run} has zero histogram integral "
            f"for channels [{channel_min}, {channel_max}), skipping."
        )
        return

    scale_factors = {ref_run: 1.0}

    for run in runs[1:]:
        integral = raw_integrals[run]

        if integral > 0:
            scale = ref_integral / integral
        else:
            scale = 1.0

        scale_factors[run] = scale

        raw_max_bin = np.max(hist_counts[run]) if len(hist_counts[run]) > 0 else 0.0
        raw_peak_bin = np.argmax(hist_counts[run]) + 1
        raw_peak_x = bin_centers[raw_peak_bin - 1]

        hist_counts[run] *= scale
        hist_errors[run] *= scale

        scaled_max_bin = np.max(hist_counts[run]) if len(hist_counts[run]) > 0 else 0.0

        print()
        print(f"Run {run} ({get_run_status(run)})")
        print(f"  raw integral       = {integral}")
        print(f"  reference integral = {ref_integral}")
        print(f"  scale factor       = {scale}")
        print(f"  raw max bin        = {raw_max_bin}")
        print(f"  raw peak x         = {raw_peak_x}")
        print(f"  scaled max bin     = {scaled_max_bin}")

    # --------------------------------------------------------
    # Chi-square / ndf vs reference
    # Same as C++ ROOT plotter
    # --------------------------------------------------------

    ref_counts = hist_counts[ref_run]
    ref_errors = hist_errors[ref_run]

    chi2_info = {}

    chi2_info[ref_run] = {
        "chi2": 0.0,
        "ndf": 0,
        "reduced_chi2": 0.0,
    }

    for run in runs[1:]:
        other_counts = hist_counts[run]
        other_errors = hist_errors[run]

        chi2 = 0.0
        n_used_bins = 0

        for a, ea, b, eb in zip(
            ref_counts,
            ref_errors,
            other_counts,
            other_errors,
        ):
            err2 = ea * ea + eb * eb

            if err2 > 0 and (a > 0 or b > 0):
                chi2 += (a - b) * (a - b) / err2
                n_used_bins += 1

        ndf = n_used_bins - 1
        reduced_chi2 = chi2 / ndf if ndf > 0 else 0.0

        chi2_info[run] = {
            "chi2": chi2,
            "ndf": ndf,
            "reduced_chi2": reduced_chi2,
        }

        print()
        print(f"Comparison: run {ref_run} vs run {run}")
        print(f"Chi2 = {chi2}")
        print(f"NDF = {ndf}")
        print(f"Reduced chi2 = {reduced_chi2}")

    # --------------------------------------------------------
    # Canvas style:
    # Top plot = histograms + error bands + error bars
    # Bottom plot = fractional difference
    # --------------------------------------------------------

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={
            "height_ratios": [2.1, 1.0],
            "hspace": 0.05,
        },
    )

    # --------------------------------------------------------
    # Top panel
    # --------------------------------------------------------

    ymax = 0.0

    for i, run in enumerate(runs):
        color = get_run_color(run, runs)
        marker = marker_styles[i % len(marker_styles)]

        y = hist_counts[run]
        yerr = hist_errors[run]

        if len(y) > 0:
            ymax = max(ymax, np.max(y + yerr))

        if run == ref_run:
            label = f"Data ID {run} reference ({get_run_status(run)})"
        else:
            label = f"Data ID {run} normalized ({get_run_status(run)})"

        # ROOT-like error band: Draw("E2")
        ax_top.fill_between(
            bin_centers,
            y - yerr,
            y + yerr,
            step="mid",
            color=color,
            alpha=0.18,
            linewidth=0,
        )

        # ROOT-like histogram line: Draw("HIST SAME")
        ax_top.step(
            bins[:-1],
            y,
            where="post",
            color=color,
            linewidth=2,
        )

        # ROOT-like error bars: Draw("E1 SAME")
        ax_top.errorbar(
            bin_centers,
            y,
            yerr=yerr,
            fmt=marker,
            color=color,
            markersize=4,
            linewidth=1,
            capsize=1.5,
            label=label,
        )

    ax_top.set_ylabel("Hits")
    ax_top.set_ylim(0, 1.15 * ymax if ymax > 0 else 1)

    ax_top.set_title(
        f"Integral of gaussian fit to ADC values Collection, "
        f"channels [{channel_min},{channel_max})"
    )

    ax_top.grid(True)

    ax_top.legend(
        loc="upper right",
        fontsize=8,
        frameon=True,
        framealpha=1.0,
    )

    # Chi-square text
    text_x = 0.58
    text_y = 0.50

    for run in runs[1:]:
        reduced = chi2_info[run]["reduced_chi2"]

        ax_top.text(
            text_x,
            text_y,
            rf"run {run}: $\chi^2$/ndf = {reduced:.6f}",
            transform=ax_top.transAxes,
            fontsize=8,
        )

        text_y -= 0.04

    # --------------------------------------------------------
    # Bottom panel:
    # fractional difference = (reference - other) / reference
    # Same as C++ ROOT plotter
    # --------------------------------------------------------

    for i, run in enumerate(runs[1:], start=1):
        color = get_run_color(run, runs)
        marker = marker_styles[i % len(marker_styles)]

        a = hist_counts[ref_run]
        ea = hist_errors[ref_run]

        b = hist_counts[run]
        eb = hist_errors[run]

        ratio = np.zeros_like(a, dtype=float)
        ratio_err = np.zeros_like(a, dtype=float)

        nonzero = a > 0

        ratio[nonzero] = (a[nonzero] - b[nonzero]) / a[nonzero]

        # Same error propagation as C++:
        # f = (a - b) / a = 1 - b/a
        # df/da = b/a^2
        # df/db = -1/a
        ratio_err[nonzero] = np.sqrt(
            ((b[nonzero] / (a[nonzero] * a[nonzero])) * ea[nonzero]) ** 2
            + (eb[nonzero] / a[nonzero]) ** 2
        )

        ax_bot.errorbar(
            bin_centers,
            ratio,
            yerr=ratio_err,
            fmt=marker,
            color=color,
            markersize=4,
            linewidth=1,
            capsize=1.5,
            label=f"run {run} ({get_run_status(run)})",
        )

    ax_bot.axhline(
        0.0,
        color="black",
        linewidth=2,
    )

    ax_bot.set_ylabel(
        rf"$\frac{{(\mathrm{{run}}\ {ref_run}) - (\mathrm{{other\ run}})}}"
        rf"{{(\mathrm{{run}}\ {ref_run})}}$"
    )

    ax_bot.set_xlabel("Integral")
    ax_bot.set_ylim(-1.0, 1.0)
    ax_bot.grid(True)

    ax_bot.legend(
        loc="upper right",
        fontsize=8,
        frameon=True,
        framealpha=1.0,
    )

    # --------------------------------------------------------
    # Save plot
    # --------------------------------------------------------

    run_tag = "_".join(str(r) for r in runs)

    out_png = os.path.join(
        out_dir,
        f"integral_comparison_many_runs_good_bad_unknown_colors_{channel_tag}_{run_tag}.png"
    )

    out_pdf = os.path.join(
        out_dir,
        f"integral_comparison_many_runs_good_bad_unknown_colors_{channel_tag}_{run_tag}.pdf"
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_pdf)
    plt.close()

    print()
    print("Saved plot:")
    print(out_png)
    print(out_pdf)

    # --------------------------------------------------------
    # Save chi-square table
    # --------------------------------------------------------

    txt_path = os.path.join(
        out_dir,
        f"chi2_table_{channel_tag}.txt"
    )

    with open(txt_path, "w") as f:
        header = (
            "\n"
            "============================================================\n"
            f"Chi-square table for channels [{channel_min}, {channel_max})\n"
            "============================================================\n"
            f"{'Run':<12}"
            f"{'Status':<12}"
            f"{'Chi2':<18}"
            f"{'NDF':<10}"
            f"{'Chi2/NDF':<18}\n"
            + "-" * 70
            + "\n"
        )

        print(header, end="")
        f.write(header)

        for run in runs:
            row = (
                f"{run:<12}"
                f"{get_run_status(run):<12}"
                f"{chi2_info[run]['chi2']:<18.6f}"
                f"{chi2_info[run]['ndf']:<10}"
                f"{chi2_info[run]['reduced_chi2']:<18.6f}"
                "\n"
            )

            print(row, end="")
            f.write(row)

        footer = "============================================================\n"

        print(footer, end="")
        f.write(footer)

    print("Saved chi-square table:")
    print(txt_path)


# ============================================================
# Plot all predefined channel ranges
# ============================================================

def plot_all_channel_ranges():
    for channel_min, channel_max in channel_ranges:
        plot_integral_for_channel_range(channel_min, channel_max)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    plot_all_channel_ranges()