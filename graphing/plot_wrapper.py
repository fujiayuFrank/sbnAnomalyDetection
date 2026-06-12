import re
import glob
import os

import numpy as np
import awkward as ak
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# List of directory paths
# The first one is the reference run
# ------------------------------------------------------------

run_dirs = [
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_19305/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20620/reco/"
]


# ------------------------------------------------------------
# Extract run number from path
# ------------------------------------------------------------

def extract_run_number(path):
    match = re.search(r"CI_build_lar_ci_([0-9]+)", path)
    return int(match.group(1)) if match else -1


# ------------------------------------------------------------
# Make summed histogram from all DQM ROOT files in one directory
# ------------------------------------------------------------

def make_hist_from_root_dir(run_dir, tree_path, branch, bins, step_size="100 MB"):
    hist = np.zeros(len(bins) - 1, dtype=float)

    root_files = sorted(glob.glob(os.path.join(run_dir, "DQMValidationTrees_*.root")))

    if len(root_files) == 0:
        raise RuntimeError(f"No DQMValidationTrees ROOT files found in directory: {run_dir}")

    print(f"Found {len(root_files)} DQMValidationTrees ROOT files in:")
    print(run_dir)

    for i, file_path in enumerate(root_files, start=1):
        print(f"[{i}/{len(root_files)}] Reading {file_path}")

        full_path = f"{file_path}:{tree_path}"

        try:
            for batch in uproot.iterate(
                full_path,
                [branch],
                step_size=step_size,
                library="ak"
            ):
                arr = batch[branch]

                arr = ak.flatten(arr, axis=None)
                arr = ak.to_numpy(arr)
                arr = arr[np.isfinite(arr)]

                h, _ = np.histogram(arr, bins=bins)
                hist += h

        except Exception as e:
            print(f"Skipping file due to error: {file_path}")
            print(f"Error: {e}")

    return hist


# ------------------------------------------------------------
# Chi-square compared with reference
# ------------------------------------------------------------

def calculate_chi2(h_ref, eh_ref, h_other, eh_other, subtract_normalization=True):
    err2 = eh_ref**2 + eh_other**2
    valid = (err2 > 0) & ((h_ref > 0) | (h_other > 0))

    chi2 = np.sum((h_ref[valid] - h_other[valid])**2 / err2[valid])
    n_used_bins = np.sum(valid)

    ndf = n_used_bins - 1 if subtract_normalization else n_used_bins
    reduced_chi2 = chi2 / ndf if ndf > 0 else 0.0

    return chi2, ndf, reduced_chi2


# ------------------------------------------------------------
# Main plotting function
# ------------------------------------------------------------

def plot_integral_many(
    dirs=run_dirs,
    tree_path="caloskim/TrackCaloSkim",
    branch="hits2.h.integral",
    nbins=100,
    xmin=0,
    xmax=4000,
    normalize=True,
    step_size="100 MB"
):
    if len(dirs) < 2:
        raise RuntimeError("Need at least two directories to compare.")

    run_numbers = [extract_run_number(d) for d in dirs]

    print("Runs:")
    for d, r in zip(dirs, run_numbers):
        print(f"  run {r}: {d}")

    bins = np.linspace(xmin, xmax, nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # --------------------------------------------------------
    # Make histograms for all directories
    # --------------------------------------------------------

    histograms = []
    errors = []

    for run_dir, run in zip(dirs, run_numbers):
        print(f"\nProcessing run {run}")

        h = make_hist_from_root_dir(
            run_dir,
            tree_path,
            branch,
            bins,
            step_size=step_size
        )

        eh = np.sqrt(h)

        histograms.append(h)
        errors.append(eh)

    # --------------------------------------------------------
    # Normalize all runs to the first run
    # --------------------------------------------------------

    h_ref = histograms[0]
    eh_ref = errors[0]
    run_ref = run_numbers[0]

    if normalize:
        ref_integral = h_ref.sum()

        for i in range(1, len(histograms)):
            h = histograms[i]
            eh = errors[i]

            if h.sum() > 0:
                scale = ref_integral / h.sum()
                histograms[i] = h * scale
                errors[i] = eh * scale

                print(f"Scale factor applied to run {run_numbers[i]} = {scale}")

    # Refresh after possible scaling
    h_ref = histograms[0]
    eh_ref = errors[0]

    # --------------------------------------------------------
    # Chi-square for each run relative to reference
    # --------------------------------------------------------

    chi2_results = {}

    for i in range(1, len(histograms)):
        chi2, ndf, reduced_chi2 = calculate_chi2(
            h_ref,
            eh_ref,
            histograms[i],
            errors[i],
            subtract_normalization=normalize
        )

        chi2_results[run_numbers[i]] = (chi2, ndf, reduced_chi2)

        print(f"\nComparison: run {run_ref} vs run {run_numbers[i]}")
        print(f"Chi2 = {chi2}")
        print(f"NDF = {ndf}")
        print(f"Reduced chi2 = {reduced_chi2}")

    # --------------------------------------------------------
    # Plot colors
    # --------------------------------------------------------

    colors = [
        "black",
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "magenta",
        "cyan",
    ]

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        gridspec_kw={"height_ratios": [2.1, 1]},
        sharex=True
    )

    # --------------------------------------------------------
    # Top plot: all histograms
    # --------------------------------------------------------

    for i, (h, eh, run) in enumerate(zip(histograms, errors, run_numbers)):
        color = colors[i % len(colors)]

        if i == 0:
            label = f"Data ID {run} reference"
        else:
            label = f"Data ID {run} normalized" if normalize else f"Data ID {run}"

        ax1.step(
            bin_centers,
            h,
            where="mid",
            linewidth=2,
            color=color,
            label=label
        )

        ax1.errorbar(
            bin_centers,
            h,
            yerr=eh,
            fmt="o",
            markersize=3,
            capsize=1,
            color=color
        )

        ax1.fill_between(
            bin_centers,
            h - eh,
            h + eh,
            step="mid",
            alpha=0.15,
            color=color
        )

    ax1.set_title("Integral of gaussian fit to ADC values Collection")
    ax1.set_ylabel("Hits")
    ax1.grid(True)
    ax1.legend(fontsize=9)

    # Add chi-square text
    text_lines = []
    for run, (_, _, reduced_chi2) in chi2_results.items():
        text_lines.append(rf"run {run}: $\chi^2/\mathrm{{ndf}}$ = {reduced_chi2:.4f}")

    ax1.text(
        0.60,
        0.58,
        "\n".join(text_lines),
        transform=ax1.transAxes,
        fontsize=10
    )

    # --------------------------------------------------------
    # Bottom plot:
    # fractional difference for each non-reference run
    # (reference - run_i) / reference
    # --------------------------------------------------------

    for i in range(1, len(histograms)):
        h = histograms[i]
        eh = errors[i]
        run = run_numbers[i]
        color = colors[i % len(colors)]

        frac = np.zeros_like(h_ref, dtype=float)
        efrac = np.zeros_like(h_ref, dtype=float)

        valid_frac = h_ref > 0

        a = h_ref[valid_frac]
        b = h[valid_frac]
        ea = eh_ref[valid_frac]
        eb = eh[valid_frac]

        frac[valid_frac] = (a - b) / a

        # f = (a - b)/a = 1 - b/a
        # df/da = b/a^2
        # df/db = -1/a
        efrac[valid_frac] = np.sqrt(
            (b / a**2 * ea)**2 + (eb / a)**2
        )

        ax2.errorbar(
            bin_centers,
            frac,
            yerr=efrac,
            fmt="o",
            markersize=3,
            capsize=1,
            color=color,
            label=f"run {run}"
        )

    ax2.axhline(
        0,
        linewidth=2,
        color="black"
    )

    ax2.set_ylabel(
        rf"$\frac{{(\mathrm{{run}}\ {run_ref}) - (\mathrm{{other\ run}})}}{{(\mathrm{{run}}\ {run_ref})}}$",
        fontsize=12
    )

    ax2.set_xlabel("Integral")
    ax2.set_ylim(-1.0, 1.0)
    ax2.grid(True)
    ax2.legend(fontsize=9)

    plt.tight_layout()

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    run_tag = "_".join(str(r) for r in run_numbers)

    png_name = f"integral_comparison_many_runs_{run_tag}.png"
    pdf_name = f"integral_comparison_many_runs_{run_tag}.pdf"

    fig.savefig(png_name, dpi=200)
    fig.savefig(pdf_name)

    print(f"Saved {png_name}")
    print(f"Saved {pdf_name}")


# ------------------------------------------------------------
# Run when calling: python3 plot.py
# ------------------------------------------------------------

if __name__ == "__main__":
    plot_integral_many()