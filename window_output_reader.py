import argparse
from pathlib import Path

import numpy as np


def print_array_values(arr, full=False, max_items=50, indent="  "):
    """
    Print array contents.

    If full=True, print the exact full array.
    Otherwise print a summary if the array is large.
    """
    if arr.size == 0:
        print(f"{indent}values: []")
        return

    # Object arrays need special handling
    if arr.dtype == object:
        print(f"{indent}values:")
        if full or arr.size <= max_items:
            for i, x in enumerate(arr.ravel()):
                print(f"{indent}  [{i}] {repr(x)}")
        else:
            flat = arr.ravel()
            print(f"{indent}  first {max_items}:")
            for i, x in enumerate(flat[:max_items]):
                print(f"{indent}    [{i}] {repr(x)}")
            print(f"{indent}  ...")
            print(f"{indent}  last {max_items}:")
            start = arr.size - max_items
            for i, x in enumerate(flat[-max_items:]):
                print(f"{indent}    [{start + i}] {repr(x)}")
        return

    # For normal numpy arrays
    if full or arr.size <= max_items:
        print(f"{indent}values:")
        print(arr)
    else:
        flat = arr.ravel()
        print(f"{indent}first {max_items}: {flat[:max_items]}")
        print(f"{indent}last  {max_items}: {flat[-max_items:]}")


def print_npz(npz_path, full=False, max_items=50):
    print("=" * 80)
    print(f"Reading NPZ file: {npz_path}")
    print("=" * 80)

    meta = np.load(npz_path, allow_pickle=True)

    print("\nKeys:")
    for key in meta.files:
        print(f"  - {key}")

    print("\nContents:")
    for key in meta.files:
        arr = meta[key]

        print(f"\n[{key}]")
        print(f"  shape: {arr.shape}")
        print(f"  dtype: {arr.dtype}")
        print(f"  ndim:  {arr.ndim}")
        print(f"  size:  {arr.size}")

        print_array_values(arr, full=full, max_items=max_items)

    # Extra sparse-mode interpretation, if possible
    print_sparse_npz_interpretation(meta, full=full, max_items=max_items)

    meta.close()


def print_sparse_npz_interpretation(meta, full=False, max_items=50):
    """
    Try to recognize common sparse-mode NPZ layouts and print them clearly.

    This does not assume one exact format. It checks for common key names.
    """

    keys = set(meta.files)

    possible_sparse_keys = {
        "indices",
        "values",
        "shape",
        "coords",
        "data",
        "window_indices",
        "channel_indices",
        "bin_indices",
        "feature_indices",
        "sparse_indices",
        "sparse_values",
        "sparse_shape",
    }

    if not (keys & possible_sparse_keys):
        return

    print("\n" + "=" * 80)
    print("Sparse-mode interpretation")
    print("=" * 80)

    # Case 1: generic sparse_indices / sparse_values / sparse_shape
    if {"sparse_indices", "sparse_values"}.issubset(keys):
        indices = meta["sparse_indices"]
        values = meta["sparse_values"]

        print("\nDetected sparse_indices + sparse_values format")
        print(f"  sparse_indices shape: {indices.shape}")
        print(f"  sparse_values shape:  {values.shape}")

        if "sparse_shape" in keys:
            print(f"  sparse_shape: {meta['sparse_shape']}")

        print_sparse_entries(indices, values, full=full, max_items=max_items)

    # Case 2: scipy-style np.savez sparse matrix: data, indices, indptr, shape
    elif {"data", "indices", "indptr", "shape"}.issubset(keys):
        data = meta["data"]
        indices = meta["indices"]
        indptr = meta["indptr"]
        shape = meta["shape"]

        print("\nDetected CSR/CSC-like sparse format")
        print(f"  data shape:    {data.shape}")
        print(f"  indices shape: {indices.shape}")
        print(f"  indptr shape:  {indptr.shape}")
        print(f"  sparse shape:  {shape}")

        print("\n[data]")
        print_array_values(data, full=full, max_items=max_items)

        print("\n[indices]")
        print_array_values(indices, full=full, max_items=max_items)

        print("\n[indptr]")
        print_array_values(indptr, full=full, max_items=max_items)

    # Case 3: coordinate format: coords + values/data
    elif "coords" in keys and ("values" in keys or "data" in keys):
        coords = meta["coords"]
        values = meta["values"] if "values" in keys else meta["data"]

        print("\nDetected coords + values/data sparse format")
        print(f"  coords shape: {coords.shape}")
        print(f"  values shape: {values.shape}")

        if "shape" in keys:
            print(f"  sparse shape: {meta['shape']}")

        print_sparse_entries(coords, values, full=full, max_items=max_items)

    # Case 4: separate coordinate arrays
    elif {
        "window_indices",
        "channel_indices",
        "bin_indices",
        "feature_indices",
        "values",
    }.issubset(keys):
        window_indices = meta["window_indices"]
        channel_indices = meta["channel_indices"]
        bin_indices = meta["bin_indices"]
        feature_indices = meta["feature_indices"]
        values = meta["values"]

        print("\nDetected separate sparse coordinate arrays")
        print(f"  entries: {values.size}")

        n = values.size if full else min(values.size, max_items)

        print("\nSparse entries:")
        print("  entry | window | channel | bin | feature | value")
        print("  " + "-" * 55)

        for i in range(n):
            print(
                f"  {i:5d} | "
                f"{int(window_indices[i]):6d} | "
                f"{int(channel_indices[i]):7d} | "
                f"{int(bin_indices[i]):3d} | "
                f"{int(feature_indices[i]):7d} | "
                f"{values[i]}"
            )

        if not full and values.size > max_items:
            print(f"  ... showing first {max_items} of {values.size} entries")
            print("  Use --full to print every sparse entry.")

    else:
        print("\nSparse-like keys were found, but the exact sparse layout was not recognized.")
        print("The raw arrays above are still printed.")


def print_sparse_entries(indices, values, full=False, max_items=50):
    """
    Print sparse coordinate/value pairs.

    Supports:
      indices shape (N, D)
      indices shape (D, N)
    """
    if indices.ndim != 2:
        print("\nCannot interpret sparse indices because indices is not 2D.")
        return

    # Normalize indices to shape (N, D)
    if indices.shape[0] == values.size:
        coords = indices
    elif indices.shape[1] == values.size:
        coords = indices.T
    else:
        print("\nCannot match indices with values.")
        print(f"  indices shape: {indices.shape}")
        print(f"  values size:   {values.size}")
        return

    n_entries = values.size
    n_show = n_entries if full else min(n_entries, max_items)

    print("\nSparse entries:")
    header_coords = " ".join([f"idx{d}" for d in range(coords.shape[1])])
    print(f"  entry | {header_coords} | value")
    print("  " + "-" * (20 + 7 * coords.shape[1]))

    for i in range(n_show):
        coord_str = " ".join([f"{int(x):5d}" for x in coords[i]])
        print(f"  {i:5d} | {coord_str} | {values[i]}")

    if not full and n_entries > max_items:
        print(f"  ... showing first {max_items} of {n_entries} entries")
        print("  Use --full to print every sparse entry.")


def print_npy(npy_path, npz_path=None):
    print("=" * 80)
    print(f"Reading NPY/windows file: {npy_path}")
    print("=" * 80)

    if npz_path is None:
        raise ValueError(
            "This file was written by np.memmap, so you must provide --npz "
            "so the reader knows the shape."
        )

    meta = np.load(npz_path, allow_pickle=True)

    n_channels = int(meta["n_channels"])
    n_bins = int(meta["n_temporal_bins"])
    n_features = len(meta["node_features"])

    file_size = Path(npy_path).stat().st_size
    bytes_per_value = np.dtype(np.float32).itemsize
    values_per_window = n_channels * n_bins * n_features

    if file_size % (bytes_per_value * values_per_window) != 0:
        raise ValueError(
            "File size is not divisible by one full window. "
            "This may mean the shape metadata is wrong, or the .npy file is incomplete."
        )

    n_windows = file_size // (bytes_per_value * values_per_window)

    windows = np.memmap(
        npy_path,
        dtype=np.float32,
        mode="r",
        shape=(n_windows, n_channels, n_bins, n_features),
    )

    print("\nArray info:")
    print(f"  shape: {windows.shape}")
    print(f"  dtype: {windows.dtype}")
    print(f"  ndim:  {windows.ndim}")

    print("\nInterpreted as:")
    print(f"  N_windows:  {n_windows}")
    print(f"  N_channels: {n_channels}")
    print(f"  n_bins:     {n_bins}")
    print(f"  n_features: {n_features}")
    print(f"  features:   {meta['node_features']}")

    if n_windows > 0 and n_channels > 0:
        first_window = windows[0]

        nonzero_channels = np.where(np.any(first_window != 0, axis=(1, 2)))[0]
        zero_channels = np.where(~np.any(first_window != 0, axis=(1, 2)))[0]

        channel_start = int(meta["channel_start"]) if "channel_start" in meta else 0

        print("\nChannel activity in first window:")
        print(f"  total local channels:        {n_channels}")
        print(f"  nonzero local channels:      {len(nonzero_channels)}")
        print(f"  zero local channels:         {len(zero_channels)}")

        print("\nFirst 50 zero local channels:")
        print(zero_channels[:50])

        print("\nFirst 50 zero physical channels:")
        print(zero_channels[:50] + channel_start)

        print("\nFirst 50 nonzero local channels:")
        print(nonzero_channels[:50])

        print("\nFirst 50 nonzero physical channels:")
        print(nonzero_channels[:50] + channel_start)

        if len(nonzero_channels) > 0:
            print("\nValues for first 10 nonzero channels only:")
            for ch in nonzero_channels[:10]:
                physical_ch = ch + channel_start
                print("-" * 60)
                print(f"local channel {ch}  physical channel {physical_ch}")
                print(windows[0, ch, :, :])

    meta.close()
    del windows


def main():
    parser = argparse.ArgumentParser(
        description="Read dense window .npy and metadata/sparse .npz files."
    )

    parser.add_argument("--npy", type=str, default=None, help="Path to dense window .npy file")
    parser.add_argument("--npz", type=str, default=None, help="Path to metadata or sparse .npz file")

    parser.add_argument(
        "--full",
        action="store_true",
        help="Print exact full contents of every array in the .npz file.",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=50,
        help="Maximum number of items to print per large array when --full is not used.",
    )

    args = parser.parse_args()

    if args.npy is None and args.npz is None:
        parser.error("Provide --npy, --npz, or both.")

    if args.npy is not None:
        npy_path = Path(args.npy)
        if not npy_path.exists():
            raise FileNotFoundError(f"Cannot find .npy file: {npy_path}")
        print_npy(npy_path, args.npz)

    if args.npz is not None:
        npz_path = Path(args.npz)
        if not npz_path.exists():
            raise FileNotFoundError(f"Cannot find .npz file: {npz_path}")
        print_npz(npz_path, full=args.full, max_items=args.max_items)


if __name__ == "__main__":
    main()