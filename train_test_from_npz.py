#!/usr/bin/env python3

from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict

import re
import numpy as np


# ============================================================
# User settings
# ============================================================

# Directory containing the input .npz files
NPZ_DIR = Path("/exp/sbnd/data/users/jiayufu/checkpoints")

# Output files
TEST_OUTPUT_PATH = NPZ_DIR / "windows_test.npz"
TRAIN_OUTPUT_PATH = NPZ_DIR / "windows_train.npz"

# Whether to search subdirectories recursively
RECURSIVE = True

# Random seed for reproducible good-file selection
RANDOM_SEED = 12345

# Whether to print detailed file-level information
PRINT_FILES = True


# ============================================================
# Channel selection settings
# ============================================================

# Keep only channels in this half-open interval:
#
#   CHANNEL_START <= channel < CHANNEL_END
#
# Example:
#   CHANNEL_START = 9600
#   CHANNEL_END   = 9650
#
# This keeps exactly 50 channels: 9600, 9601, ..., 9649.
CHANNEL_START = 9600
CHANNEL_END = 9650

# If True:
#   original channels 9600, 9601, ..., 9649 become 0, 1, ..., 49
#   and n_channels is saved as CHANNEL_END - CHANNEL_START.
#
# If False:
#   original channel numbers are kept unchanged
#   and n_channels is kept from the original files.
#
# For training, True is usually better.
REINDEX_CHANNELS = True


# --------------------------
# Good/Bad Runs List
# --------------------------
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

# ============================================================
# NPZ key settings
# ============================================================

# Sparse flat arrays. These are filtered by channels_flat and concatenated.
FLAT_KEYS = [
    "channels_flat",
    "integrals_flat",
    "times_flat",
    "wires_flat",
    "planes_flat",
    "tpcs_flat",
]

# Per-window arrays. These should have length equal to number of windows.
PER_WINDOW_KEYS = [
    "evt_run",
    "evt_subrun",
    "evt_num",
    "evt_file_idx",
]

# Special keys
OFFSETS_KEY = "offsets"
N_CHANNELS_KEY = "n_channels"
FILENAMES_KEY = "filenames"


def extract_last_integer(path: Path) -> int:
    """
    Extract the last integer from a filename.

    Example:
      tpc_data_20620.npz -> 20620

    This is used to sort files inside the same run.
    """
    numbers = re.findall(r"\d+", path.stem)

    if not numbers:
        return -1

    return int(numbers[-1])


def get_npz_files() -> List[Path]:
    """Return all .npz files in NPZ_DIR, excluding output files."""
    if not NPZ_DIR.exists():
        raise FileNotFoundError(f"Directory does not exist: {NPZ_DIR}")

    if not NPZ_DIR.is_dir():
        raise NotADirectoryError(f"Not a directory: {NPZ_DIR}")

    if RECURSIVE:
        files = sorted(NPZ_DIR.rglob("*.npz"))
    else:
        files = sorted(NPZ_DIR.glob("*.npz"))

    output_paths = {
        TEST_OUTPUT_PATH.resolve(),
        TRAIN_OUTPUT_PATH.resolve(),
    }

    files = [
        path
        for path in files
        if path.resolve() not in output_paths
    ]

    return files


def read_file_info(npz_path: Path) -> Optional[Dict]:
    """
    Read basic information from one .npz file.

    Returns:
      {
        "path": Path,
        "run": int,
        "n_windows": int,
        "file_number": int,
      }

    Returns None if the file cannot be used.
    """
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "evt_run" not in data.files:
                print(f"[WARNING] Missing evt_run: {npz_path}")
                return None

            evt_run = data["evt_run"]

            if evt_run.size == 0:
                print(f"[WARNING] Empty evt_run: {npz_path}")
                return None

            run = int(evt_run.flat[0])
            n_windows = int(evt_run.shape[0])
            file_number = extract_last_integer(npz_path)

            return {
                "path": npz_path,
                "run": run,
                "n_windows": n_windows,
                "file_number": file_number,
            }

    except Exception as e:
        print(f"[ERROR] Could not read {npz_path}: {e}")
        return None


def sort_infos_by_run_then_file(infos: List[Dict]) -> List[Dict]:
    """
    Sort files so that:

      1. Files from the same run stay together.
      2. Within one run, files are sorted by filename number.

    This prevents interleaving two different runs.
    """
    return sorted(
        infos,
        key=lambda x: (
            x["run"],
            x["file_number"],
            str(x["path"]),
        ),
    )


def select_good_files_for_test(
    good_infos: List[Dict],
    target_windows: int,
    random_seed: int,
) -> List[Dict]:
    """
    Randomly select good-run files until the number of selected good windows
    is close to the target number of bad windows.

    This selects whole .npz files, never partial files.
    """
    rng = np.random.default_rng(random_seed)

    shuffled = list(good_infos)
    rng.shuffle(shuffled)

    selected = []
    selected_windows = 0

    for info in shuffled:
        selected.append(info)
        selected_windows += info["n_windows"]

        if selected_windows >= target_windows:
            break

    return selected


def validate_channel_settings() -> None:
    """Check that the channel range is valid."""
    if CHANNEL_START < 0:
        raise ValueError(f"CHANNEL_START must be >= 0, got {CHANNEL_START}")

    if CHANNEL_END <= CHANNEL_START:
        raise ValueError(
            f"CHANNEL_END must be larger than CHANNEL_START, "
            f"got CHANNEL_START={CHANNEL_START}, CHANNEL_END={CHANNEL_END}"
        )


def validate_same_n_channels(existing_n_channels, new_n_channels, npz_path: Path):
    """Make sure all files have the same original n_channels."""
    if existing_n_channels is None:
        return new_n_channels

    if int(existing_n_channels) != int(new_n_channels):
        raise ValueError(
            f"n_channels mismatch in {npz_path}: "
            f"expected {existing_n_channels}, got {new_n_channels}"
        )

    return existing_n_channels


def filter_one_file_by_channel_range(data, npz_path: Path):
    """
    Filter one loaded npz file by CHANNEL_START <= channels_flat < CHANNEL_END.

    Returns:
      filtered_flat_parts:
        dict of filtered flat arrays for this file

      new_offsets:
        offsets after filtering, starting from 0

      n_kept_hits:
        total number of selected flat entries in this file
    """
    channels = data["channels_flat"]
    offsets = data[OFFSETS_KEY]
    evt_run = data["evt_run"]

    n_windows = int(evt_run.shape[0])

    if len(offsets) != n_windows + 1:
        raise ValueError(
            f"Bad offsets length in {npz_path}: "
            f"len(offsets)={len(offsets)}, n_windows={n_windows}. "
            f"Expected len(offsets)=n_windows+1."
        )

    filtered_flat_parts = {key: [] for key in FLAT_KEYS}
    new_offsets = [0]
    n_kept_hits = 0

    for i_window in range(n_windows):
        start = int(offsets[i_window])
        end = int(offsets[i_window + 1])

        window_channels = channels[start:end]

        mask = (window_channels >= CHANNEL_START) & (window_channels < CHANNEL_END)

        n_selected_this_window = int(np.count_nonzero(mask))
        n_kept_hits += n_selected_this_window

        for key in FLAT_KEYS:
            arr = data[key]
            selected = arr[start:end][mask]

            if key == "channels_flat" and REINDEX_CHANNELS:
                selected = selected - CHANNEL_START

            filtered_flat_parts[key].append(selected)

        new_offsets.append(n_kept_hits)

    for key in FLAT_KEYS:
        if filtered_flat_parts[key]:
            filtered_flat_parts[key] = np.concatenate(filtered_flat_parts[key], axis=0)
        else:
            filtered_flat_parts[key] = np.asarray([], dtype=data[key].dtype)

    new_offsets = np.asarray(new_offsets, dtype=np.int64)

    return filtered_flat_parts, new_offsets, n_kept_hits


def concatenate_npz_files(input_paths: List[Path], output_path: Path) -> None:
    """
    Concatenate sparse window .npz files into one .npz file.

    Important handling:
      - flat arrays are filtered by channel range before concatenation
      - flat arrays are concatenated directly after filtering
      - offsets are rebuilt after filtering
      - evt_run, evt_subrun, evt_num, evt_file_idx are concatenated per window
      - filenames are concatenated, and evt_file_idx is shifted accordingly
      - n_channels is set to CHANNEL_END - CHANNEL_START if REINDEX_CHANNELS=True
    """
    if not input_paths:
        raise ValueError(f"No input files provided for {output_path}")

    flat_parts = {key: [] for key in FLAT_KEYS}
    per_window_parts = {key: [] for key in PER_WINDOW_KEYS}

    combined_offsets = [0]
    combined_filenames = []

    total_flat_length = 0
    filename_offset = 0
    original_n_channels_value = None

    total_windows = 0
    total_kept_hits = 0

    for npz_path in input_paths:
        with np.load(npz_path, allow_pickle=True) as data:
            keys = set(data.files)

            required_keys = set(FLAT_KEYS + PER_WINDOW_KEYS + [OFFSETS_KEY, N_CHANNELS_KEY])
            missing_keys = required_keys - keys

            if missing_keys:
                raise KeyError(
                    f"{npz_path} is missing required keys: {sorted(missing_keys)}"
                )

            evt_run = data["evt_run"]

            if evt_run.size == 0:
                raise ValueError(f"Empty evt_run in {npz_path}")

            n_windows = int(evt_run.shape[0])

            # Check original n_channels consistency.
            n_channels_current = int(np.asarray(data[N_CHANNELS_KEY]).flat[0])
            original_n_channels_value = validate_same_n_channels(
                original_n_channels_value,
                n_channels_current,
                npz_path,
            )

            # Filter flat arrays by channel range.
            filtered_flat_parts, filtered_offsets, n_kept_hits = filter_one_file_by_channel_range(
                data,
                npz_path,
            )

            for key in FLAT_KEYS:
                flat_parts[key].append(filtered_flat_parts[key])

            # Shift and append offsets.
            #
            # filtered_offsets starts from 0 for this individual file.
            # We skip filtered_offsets[0] and append filtered_offsets[1:]
            # shifted by the current total flat length.
            shifted_offsets = filtered_offsets[1:] + total_flat_length
            combined_offsets.extend(shifted_offsets.tolist())

            total_flat_length += n_kept_hits
            total_kept_hits += n_kept_hits

            # Concatenate per-window arrays.
            # These are NOT filtered because windows are kept even if they have zero selected hits.
            for key in PER_WINDOW_KEYS:
                arr = data[key]

                if arr.shape[0] != n_windows:
                    raise ValueError(
                        f"Per-window key {key} has wrong length in {npz_path}: "
                        f"{arr.shape[0]} vs n_windows={n_windows}"
                    )

                if key == "evt_file_idx":
                    # evt_file_idx points into the filenames array.
                    # Since we concatenate filenames, shift indices.
                    arr = arr + filename_offset

                per_window_parts[key].append(arr)

            # Concatenate filenames.
            if FILENAMES_KEY in data.files:
                filenames = data[FILENAMES_KEY]
                combined_filenames.extend(filenames.tolist())
                filename_offset += len(filenames)
            else:
                combined_filenames.append(str(npz_path))
                filename_offset += 1

            total_windows += n_windows

    output = {}

    for key in FLAT_KEYS:
        if flat_parts[key]:
            output[key] = np.concatenate(flat_parts[key], axis=0)
        else:
            output[key] = np.asarray([])

    for key in PER_WINDOW_KEYS:
        output[key] = np.concatenate(per_window_parts[key], axis=0)

    output[OFFSETS_KEY] = np.asarray(combined_offsets, dtype=np.int64)

    if REINDEX_CHANNELS:
        output[N_CHANNELS_KEY] = np.asarray(CHANNEL_END - CHANNEL_START)
    else:
        output[N_CHANNELS_KEY] = np.asarray(original_n_channels_value)

    output[FILENAMES_KEY] = np.asarray(combined_filenames, dtype=object)

    # Final consistency checks
    if output["evt_run"].shape[0] != total_windows:
        raise RuntimeError(
            f"Internal error for {output_path}: "
            f"evt_run length {output['evt_run'].shape[0]} != total_windows {total_windows}"
        )

    if output[OFFSETS_KEY].shape[0] != total_windows + 1:
        raise RuntimeError(
            f"Internal error for {output_path}: "
            f"offsets length {output[OFFSETS_KEY].shape[0]} != total_windows + 1 {total_windows + 1}"
        )

    if output[OFFSETS_KEY][-1] != output["channels_flat"].shape[0]:
        raise RuntimeError(
            f"Internal error for {output_path}: "
            f"last offset {output[OFFSETS_KEY][-1]} != "
            f"channels_flat length {output['channels_flat'].shape[0]}"
        )

    if output["channels_flat"].size > 0:
        min_channel = int(output["channels_flat"].min())
        max_channel = int(output["channels_flat"].max())

        if REINDEX_CHANNELS:
            if min_channel < 0 or max_channel >= CHANNEL_END - CHANNEL_START:
                raise RuntimeError(
                    f"Reindexed channel range error in {output_path}: "
                    f"min={min_channel}, max={max_channel}, "
                    f"expected [0, {CHANNEL_END - CHANNEL_START})"
                )
        else:
            if min_channel < CHANNEL_START or max_channel >= CHANNEL_END:
                raise RuntimeError(
                    f"Channel range error in {output_path}: "
                    f"min={min_channel}, max={max_channel}, "
                    f"expected [{CHANNEL_START}, {CHANNEL_END})"
                )

    np.savez_compressed(output_path, **output)

    print(f"[SAVED] {output_path}")
    print(f"        windows: {total_windows}")
    print(f"        selected flat entries: {total_kept_hits}")
    print(f"        n_channels: {int(output[N_CHANNELS_KEY])}")
    print(f"        channel range: [{CHANNEL_START}, {CHANNEL_END})")
    print(f"        reindex channels: {REINDEX_CHANNELS}")


def main() -> None:
    validate_channel_settings()

    npz_files = get_npz_files()

    all_infos = []
    skipped_files = []

    for npz_path in npz_files:
        info = read_file_info(npz_path)

        if info is None:
            skipped_files.append(npz_path)
        else:
            all_infos.append(info)

    good_infos = []
    bad_infos = []

    for info in all_infos:
        run = info["run"]

        if run in BAD_RUNS:
            bad_infos.append(info)
        else:
            # Default behavior:
            # any run not explicitly listed as bad is treated as good.
            good_infos.append(info)

    total_good_windows = sum(info["n_windows"] for info in good_infos)
    total_bad_windows = sum(info["n_windows"] for info in bad_infos)
    default_good_infos = [
        info
        for info in good_infos
        if info["run"] not in GOOD_RUNS
    ]

    total_default_good_windows = sum(info["n_windows"] for info in default_good_infos)

    # Select good files for test.
    selected_good_infos = select_good_files_for_test(
        good_infos=good_infos,
        target_windows=total_bad_windows,
        random_seed=RANDOM_SEED,
    )

    selected_good_paths = {info["path"] for info in selected_good_infos}
    bad_paths = {info["path"] for info in bad_infos}

    test_paths_set = selected_good_paths | bad_paths

    # Everything not selected for test goes to train.
    train_infos = [
        info
        for info in all_infos
        if info["path"] not in test_paths_set
    ]

    test_infos = bad_infos + selected_good_infos

    # Sort so runs are never interleaved, and files inside each run are ordered.
    test_infos = sort_infos_by_run_then_file(test_infos)
    train_infos = sort_infos_by_run_then_file(train_infos)

    test_paths = [info["path"] for info in test_infos]
    train_paths = [info["path"] for info in train_infos]

    selected_good_windows = sum(info["n_windows"] for info in selected_good_infos)
    test_bad_windows = sum(info["n_windows"] for info in bad_infos)
    test_total_windows = sum(info["n_windows"] for info in test_infos)
    train_total_windows = sum(info["n_windows"] for info in train_infos)

    # Safety check: no overlap between train and test
    overlap = set(test_paths) & set(train_paths)
    if overlap:
        raise RuntimeError(
            "Some files are in both train and test. This should never happen:\n"
            + "\n".join(str(p) for p in sorted(overlap))
        )

    # Print summary before writing
    print("=" * 80)
    print("Input summary")
    print("=" * 80)
    print(f"Directory: {NPZ_DIR}")
    print(f"Recursive search: {RECURSIVE}")
    print(f"Total usable .npz files: {len(all_infos)}")
    print(f"Skipped/error files: {len(skipped_files)}")
    print("-" * 80)
    print(f"Good-run files: {len(good_infos)}")
    print(f"Good-run windows: {total_good_windows}")
    print(f"Bad-run files: {len(bad_infos)}")
    print(f"Bad-run windows: {total_bad_windows}")
    print(f"Explicit good-run files: {sum(1 for info in good_infos if info['run'] in GOOD_RUNS)}")
    print(f"Default good-run (unknown) files: {len(default_good_infos)}")
    print(f"Default good-run (unknown) windows: {total_default_good_windows}")
    print("-" * 80)
    print("Channel selection")
    print("-" * 80)
    print(f"Channel start: {CHANNEL_START}")
    print(f"Channel end: {CHANNEL_END}")
    print(f"Number of selected channels: {CHANNEL_END - CHANNEL_START}")
    print(f"Reindex channels: {REINDEX_CHANNELS}")
    print("-" * 80)
    print("Test selection")
    print("-" * 80)
    print(f"Bad files selected for test: {len(bad_infos)}")
    print(f"Bad windows selected for test: {test_bad_windows}")
    print(f"Good files selected for test: {len(selected_good_infos)}")
    print(f"Good windows selected for test: {selected_good_windows}")
    print(f"Total test files: {len(test_infos)}")
    print(f"Total test windows: {test_total_windows}")
    print("-" * 80)
    print("Train selection")
    print("-" * 80)
    print(f"Total train files: {len(train_infos)}")
    print(f"Total train windows: {train_total_windows}")
    print("=" * 80)

    if PRINT_FILES:
        print("\nTest files:")
        for info in test_infos:
            label = "BAD" if info["run"] in BAD_RUNS else "GOOD"
            print(
                f"  [{label}] run={info['run']} "
                f"windows={info['n_windows']} "
                f"file_number={info['file_number']} "
                f"path={info['path']}"
            )

        print("\nTrain files:")
        for info in train_infos:
            if info["run"] in BAD_RUNS:
                label = "BAD"
            elif info["run"] in GOOD_RUNS:
                label = "GOOD"
            else:
                label = "UNKNOWN"

            print(
                f"  [{label}] run={info['run']} "
                f"windows={info['n_windows']} "
                f"file_number={info['file_number']} "
                f"path={info['path']}"
            )

    # Write output files
    print("\n" + "=" * 80)
    print("Writing output files")
    print("=" * 80)

    concatenate_npz_files(test_paths, TEST_OUTPUT_PATH)
    concatenate_npz_files(train_paths, TRAIN_OUTPUT_PATH)

    print("=" * 80)
    print("Done")
    print("=" * 80)
    print(f"Test output:  {TEST_OUTPUT_PATH}")
    print(f"Train output: {TRAIN_OUTPUT_PATH}")


if __name__ == "__main__":
    main()