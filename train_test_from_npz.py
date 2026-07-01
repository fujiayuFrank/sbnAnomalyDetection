#!/usr/bin/env python3

from pathlib import Path
from typing import Optional, Dict, List

import re
import numpy as np


# ============================================================
# User settings
# ============================================================

# Directory containing the input .npz files
NPZ_DIR = Path("/exp/sbnd/data/users/micarrig/DQM/tpc_data")

# Output files
OUTPUT_PARENT_PATH = Path("/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection")
TEST_OUTPUT_PATH = OUTPUT_PARENT_PATH / "windows_test.npz"
TRAIN_OUTPUT_PATH = OUTPUT_PARENT_PATH / "windows_train.npz"

# Whether to search subdirectories recursively
RECURSIVE = True

# Random seed for reproducible good-file selection
RANDOM_SEED = 12345

# Whether to print detailed file-level information
PRINT_FILES = True

# If True, use np.savez_compressed.
# If False, use np.savez, which is usually much faster but creates larger files.
COMPRESS_OUTPUT = False


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


# If None:
#   use all remaining training files, same behavior as before.
#
# If an integer:
#   write exactly this many windows/events into windows_train.npz.
#   The script will use train files in sorted order and, if needed, take
#   only the first part of the last file to hit this exact window count.
MAX_TRAIN_WINDOW_NUM = None
# MAX_TRAIN_WINDOW_NUM = 100000


# If None:
#   use the original behavior:
#   select all bad-run files for test, randomly select enough good-run files
#   to approximately match the bad-window count, and put everything else
#   into the training set.
#
# If an integer:
#   first reserve good-run files for training until at least this many good
#   windows/events are available for training. Then the test set is built
#   only from the remaining good-run files and the bad-run files.
#
#   In this mode, if there are not enough remaining good windows to match all
#   bad windows, or not enough bad windows to match all remaining good windows,
#   the script uses the lower available window count as the target for each
#   test class.
#
# Note:
#   File selection is whole-file based, so final selected counts can be a bit
#   above the target if the last selected file crosses the target.
MIN_TRAIN_WINDOW_NUM = 100000
# MIN_TRAIN_WINDOW_NUM = 100000


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

    Note:
      n_windows is the number of windows/events in this sparse file before
      channel filtering. Zero-hit windows after channel filtering are still kept.
    """
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "evt_run" not in data.files:
                print(f"[WARNING] Missing evt_run: {npz_path}", flush=True)
                return None

            evt_run = data["evt_run"]

            if evt_run.size == 0:
                print(f"[WARNING] Empty evt_run: {npz_path}", flush=True)
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
        print(f"[ERROR] Could not read {npz_path}: {e}", flush=True)
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

    The selection uses the original number of windows/events before channel
    filtering. Since zero-hit windows are still kept, this is also the written
    window count after filtering.
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


def select_files_until_min_windows(
    infos: List[Dict],
    min_windows: int,
) -> List[Dict]:
    """
    Select whole files in sorted order until at least min_windows are reached.

    This is used to reserve good-run files for training before constructing
    the test set when MIN_TRAIN_WINDOW_NUM is enabled.

    This selects whole .npz files, never partial files.
    """
    min_windows = int(min_windows)

    if min_windows <= 0:
        raise ValueError(f"min_windows must be positive or None, got {min_windows}")

    selected = []
    selected_windows = 0

    for info in sort_infos_by_run_then_file(infos):
        selected.append(info)
        selected_windows += info["n_windows"]

        if selected_windows >= min_windows:
            break

    if selected_windows < min_windows:
        raise ValueError(
            f"Could not reserve the requested minimum training windows. "
            f"Requested MIN_TRAIN_WINDOW_NUM={min_windows}, "
            f"but only {selected_windows} good-run windows are available."
        )

    return selected


def select_files_for_window_target(
    infos: List[Dict],
    target_windows: int,
    random_seed: int,
) -> List[Dict]:
    """
    Randomly select whole files until reaching target_windows.

    This is used for class-balanced test selection when MIN_TRAIN_WINDOW_NUM
    is enabled. Because files are selected whole, the final selected window
    count may be slightly larger than target_windows.
    """
    target_windows = int(target_windows)

    if target_windows <= 0:
        return []

    rng = np.random.default_rng(random_seed)

    shuffled = list(infos)
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


def filter_one_file_by_channel_range(data, npz_path: Path):
    """
    Fast vectorized filtering by CHANNEL_START <= channels_flat < CHANNEL_END.

    This version keeps ALL windows/events, even if a window has zero selected hits.

    Functionally:
      - keeps every original window/event
      - removes flat hit entries outside the channel range
      - rebuilds offsets correctly
      - optionally reindexes selected channels to 0...(CHANNEL_END-CHANNEL_START-1)

    Returns:
      filtered_flat_parts:
        dict of filtered flat arrays for this file

      new_offsets:
        offsets after filtering, with length n_windows + 1

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

    # One global hit-level mask for all flat entries.
    hit_mask = (channels >= CHANNEL_START) & (channels < CHANNEL_END)

    filtered_flat_parts = {}

    for key in FLAT_KEYS:
        selected = data[key][hit_mask]

        if key == "channels_flat" and REINDEX_CHANNELS:
            selected = selected - CHANNEL_START

        filtered_flat_parts[key] = selected

    # Rebuild offsets after filtering.
    #
    # selected_cumsum[j] gives the number of kept hits before flat index j.
    # Therefore selected_cumsum[offsets] gives the new offsets for every
    # original window, including windows with zero selected hits.
    selected_cumsum = np.empty(hit_mask.shape[0] + 1, dtype=np.int64)
    selected_cumsum[0] = 0
    selected_cumsum[1:] = np.cumsum(hit_mask, dtype=np.int64)

    new_offsets = selected_cumsum[offsets]
    n_kept_hits = int(new_offsets[-1])

    return filtered_flat_parts, new_offsets, n_kept_hits


def concatenate_npz_files(
    input_paths: List[Path],
    output_path: Path,
    max_windows: Optional[int] = None,
) -> None:
    """
    Concatenate sparse window/event .npz files into one .npz file.

    Important handling:
      - flat arrays are filtered by channel range before concatenation
      - windows/events with zero selected hits are still kept
      - per-window arrays are NOT filtered by channel selection
      - offsets are rebuilt after hit filtering
      - evt_file_idx is shifted according to concatenated filenames
      - n_channels is set to CHANNEL_END - CHANNEL_START if REINDEX_CHANNELS=True

    max_windows:
      If None, write all windows/events from input_paths.
      If an integer, write exactly up to max_windows windows/events.
      If the limit falls inside a file, only the first needed windows/events
      from that file are written.
    """
    if not input_paths:
        raise ValueError(f"No input files provided for {output_path}")

    if max_windows is not None:
        max_windows = int(max_windows)
        if max_windows <= 0:
            raise ValueError(f"max_windows must be positive or None, got {max_windows}")

    flat_parts = {key: [] for key in FLAT_KEYS}
    per_window_parts = {key: [] for key in PER_WINDOW_KEYS}

    combined_offsets = [0]
    combined_filenames = []

    total_flat_length = 0
    filename_offset = 0
    original_n_channels_value = None

    total_windows = 0
    total_kept_hits = 0

    reached_window_limit = False

    for i_file, npz_path in enumerate(input_paths, start=1):
        if max_windows is not None and total_windows >= max_windows:
            reached_window_limit = True
            break

        print(
            f"[{output_path.name}] reading/filtering "
            f"{i_file}/{len(input_paths)}: {npz_path}",
            flush=True,
        )

        with np.load(npz_path, allow_pickle=True) as data:
            keys = set(data.files)

            required_keys = set(
                FLAT_KEYS
                + PER_WINDOW_KEYS
                + [OFFSETS_KEY, N_CHANNELS_KEY]
            )
            missing_keys = required_keys - keys

            if missing_keys:
                raise KeyError(
                    f"{npz_path} is missing required keys: {sorted(missing_keys)}"
                )

            evt_run = data["evt_run"]

            if evt_run.size == 0:
                raise ValueError(f"Empty evt_run in {npz_path}")

            n_windows_in_file = int(evt_run.shape[0])

            # Decide how many windows to take from this file.
            # If max_windows is None, take the whole file.
            # If max_windows is set, take only what is needed to reach the limit.
            if max_windows is None:
                n_windows_to_take = n_windows_in_file
            else:
                remaining_windows = max_windows - total_windows
                n_windows_to_take = min(n_windows_in_file, remaining_windows)

            if n_windows_to_take <= 0:
                reached_window_limit = True
                break

            # Original files may have slightly different n_channels if some
            # channels were not recorded. Do not require consistency.
            n_channels_current = int(np.asarray(data[N_CHANNELS_KEY]).flat[0])

            if original_n_channels_value is None:
                original_n_channels_value = n_channels_current

            if CHANNEL_START >= n_channels_current:
                print(
                    f"[WARNING] CHANNEL_START={CHANNEL_START} is outside "
                    f"n_channels={n_channels_current} for {npz_path}. "
                    f"This file may contribute no hits in the selected range.",
                    flush=True,
                )

            # Filter flat arrays by channel range.
            # This keeps all windows/events, even if some windows have zero selected hits.
            filtered_flat_parts, filtered_offsets, n_kept_hits_full_file = filter_one_file_by_channel_range(
                data,
                npz_path,
            )

            # If taking only part of this file, trim the filtered result to the
            # first n_windows_to_take windows.
            #
            # filtered_offsets has length n_windows_in_file + 1.
            # filtered_offsets[n_windows_to_take] gives the number of selected
            # flat entries belonging to the first n_windows_to_take windows.
            local_flat_length_to_take = int(filtered_offsets[n_windows_to_take])

            for key in FLAT_KEYS:
                flat_parts[key].append(filtered_flat_parts[key][:local_flat_length_to_take])

            filtered_offsets_to_take = filtered_offsets[: n_windows_to_take + 1]
            n_kept_hits = local_flat_length_to_take

            # Shift and append offsets.
            #
            # filtered_offsets_to_take starts from 0 for this individual file slice.
            # We skip filtered_offsets_to_take[0] and append the rest shifted by
            # the current total flat length.
            shifted_offsets = filtered_offsets_to_take[1:] + total_flat_length
            combined_offsets.extend(shifted_offsets.tolist())

            total_flat_length += n_kept_hits
            total_kept_hits += n_kept_hits

            # Concatenate per-window arrays.
            # These are sliced only if max_windows cuts through this file.
            for key in PER_WINDOW_KEYS:
                arr = data[key]

                if arr.shape[0] != n_windows_in_file:
                    raise ValueError(
                        f"Per-window key {key} has wrong length in {npz_path}: "
                        f"{arr.shape[0]} vs n_windows={n_windows_in_file}"
                    )

                arr = arr[:n_windows_to_take]

                if key == "evt_file_idx":
                    # evt_file_idx points into the filenames array.
                    # Since we concatenate filenames, shift indices.
                    arr = arr + filename_offset

                per_window_parts[key].append(arr)

            # Concatenate filenames.
            #
            # We keep the full filenames array from any file that contributes
            # at least one window. This is safe because evt_file_idx values still
            # point into this combined filenames array.
            if FILENAMES_KEY in data.files:
                filenames = data[FILENAMES_KEY]
                combined_filenames.extend(filenames.tolist())
                filename_offset += len(filenames)
            else:
                combined_filenames.append(str(npz_path))
                filename_offset += 1

            total_windows += n_windows_to_take

            if max_windows is not None and total_windows >= max_windows:
                reached_window_limit = True
                break

    if total_windows == 0:
        raise ValueError(
            f"No windows/events were written for {output_path}. "
            f"Check input_paths and evt_run arrays."
        )

    if max_windows is not None and total_windows != max_windows:
        raise ValueError(
            f"Could not write requested max_windows={max_windows} for {output_path}. "
            f"Only {total_windows} windows/events were available."
        )

    print(f"[{output_path.name}] concatenating arrays...", flush=True)

    output = {}

    for key in FLAT_KEYS:
        if flat_parts[key]:
            output[key] = np.concatenate(flat_parts[key], axis=0)
        else:
            output[key] = np.asarray([])

    for key in PER_WINDOW_KEYS:
        if per_window_parts[key]:
            output[key] = np.concatenate(per_window_parts[key], axis=0)
        else:
            output[key] = np.asarray([])

    output[OFFSETS_KEY] = np.asarray(combined_offsets, dtype=np.int64)

    if REINDEX_CHANNELS:
        output[N_CHANNELS_KEY] = np.asarray(CHANNEL_END - CHANNEL_START)
    else:
        output[N_CHANNELS_KEY] = np.asarray(original_n_channels_value)

    output[FILENAMES_KEY] = np.asarray([str(f) for f in combined_filenames], dtype=str)

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

    # Zero-hit windows/events are allowed.
    # Therefore, do NOT require np.diff(offsets) > 0.

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

    print(f"[{output_path.name}] saving to disk...", flush=True)

    if COMPRESS_OUTPUT:
        np.savez_compressed(output_path, **output)
    else:
        np.savez(output_path, **output)

    print(f"[SAVED] {output_path}", flush=True)
    print(f"        windows written: {total_windows}", flush=True)
    print(f"        selected flat entries: {total_kept_hits}", flush=True)
    print(f"        n_channels: {int(output[N_CHANNELS_KEY])}", flush=True)
    print(f"        channel range: [{CHANNEL_START}, {CHANNEL_END})", flush=True)
    print(f"        reindex channels: {REINDEX_CHANNELS}", flush=True)
    print(f"        compressed output: {COMPRESS_OUTPUT}", flush=True)
    print(f"        max_windows limit: {max_windows}", flush=True)


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

    explicit_good_infos = [
        info
        for info in good_infos
        if info["run"] in GOOD_RUNS
    ]

    default_good_infos = [
        info
        for info in good_infos
        if info["run"] not in GOOD_RUNS
    ]

    total_default_good_windows = sum(info["n_windows"] for info in default_good_infos)

    if MIN_TRAIN_WINDOW_NUM is None:
        # Original behavior:
        #   - all bad-run files go to test
        #   - randomly select enough good-run files to approximately match
        #     the total bad-window count
        #   - everything else goes to train
        selected_bad_infos = list(bad_infos)

        selected_good_infos = select_good_files_for_test(
            good_infos=good_infos,
            target_windows=total_bad_windows,
            random_seed=RANDOM_SEED,
        )

        selected_good_paths = {info["path"] for info in selected_good_infos}
        selected_bad_paths = {info["path"] for info in selected_bad_infos}

        test_paths_set = selected_good_paths | selected_bad_paths

        # Everything not selected for test goes to train.
        train_infos = [
            info
            for info in all_infos
            if info["path"] not in test_paths_set
        ]

        test_infos = selected_bad_infos + selected_good_infos

        remaining_good_windows_after_train_reserve = None
        test_target_windows_per_class = None

    else:
        # New behavior:
        #   1. Reserve good-run files for training until MIN_TRAIN_WINDOW_NUM
        #      is reached.
        #   2. Build the test set only from the remaining good-run files and
        #      bad-run files.
        #   3. Use the lower of remaining-good windows and available-bad
        #      windows as the per-class test target.
        if MAX_TRAIN_WINDOW_NUM is not None and MAX_TRAIN_WINDOW_NUM < MIN_TRAIN_WINDOW_NUM:
            raise ValueError(
                f"MAX_TRAIN_WINDOW_NUM={MAX_TRAIN_WINDOW_NUM} is smaller than "
                f"MIN_TRAIN_WINDOW_NUM={MIN_TRAIN_WINDOW_NUM}. "
                f"This would cap the written training set below the requested minimum."
            )

        reserved_train_good_infos = select_files_until_min_windows(
            infos=good_infos,
            min_windows=MIN_TRAIN_WINDOW_NUM,
        )

        reserved_train_good_paths = {
            info["path"]
            for info in reserved_train_good_infos
        }

        remaining_good_infos = [
            info
            for info in good_infos
            if info["path"] not in reserved_train_good_paths
        ]

        remaining_good_windows_after_train_reserve = sum(
            info["n_windows"]
            for info in remaining_good_infos
        )
        available_bad_windows = sum(info["n_windows"] for info in bad_infos)

        test_target_windows_per_class = min(
            remaining_good_windows_after_train_reserve,
            available_bad_windows,
        )

        if test_target_windows_per_class <= 0:
            raise ValueError(
                "Could not build a balanced test set after reserving the minimum "
                "training windows. Either no good windows remain for test, or no "
                "bad windows are available."
            )

        selected_good_infos = select_files_for_window_target(
            infos=remaining_good_infos,
            target_windows=test_target_windows_per_class,
            random_seed=RANDOM_SEED,
        )

        selected_bad_infos = select_files_for_window_target(
            infos=bad_infos,
            target_windows=test_target_windows_per_class,
            random_seed=RANDOM_SEED + 1,
        )

        selected_good_paths = {info["path"] for info in selected_good_infos}
        selected_bad_paths = {info["path"] for info in selected_bad_infos}

        # In the new mode, train contains only good-run files.
        # Bad-run files not selected for test are intentionally unused.
        train_infos = [
            info
            for info in good_infos
            if info["path"] not in selected_good_paths
        ]

        test_infos = selected_bad_infos + selected_good_infos

    # Sort so runs are never interleaved, and files inside each run are ordered.
    test_infos = sort_infos_by_run_then_file(test_infos)
    train_infos = sort_infos_by_run_then_file(train_infos)

    test_paths = [info["path"] for info in test_infos]
    train_paths = [info["path"] for info in train_infos]

    selected_good_windows = sum(info["n_windows"] for info in selected_good_infos)
    test_bad_windows = sum(info["n_windows"] for info in selected_bad_infos)
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
    print(f"Good-run windows before channel filtering: {total_good_windows}")
    print(f"Bad-run files: {len(bad_infos)}")
    print(f"Bad-run windows before channel filtering: {total_bad_windows}")
    print(f"Explicit good-run files: {len(explicit_good_infos)}")
    print(f"Default good-run files: {len(default_good_infos)}")
    print(f"Default good-run windows before channel filtering: {total_default_good_windows}")
    print("-" * 80)
    print("Channel selection")
    print("-" * 80)
    print(f"Channel start: {CHANNEL_START}")
    print(f"Channel end: {CHANNEL_END}")
    print(f"Number of selected channels: {CHANNEL_END - CHANNEL_START}")
    print(f"Reindex channels: {REINDEX_CHANNELS}")
    print("-" * 80)
    print("Train/test sizing settings")
    print("-" * 80)
    print(f"Minimum training windows setting: {MIN_TRAIN_WINDOW_NUM}")
    print(f"Max training windows output cap: {MAX_TRAIN_WINDOW_NUM}")
    if test_target_windows_per_class is not None:
        print(f"Remaining good windows after training reserve: {remaining_good_windows_after_train_reserve}")
        print(f"Test target windows per class: {test_target_windows_per_class}")
    print("-" * 80)
    print("Test selection before channel filtering")
    print("-" * 80)
    print(f"Bad files selected for test: {len(selected_bad_infos)}")
    print(f"Bad windows selected for test before filtering: {test_bad_windows}")
    print(f"Good files selected for test: {len(selected_good_infos)}")
    print(f"Good windows selected for test before filtering: {selected_good_windows}")
    print(f"Total test files: {len(test_infos)}")
    print(f"Total test windows before filtering: {test_total_windows}")
    print("-" * 80)
    print("Train selection before channel filtering")
    print("-" * 80)
    print(f"Total train files: {len(train_infos)}")
    print(f"Total train windows before filtering: {train_total_windows}")
    print("=" * 80)

    if PRINT_FILES:
        print("\nTest files:")
        for info in test_infos:
            if info["run"] in BAD_RUNS:
                label = "BAD"
            elif info["run"] in GOOD_RUNS:
                label = "GOOD"
            else:
                label = "GOOD_DEFAULT"

            print(
                f"  [{label}] run={info['run']} "
                f"windows_before_filtering={info['n_windows']} "
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
                label = "GOOD_DEFAULT"

            print(
                f"  [{label}] run={info['run']} "
                f"windows_before_filtering={info['n_windows']} "
                f"file_number={info['file_number']} "
                f"path={info['path']}"
            )

    # Write output files
    print("\n" + "=" * 80)
    print("Writing output files")
    print("=" * 80)

    concatenate_npz_files(test_paths, TEST_OUTPUT_PATH)
    concatenate_npz_files(train_paths, TRAIN_OUTPUT_PATH, max_windows=MAX_TRAIN_WINDOW_NUM)

    print("=" * 80)
    print("Done")
    print("=" * 80)
    print(f"Test output:  {TEST_OUTPUT_PATH}")
    print(f"Train output: {TRAIN_OUTPUT_PATH}")


if __name__ == "__main__":
    main()