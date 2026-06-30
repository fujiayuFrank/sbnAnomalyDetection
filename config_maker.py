#!/usr/bin/env python3
"""
Generate GNN sweep YAML configuration files.

This script reads one base YAML file, modifies selected sweep parameters,
and writes one YAML config per parameter combination. Each generated config
gets its own run name, checkpoint directory, final model path, and inference
output path.

Typical usage
-------------
Run the default sweep using the values set in this file:

    python generate_gnn_sweep.py

Run the sweep while forcing stride = window_size for every window size:

    python generate_gnn_sweep.py --same-stride

Input / output settings
-----------------------
BASE_YAML_PATH:
    Path to the base YAML config that will be copied and modified.

OUTPUT_YAML_DIR:
    Directory where generated YAML files will be written.

START_INDEX:
    Starting integer index used in generated YAML filenames and run names.
    For example, START_INDEX = 121 produces names starting with 0121_....

Sweep settings
--------------
WINDOW_SIZES:
    List of data.window_size values to test.

STRIDES:
    List of data.stride values to test.
    This list is ignored when --same-stride is used.

HISTORIES:
    List of model.history values to test.

BATCH_SIZES:
    List of training.batch_size values to test.

LEARNING_RATES:
    List of training.lr values to test.

The sweep uses a full Cartesian product:

    WINDOW_SIZES × STRIDES × HISTORIES × BATCH_SIZES × LEARNING_RATES

When --same-stride is used, the sweep becomes:

    for each window_size:
        stride = window_size

    WINDOW_SIZES × HISTORIES × BATCH_SIZES × LEARNING_RATES

Default training settings
-------------------------
DEFAULT_WEIGHT_DECAY:
    Value written to training.weight_decay for every generated config.

DEFAULT_MAX_EPOCHS:
    Value written to training.max_epochs for every generated config.

Run naming / checkpoint settings
--------------------------------
RUN_PREFIX:
    Name prefix reserved for generated runs.
    Currently this variable is defined for clarity, but the actual run name
    is produced by make_run_name() using index, window size, stride, history,
    batch size, and learning rate.

CHECKPOINT_BASE_DIR:
    Base directory where each generated run's checkpoint folder is placed.

Each generated run gets paths like:

    checkpoints/gnn/<run_name>/gnn_final.pt
    checkpoints/gnn/<run_name>/inference_scores.npz

Validation behavior
-------------------
The script skips invalid combinations where:

    stride > window_size
    history < 1

It also checks that BATCH_SIZES and LEARNING_RATES are not empty.

Command-line flags
------------------
--same-stride:
    Ignore the STRIDES list and automatically set stride = window_size
    for every window size in WINDOW_SIZES.

Example
-------
If:

    WINDOW_SIZES = [500, 1000]
    STRIDES = [100, 250, 500]

Then normal mode tests:

    500 with strides 100, 250, 500
    1000 with strides 100, 250, 500

But with --same-stride, it tests only:

    window_size=500, stride=500
    window_size=1000, stride=1000
"""

from pathlib import Path
from itertools import product
import copy
import argparse
import yaml


# ============================================================
# User settings
# ============================================================

# Base YAML file to vary
BASE_YAML_PATH = Path("configs/gnn.yaml")

# Directory where generated YAML files will be saved
OUTPUT_YAML_DIR = Path("tuning_configs/gnn_sweep")

# Starting index for generated YAML names.
# Example:
#   START_INDEX = 0    -> 0000_win20_stride10_hist1_bs64_lr0p003.yaml
#   START_INDEX = 81   -> 0081_win20_stride10_hist1_bs64_lr0p003.yaml
START_INDEX = 2

# Parameter values to sweep
WINDOW_SIZES = [500, 1000]
STRIDES = [500, 1000]
HISTORIES = [10]

# Training settings.
# These are now iterated as a full Cartesian product:
#   every batch size × every learning rate
BATCH_SIZES = [64]
LEARNING_RATES = [0.003]

DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_MAX_EPOCHS = 100

# Name prefix for generated runs
RUN_PREFIX = "gnn_sweep"

# Project-relative checkpoint base directory
CHECKPOINT_BASE_DIR = Path("checkpoints/gnn")


# ============================================================
# CLI arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GNN sweep YAML configs."
    )

    parser.add_argument(
        "--same-stride",
        "--same_stride",
        "--same",
        action="store_true",
        help=(
            "Ignore STRIDES and automatically set stride = window_size "
            "for every window size in the sweep."
        ),
    )

    return parser.parse_args()


# ============================================================
# YAML generation
# ============================================================

def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Base YAML did not load as a dictionary: {path}")

    return cfg


def save_yaml(config: dict, path: Path) -> None:
    with path.open("w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            default_flow_style=False,
        )


def format_float_for_name(value: float) -> str:
    """
    Convert a float to a filename-safe string.

    Examples:
        0.003 -> 0p003
        0.001 -> 0p001
        1e-4 -> 0p0001
    """
    text = f"{value:g}"
    text = text.replace(".", "p")
    text = text.replace("-", "m")
    return text


def validate_training_sweep_lists() -> None:
    """Validate sweep lists for full Cartesian-product training settings."""
    if len(BATCH_SIZES) == 0:
        raise ValueError("BATCH_SIZES cannot be empty.")

    if len(LEARNING_RATES) == 0:
        raise ValueError("LEARNING_RATES cannot be empty.")


def make_run_name(
    index: int,
    window_size: int,
    stride: int,
    history: int,
    batch_size: int,
    lr: float,
) -> str:
    lr_name = format_float_for_name(lr)

    return (
        f"{index:04d}_"
        f"win{window_size}_"
        f"stride{stride}_"
        f"hist{history}_"
        f"bs{batch_size}_"
        f"lr{lr_name}"
    )


def make_config(
    base_config: dict,
    run_name: str,
    window_size: int,
    stride: int,
    history: int,
    batch_size: int,
    lr: float,
) -> dict:
    cfg = copy.deepcopy(base_config)

    # Make sure required sections exist
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg.setdefault("inference", {})

    # Vary data rolling-window settings
    cfg["data"]["window_size"] = window_size
    cfg["data"]["stride"] = stride

    # Vary model history
    cfg["model"]["history"] = history

    # Apply training settings
    cfg["training"]["lr"] = lr
    cfg["training"]["weight_decay"] = DEFAULT_WEIGHT_DECAY
    cfg["training"]["batch_size"] = batch_size
    cfg["training"]["max_epochs"] = DEFAULT_MAX_EPOCHS

    # Give each run its own checkpoint directory
    checkpoint_dir = CHECKPOINT_BASE_DIR / run_name

    cfg["training"]["checkpoint_dir"] = str(checkpoint_dir)
    cfg["training"]["output_path"] = str(checkpoint_dir / "gnn_final.pt")

    # Match inference paths to this run
    cfg["inference"]["checkpoint_path"] = str(checkpoint_dir / "gnn_final.pt")
    cfg["inference"]["output_path"] = str(checkpoint_dir / "inference_scores.npz")

    return cfg


def main() -> None:
    args = parse_args()

    validate_training_sweep_lists()

    OUTPUT_YAML_DIR.mkdir(parents=True, exist_ok=True)

    base_config = load_yaml(BASE_YAML_PATH)

    index = START_INDEX
    generated = 0
    skipped = 0

    for window_size in WINDOW_SIZES:
        if args.same_stride:
            stride_values = [window_size]
        else:
            stride_values = STRIDES

        for stride, history, batch_size, lr in product(
            stride_values,
            HISTORIES,
            BATCH_SIZES,
            LEARNING_RATES,
        ):
            if stride > window_size:
                print(
                    f"Skipping invalid combination: "
                    f"window_size={window_size}, stride={stride}, history={history}, "
                    f"batch_size={batch_size}, lr={lr} "
                    f"(stride must be <= window_size)"
                )
                skipped += 1
                continue

            if history < 1:
                print(
                    f"Skipping invalid combination: "
                    f"window_size={window_size}, stride={stride}, history={history}, "
                    f"batch_size={batch_size}, lr={lr} "
                    f"(history must be >= 1 for forecasting)"
                )
                skipped += 1
                continue

            run_name = make_run_name(
                index=index,
                window_size=window_size,
                stride=stride,
                history=history,
                batch_size=batch_size,
                lr=lr,
            )

            cfg = make_config(
                base_config=base_config,
                run_name=run_name,
                window_size=window_size,
                stride=stride,
                history=history,
                batch_size=batch_size,
                lr=lr,
            )

            output_yaml_path = OUTPUT_YAML_DIR / f"{run_name}.yaml"
            save_yaml(cfg, output_yaml_path)

            print(f"Wrote {output_yaml_path}")

            index += 1
            generated += 1

    print()
    print(f"Generated {generated} YAML files in {OUTPUT_YAML_DIR}")
    print(f"Skipped {skipped} invalid combinations")
    print(f"First index: {START_INDEX:04d}")
    print(f"Last index: {index - 1:04d}" if generated > 0 else "No YAML files generated")


if __name__ == "__main__":
    main()