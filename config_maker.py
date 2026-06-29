#!/usr/bin/env python3

from pathlib import Path
from itertools import product
import copy
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
START_INDEX = 81

# Parameter values to sweep
WINDOW_SIZES = [200, 500, 700, 1000]
STRIDES = [200, 500, 700, 1000]
HISTORIES = [1, 10, 15, 20]

# Paired training settings.
# These are iterated pair by pair:
#   BATCH_SIZES[0] uses LEARNING_RATES[0]
#   BATCH_SIZES[1] uses LEARNING_RATES[1]
#   etc.
BATCH_SIZES = [64]
LEARNING_RATES = [0.003]

DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_MAX_EPOCHS = 100

# Name prefix for generated runs
RUN_PREFIX = "gnn_sweep"

# Project-relative checkpoint base directory
CHECKPOINT_BASE_DIR = Path("checkpoints/gnn")


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
    if len(BATCH_SIZES) != len(LEARNING_RATES):
        raise ValueError(
            "BATCH_SIZES and LEARNING_RATES must have the same length because "
            "they are iterated pair by pair.\n"
            f"len(BATCH_SIZES) = {len(BATCH_SIZES)}\n"
            f"len(LEARNING_RATES) = {len(LEARNING_RATES)}"
        )

    if len(BATCH_SIZES) == 0:
        raise ValueError("BATCH_SIZES and LEARNING_RATES cannot be empty.")


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
    validate_training_sweep_lists()

    OUTPUT_YAML_DIR.mkdir(parents=True, exist_ok=True)

    base_config = load_yaml(BASE_YAML_PATH)

    index = START_INDEX
    generated = 0
    skipped = 0

    training_pairs = list(zip(BATCH_SIZES, LEARNING_RATES))

    for window_size, stride, history, training_pair in product(
        WINDOW_SIZES,
        STRIDES,
        HISTORIES,
        training_pairs,
    ):
        batch_size, lr = training_pair

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