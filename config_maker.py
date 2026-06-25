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

# Parameter values to sweep
WINDOW_SIZES = [20, 50, 70, 100]
STRIDES = [10, 20, 30, 40, 50, 70, 100]
HISTORIES = [1, 2, 6, 10]

# Default training settings to apply to every generated YAML
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_BATCH_SIZE = 64
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


def make_run_name(index: int, window_size: int, stride: int, history: int) -> str:
    return (
        f"{index:04d}_"
        f"win{window_size}_"
        f"stride{stride}_"
        f"hist{history}"
    )


def make_config(
    base_config: dict,
    run_name: str,
    window_size: int,
    stride: int,
    history: int,
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

    # Apply default training settings
    cfg["training"]["lr"] = DEFAULT_LR
    cfg["training"]["weight_decay"] = DEFAULT_WEIGHT_DECAY
    cfg["training"]["batch_size"] = DEFAULT_BATCH_SIZE
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
    OUTPUT_YAML_DIR.mkdir(parents=True, exist_ok=True)

    base_config = load_yaml(BASE_YAML_PATH)

    index = 1
    skipped = 0

    for window_size, stride, history in product(WINDOW_SIZES, STRIDES, HISTORIES):
        if stride > window_size:
            print(
                f"Skipping invalid combination: "
                f"window_size={window_size}, stride={stride}, history={history} "
                f"(stride must be <= window_size)"
            )
            skipped += 1
            continue

        if history < 1:
            print(
                f"Skipping invalid combination: "
                f"window_size={window_size}, stride={stride}, history={history} "
                f"(history must be >= 1 for forecasting)"
            )
            skipped += 1
            continue

        run_name = make_run_name(index, window_size, stride, history)

        cfg = make_config(
            base_config=base_config,
            run_name=run_name,
            window_size=window_size,
            stride=stride,
            history=history,
        )

        output_yaml_path = OUTPUT_YAML_DIR / f"{run_name}.yaml"
        save_yaml(cfg, output_yaml_path)

        print(f"Wrote {output_yaml_path}")

        index += 1

    print()
    print(f"Generated {index - 1} YAML files in {OUTPUT_YAML_DIR}")
    print(f"Skipped {skipped} invalid combinations")


if __name__ == "__main__":
    main()