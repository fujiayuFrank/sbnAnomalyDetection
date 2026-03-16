"""Inference CLI entry-point.

Usage::

    sbn-infer --config configs/tpc.yaml --input /data/run.root --output scores.npy
    python -m sbn_anomaly.infer.cli --config configs/fusion.yaml \\
        --tpc-input tpc_features.npy --pmt-input pmt_features.npy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

from sbn_anomaly.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run inference."""
    parser = argparse.ArgumentParser(description="Run SBN anomaly detection inference.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to .npy feature file (tpc/pmt/window modes).",
    )
    parser.add_argument(
        "--tpc-input",
        type=str,
        default=None,
        help="Path to TPC .npy feature file (fusion mode).",
    )
    parser.add_argument(
        "--pmt-input",
        type=str,
        default=None,
        help="Path to PMT .npy feature file (fusion mode).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scores.npy",
        help="Path to save anomaly scores (.npy).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 1

    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)

    model_type = cfg.get("model_type", "").lower()
    infer_cfg = cfg.get("inference", {})
    checkpoint = infer_cfg.get("checkpoint_path")
    if not checkpoint:
        logger.error("inference.checkpoint_path not set in config.")
        return 1

    scorer = _build_scorer(cfg, model_type, checkpoint)

    if model_type == "fusion":
        tpc_path = args.tpc_input or infer_cfg.get("tpc_input_path")
        pmt_path = args.pmt_input or infer_cfg.get("pmt_input_path")
        if not tpc_path or not pmt_path:
            logger.error("Fusion mode requires --tpc-input and --pmt-input.")
            return 1
        tpc_feat = np.load(tpc_path)
        pmt_feat = np.load(pmt_path)
        scores = scorer.score(tpc_feat, pmt_feat)
    else:
        input_path = args.input or infer_cfg.get("input_path")
        if not input_path:
            logger.error("Provide --input or set inference.input_path in config.")
            return 1
        features = np.load(input_path)
        scores = scorer.score(features)

    np.save(args.output, scores)
    logger.info("Saved %d anomaly scores to %s", len(scores), args.output)
    return 0


def _build_scorer(cfg: dict, model_type: str, checkpoint: str):
    from sbn_anomaly.infer.inferrer import AnomalyScorer

    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("inference", {})
    threshold = infer_cfg.get("threshold")

    if model_type == "tpc":
        from sbn_anomaly.models.tpc_model import TPCAutoencoder

        model = TPCAutoencoder(
            input_dim=model_cfg.get("input_dim", 256),
            latent_dim=model_cfg.get("latent_dim", 32),
        )
    elif model_type == "pmt":
        from sbn_anomaly.models.pmt_model import PMTAutoencoder

        model = PMTAutoencoder(
            input_dim=model_cfg.get("input_dim", 128),
            latent_dim=model_cfg.get("latent_dim", 16),
        )
    elif model_type == "fusion":
        from sbn_anomaly.models.fusion_model import FusionAutoencoder

        model = FusionAutoencoder(
            tpc_input_dim=model_cfg.get("tpc_input_dim", 256),
            pmt_input_dim=model_cfg.get("pmt_input_dim", 128),
            latent_dim=model_cfg.get("latent_dim", 32),
        )
    elif model_type == "window":
        from sbn_anomaly.models.window_model import WindowAutoencoder

        model = WindowAutoencoder(
            window_size=model_cfg.get("window_size", 256),
            n_channels=model_cfg.get("n_channels", 1),
            latent_dim=model_cfg.get("latent_dim", 64),
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'.")

    return AnomalyScorer.from_checkpoint(
        checkpoint_path=checkpoint,
        model=model,
        model_type=model_type,
        threshold=threshold,
    )


if __name__ == "__main__":
    sys.exit(main())
