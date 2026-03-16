"""Tests for the AnomalyScorer inference wrapper."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sbn_anomaly.infer.inferrer import AnomalyScorer
from sbn_anomaly.models.tpc_model import TPCAutoencoder
from sbn_anomaly.models.pmt_model import PMTAutoencoder
from sbn_anomaly.models.fusion_model import FusionAutoencoder
from sbn_anomaly.models.window_model import WindowAutoencoder


class TestAnomalyScorer:
    def test_tpc_score_shape(self):
        model = TPCAutoencoder(input_dim=32, latent_dim=4)
        scorer = AnomalyScorer(model, model_type="tpc")
        features = np.random.randn(10, 32).astype(np.float32)
        scores = scorer.score(features)
        assert scores.shape == (10,)
        assert (scores >= 0).all()

    def test_pmt_score_shape(self):
        model = PMTAutoencoder(input_dim=16, latent_dim=4)
        scorer = AnomalyScorer(model, model_type="pmt")
        features = np.random.randn(8, 16).astype(np.float32)
        scores = scorer.score(features)
        assert scores.shape == (8,)

    def test_fusion_score_shape(self):
        model = FusionAutoencoder(tpc_input_dim=32, pmt_input_dim=16, latent_dim=8)
        scorer = AnomalyScorer(model, model_type="fusion")
        tpc_feat = np.random.randn(6, 32).astype(np.float32)
        pmt_feat = np.random.randn(6, 16).astype(np.float32)
        scores = scorer.score(tpc_feat, pmt_feat)
        assert scores.shape == (6,)

    def test_window_score_shape(self):
        model = WindowAutoencoder(window_size=32, n_channels=1, latent_dim=8)
        scorer = AnomalyScorer(model, model_type="window")
        # Input shape (B, C, L)
        features = np.random.randn(4, 1, 32).astype(np.float32)
        scores = scorer.score(features)
        assert scores.shape == (4,)

    def test_window_score_auto_channel_dim(self):
        model = WindowAutoencoder(window_size=32, n_channels=1, latent_dim=8)
        scorer = AnomalyScorer(model, model_type="window")
        # (B, L) – missing channel dim, should be added automatically
        features = np.random.randn(4, 32).astype(np.float32)
        scores = scorer.score(features)
        assert scores.shape == (4,)

    def test_is_anomaly_above_threshold(self):
        model = TPCAutoencoder(input_dim=32, latent_dim=4)
        scorer = AnomalyScorer(model, model_type="tpc", threshold=0.0)
        features = np.random.randn(5, 32).astype(np.float32)
        flags = scorer.is_anomaly(features)
        assert flags.dtype == bool
        assert flags.shape == (5,)

    def test_is_anomaly_no_threshold_raises(self):
        model = TPCAutoencoder(input_dim=32, latent_dim=4)
        scorer = AnomalyScorer(model, model_type="tpc")
        with pytest.raises(ValueError, match="threshold"):
            scorer.is_anomaly(np.random.randn(5, 32).astype(np.float32))

    def test_invalid_model_type_raises(self):
        model = TPCAutoencoder(input_dim=32, latent_dim=4)
        with pytest.raises(ValueError, match="model_type"):
            AnomalyScorer(model, model_type="unknown")

    def test_from_checkpoint(self, tmp_path):
        model = TPCAutoencoder(input_dim=32, latent_dim=4)
        ckpt = tmp_path / "model.pt"
        torch.save(model.state_dict(), str(ckpt))

        model2 = TPCAutoencoder(input_dim=32, latent_dim=4)
        scorer = AnomalyScorer.from_checkpoint(ckpt, model2, model_type="tpc")
        features = np.random.randn(3, 32).astype(np.float32)
        scores = scorer.score(features)
        assert scores.shape == (3,)

    def test_from_training_checkpoint(self, tmp_path):
        """AnomalyScorer.from_checkpoint should also accept training checkpoint dicts."""
        model = TPCAutoencoder(input_dim=32, latent_dim=4)
        ckpt = tmp_path / "train_ckpt.pt"
        torch.save(
            {"epoch": 1, "model_state_dict": model.state_dict(), "loss": 0.01},
            str(ckpt),
        )
        model2 = TPCAutoencoder(input_dim=32, latent_dim=4)
        scorer = AnomalyScorer.from_checkpoint(ckpt, model2, model_type="tpc")
        scores = scorer.score(np.random.randn(2, 32).astype(np.float32))
        assert scores.shape == (2,)
