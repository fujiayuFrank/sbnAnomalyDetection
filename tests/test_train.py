"""Tests for the BaseTrainer training loop."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from sbn_anomaly.data.dataset import TPCDataset, PMTDataset, FusionDataset
from sbn_anomaly.models.tpc_model import TPCAutoencoder
from sbn_anomaly.models.pmt_model import PMTAutoencoder
from sbn_anomaly.models.fusion_model import FusionAutoencoder
from sbn_anomaly.train.tpc_trainer import TPCTrainer
from sbn_anomaly.train.pmt_trainer import PMTTrainer
from sbn_anomaly.train.fusion_trainer import FusionTrainer
from sbn_anomaly.train.window_trainer import WindowTrainer
from sbn_anomaly.models.window_model import WindowAutoencoder
from sbn_anomaly.data.dataset import WindowDataset


class TestTPCTrainer:
    def test_train_one_epoch_returns_loss(self, tmp_path):
        feat = np.random.randn(64, 32).astype(np.float32)
        dataset = TPCDataset(feat)
        loader = DataLoader(dataset, batch_size=16)
        model = TPCAutoencoder(input_dim=32, latent_dim=4, hidden_dims=(16,))
        trainer = TPCTrainer(model=model, max_epochs=1, checkpoint_dir=str(tmp_path))
        losses = trainer.train(loader)
        assert len(losses) == 1
        assert losses[0] > 0

    def test_save_and_load(self, tmp_path):
        model = TPCAutoencoder(input_dim=32, latent_dim=4)
        trainer = TPCTrainer(model=model, max_epochs=1)
        ckpt = str(tmp_path / "model.pt")
        trainer.save(ckpt)
        trainer.load(ckpt)


class TestPMTTrainer:
    def test_train_one_epoch(self):
        feat = np.random.randn(32, 16).astype(np.float32)
        dataset = PMTDataset(feat)
        loader = DataLoader(dataset, batch_size=8)
        model = PMTAutoencoder(input_dim=16, latent_dim=4, hidden_dims=(8,))
        trainer = PMTTrainer(model=model, max_epochs=1)
        losses = trainer.train(loader)
        assert len(losses) == 1


class TestFusionTrainer:
    def test_train_one_epoch(self):
        tpc_feat = np.random.randn(32, 32).astype(np.float32)
        pmt_feat = np.random.randn(32, 16).astype(np.float32)
        dataset = FusionDataset(tpc_feat, pmt_feat)
        loader = DataLoader(dataset, batch_size=8)
        model = FusionAutoencoder(tpc_input_dim=32, pmt_input_dim=16, latent_dim=8)
        trainer = FusionTrainer(model=model, max_epochs=1)
        losses = trainer.train(loader)
        assert len(losses) == 1


class TestWindowTrainer:
    def test_train_one_epoch(self):
        signal = np.random.randn(512).astype(np.float32)
        dataset = WindowDataset(signal, window_size=32, stride=16)
        loader = DataLoader(dataset, batch_size=16)
        model = WindowAutoencoder(window_size=32, n_channels=1, latent_dim=8)
        trainer = WindowTrainer(model=model, max_epochs=1)
        losses = trainer.train(loader)
        assert len(losses) == 1
