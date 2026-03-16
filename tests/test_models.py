"""Tests for TPC, PMT, Fusion, and Window autoencoder models."""

from __future__ import annotations

import pytest
import torch

from sbn_anomaly.models.tpc_model import TPCAutoencoder
from sbn_anomaly.models.pmt_model import PMTAutoencoder
from sbn_anomaly.models.fusion_model import FusionAutoencoder
from sbn_anomaly.models.window_model import WindowAutoencoder


# ---------------------------------------------------------------------------
# TPC Autoencoder
# ---------------------------------------------------------------------------


class TestTPCAutoencoder:
    def test_forward_shape(self):
        model = TPCAutoencoder(input_dim=64, latent_dim=8, hidden_dims=(32, 16))
        x = torch.randn(4, 64)
        x_hat, z = model(x)
        assert x_hat.shape == (4, 64)
        assert z.shape == (4, 8)

    def test_reconstruction_error_shape(self):
        model = TPCAutoencoder(input_dim=64, latent_dim=8)
        x = torch.randn(10, 64)
        errors = model.reconstruction_error(x)
        assert errors.shape == (10,)

    def test_reconstruction_error_non_negative(self):
        model = TPCAutoencoder(input_dim=64, latent_dim=8)
        x = torch.randn(10, 64)
        errors = model.reconstruction_error(x)
        assert (errors >= 0).all()


# ---------------------------------------------------------------------------
# PMT Autoencoder
# ---------------------------------------------------------------------------


class TestPMTAutoencoder:
    def test_forward_shape(self):
        model = PMTAutoencoder(input_dim=32, latent_dim=4, hidden_dims=(16,))
        x = torch.randn(8, 32)
        x_hat, z = model(x)
        assert x_hat.shape == (8, 32)
        assert z.shape == (8, 4)

    def test_reconstruction_error_shape(self):
        model = PMTAutoencoder(input_dim=32, latent_dim=4)
        errors = model.reconstruction_error(torch.randn(5, 32))
        assert errors.shape == (5,)


# ---------------------------------------------------------------------------
# Fusion Autoencoder
# ---------------------------------------------------------------------------


class TestFusionAutoencoder:
    def test_joint_forward_shape(self):
        model = FusionAutoencoder(
            tpc_input_dim=64, pmt_input_dim=32, latent_dim=16, hidden_dims=(48, 24)
        )
        x_tpc = torch.randn(6, 64)
        x_pmt = torch.randn(6, 32)
        recon, z, combined = model(x_tpc, x_pmt)
        assert recon.shape == (6, 96)     # 64 + 32
        assert z.shape == (6, 16)
        assert combined.shape == (6, 96)

    def test_reconstruction_error_shape(self):
        model = FusionAutoencoder(tpc_input_dim=64, pmt_input_dim=32, latent_dim=16)
        errors = model.reconstruction_error(torch.randn(4, 64), torch.randn(4, 32))
        assert errors.shape == (4,)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            FusionAutoencoder(mode="invalid")

    def test_late_mode_requires_encoders(self):
        with pytest.raises(ValueError, match="tpc_encoder"):
            FusionAutoencoder(mode="late")

    def test_late_fusion_forward(self):
        tpc_enc = TPCAutoencoder(input_dim=64, latent_dim=8)
        pmt_enc = PMTAutoencoder(input_dim=32, latent_dim=4)
        model = FusionAutoencoder(
            tpc_input_dim=64,
            pmt_input_dim=32,
            latent_dim=8,
            hidden_dims=(8,),
            mode="late",
            tpc_encoder=tpc_enc,
            pmt_encoder=pmt_enc,
        )
        x_tpc = torch.randn(3, 64)
        x_pmt = torch.randn(3, 32)
        recon, z, combined = model(x_tpc, x_pmt)
        # combined dim = tpc latent + pmt latent = 8 + 4 = 12
        assert recon.shape == (3, 12)
        assert z.shape == (3, 8)


# ---------------------------------------------------------------------------
# Window Autoencoder
# ---------------------------------------------------------------------------


class TestWindowAutoencoder:
    def test_forward_shape(self):
        model = WindowAutoencoder(window_size=64, n_channels=1, latent_dim=16)
        x = torch.randn(4, 1, 64)
        x_hat, z = model(x)
        assert x_hat.shape == (4, 1, 64)
        assert z.shape == (4, 16)

    def test_forward_multi_channel(self):
        model = WindowAutoencoder(window_size=64, n_channels=4, latent_dim=32)
        x = torch.randn(2, 4, 64)
        x_hat, z = model(x)
        assert x_hat.shape == (2, 4, 64)

    def test_reconstruction_error_shape(self):
        model = WindowAutoencoder(window_size=64, n_channels=1, latent_dim=16)
        errors = model.reconstruction_error(torch.randn(5, 1, 64))
        assert errors.shape == (5,)
