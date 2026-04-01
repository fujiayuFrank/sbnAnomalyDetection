"""Window autoencoder trainer for time-series waveform anomaly detection."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from sbn_anomaly.models.window_model import WindowAutoencoder
from sbn_anomaly.train.trainer import BaseTrainer


class WindowTrainer(BaseTrainer):
    """Trainer for :class:`~sbn_anomaly.models.WindowAutoencoder`.

    Parameters
    ----------
    model:
        Window autoencoder instance.
    lr:
        Learning rate.
    weight_decay:
        L2 regularisation.
    device:
        Compute device.
    max_epochs:
        Training epochs.
    checkpoint_dir:
        Checkpoint save directory.
    log_interval:
        Logging frequency in batches.
    """

    def __init__(
        self,
        model: Optional[WindowAutoencoder] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "auto",
        max_epochs: int = 50,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 50,
    ) -> None:
        if model is None:
            model = WindowAutoencoder()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            max_epochs=max_epochs,
            checkpoint_dir=checkpoint_dir,
            log_interval=log_interval,
        )
        self.criterion = nn.MSELoss()

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        """MSE reconstruction loss for a window batch.

        Expects the DataLoader to yield ``(windows,)`` or ``(windows, labels)``
        where *windows* has shape ``(B, n_channels, window_size)``.
        """
        x = batch[0].to(self.device)
        # Ensure shape is (B, C, L); add channel dim if missing.
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x_hat, _ = self.model(x)
        return self.criterion(x_hat, x)
