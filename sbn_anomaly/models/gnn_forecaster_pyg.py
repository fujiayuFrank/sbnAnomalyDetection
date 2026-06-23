"""PyTorch Geometric GNN forecaster model.

Uses PyG's GCNConv layers for message passing and a GRU temporal encoder
for sequence-to-sequence prediction on graph-structured data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GNNForecasterPyG(nn.Module):
    """PyG-based GNN + temporal GRU forecaster for next-window prediction.

    Accepts a PyG Batch produced by torch_geometric.loader.DataLoader.
    For each time step in the history, applies stacked GCN layers to aggregate
    spatial node features, then runs a GRU across time to predict the next window.

    Expected Data object fields (per graph):
        x:          (N, 1 + T*frame_feat_dim) — channel_idx followed by T frames of F features
        y:          (N, target_dim) — prediction target
        edge_index: (2, E) sparse edges in COO format

    Args:
        frame_feat_dim: feature dimension per time step F (excluding channel index)
        target_dim: output feature dimension per node
        gnn_hidden_dims: hidden dimensions for stacked GCN layers.
            Number of GCN layers = len(gnn_hidden_dims).
        gru_hidden_dims: hidden dimensions for stacked one-layer GRUs.
            Number of GRU layers = len(gru_hidden_dims).
        history: number of past time steps T
        dropout: dropout probability applied after each GCN layer
    """

    def __init__(
        self,
        frame_feat_dim: int,
        target_dim: int,
        gnn_hidden_dims: list[int] | tuple[int, ...] = (64, 64),
        gru_hidden_dims: list[int] | tuple[int, ...] = (128,),
        history: int = 4,
        dropout: float = 0.1,
        norm_type: str = "none",
    ) -> None:
        super().__init__()

        gnn_hidden_dims = [int(d) for d in gnn_hidden_dims]
        gru_hidden_dims = [int(d) for d in gru_hidden_dims]

        if len(gnn_hidden_dims) == 0:
            raise ValueError("gnn_hidden_dims must be non-empty")
        if len(gru_hidden_dims) == 0:
            raise ValueError("gru_hidden_dims must be non-empty")
        if any(d <= 0 for d in gnn_hidden_dims):
            raise ValueError(f"gnn_hidden_dims must contain positive integers, got {gnn_hidden_dims}")
        if any(d <= 0 for d in gru_hidden_dims):
            raise ValueError(f"gru_hidden_dims must contain positive integers, got {gru_hidden_dims}")

        self.frame_feat_dim = int(frame_feat_dim)
        self.target_dim = int(target_dim)
        self.gnn_hidden_dims = gnn_hidden_dims
        self.gru_hidden_dims = gru_hidden_dims
        self.gnn_layers = len(gnn_hidden_dims)
        self.gru_layers = len(gru_hidden_dims)
        self.history = int(history)
        self.dropout = float(dropout)
        self.norm_type = str(norm_type)

        # GCN input per time step:
        #   channel_idx (1) + per-frame features (frame_feat_dim)
        gcn_in = 1 + self.frame_feat_dim

        if self.norm_type == "batch":
            self.input_norm: nn.Module | None = nn.BatchNorm1d(gcn_in)
        elif self.norm_type == "layer":
            self.input_norm = nn.LayerNorm(gcn_in)
        elif self.norm_type == "none":
            self.input_norm = None
        else:
            raise ValueError(
                f"Unknown norm_type={self.norm_type!r}. "
                "Expected one of: 'none', 'batch', 'layer'."
            )

        # Variable-width GCN stack:
        # Example:
        #   gnn_hidden_dims = [128, 128, 64]
        #   GCN: gcn_in -> 128 -> 128 -> 64
        gcn_layers: list[nn.Module] = []
        current_dim = gcn_in

        for hidden_dim in self.gnn_hidden_dims:
            gcn_layers.append(GCNConv(current_dim, hidden_dim))
            current_dim = hidden_dim

        self.gcn_layers = nn.ModuleList(gcn_layers)

        # Variable-width GRU stack.
        #
        # Important:
        #   A single nn.GRU with num_layers > 1 requires the same hidden_size
        #   for every layer. To support gru_hidden_dims = [256, 128],
        #   we use one GRU module per layer.
        #
        # Example:
        #   GCN output dim = 64
        #   gru_hidden_dims = [256, 128]
        #   GRU stack: 64 -> 256 -> 128
        gru_layers: list[nn.Module] = []
        current_dim = self.gnn_hidden_dims[-1]

        for hidden_dim in self.gru_hidden_dims:
            gru_layers.append(
                nn.GRU(
                    input_size=current_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=False,
                )
            )
            current_dim = hidden_dim

        self.gru_layers_list = nn.ModuleList(gru_layers)

        # Final decoder uses the last GRU hidden dimension.
        self.decoder = nn.Linear(self.gru_hidden_dims[-1], self.target_dim)

    def _encode_frame(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Apply stacked GCN layers to a single time frame.

        Args:
            x: (total_nodes, 1 + frame_feat_dim)
            edge_index: (2, E)

        Returns:
            h: (total_nodes, gnn_hidden_dims[-1])
        """
        h = self.input_norm(x) if self.input_norm is not None else x
        for gcn in self.gcn_layers:
            h = gcn(h, edge_index)
            h = F.relu(h)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout, training=True)
        return h

    def forward(self, data) -> torch.Tensor:
        """Predict next window for a PyG batched graph.

        Args:
            data: PyG Batch object from torch_geometric.loader.DataLoader.
                Must have x (total_nodes, 1+T*F), edge_index (2, E),
                and batch (total_nodes,).

        Returns:
            pred: (total_nodes, target_dim)
        """
        x = data.x  # (total_nodes, 1 + T*F)
        edge_index = data.edge_index

        channel_idx = x[:, :1]   # (total_nodes, 1)
        temporal = x[:, 1:]      # (total_nodes, T*F)

        T = self.history
        t_f = temporal.shape[1]

        if t_f % T != 0:
            raise ValueError(
                f"Temporal feature dim {t_f} is not divisible by history {T}. "
                f"Expected x of shape (N, 1 + {T}*F) for integer F."
            )

        feature_dim = t_f // T

        if feature_dim != self.frame_feat_dim:
            raise ValueError(
                f"Input frame feature dim {feature_dim} does not match "
                f"model frame_feat_dim={self.frame_feat_dim}."
            )

        temporal = temporal.view(-1, T, feature_dim)  # (total_nodes, T, F)

        enc_slices = []
        for t in range(T):
            x_t = torch.cat(
                [channel_idx, temporal[:, t, :]],
                dim=1,
            )  # (total_nodes, 1+F)

            h_t = self._encode_frame(
                x_t,
                edge_index,
            )  # (total_nodes, gnn_hidden_dims[-1])

            enc_slices.append(h_t)

        # Shape: (T, total_nodes, gnn_hidden_dims[-1])
        enc_seq = torch.stack(enc_slices, dim=0)

        # Variable-width GRU stack.
        # Example:
        #   enc_seq:  (T, N, 64)
        #   GRU 1:    64 -> 256
        #   GRU 2:    256 -> 128
        #   output:   (T, N, 128)
        out_seq = enc_seq

        for i, gru in enumerate(self.gru_layers_list):
            out_seq, _ = gru(out_seq)

            # Optional dropout between GRU layers, not after the final GRU layer.
            if self.dropout > 0 and self.training and i < len(self.gru_layers_list) - 1:
                out_seq = F.dropout(out_seq, p=self.dropout, training=True)

        # Use final time step to predict next window.
        pred = self.decoder(out_seq[-1])  # (total_nodes, target_dim)

        return pred
