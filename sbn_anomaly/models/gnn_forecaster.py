from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGraphConv(nn.Module):
    """Very small graph convolution implemented with dense adjacency.

    x' = W1 x + W2 (A x)
    """

    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.lin_self = nn.Linear(in_feats, out_feats)
        self.lin_neigh = nn.Linear(in_feats, out_feats)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x:   (B, N, F) or (N, F)
        # adj: (N, N) or (B, N, N)

        was_batched = x.dim() == 3
        if not was_batched:
            x = x.unsqueeze(0)

        # aggregate neighbor messages
        # If adj is (N, N), PyTorch broadcasts over B.
        # If adj is (B, N, N), it does batched matmul.
        neigh = torch.matmul(adj.to(x.device), x)  # (B, N, F)

        out = self.lin_self(x) + self.lin_neigh(neigh)
        out = F.relu(out)

        if not was_batched:
            out = out.squeeze(0)

        return out


class GNNForecaster(nn.Module):
    """Dense-adjacency GNN + temporal GRU forecaster for next-window prediction.

    Input:
        past: (B, T, N, F)
        adj:  (N, N) or (B, N, N)

    Output:
        pred: (B, N, F)

    Args:
        node_feat_dim: feature dimension per node
        gnn_hidden_dims: hidden dimensions for graph convolution stack.
            Number of GNN layers = len(gnn_hidden_dims).
        gru_hidden_dims: hidden dimensions for temporal GRU stack.
            Number of GRU layers = len(gru_hidden_dims).
        history: number of past time windows T
        dropout: dropout probability after GNN layers and between GRU layers
    """

    def __init__(
        self,
        node_feat_dim: int,
        gnn_hidden_dims: list[int] | tuple[int, ...] = (64, 64),
        gru_hidden_dims: list[int] | tuple[int, ...] = (128,),
        history: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        gnn_hidden_dims = [int(d) for d in gnn_hidden_dims]
        gru_hidden_dims = [int(d) for d in gru_hidden_dims]

        if len(gnn_hidden_dims) == 0:
            raise ValueError("gnn_hidden_dims must be non-empty")
        if len(gru_hidden_dims) == 0:
            raise ValueError("gru_hidden_dims must be non-empty")
        if any(d <= 0 for d in gnn_hidden_dims):
            raise ValueError(
                f"gnn_hidden_dims must contain positive integers, got {gnn_hidden_dims}"
            )
        if any(d <= 0 for d in gru_hidden_dims):
            raise ValueError(
                f"gru_hidden_dims must contain positive integers, got {gru_hidden_dims}"
            )

        self.node_feat_dim = int(node_feat_dim)
        self.gnn_hidden_dims = gnn_hidden_dims
        self.gru_hidden_dims = gru_hidden_dims
        self.gnn_layers = len(gnn_hidden_dims)
        self.gru_layers = len(gru_hidden_dims)
        self.history = int(history)
        self.dropout = float(dropout)

        # Variable-width GNN encoder per time slice.
        #
        # Example:
        #   node_feat_dim = 24
        #   gnn_hidden_dims = [128, 128, 64]
        #   GNN: 24 -> 128 -> 128 -> 64
        convs: list[nn.Module] = []
        in_f = self.node_feat_dim

        for hidden_dim in self.gnn_hidden_dims:
            convs.append(SimpleGraphConv(in_f, hidden_dim))
            in_f = hidden_dim

        self.convs = nn.ModuleList(convs)

        # Variable-width GRU stack.
        #
        # Important:
        #   A single nn.GRU with num_layers > 1 requires the same hidden_size
        #   for every GRU layer. To support [256, 128], we use one GRU per layer.
        #
        # Example:
        #   GNN output:      64
        #   gru_hidden_dims: [256, 128]
        #   GRU stack:      64 -> 256 -> 128
        gru_layers_list: list[nn.Module] = []
        in_f = self.gnn_hidden_dims[-1]

        for hidden_dim in self.gru_hidden_dims:
            gru_layers_list.append(
                nn.GRU(
                    input_size=in_f,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=False,
                )
            )
            in_f = hidden_dim

        self.gru_layers_list = nn.ModuleList(gru_layers_list)

        # Decoder: final GRU hidden dim -> node feature prediction.
        self.dec = nn.Linear(self.gru_hidden_dims[-1], self.node_feat_dim)

    def encode_one(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Encode single window.

        Args:
            x:   (B, N, F)
            adj: (N, N) or (B, N, N)

        Returns:
            h: (B, N, gnn_hidden_dims[-1])
        """
        h = x

        for conv in self.convs:
            h = conv(h, adj)

            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout, training=True)

        return h

    def forward(self, past: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Predict next window.

        Args:
            past: (B, T, N, F)
            adj:  (N, N) or (B, N, N)

        Returns:
            pred: (B, N, F)
        """
        B, T, N, feature_dim = past.shape

        if T != self.history:
            raise ValueError(
                f"Input history T={T} does not match model history={self.history}."
            )

        if feature_dim != self.node_feat_dim:
            raise ValueError(
                f"Input node feature dim {feature_dim} does not match "
                f"model node_feat_dim={self.node_feat_dim}."
            )

        # Encode each time slice.
        enc_slices = []

        for t in range(T):
            x_t = past[:, t, :, :]          # (B, N, F)
            h_t = self.encode_one(x_t, adj) # (B, N, gnn_hidden_dims[-1])
            enc_slices.append(h_t)

        # Stack encoded history:
        #   enc: (B, T, N, H_gnn)
        enc = torch.stack(enc_slices, dim=1)

        # Prepare for GRU:
        #   GRU wants (seq_len, batch, input_size)
        #   Treat each node in each graph as an independent temporal sequence.
        #
        #   enc_perm: (T, B, N, H_gnn)
        #   enc_flat: (T, B*N, H_gnn)
        enc_perm = enc.permute(1, 0, 2, 3).contiguous()
        out_seq = enc_perm.view(T, B * N, -1)

        # Variable-width GRU stack.
        for i, gru in enumerate(self.gru_layers_list):
            out_seq, _ = gru(out_seq)

            # Dropout between GRU layers, not after the final layer.
            if self.dropout > 0 and self.training and i < len(self.gru_layers_list) - 1:
                out_seq = F.dropout(out_seq, p=self.dropout, training=True)

        # Final time step.
        last = out_seq[-1]                 # (B*N, gru_hidden_dims[-1])
        pred_flat = self.dec(last)         # (B*N, node_feat_dim)
        pred = pred_flat.view(B, N, -1)    # (B, N, node_feat_dim)

        return pred