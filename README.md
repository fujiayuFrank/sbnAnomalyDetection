# sbnAnomalyDetection

Streaming anomaly detection pipeline for the [Short-Baseline Neutrino (SBN)](https://sbn.fnal.gov/) experiment at Fermilab.

## Primary Architecture — GNN Forecaster

The current model is a **graph neural network forecaster** (`GNNForecasterPyG`) that treats TPC channels as nodes in a spatial graph and learns to predict the next time window from a history of past windows. Anomaly scores are the per-channel MSE between the predicted and actual next window.

```
Past windows (history × window_size events)
        │
        ▼
  ┌─────────────┐   for each time step
  │  GCN layers │◄── spatial message passing between neighboring channels
  └──────┬──────┘
         │ node embeddings
         ▼
  ┌─────────────┐
  │     GRU     │   temporal encoding over history steps
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │   Linear    │   predict next-window features per node
  └──────┬──────┘
         │
         ▼
  MSE vs actual next window  →  anomaly score per channel and per window
```

### Key design choices

| Component | Detail |
|-----------|--------|
| Input | Sparse TPC hit integrals from ROOT files |
| Graph nodes | TPC channels (wires) |
| Graph edges | Channels within configurable `adjacency_radius` of each other |
| Node features | Per-channel aggregates per temporal bin: `sum`, `min`, `max`, `mean`, `stdev`, `count` |
| Temporal model | GRU over `history` past frames |
| Anomaly score | Per-channel MSE between predicted and actual next frame |
| Pruning | Only active channels (non-zero across history) are included in each graph |

## Project Structure

```
sbnAnomalyDetection/
├── sbn_anomaly/
│   ├── data/
│   │   ├── sparse_window_dataset.py  # SparseWindowDatasetPyG — main GNN dataset
│   │   ├── graph_window_dataset_pyg.py  # PyG Data/Batch construction, edge index
│   │   ├── streaming.py              # RootStreamer — uproot-based ROOT streaming
│   │   ├── materialize_windows.py    # (legacy) dense window materializer
│   │   ├── stream_dataset.py         # TPCStreamDataset for TPC autoencoder
│   │   └── dataset.py                # Map-style datasets (TPC, PMT, Fusion, Window)
│   ├── models/
│   │   ├── gnn_forecaster_pyg.py     # GNNForecasterPyG — primary model
│   │   ├── tpc_model.py              # TPCAutoencoder
│   │   ├── pmt_model.py              # PMTAutoencoder
│   │   ├── fusion_model.py           # FusionAutoencoder
│   │   └── window_model.py           # WindowAutoencoder
│   ├── train/
│   │   ├── cli.py                    # sbn-train entry point
│   │   ├── gnn_trainer.py            # GNNTrainerPyG
│   │   └── trainer.py                # BaseTrainer (shared training loop)
│   ├── infer/
│   │   ├── cli.py                    # sbn-infer entry point
│   │   └── inferrer.py               # GNNScorer, AnomalyScorer
│   └── utils/
├── configs/
│   ├── gnn.yaml                      # GNN configuration (primary)
│   ├── tpc.yaml
│   ├── pmt.yaml
│   └── ...
├── tests/
├── pyproject.toml
└── graphing/
    ├── plot_wrapper.C               # hits2.h.integral Histogram plotter
    └── plot_wrapper.py
```

## Installation

```bash
pixi install
pixi shell
```

Or with pip:

```bash
pip install -e ".[dev]"
```

## GNN Training

### From ROOT files (recommended)

Stream directly from ROOT files. Events are loaded into a `SparseWindowDatasetPyG` and optionally cached as a compact `.npz` for subsequent runs.

```bash
sbn-train --config configs/gnn.yaml \
  --root-files /data/run1.root /data/run2.root

# Or from a file list manifest (one path per line, # comments allowed)
sbn-train --config configs/gnn.yaml \
  --root-file-list data/train_files.txt
```

To cache the loaded events for faster future runs, set `training.save_events_path` in `configs/gnn.yaml`:

```yaml
training:
  save_events_path: data/events_cache.npz
```

### From a cached events file

Once an events NPZ has been saved (either from a prior training run or explicitly), point `data.events_path` at it:

```yaml
data:
  events_path: data/events_cache.npz
```

Then run without `--root-files`:

```bash
sbn-train --config configs/gnn.yaml
```

### Key config parameters

```yaml
model_type: gnn

data:
  events_path: data/events_cache.npz   # sparse events NPZ (or use --root-files)
  tree_name: caloskim/TrackCaloSkim    # ROOT TTree path
  hit_branches:                         # which hit branches to read
    - hits0.h.integral
    - hits0.h.channel
    - hits2.h.integral
    - hits2.h.channel
  window_size: 20        # events per frame
  n_temporal_bins: 4     # bins within each frame
  stride: 5              # step between consecutive windows
  adjacency_radius: 4    # channels within this radius are connected
  node_features:         # per-channel aggregates per bin
    - sum
    - min
    - max
    - mean
    - stdev
    - count

model:
  history: 4             # number of past frames used to predict the next frame
  gnn_hidden: 64
  gnn_layers: 2
  gru_hidden: 128

training:
  batch_size: 32
  num_workers: 4
  lr: 0.001
  max_epochs: 10
  validation_split: 0.1
  checkpoint_dir: checkpoints/gnn/
  output_path: checkpoints/gnn/gnn_final.pt
  save_events_path: data/events_cache.npz  # optional: cache events after loading
```

### Training output

After training, the checkpoint directory contains:
- `gnn_final.pt` — trained model weights
- `training_history.csv` — per-epoch loss and validation loss
- `training_curves.png` — loss curves
- `score_distribution.png` — histogram of window anomaly scores on the training set
- `score_over_time.png` — anomaly score vs window index (temporal trend)
- `node_mse.png` — per-channel average prediction MSE (shows which wires are hardest to predict)

## GNN Inference

```bash
sbn-infer --config configs/gnn.yaml --output scores.npz
```

The `inference.input_path` in `configs/gnn.yaml` points at the windows/events file to score. The output is a compressed `.npz` containing:

| Array | Shape | Description |
|-------|-------|-------------|
| `scores` | `(N_windows,)` | Mean active-channel MSE per window |
| `scores_max` | `(N_windows,)` | Max active-channel MSE per window |
| `node_scores` | `(N_windows, N_channels)` | Per-channel MSE, NaN for inactive channels |
| `event_index` | `(N_windows,)` | Window index |
| `is_anomaly` | `(N_windows,)` | Boolean flag (only when `inference.threshold` is set) |

## Data Pipeline

### Sparse event representation

Raw TPC hits are stored as a CSR-style flat array in the events NPZ:

| Key | Dtype | Description |
|-----|-------|-------------|
| `channels_flat` | int64 | Concatenated channel indices across all events |
| `integrals_flat` | float32 | Corresponding hit integrals |
| `offsets` | int64 | CSR row pointers — event `i` spans `[offsets[i], offsets[i+1])` |
| `n_channels` | int64 | Total channel count |

This format is ~1000× more compact than a dense window array because most channels are inactive in any given event.

### Window construction

`SparseWindowDatasetPyG` assembles windows lazily in `__getitem__`:

```
Window i covers events[start : start + (history+1)*window_size]
  └─ split into (history+1) frames of window_size events each
       └─ each frame split into n_bins temporal bins
            └─ hits per bin aggregated per channel → node features
```

Per-event features are pre-aggregated once at construction time, so `__getitem__` only needs to combine `~events_per_bin` already-deduplicated arrays per bin (no per-sample sorting).

### Streaming from ROOT

```python
from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG

dataset = SparseWindowDatasetPyG.from_root(
    root_files=["run1.root", "run2.root"],
    tree_name="caloskim/TrackCaloSkim",
    hit_branches=["hits0.h.integral", "hits0.h.channel"],
    history=4,
    window_size=20,
    n_bins=4,
    stride=5,
    radius=4,
)
dataset.save_events("events_cache.npz")  # cache for future runs
```

## Legacy Autoencoder Models

The original per-subsystem autoencoders are still available for comparison or independent deployment:

| Model | `model_type` | Input |
|-------|-------------|-------|
| `TPCAutoencoder` | `tpc` | TPC waveform/hit features |
| `PMTAutoencoder` | `pmt` | PMT waveform features |
| `FusionAutoencoder` | `fusion` | TPC + PMT (joint or late fusion) |
| `WindowAutoencoder` | `window` | Raw waveform windows (1-D conv) |

```bash
sbn-train --config configs/tpc.yaml
sbn-train --config configs/tpc.yaml --root-files /data/run1.root  # stream from ROOT
sbn-infer --config configs/tpc.yaml --root-file-list data/test_files.txt
```

## Tests

```bash
# With pixi environment active
python -m pytest tests/

# Or point directly at the pixi Python
.pixi/envs/default/bin/python -m pytest tests/
```
