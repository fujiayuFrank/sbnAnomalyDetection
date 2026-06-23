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
├── graphing/
│    ├── plot_wrapper.C               # hits2.h.integral histogram plotter
│    ├── plot_wrapper.py
│    ├── materialize_window_plot.py   # materialize_window .npz file plotter
│    ├── plot_channelhist.C           # hits2.h.channel histogram plotter
│    ├── plot_channelhist.py
│    ├── channel_time_plot.C          # per channel integral value with time line chart plotter
│    └── inference_result_plotter.py  # model inference score historgram plotter
└── pyproject.toml
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
  hit_branches:                        # which hit branches to read
    - hits0.h.integral
    - hits0.h.channel
    - hits1.h.integral
    - hits1.h.channel
    - hits2.h.integral
    - hits2.h.channel
  tpc_branches:                        # optional event provenance branches
    - meta.run
    - meta.subrun
    - meta.evt
  window_size: 20        # events per frame
  n_temporal_bins: 4     # bins within each frame
  stride: 20             # step between consecutive windows
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
  gnn_layers: 3
  gru_hidden: 256
  gru_layers: 2
  norm_type: batch       # options: none, batch, layer

training:
  batch_size: 20
  num_workers: 5
  lr: 0.005
  weight_decay: 1.0e-4
  max_epochs: 100
  validation_split: 0.15
  checkpoint_dir: checkpoints/gnn/June-22-test
  output_path: checkpoints/gnn/June-22-test/gnn_final.pt
  log_interval: 50
  use_amp: true

  # Save weights/epoch_XXXX.pt after epochs.
  # false saves disk space; final weights are still saved to output_path.
  save_epoch_checkpoints: false

  # How to reduce the two per-window anomaly scores returned by the GNN trainer.
  # The trainer computes per-node MSE, then returns [mean, max] across nodes.
  # score_mode=mean selects the mean across nodes; score_mode=max selects the max.
  score_mode: mean

inference:
  checkpoint_path: checkpoints/gnn/June-22-test/gnn_final.pt
  input_path: /exp/sbnd/app/users/jiayufu/sbnAnomalyDetection/data/windows_test.npz
  output_path: checkpoints/gnn/June-22-test/inference_scores.npz
  threshold: null
  max_windows: null  # set to an integer to score only the first N windows
```

### Training output

After training, the checkpoint directory contains:

* `gnn_final.pt` — final trained model weights saved from `training.output_path`
* `training_history.csv` — per-epoch loss, validation loss, score percentiles, and timing metrics
* `training_curves.png` — training and validation loss curves
* `score_distribution.png` — histogram of window anomaly scores on the training set
* `score_over_time.png` — anomaly score vs. window index using the non-shuffled training loader
* `node_mse.png` — per-channel average prediction MSE, showing which wires are hardest to predict

If `training.save_epoch_checkpoints: true`, the trainer also saves per-epoch checkpoints under:

```text
weights/
  epoch_0001.pt
  epoch_0002.pt
  ...
```

If `training.save_epoch_checkpoints: false`, these per-epoch weight files are skipped to save disk space. The final model is still saved to `training.output_path`.


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

### Materialize windows / events (.npz)

Use the materializer to produce a compact events `.npz` by default. This is
the preferred format for GNN training because windows are built lazily by
`SparseWindowDatasetPyG`. Pass `--windows` only when you explicitly want the
legacy dense `.npy` window array plus a companion `_meta.npz` file.

Sparse events NPZ (default, recommended for GNN training; saved as `data/events_cache.npz`):

```bash
python -m sbn_anomaly.data.materialize_windows \
  --root-files /data/run1.root /data/run2.root \
  --output data/events_cache.npz \
  --window-size 20 \
  --n-bins 4 \
  --stride 5
```

Dense windows (legacy opt-in, saved as `data/windows.npy` + `data/windows_meta.npz`):

```bash
python -m sbn_anomaly.data.materialize_windows \
  --root-files /data/run1.root /data/run2.root \
  --output data/windows \
  --window-size 20 \
  --n-bins 4 \
  --stride 5 \
  --windows
```

Alternatively, build and save the sparse events programmatically (same result):

```bash
python - <<'PY'
from sbn_anomaly.data.sparse_window_dataset import SparseWindowDatasetPyG

ds = SparseWindowDatasetPyG.from_root(
    root_files=["/data/run1.root", "/data/run2.root"],
    tree_name="caloskim/TrackCaloSkim",
    hit_branches=["hits0.h.integral", "hits0.h.channel"],
    history=4,
    window_size=20,
    n_bins=4,
    stride=5,
    radius=4,
)
ds.save_events("data/events_cache.npz")
PY
```

Note on defaults and precedence
--------------------------------

If you omit `--window-size`, `--n-bins` (temporal bins), or `--stride` on the
command line, the materializer and the train/infer CLIs will read those values
from the provided YAML config under the `data` section (`data.window_size`,
`data.n_temporal_bins`, `data.stride`). Command-line flags override values in
the YAML. If neither CLI flags nor the config supply a value, the materializer
falls back to sensible built-in defaults (e.g. `window_size=20`,
`n_temporal_bins=4`, `stride=1`).

This precedence also applies when training or running inference: the CLI will
use values from the config unless you explicitly pass overriding flags.


### Using a ROOT file list (manifest)

You can pass a file containing ROOT paths (one per line, `#` allowed for comments)
instead of listing files on the command line. This is convenient for long runs
or reproducible manifests.

Materialize sparse events from a manifest file:

```bash
python -m sbn_anomaly.data.materialize_windows \
  --root-file-list data/train_files.txt \
  --output data/events_cache.npz \
  --window-size 20 \
  --n-bins 4 \
  --stride 5
```

Materialize dense windows from a manifest file:

```bash
python -m sbn_anomaly.data.materialize_windows \
  --root-file-list data/train_files.txt \
  --output data/windows \
  --window-size 20 \
  --n-bins 4 \
  --stride 5 \
  --windows
```

Train using a manifest (streaming path):

```bash
sbn-train --config configs/gnn.yaml \
  --root-file-list data/train_files.txt
```

Infer/score using a manifest (streaming TPC path):

```bash
sbn-infer --config configs/gnn.yaml \
  --root-file-list data/test_files.txt \
  --output scores_from_manifest.npz
```

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
