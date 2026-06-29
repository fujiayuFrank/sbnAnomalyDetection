# sbnAnomalyDetection

Streaming anomaly detection pipeline for the [Short-Baseline Neutrino (SBN)](https://sbn.fnal.gov/) experiment at Fermilab.

## Primary Architecture — GNN Forecaster

The current model is a **graph neural network forecaster** (`GNNForecasterPyG`) that treats TPC channels as nodes in a spatial graph and learns to predict the next time window from a history of past windows. The graph encoder uses PyG `GCNConv` layers, and both the graph encoder and temporal GRU stack are configured with variable hidden-dimension lists. Anomaly scores are the per-channel MSE between the predicted and actual next window.

```
Past windows (history × window_size events)
        │
        ▼
  ┌─────────────┐   for each time step
  │ GNN encoder │◄── GCNConv message passing between neighboring channels
  └──────┬──────┘
         │ node embeddings
         ▼
  ┌─────────────┐
  │  GRU stack  │   temporal encoding over history steps
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
| Graph model | Variable-width `GCNConv` stack configured by `model.gnn_hidden_dims` |
| Temporal model | Variable-width GRU stack configured by `model.gru_hidden_dims` over `history` past frames |
| Anomaly score | Per-channel MSE between predicted and actual next frame |
| Pruning | Only active channels (non-zero across history) are included in each graph |


## Project Structure

```text
sbnAnomalyDetection/
├── sbn_anomaly/
│   ├── data/
│   │   ├── sparse_window_dataset.py     # SparseWindowDatasetPyG — main GNN dataset
│   │   ├── graph_window_dataset_pyg.py  # PyG Data/Batch construction, edge index
│   │   ├── streaming.py                 # RootStreamer — uproot-based ROOT streaming
│   │   ├── materialize_windows.py       # Window/event materialization CLI
│   │   ├── stream_dataset.py            # TPCStreamDataset for TPC autoencoder
│   │   └── dataset.py                   # Map-style datasets (TPC, PMT, Fusion, Window)
│   ├── models/
│   │   ├── gnn_forecaster_pyg.py        # GNNForecasterPyG — primary sparse PyG model
│   │   ├── gnn_forecaster.py            # Dense-adjacency legacy GNNForecaster
│   │   ├── tpc_model.py                 # TPCAutoencoder
│   │   ├── pmt_model.py                 # PMTAutoencoder
│   │   ├── fusion_model.py              # FusionAutoencoder
│   │   └── window_model.py              # WindowAutoencoder
│   ├── train/
│   │   ├── cli.py                       # sbn-train entry point
│   │   ├── gnn_trainer.py               # GNNTrainerPyG
│   │   └── trainer.py                   # BaseTrainer (shared training loop)
│   ├── infer/
│   │   ├── cli.py                       # sbn-infer entry point
│   │   └── inferrer.py                  # GNNScorer, AnomalyScorer
│   └── utils/
├── configs/
│   ├── gnn.yaml                         # GNN configuration (primary)
│   ├── materialize_windows.yaml         # Window/event materialization config
│   ├── default.yaml                     # Shared/default configuration
│   ├── tpc.yaml
│   ├── pmt.yaml
│   ├── fusion.yaml
│   ├── window.yaml
│   └── ...
├── tuning_configs/
│   └── test_configs/
│       ├── gnn_test1.yaml
│       └── ...
├── config_maker.py                     # Generate sweep/tuning YAML configs
├── run_gnn_sweep.py                    # Run GNN sweep configs and track results
├── evaluate_gnn_sweep.py               # Evaluate per-model inference_scores.npz threshold metrics
├── run_materialize_windows.sh          # Bash wrapper for materializing windows/events
├── run_multi_branch_inference.py       # Multi-branch inference runner
├── npz_npy_reader.py                   # Inspect .npz/.npy files
├── tests/
├── graphing/
│    ├── plot_sweep_metrics.py          # Plot threshold_metrics_summary.csv sweep metrics
│    ├── plot_wrapper.C                 # hits2.h.integral histogram plotter
│    ├── plot_wrapper.py
│    ├── plot_materialize_window.py     # materialize_window .npz file plotter
│    ├── plot_channelhist.C             # hits2.h.channel histogram plotter
│    ├── plot_channelhist.py
│    ├── plot_channel_time.C            # per channel integral value with time line chart plotter
│    ├── plot_inference_result.py       # model inference score histogram plotter
│    └── plot_sweep_metrics.py          # Plot the performance of the models in gnn sweep 
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

### Running a GNN parameter sweep

The top-level `run_gnn_sweep.py` script is used for batch GNN experiments. It runs YAML configuration files from `tuning_configs/` / `tunning_configs/` and gives each config its own run name based on the config filename stem. For example, `tuning_configs/gnn_test2.yaml` becomes run name `gnn_test2`.

Typical usage:

```bash
python run_gnn_sweep.py
```

#### Generating sweep YAML configs

Use `config_maker.py` to batch-generate YAML files for GNN sweep experiments. The generator copies a base YAML such as:

```text
configs/gnn.yaml
```

and writes one generated config per selected parameter combination under:

```text
tuning_configs/gnn_sweep/
```

The generated config names use a zero-padded numeric prefix plus the swept model settings, for example:

```text
0081_win20_stride10_hist1_bs64_lr0p003.yaml
```

The numeric prefix is controlled by the in-script setting:

```python
START_INDEX = 0
```

Set it to a later value when you are continuing a sweep and do not want the next generated configs to reuse old names:

```python
START_INDEX = 81
```

With this setting, the first generated file starts with `0081_...` instead of `0000_...`.

The generator also supports paired learning-rate and batch-size sweeps:

```python
BATCH_SIZES = [64, 128]
LEARNING_RATES = [0.003, 0.001]
```

These lists are iterated **pair by pair**, not as a full Cartesian product:

```text
batch_size=64,  lr=0.003
batch_size=128, lr=0.001
```

This is useful when each batch size has a matching learning rate that should stay tied together. The two lists must have the same length; otherwise the generator raises an error instead of silently making mismatched configs.


For each YAML file, the wrapper creates/reuses a per-run directory under:

```text
checkpoints/gnn/<config_stem>/
```

Inside that directory, it writes a patched `config_run.yaml`, trains the model, and then runs inference using the newly saved checkpoint. The SQLite database, CSV summary, and Excel summary are stored in the main project directory by default.

Example:

```bash
python run_gnn_sweep.py \
  --config-dir tuning_configs \
  --pattern "gnn_test*.yaml" \
  --monitor-interval 30
```

Useful options:

| Option | Meaning |
|--------|---------|
| `--config-dir` | Directory containing sweep YAML files. Default: `tuning_configs/`. |
| `--pattern` | Glob pattern for selecting configs, e.g. `gnn_test*.yaml`. |
| `--runs-root` | Root directory for per-model output folders. Default: `checkpoints/gnn/`. |
| `--db-path` | SQLite summary database path. Default: `gnn_sweep.sqlite3` in the project directory. |
| `--train-cmd` | Training command. Default: `sbn-train`. Useful if you want `python -m sbn_anomaly.train.cli`. |
| `--infer-cmd` | Inference command. Default: `sbn-infer`. Useful if you want `python -m sbn_anomaly.infer.cli`. |
| `--timeout` | Optional timeout in seconds for each train/infer subprocess. Default: no timeout. |
| `--skip-infer` | Train only; do not run inference after training. |
| `--stop-on-error` | Stop the sweep after the first failed config. Without this flag, failed configs are skipped and later configs continue running. |
| `--monitor-interval N` / `--monitor_interval N` | Print CPU/RAM usage every `N` seconds while training or inference is running. Use `0` to disable. |
| `--force-rewrite` / `--force_rewrite` | If a run name already exists in the database, delete the old database record and rerun training/inference, overwriting files in that run directory. |
| `--missing-rewrite` / `--missing_rewrite` | If a successful run already exists in the database but its expected output directory is missing or incomplete, delete the old database record and rerun only that model. |

#### Name-conflict and rerun behavior

The wrapper uses `run_name = <config filename stem>`, so rerunning the same YAML will hit the same database record and the same directory:

```text
checkpoints/gnn/<run_name>/
```

For example:

```text
tuning_configs/test.yaml
        ↓
run_name = test
run_dir  = checkpoints/gnn/test/
```

By default, the wrapper does **not** overwrite successful existing runs. This avoids accidentally retraining a model that already completed.

Default behavior:

| Existing DB status | Behavior without rewrite flags |
|--------------------|--------------------------------|
| `success` | Skip training/inference and keep the existing model/database record. |
| `trained_no_infer` | Skip training and keep the existing model/database record. |
| `failed` | Delete the failed database record and rerun automatically. |
| `running` | Delete the stale/interrupted database record and rerun automatically. |

Use forced rewrite when you intentionally want to retrain every selected model even though completed records already exist:

```bash
python run_gnn_sweep.py --force-rewrite
```

This deletes the old database rows for that `run_name`, then reuses the same output directory. Files with the same names, such as `config_run.yaml`, `gnn_final.pt`, `scores.npz`, `training_history.csv`, and plots, will be overwritten by the new run. The wrapper does not delete the entire directory first, so unrelated old files with different names may remain.

Use missing rewrite when the database says a model exists, but the corresponding checkpoint directory is missing or clearly incomplete:

```bash
python run_gnn_sweep.py --missing-rewrite
```

With `--missing-rewrite`, the wrapper checks the expected directory:

```text
checkpoints/gnn/<config_stem>/
```

A model is rerun only when one of these is true:

| Directory check | Behavior with `--missing-rewrite` |
|-----------------|------------------------------------|
| `checkpoints/gnn/<config_stem>/` is missing | Delete the old database record and rerun that model. |
| The path exists but is not a directory | Delete the old database record and rerun that model. |
| The directory contains only one `.yaml` / `.yml` file | Treat it as incomplete, delete the old database record, and rerun that model. |
| The directory contains additional artifacts | Keep the existing database record and skip, unless `--force-rewrite` is also passed. |

This is useful after an interrupted or partially failed sweep where SQLite still contains a successful-looking record, but the model folder is missing `gnn_final.pt`, `scores.npz`, plots, or other output files.

`--force-rewrite` has higher priority than `--missing-rewrite`. If both are passed, `--force-rewrite` reruns every selected config with an existing database record, regardless of whether the output directory looks complete.

Do **not** run two copies of the sweep wrapper on the same configs at the same time if `running` records are auto-rewritable. A second wrapper process could see the first process's `running` record, delete it, and start another training job writing to the same directory.

#### Resource monitoring and terminal output

CSV/XLSX summary export is optional. The sweep always updates the SQLite database at `--db-path`, but it does not write `gnn_sweep_summary.csv` or `gnn_sweep_summary.xlsx` unless explicitly requested:

```bash
python run_gnn_sweep.py --export-summary
```

This is useful when running many sweeps because the database remains the source of truth, while CSV/XLSX files can be regenerated only when needed.

The monitor output is injected directly into the visible terminal stream, so it remains visible while training progress bars and logging output are being printed. A typical line looks like:

```text
[2026-06-24T20:27:19] during command usage
  Process tree: 11 process(es) | CPU: 3072.3% over 0.5s sample | PSS: 8.3 GB | RSS: 10.5 GB
```

Interpretation:

* `Process tree` means the wrapper is monitoring the launched `sbn-train` or `sbn-infer` process plus its child processes, such as PyTorch DataLoader workers.
* `CPU` is sampled over a short window. `100%` means one full CPU core during that sample; `200%` means about two cores; values above `100%` are normal for multi-process or multi-threaded PyTorch jobs.
* `PSS`, when available on Linux, is the better estimate of real memory pressure because it divides shared memory pages among processes.
* `RSS` is still shown as a useful upper-bound-like number, but summed RSS can over-count shared pages between forked workers.

The monitor interval controls how often the resource block is printed, not the averaging window of the CPU value. For example, `--monitor-interval 30` prints once every 30 seconds, but each CPU value is still a short sample. Use a smaller interval such as `10` for more frequent feedback, or `0` to turn monitoring off:

```bash
python run_gnn_sweep.py --monitor-interval 10
python run_gnn_sweep.py --monitor-interval 0
```

When running through `nohup`, output is usually redirected to a file instead of staying attached to the terminal. Use an explicit log file and watch it with `tail -f`:

```bash
nohup python run_gnn_sweep.py > gnn_sweep.log 2>&1 &
tail -f gnn_sweep.log
```

### Evaluating GNN sweep inference results

After a sweep has produced per-model inference outputs, use the top-level `evaluate_gnn_sweep.py` script to compare models over one or more anomaly thresholds. The script scans model directories under:

```text
checkpoints/gnn/<model_name>/inference_scores.npz
```

Model directories without `inference_scores.npz` are skipped automatically. Each NPZ must contain at least:

```text
scores
scores_max
first_run
```

The script normalizes `scores` and `scores_max`, applies every threshold listed in `THRESHOLDS`, and compares the predicted good/bad window labels against the known run labels:

```python
good_runs = {18445, 19724, 20141, 20142, 20144}
bad_runs = {19627, 19946, 20104}
```

Confusion matrix convention:

| Term | Meaning |
|------|---------|
| `TP` | True bad run window predicted bad |
| `TN` | True good run window predicted good |
| `FP` | True good run window predicted bad |
| `FN` | True bad run window predicted good |

The compact output is written by default to:

```text
threshold_evaluation/threshold_metrics_summary.csv
```

with the columns:

```text
model_name, method, threshold, normalization_model, TP, TN, FP, FN, precision, recall, F1, accuracy
```

Typical usage:

```bash
python evaluate_gnn_sweep.py
```

#### Starting evaluation from a selected model index

By default, `evaluate_gnn_sweep.py` scans all model folders under:

```text
checkpoints/gnn/
```

You can restrict evaluation and plotting to only model directories whose leading numeric prefix is greater than or equal to a chosen start index. This can be set directly in the script:

```python
MODEL_START_INDEX = None  # use all model directories
MODEL_START_INDEX = 81    # use 0081_..., 0082_..., ...
```

You can also pass the start index from the command line:

```bash
python evaluate_gnn_sweep.py --start 81
```

The command-line flag has higher priority than the in-code setting:

```text
--start value  >  MODEL_START_INDEX variable  >  no filtering
```

For example, if the script has:

```python
MODEL_START_INDEX = 40
```

but you run:

```bash
python evaluate_gnn_sweep.py --start 81
```

then the evaluator starts from `0081_...`, not `0040_...`.

This filter also affects automatic plotting. If you run:

```bash
python evaluate_gnn_sweep.py --start 81 --with-plot
```

then the evaluator first writes a filtered `threshold_metrics_summary.csv`, and the plotting step uses that filtered summary. The resulting `02_metric_relations/` plots therefore include only models from `0081_...` onward.

If no matching model directories exist, the evaluator writes an empty compact summary CSV with only headers, prints a warning, evaluates no metrics, and produces no plots. This prevents the plotter from accidentally reusing an older non-empty `threshold_metrics_summary.csv`.

#### Multiple threshold evaluation

The threshold is configured as a list in the script:

```python
THRESHOLDS = [0.5]
```

To evaluate several thresholds in one pass, edit it to something like:

```python
THRESHOLDS = [0.5, 0.3, 0.7]
```

The script produces one metrics row for each model, method, and threshold combination. By default, rows are written in **model-first** order:

```text
model_A, threshold 0.5
model_A, threshold 0.3
model_A, threshold 0.7
model_B, threshold 0.5
model_B, threshold 0.3
model_B, threshold 0.7
```

Use `--threshold-first` to write rows in **threshold-first** order instead:

```bash
python evaluate_gnn_sweep.py --threshold-first
```

This produces output ordered like:

```text
threshold 0.5, model_A
threshold 0.5, model_B
threshold 0.3, model_A
threshold 0.3, model_B
```

This only controls CSV row order. It does not change the computed metrics. If a ranking flag is also passed, ranking overrides the model-first or threshold-first order.

By default, all three evaluation methods are included:

| Method | Meaning |
|--------|---------|
| `scores_only` | Use only the normalized `scores` array |
| `scores_max_only` | Use only the normalized `scores_max` array |
| `both_or` | Predict bad if either `scores` or `scores_max` is above threshold |

To export only one score mode, use one of the mutually exclusive method-selection flags:

```bash
python evaluate_gnn_sweep.py --scores-only
python evaluate_gnn_sweep.py --scores-max-only
```

To also write the detailed summary and per-run ratios, pass `--full-summary`:

```bash
python evaluate_gnn_sweep.py --full-summary
```

This additionally writes:

```text
threshold_evaluation/threshold_metrics_full_summary.csv
threshold_evaluation/threshold_per_run_ratios.csv
```

The full summary contains the same model/method/threshold combinations as the compact summary, plus normalization scales, evaluated-window counts, and true-class prediction ratios. The per-run ratio CSV also includes the threshold column so each run-level breakdown is tied to the threshold that produced it.

#### Ranking evaluation output

By default, the compact CSV follows the normal model-first order, or threshold-first order when `--threshold-first` is used. To sort the output by a metric instead, pass one ranking flag:

```bash
python evaluate_gnn_sweep.py --precision-rank
python evaluate_gnn_sweep.py --recall-rank
python evaluate_gnn_sweep.py --f1-rank
python evaluate_gnn_sweep.py --accuracy-rank
```

Ranking flags are mutually exclusive. Each ranking sorts from highest value to lowest value across all produced rows. For example, if `THRESHOLDS = [0.5, 0.3]`, then:

```bash
python evaluate_gnn_sweep.py --scores-only --f1-rank
```

evaluates only the `scores_only` method and ranks all `model × threshold` rows by F1 score. If `--full-summary` is also given, the full summary CSV is ranked the same way. The per-run ratio CSV is still written in model/run/threshold order because it is mainly for debugging.

#### Automatic plotting after evaluation

The evaluator can also run the plotting script automatically after it writes the compact summary CSV. This is controlled by `--with-plot`:

```bash
python evaluate_gnn_sweep.py --with-plot
```

This first writes:

```text
threshold_evaluation/threshold_metrics_summary.csv
```

and then runs the plotting script under the `graphing/` directory. The plotting script path is configured inside `evaluate_gnn_sweep.py` with:

```python
PLOT_SCRIPT_RELATIVE = Path("graphing/plot_sweep_metrics.py")
```

The path is relative to the directory containing `evaluate_gnn_sweep.py`. If the plotting script is renamed or moved, update only `PLOT_SCRIPT_RELATIVE`.

The evaluator forwards only the plotting-relevant flags to the plotting script:

| Evaluator flag | Used by evaluator | Forwarded to plotter | Reason |
|---------------|-------------------|----------------------|--------|
| `--scores-only` | Yes | Yes | Keeps evaluation and plotting restricted to `scores_only`. |
| `--scores-max-only` | Yes | Yes | Keeps evaluation and plotting restricted to `scores_max_only`. |
| `--plot-summary` | No | Yes, when `--with-plot` is used | Asks the plotter to save its own CSV summary files. |
| `--full-summary` | Yes | No | Writes evaluator full/per-run CSV files; does not control plotting CSV output. |
| Ranking flags | Yes | No | Ranking only changes CSV row order; plots regroup the rows anyway. |
| `--threshold-first` | Yes | No | This only changes CSV row order and does not affect plot contents. |
| `--with-plot` | Yes | No | This only tells the evaluator to launch the plotter. |
| `--start` | Yes | No | Filters which model directories enter the summary before plotting begins. |

Examples:

```bash
# Evaluate normally, then plot the compact summary
python evaluate_gnn_sweep.py --with-plot

# Evaluate and plot only the scores_only method
python evaluate_gnn_sweep.py --scores-only --with-plot

# Write evaluator full summaries and also make plots
python evaluate_gnn_sweep.py --full-summary --with-plot

# Ask the plotter to also save its own plotting-side CSV summaries
python evaluate_gnn_sweep.py --with-plot --plot-summary

# Full evaluator summaries plus plotting-side summaries
python evaluate_gnn_sweep.py --full-summary --with-plot --plot-summary
```

`--full-summary` and `--plot-summary` are intentionally separate. `--full-summary` belongs to the evaluator and writes `threshold_metrics_full_summary.csv` plus `threshold_per_run_ratios.csv`. `--plot-summary` belongs to the plotting step and writes the plotter's parsed/aggregated CSV outputs.

#### Plotting threshold metrics directly

You can also run the plotting script directly after `threshold_metrics_summary.csv` exists:

```bash
python graphing/plot_sweep_metrics.py
```

The plotting script uses internal path settings by default. Its input path should point to:

```python
INPUT_PATH = Path("threshold_evaluation/threshold_metrics_summary.csv")
```

The plotter creates a compact set of plots instead of the older heatmap-heavy output:

```text
plots/
  01_metric_histograms/
    accuracy_histogram.png
    precision_histogram.png
    recall_histogram.png
    F1_histogram.png

  02_metric_relations/
    accuracy/
      vary_window_size/
      vary_window_stride/
      vary_history/
    precision/
    recall/
    F1/
```

For each metric, it first plots a histogram over all filtered rows. Then it fixes two model variables and plots the metric against the third variable. The three model variables parsed from `model_name` are:

```text
window_size
window_stride
history
```

The histogram binning is controlled inside `graphing/plot_sweep_metrics.py` with:

```python
HIST_BINS = 30
```

The relation plots use automatic y-axis zooming by default so small metric differences do not appear flat:

```python
AUTO_YLIM = True
Y_PAD_FRACTION = 0.10
Y_MIN_PAD = 1e-4
```

By default, the plotting script saves only plot images. It writes CSV summary files only when `--plot-summary` is passed:

```bash
python graphing/plot_sweep_metrics.py --plot-summary
```

Available plotting flags:

| Flag | Meaning |
|------|---------|
| `--scores-only` | Plot only rows with `method == scores_only`. |
| `--scores-max-only` | Plot only rows with `method == scores_max_only`. |
| `--plot-summary` | Also save plotter-side CSV summaries. Without this flag, only plots are saved. |

Available command-line flags for `evaluate_gnn_sweep.py`:

| Flag | Meaning |
|------|---------|
| `--start N` | Only evaluate/plot model directories whose leading numeric prefix is `>= N`. This overrides the in-code `MODEL_START_INDEX` setting. |
| `--full-summary` | Also write `threshold_metrics_full_summary.csv` and `threshold_per_run_ratios.csv`. Without this flag, only the compact summary CSV is written. |
| `--with-plot` | After writing the compact summary CSV, run `graphing/plot_sweep_metrics.py` automatically. |
| `--plot-summary` | Only meaningful with `--with-plot`; forwards `--plot-summary` to the plotting script so it also writes plotter-side CSV summaries. |
| `--scores-only` | Only evaluate and store the `scores_only` method. Mutually exclusive with `--scores-max-only`. |
| `--scores-max-only` | Only evaluate and store the `scores_max_only` method. Mutually exclusive with `--scores-only`. |
| `--threshold-first` | Write rows by threshold first, then model. Without this flag, rows are written by model first, then threshold. Ranking flags override this order. |
| `--precision-rank` | Rank CSV output by precision, highest first. |
| `--recall-rank` | Rank CSV output by recall, highest first. |
| `--f1-rank` | Rank CSV output by F1 score, highest first. |
| `--accuracy-rank` | Rank CSV output by accuracy, highest first. |

Example combinations:

```bash
# Default: all methods, all thresholds, model-first order
python evaluate_gnn_sweep.py

# All methods, all thresholds, threshold-first order
python evaluate_gnn_sweep.py --threshold-first

# All methods, compact CSV ranked by F1 across all model/threshold rows
python evaluate_gnn_sweep.py --f1-rank

# Only scores_max, compact CSV ranked by recall
python evaluate_gnn_sweep.py --scores-max-only --recall-rank

# All methods, compact and full summaries ranked by accuracy
python evaluate_gnn_sweep.py --full-summary --accuracy-rank

# Evaluate and plot automatically
python evaluate_gnn_sweep.py --with-plot

# Start from model 0081_... and plot only that filtered range
python evaluate_gnn_sweep.py --start 81 --with-plot

# Evaluate only scores_only and plot only scores_only
python evaluate_gnn_sweep.py --scores-only --with-plot

# Generate evaluator full summaries and plotter-side summaries
python evaluate_gnn_sweep.py --full-summary --with-plot --plot-summary
```

Important top-level settings in the script:

| Setting | Meaning |
|---------|---------|
| `MODEL_START_INDEX` | Optional in-code lower bound on the leading numeric model directory prefix. Overridden by `--start`. |
| `NORMALIZATION_MODE` | Score transform: `tanh`, `sigmoid`, `global_max`, or `none` |
| `NORMALIZATION_SCOPE` | Use per-model or global normalization scale |
| `NORMALIZATION_SCALE_MODE` | Use max, percentile, or manual scale |
| `THRESHOLDS` | List of anomaly thresholds after normalization, each in `[0, 1]` |
| `BOTH_RULE` | How to combine `scores` and `scores_max`: `or`, `and`, `mean`, or `max` |
| `PLOT_SCRIPT_RELATIVE` | Plotting script path used by `--with-plot`, relative to `evaluate_gnn_sweep.py` |

#### Sweep config format

Each YAML in `tuning_configs/` / `tunning_configs/` should follow the same structure as `configs/gnn.yaml`, including list-style hidden dimensions such as:

```yaml
model:
  gnn_hidden_dims: [128, 128, 64]
  gru_hidden_dims: [256, 128]
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

  # Variable graph encoder dimensions.
  # Number of GNN/GCNConv layers = len(gnn_hidden_dims).
  # Example: [128, 128, 64] means input -> 128 -> 128 -> 64.
  gnn_hidden_dims: [128, 128, 64]

  # Variable temporal GRU dimensions.
  # Number of GRU layers = len(gru_hidden_dims).
  # Example: [256, 128] means GNN output -> 256 -> 128.
  gru_hidden_dims: [256, 128]

  norm_type: batch       # options: none, batch, layer
  dropout: 0.1

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

The hidden-dimension lists replace the old fixed-size fields:

```yaml
# Old style — no longer preferred
gnn_hidden: 64
gnn_layers: 3
gru_hidden: 256
gru_layers: 2

# Current style
gnn_hidden_dims: [128, 128, 64]
gru_hidden_dims: [256, 128]
```

`gnn_hidden_dims` configures the general graph-neural-network encoder. In the current implementation, that encoder is built from PyG `GCNConv` layers, so the code may call these internal modules `gcn_layers` or `gnn_layers_list`. `gru_hidden_dims` configures a stack of one-layer GRUs, which allows different hidden sizes per temporal layer. A single multi-layer `nn.GRU` is not used for variable hidden dimensions because PyTorch requires all layers in one `nn.GRU` to share the same `hidden_size`.

### Training output

After training, the checkpoint directory contains the artifacts for that model run:

* `gnn_final.pt` — final trained model weights saved from `training.output_path`
* `config_original.yaml` or `config_run.yaml` — a copy of the YAML used for the run, when config-copying is enabled in the training/sweep wrapper
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

For sweep or batch-training workflows, keep each model's full artifacts under its own directory such as `checkpoints/gnn/model1/`, `checkpoints/gnn/model2/`, etc. The SQLite sweep database remains in the main project directory by default. CSV/XLSX summary files are produced only when `--export-summary` is passed.


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
