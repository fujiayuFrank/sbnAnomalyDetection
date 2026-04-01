#!/usr/bin/env bash
# Train the Fusion autoencoder (joint TPC + PMT).
# Run train_tpc.sh and train_pmt.sh first if using late-fusion mode.
set -euo pipefail

CONFIG="${1:-configs/fusion.yaml}"
echo "[train_fusion] Using config: $CONFIG"
sbn-train --config "$CONFIG"
