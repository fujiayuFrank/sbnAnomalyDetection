#!/usr/bin/env bash
# Train the TPC autoencoder.
set -euo pipefail

CONFIG="${1:-configs/tpc.yaml}"
echo "[train_tpc] Using config: $CONFIG"
sbn-train --config "$CONFIG"
