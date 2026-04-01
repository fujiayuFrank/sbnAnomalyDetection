#!/usr/bin/env bash
# Train the PMT autoencoder.
set -euo pipefail

CONFIG="${1:-configs/pmt.yaml}"
echo "[train_pmt] Using config: $CONFIG"
sbn-train --config "$CONFIG"
