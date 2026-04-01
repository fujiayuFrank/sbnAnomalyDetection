#!/usr/bin/env bash
# Train the Window (1-D conv) autoencoder.
set -euo pipefail

CONFIG="${1:-configs/window.yaml}"
echo "[train_window] Using config: $CONFIG"
sbn-train --config "$CONFIG"
