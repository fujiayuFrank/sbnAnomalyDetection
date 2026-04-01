#!/usr/bin/env bash
# Run inference with a trained SBN anomaly detection model.
#
# Usage:
#   ./scripts/run_inference.sh configs/tpc.yaml --input data/processed/tpc_features_test.npy
#   ./scripts/run_inference.sh configs/fusion.yaml \
#       --tpc-input data/processed/tpc_features_test.npy \
#       --pmt-input data/processed/pmt_features_test.npy
set -euo pipefail

CONFIG="${1:-configs/tpc.yaml}"
shift
echo "[run_inference] Using config: $CONFIG"
sbn-infer --config "$CONFIG" "$@"
