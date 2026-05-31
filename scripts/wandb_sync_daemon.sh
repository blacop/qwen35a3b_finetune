#!/usr/bin/env bash
set -euo pipefail

WANDB_ROOT=${WANDB_ROOT:-/home/ubuntu/qwen35a3b_finetune/tracking/wandb}
SYNC_INTERVAL=${SYNC_INTERVAL:-60}
WANDB_PROJECT=${WANDB_PROJECT:-qwen35a3b}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_SYNC=${WANDB_SYNC:-1}

mkdir -p "$WANDB_ROOT"

if [[ "$WANDB_SYNC" != "1" ]]; then
  echo "[wandb-sync] WANDB_SYNC=0, sleeping indefinitely"
  exec sleep infinity
fi

while true; do
  if [[ -n "${WANDB_ENTITY}" ]]; then
    /home/ubuntu/.local/bin/wandb sync "$WANDB_ROOT" --sync-all --include-offline --include-online --mark-synced --sync-tensorboard -p "$WANDB_PROJECT" -e "$WANDB_ENTITY" || true
  else
    /home/ubuntu/.local/bin/wandb sync "$WANDB_ROOT" --sync-all --include-offline --include-online --mark-synced --sync-tensorboard -p "$WANDB_PROJECT" || true
  fi
  sleep "$SYNC_INTERVAL"
done
