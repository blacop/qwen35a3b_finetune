#!/usr/bin/env bash
set -euo pipefail

# Roll back MODEL_DIR in vLLM env file and restart related services.
#
# Example:
#   ./scripts/rollback_vllm_model.sh \
#     --target-model-dir /home/ubuntu/qwen35a3b_finetune/outputs/swift_dpo_clean_v2_gpu5_merged_20260413_vllm \
#     --reason "canary p95 breach"

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
ENV_FILE="$PROJECT_ROOT/services/vllm-gpu5-dpo.env"
TARGET_MODEL_DIR=""
REASON="manual"
SERVICES="vllm-qwen35-dpo-gpu5.service dpo-guardrails-proxy.service"
STATE_DIR="$PROJECT_ROOT/runtime/canary_guard"

usage() {
  cat <<'EOF'
Usage: rollback_vllm_model.sh --target-model-dir <path> [options]

Options:
  --env-file <path>      vLLM env file (default: services/vllm-gpu5-dpo.env)
  --target-model-dir <p> target MODEL_DIR to roll back to (required)
  --services "<a b>"     systemd user services to restart
  --reason "<text>"      reason recorded in rollback log
  --state-dir <path>     state/log directory
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --target-model-dir)
      TARGET_MODEL_DIR="$2"
      shift 2
      ;;
    --services)
      SERVICES="$2"
      shift 2
      ;;
    --reason)
      REASON="$2"
      shift 2
      ;;
    --state-dir)
      STATE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$TARGET_MODEL_DIR" ]]; then
  echo "[ERROR] --target-model-dir is required" >&2
  exit 1
fi
if [[ ! -d "$TARGET_MODEL_DIR" ]]; then
  echo "[ERROR] target model dir not found: $TARGET_MODEL_DIR" >&2
  exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ERROR] env file not found: $ENV_FILE" >&2
  exit 1
fi

mkdir -p "$STATE_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_FILE="$STATE_DIR/vllm-gpu5-dpo.env.${TS}.bak"
cp "$ENV_FILE" "$BACKUP_FILE"

CURRENT_MODEL_DIR="$(sed -n 's/^MODEL_DIR=//p' "$ENV_FILE" | head -n1)"
if [[ -z "$CURRENT_MODEL_DIR" ]]; then
  echo "[ERROR] MODEL_DIR not found in $ENV_FILE" >&2
  exit 1
fi

if [[ "$CURRENT_MODEL_DIR" == "$TARGET_MODEL_DIR" ]]; then
  echo "[INFO] MODEL_DIR already at target: $TARGET_MODEL_DIR"
else
  sed -i "s|^MODEL_DIR=.*$|MODEL_DIR=$TARGET_MODEL_DIR|g" "$ENV_FILE"
fi

systemctl --user daemon-reload
systemctl --user restart $SERVICES

python3 - <<PY
import json, pathlib, datetime
log_path = pathlib.Path("$STATE_DIR") / "rollback_actions.jsonl"
event = {
    "ts_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "reason": "$REASON",
    "env_file": "$ENV_FILE",
    "backup_file": "$BACKUP_FILE",
    "from_model_dir": "$CURRENT_MODEL_DIR",
    "to_model_dir": "$TARGET_MODEL_DIR",
    "services": "$SERVICES".split(),
}
with log_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(event, ensure_ascii=False) + "\\n")
print(json.dumps(event, ensure_ascii=False))
print(f"[INFO] rollback log: {log_path}")
PY

echo "[DONE] rollback applied"
