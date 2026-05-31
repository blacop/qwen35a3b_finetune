#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
MERGE_SCRIPT="${MERGE_SCRIPT:-$PROJECT_ROOT/scripts/post_train_merge_deploy_text_vl_unified_gpu5.sh}"
VLLM_ENV_FILE="${VLLM_ENV_FILE:-$PROJECT_ROOT/services/vllm-gpu5-qwen35-domain-text-vl.env}"
TEXT_TRAIN_SERVICE="${TEXT_TRAIN_SERVICE:-swift-sft-text-relation-gpu5.service}"
VL_TRAIN_SERVICE="${VL_TRAIN_SERVICE:-swift-sft-vl-relation-gpu4.service}"
AB_SCRIPT="${AB_SCRIPT:-$PROJECT_ROOT/scripts/run_domain_text_vl_ab.py}"
AB_OUT_ROOT="${AB_OUT_ROOT:-$PROJECT_ROOT/rag/eval/domain_text_vl_ab}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/tracking/logs/post_train_merge_deploy_and_ab_text_vl_unified_gpu5.log}"
RUN_POST_DEPLOY_KNOWLEDGE_GATE="${RUN_POST_DEPLOY_KNOWLEDGE_GATE:-1}"
RELEASE_LOG_INDEX_FILE="${RELEASE_LOG_INDEX_FILE:-$PROJECT_ROOT/tracking/logs/release_log_index.jsonl}"
RELEASE_NOTIFY_LOG="${RELEASE_NOTIFY_LOG:-$PROJECT_ROOT/tracking/logs/release_notifications.log}"

mkdir -p "$(dirname "$LOG_FILE")"
exec >>"$LOG_FILE" 2>&1

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

read_env_value() {
  local file="$1"
  local key="$2"
  awk -F'=' -v k="$key" '$1==k {print substr($0, index($0, "=")+1)}' "$file" | tail -n1
}

json_get() {
  local path="$1"
  local expr="$2"
  python3 - "$path" "$expr" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expr = sys.argv[2]
cur = data
for part in expr.split("."):
    if not part:
        continue
    cur = cur[part]
print(cur)
PY
}

wait_http_ready() {
  local port="$1"
  local api_key="$2"
  for _ in $(seq 1 60); do
    if [[ -n "${api_key:-}" ]]; then
      if curl -fsS -H "Authorization: Bearer ${api_key}" "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
        return 0
      fi
    else
      if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
        return 0
      fi
    fi
    sleep 2
  done
  return 1
}

log "pipeline start"
pipeline_ts="$(date -u +%Y%m%dT%H%M%SZ)"
KNOWLEDGE_GATE_OUT_DIR="${KNOWLEDGE_GATE_OUT_DIR:-$PROJECT_ROOT/eval_outputs/post_deploy_knowledge_gate_text_vl_gpu5_${pipeline_ts}}"
log "reserved knowledge gate out dir: $KNOWLEDGE_GATE_OUT_DIR"
RUN_POST_DEPLOY_KNOWLEDGE_GATE="$RUN_POST_DEPLOY_KNOWLEDGE_GATE" \
KNOWLEDGE_GATE_OUT_DIR="$KNOWLEDGE_GATE_OUT_DIR" \
"$MERGE_SCRIPT"

text_result="$(systemctl --user show -p Result --value "$TEXT_TRAIN_SERVICE" || true)"
vl_result="$(systemctl --user show -p Result --value "$VL_TRAIN_SERVICE" || true)"
log "text training result=${text_result:-unknown}"
log "vl training result=${vl_result:-unknown}"
if [[ "$text_result" != "success" || "$vl_result" != "success" ]]; then
  log "skip ab because one or both training services were not successful"
  exit 1
fi

PORT="$(read_env_value "$VLLM_ENV_FILE" "PORT" || true)"
API_KEY="$(read_env_value "$VLLM_ENV_FILE" "API_KEY" || true)"
MODEL_DIR="$(read_env_value "$VLLM_ENV_FILE" "MODEL_DIR" || true)"
if [[ -z "${PORT:-}" || -z "${MODEL_DIR:-}" ]]; then
  log "missing PORT or MODEL_DIR in $VLLM_ENV_FILE"
  exit 1
fi

if ! wait_http_ready "$PORT" "$API_KEY"; then
  log "gpu5 vllm did not become ready on port=$PORT"
  exit 1
fi

ab_out_dir="$AB_OUT_ROOT/run_${pipeline_ts}"
log "run ab: out_dir=$ab_out_dir model_dir=$MODEL_DIR"
python3 "$AB_SCRIPT" --out-dir "$ab_out_dir"

summary_path="$ab_out_dir/summary.json"
if [[ ! -f "$summary_path" ]]; then
  log "missing ab summary: $summary_path"
  exit 1
fi

text_avg="$(json_get "$summary_path" "text_avg_score")"
vl_avg="$(json_get "$summary_path" "vl_avg_score")"
relation_text="$(json_get "$summary_path" "by_dimension.relation.text_avg_score")"
relation_vl="$(json_get "$summary_path" "by_dimension.relation.vl_avg_score")"

knowledge_gate_gpu5_summary="$KNOWLEDGE_GATE_OUT_DIR/gpu5_20260506/summary.json"
knowledge_gate_gpu7_summary="$KNOWLEDGE_GATE_OUT_DIR/gpu7_20260506/summary.json"
knowledge_gate_kb40_summary="$KNOWLEDGE_GATE_OUT_DIR/kb40_20260428/summary.json"
knowledge_gate_gpu7_ops8_summary="$KNOWLEDGE_GATE_OUT_DIR/gpu7_ops_feedback_8_20260509/summary.json"
release_summary_path="$ab_out_dir/release_gate_summary.json"

python3 - "$release_summary_path" "$MODEL_DIR" "$summary_path" "$KNOWLEDGE_GATE_OUT_DIR" "$knowledge_gate_gpu5_summary" "$knowledge_gate_gpu7_summary" "$knowledge_gate_kb40_summary" "$knowledge_gate_gpu7_ops8_summary" "$RUN_POST_DEPLOY_KNOWLEDGE_GATE" "$text_avg" "$vl_avg" "$relation_text" "$relation_vl" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
payload = {
    "model_dir": sys.argv[2],
    "ab_summary_path": sys.argv[3],
    "knowledge_gate": {
        "enabled": sys.argv[9].lower() in {"1", "true", "yes"},
        "output_dir": sys.argv[4],
        "gpu5_20260506_summary": sys.argv[5],
        "gpu7_20260506_summary": sys.argv[6],
        "kb40_20260428_summary": sys.argv[7],
        "gpu7_ops_feedback_8_20260509_summary": sys.argv[8],
    },
    "ab_metrics": {
        "text_avg_score": float(sys.argv[10]),
        "vl_avg_score": float(sys.argv[11]),
        "relation_text_avg_score": float(sys.argv[12]),
        "relation_vl_avg_score": float(sys.argv[13]),
    },
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

python3 "$PROJECT_ROOT/scripts/append_release_log_index.py" \
  --index-file "$RELEASE_LOG_INDEX_FILE" \
  --ts "$pipeline_ts" \
  --pipeline "post_train_merge_deploy_and_ab_text_vl_unified_gpu5" \
  --model-dir "$MODEL_DIR" \
  --ab-summary-path "$summary_path" \
  --release-gate-summary-path "$release_summary_path" \
  --knowledge-gate-out-dir "$KNOWLEDGE_GATE_OUT_DIR" \
  --knowledge-gate-gpu5-summary "$knowledge_gate_gpu5_summary" \
  --knowledge-gate-gpu7-summary "$knowledge_gate_gpu7_summary" \
  --knowledge-gate-kb40-summary "$knowledge_gate_kb40_summary" \
  --extra-json "{\"knowledge_gate_gpu7_ops8_summary\":\"$knowledge_gate_gpu7_ops8_summary\"}"

printf '[%s] release_gate_summary=%s knowledge_gate_out_dir=%s ab_summary=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$release_summary_path" "$KNOWLEDGE_GATE_OUT_DIR" "$summary_path" >>"$RELEASE_NOTIFY_LOG"

log "ab done: summary_path=$summary_path"
log "ab metrics: text_avg=$text_avg vl_avg=$vl_avg relation_text=$relation_text relation_vl=$relation_vl"
log "knowledge gate out dir: $KNOWLEDGE_GATE_OUT_DIR"
log "knowledge gate summaries: gpu5=$knowledge_gate_gpu5_summary gpu7=$knowledge_gate_gpu7_summary kb40=$knowledge_gate_kb40_summary gpu7_ops8=$knowledge_gate_gpu7_ops8_summary"
log "release gate summary: $release_summary_path"
log "release log index: $RELEASE_LOG_INDEX_FILE"
log "release notify log: $RELEASE_NOTIFY_LOG"
