# Typlay RAG Remediation 2026-05-27

## What Was Restored
- Restored the `typlay` source archive from `/tmp/ragdata/各类型玩法.zip` into a persistent location.
- Rebuilt extracted text files and the default `typlay` index.
- Repaired runtime path drift from `/home/ubuntu/tydata_rag` to `/home/ubuntu/generate/tydata_rag` in active units and scripts.
- Repaired GPU7 guardrails proxy pathing so `tydata-rag-api-gpu7` points to the live proxy on `http://127.0.0.1:8010/v1`.

## Persistent Typlay Layout
- Archive: `/video-storage/ai-customer/qwen35a3b_finetune/rag/typlay/archives/各类型玩法.zip`
- Raw files: `/video-storage/ai-customer/qwen35a3b_finetune/rag/typlay/raw`
- Extracted text: `/video-storage/ai-customer/qwen35a3b_finetune/rag/typlay/text_all`
- Default index: `/video-storage/ai-customer/qwen35a3b_finetune/rag/typlay/rag_index_typlay_default_560_112`
- Manifest: `/video-storage/ai-customer/qwen35a3b_finetune/rag/typlay/restore_manifest.json`

## Active Runtime Fixes
- `~/.config/systemd/user/tydata-rag-api.service`
- `~/.config/systemd/user/tydata-rag-api-gpu7.service`
- `~/.config/systemd/user/tydata-rag-guard.service`
- `~/.config/systemd/user/mm-rag-api.service`
- `~/.config/systemd/user/mm-rag-guard.service`
- `~/.config/systemd/user/dpo-guardrails-proxy.service`
- `~/.config/systemd/user/dpo-guardrails-proxy-gpu5.service`
- `~/.config/systemd/user/vllm-qwen35-gpu5-agent-upstream.service`
- `~/.config/systemd/user/vllm-qwen35-dpo-gpu5.service`
- `~/.config/systemd/user/vllm-qwen35-gpu7-baseline.service`
- `generate/tydata_rag/start_rag_api.sh`
- `generate/tydata_rag/run_rag_guard.sh`
- `generate/tydata_rag/install_rag_guard_timer.sh`
- `generate/tydata_rag/rag_guard.env`
- `generate/tydata_rag/rag_api.py`
- `generate/tydata_rag/rag_guard.py`
- `generate/tydata_rag/build_rag_index.py`
- `generate/tydata_rag/query_rag.py`
- `generate/tydata_rag/set_rag_gray.py`
- `generate/tydata_rag/ab_compare_retrieval.py`
- `generate/tydata_rag/mm_rag/mm_rag.env`
- `generate/tydata_rag/mm_rag/mm_rag_guard.env`
- `generate/tydata_rag/mm_rag/mm_rag_api.py`
- `generate/tydata_rag/mm_rag/start_mm_rag_api.sh`
- `generate/tydata_rag/mm_rag/run_mm_rag_guard.sh`
- `generate/tydata_rag/mm_rag/run_mm_pipeline.sh`
- `generate/tydata_rag/mm_rag/extract_docx_assets.py`
- `generate/tydata_rag/mm_rag/build_mm_index.py`
- `generate/tydata_rag/mm_rag/generate_mm_training_jsonl.py`
- `generate/tydata_rag/mm_rag/capture_site_assets.py`
- `generate/tydata_rag/mm_rag/eval_compare.py`
- `generate/qwen35a3b_finetune/services/dpo-guardrails-proxy.env`
- `generate/qwen35a3b_finetune/services/dpo-guardrails-proxy-gpu5.env`
- `generate/qwen35a3b_finetune/services/vllm-gpu5-agent-upstream.env`
- `generate/qwen35a3b_finetune/services/vllm-gpu5-dpo.env`
- `generate/qwen35a3b_finetune/services/vllm-gpu7-baseline.env`
- `generate/qwen35a3b_finetune/scripts/dpo_guardrails_proxy.py`

## Runtime Status At Handoff
- `tydata-rag-api.service` is active on `18080` and serves the restored `typlay` index.
- `tydata-rag-api-gpu7.service` is active on `18081` and points at `dpo-guardrails-proxy.service`.
- `mm-rag-api.service` is active on `18100`.
- `dpo-guardrails-proxy.service` is active on `8010`.
- `dpo-guardrails-proxy-gpu5.service` is active on `8001`.
- `vllm-qwen35-gpu7-baseline.service` is active on `8013`.
- `vllm-qwen35-gpu5-agent-upstream.service` is active on `8014`.

## Open Items
- The legacy `llama-server` process that had occupied `8001` on GPU5 was terminated, and the official `dpo-guardrails-proxy-gpu5.service` now owns `8001`.
- The legacy `vllm-qwen35-dpo-gpu5.service` was disabled so it no longer competes for the same GPU5 path / historical `8001` role.
- `run_mm_pipeline.sh` still defaults its input docx path to `/tmp/tydata/...` because that auxiliary JT manual pipeline has not yet been assigned a new persistent source-of-truth path.

## Still Worth Cleaning Up
These are mostly docs, examples, or generated historical artifacts. They do not block the live chain, but they still mention old paths and should be cleaned if you want the tree fully consistent.
- `generate/tydata_rag/README.md`
- `generate/tydata_rag/RAG_PROXY_MINIMAL_INTEGRATION_20260523.md`
- `generate/tydata_rag/EXTERNAL_API_EXPOSURE_SOP_20260526.md`
- `generate/tydata_rag/CURRENT_RAG_GATEWAY_GUARDRAILS_GPU5_GPU7_CHAIN_20260526.md`
- `generate/qwen35a3b_finetune/docs/GATEWAY_DEPLOYMENT_ARCHITECTURE_ZH_20260525.md`
- `generate/qwen35a3b_finetune/docs/GATEWAY_DEPLOYMENT_ARCHITECTURE_20260525.md`
- historical `rag_mvp/outputs/**`
- historical `index*/chunks.jsonl`
- historical `mm_rag/work/**`

## Notes
- I intentionally did not bulk-edit historical outputs or provenance files.
- `8010` was reclaimed from a stray debug process so the GPU7 proxy could bind its normal port again.
