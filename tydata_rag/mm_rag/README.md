# MultiModal RAG Pipeline (Docx + Website)

This folder provides directly runnable scripts for:

1. `docx` extraction (text + images + image contexts)
2. website asset capture (screenshots/crops if Playwright available, fallback to HTML + image downloads)
3. multimodal indexing (text TF-IDF + image histogram features)
4. retrieval API (FastAPI)

## 1) Quick Start (One Command)

```bash
chmod +x /home/ubuntu/tydata_rag/mm_rag/*.sh
/home/ubuntu/tydata_rag/mm_rag/run_mm_pipeline.sh \
  /tmp/tydata/JT包网后台操作手册.docx \
  "https://your-backend-site.example.com"
```

If you only want docx ingestion:

```bash
/home/ubuntu/tydata_rag/mm_rag/run_mm_pipeline.sh /tmp/tydata/JT包网后台操作手册.docx
```

Default chunk params are now finer-grained: `chunk_size=360`, `overlap=80`.
You can override by passing arg4/arg5:

```bash
/home/ubuntu/tydata_rag/mm_rag/run_mm_pipeline.sh /tmp/tydata/JT包网后台操作手册.docx "" /home/ubuntu/tydata_rag/mm_rag/work 600 120
```

## 2) Website Capture (Detailed)

Basic:

```bash
/usr/bin/python3 /home/ubuntu/tydata_rag/mm_rag/capture_site_assets.py \
  --base-url "https://your-backend-site.example.com" \
  --output-dir /home/ubuntu/tydata_rag/mm_rag/work/web_assets
```

With login (Playwright mode):

```bash
/usr/bin/python3 /home/ubuntu/tydata_rag/mm_rag/capture_site_assets.py \
  --base-url "https://your-backend-site.example.com" \
  --login-url "https://your-backend-site.example.com/login" \
  --username "admin" \
  --password "password" \
  --user-selector "input[name='username']" \
  --pass-selector "input[name='password']" \
  --submit-selector "button[type='submit']" \
  --max-pages 50 \
  --max-depth 2
```

Optional Playwright install:

```bash
/usr/bin/python3 -m pip install --user playwright
/home/ubuntu/.local/bin/playwright install chromium
```

## 3) Build Multimodal Index

```bash
/usr/bin/python3 /home/ubuntu/tydata_rag/mm_rag/build_mm_index.py \
  --docx-dir /home/ubuntu/tydata_rag/mm_rag/work/docx_assets \
  --web-dir /home/ubuntu/tydata_rag/mm_rag/work/web_assets \
  --output-dir /home/ubuntu/tydata_rag/mm_rag/index
```

## 3.1) Baseline vs Improved Compare

```bash
/usr/bin/python3 /home/ubuntu/tydata_rag/mm_rag/eval_compare.py \
  --baseline-index /home/ubuntu/tydata_rag/mm_rag/index_baseline_600 \
  --improved-index /home/ubuntu/tydata_rag/mm_rag/index_improved_360
```

Outputs:
- `eval_outputs/comparison_*/comparison_cases.csv`
- `eval_outputs/comparison_*/comparison_summary.json`

## 3.2) Generate Multimodal Training JSONL

```bash
/usr/bin/python3 /home/ubuntu/tydata_rag/mm_rag/generate_mm_training_jsonl.py
```

Default outputs:
- `train_datasets/jt_manual_mm_train_*/multimodal_train.canonical.jsonl`
- `train_datasets/jt_manual_mm_train_*/multimodal_train.swift_vl.jsonl`
- `train_datasets/jt_manual_mm_train_*/multimodal_train.llava.jsonl`
- `train_datasets/jt_manual_mm_train_*/multimodal_train.openai_vision.jsonl`
- `train_datasets/jt_manual_mm_train_*/summary.json`

Disable redaction (not recommended):

```bash
/usr/bin/python3 /home/ubuntu/tydata_rag/mm_rag/generate_mm_training_jsonl.py --no-redact
```

## 4) Start Retrieval API

```bash
chmod +x /home/ubuntu/tydata_rag/mm_rag/start_mm_rag_api.sh
/home/ubuntu/tydata_rag/mm_rag/start_mm_rag_api.sh
```

Health:

```bash
curl -sS http://127.0.0.1:18100/health
```

Text retrieval:

```bash
curl -sS http://127.0.0.1:18100/retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"如何配置后台角色权限",
    "top_k":8
  }'
```

Image + text retrieval:

```bash
curl -sS http://127.0.0.1:18100/retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"这个页面的提款审核在哪里",
    "query_image_path":"/absolute/path/to/screenshot.png",
    "top_k":8
  }'
```

RAG answer (if `LLM_API_BASE` + `LLM_MODEL` available):

```bash
curl -sS http://127.0.0.1:18100/answer \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"如何做资金修正审核",
    "top_k":8,
    "include_hits":true
  }'
```

## Output Artifacts

- `work/docx_assets/text_chunks.jsonl`
- `work/docx_assets/image_records.jsonl`
- `work/web_assets/page_records.jsonl`
- `index/units.jsonl`
- `index/text_matrix.npz`
- `index/text_vectorizer.pkl`
- `index/image_features.npy`
- `index/image_unit_indices.npy`
