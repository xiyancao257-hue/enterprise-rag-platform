#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEMO_DIR="${DEMO_DIR:-data/demo}"
INDEX_PATH="$DEMO_DIR/chunks.json"
QUERY_LOG="$DEMO_DIR/query_log.jsonl"
SELF_HEALING_DIR="$DEMO_DIR/self_healing"
CONFIG_PATH="${CONFIG_PATH:-config/default.json}"

mkdir -p "$DEMO_DIR" "$SELF_HEALING_DIR"
rm -f "$INDEX_PATH" "$QUERY_LOG" "$DEMO_DIR/source_manifest.json" "$DEMO_DIR/index_version.json"

echo "== Enterprise RAG demo =="
echo "Config: $CONFIG_PATH"
echo "Artifacts: $DEMO_DIR"
echo

echo "== 1. Ingest sample documents =="
uv run enterprise-rag ingest data/raw \
  --index "$INDEX_PATH" \
  --config "$CONFIG_PATH"
echo

echo "== 2. Query with graph retrieval, trace, citations, and query log =="
uv run enterprise-rag query "Which product is affected by AUTH-429?" \
  --index "$INDEX_PATH" \
  --config "$CONFIG_PATH" \
  --enable-graph \
  --top-k 3 \
  --trace \
  --log-query "$QUERY_LOG"
echo

echo "== 3. Run retrieval regression eval =="
uv run enterprise-rag eval data/eval/retrieval_eval.json \
  --index "$INDEX_PATH" \
  --config "$CONFIG_PATH" \
  --k 5
echo

echo "== 4. Run top-k experiment =="
uv run enterprise-rag experiment data/eval/retrieval_eval.json \
  --index "$INDEX_PATH" \
  --config "$CONFIG_PATH" \
  --k-values 1 3 5 8
echo

echo "== 5. Readiness report =="
uv run enterprise-rag readiness-report \
  --config "$CONFIG_PATH" \
  --index "$INDEX_PATH" \
  --eval data/eval/retrieval_eval.json \
  --query-log "$QUERY_LOG" \
  --self-healing-dir "$SELF_HEALING_DIR" \
  --k 5
echo

echo "Demo complete."
echo "Generated artifacts:"
echo "- $INDEX_PATH"
echo "- $QUERY_LOG"
echo "- $DEMO_DIR/source_manifest.json"
echo "- $DEMO_DIR/index_version.json"
