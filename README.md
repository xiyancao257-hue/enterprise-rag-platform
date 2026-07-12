# Enterprise RAG

Enterprise RAG is a learning and portfolio project for building production-style retrieval systems.
It starts small, but the architecture is shaped around real enterprise RAG problems:

- noisy document ingestion and cleanup
- structure-aware parsing instead of naive text chunking
- hybrid retrieval with BM25 + dense-style vectors
- rank fusion, reranking, and context compression
- query engine features such as ambiguity detection, spelling correction, and query rewriting
- knowledge graph enhanced retrieval for entity relationship questions
- observability, retrieval evaluation, and human-in-the-loop RAG self-healing

## Quick Start

```bash
cd enterprise-rag
uv sync --extra dev
enterprise-rag ingest data/raw
enterprise-rag query "What does AUTH-429 affect?" --enable-graph --trace
```

Add `.txt` or `.md` files into `data/raw` before running ingestion.

## CLI Demo

```bash
enterprise-rag ingest data/raw

enterprise-rag query "What does AUTH-429 affect?" \
  --config config/default.json \
  --enable-graph \
  --top-k 3 \
  --trace \
  --log-query data/logs/query_log.jsonl

enterprise-rag eval data/eval/retrieval_eval.json --k 5

enterprise-rag experiment data/eval/retrieval_eval.json --k-values 1 3 5 8

enterprise-rag inspect-index
```

## API Service

The project also exposes the same RAG pipeline through FastAPI.

```bash
enterprise-rag-api
```

Or with Docker Compose:

```bash
docker compose up api
```

Production-style Compose wiring:

```bash
export ENTERPRISE_RAG_API_KEY="replace-with-real-api-key"
export ENTERPRISE_RAG_API_KEYS="$ENTERPRISE_RAG_API_KEY"
docker compose -f docker-compose.yml -f docker-compose.prod.yml up api worker qdrant redis
```

This uses `config/production.example.json`, Qdrant, Redis cache, Redis leases, persistent app data,
API key auth, audit logging, and guardrail budgets.

Health check:

```bash
curl http://localhost:8000/health
```

Query:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What does AUTH-429 affect?","top_k":3,"include_trace":true}'
```

The API returns the grounded answer, query plan, citations, and optional retrieval trace.

For production-style deployments, enable API key auth with `api_security.require_api_key`
and provide keys through the configured environment variable instead of committing secrets.
Protected endpoints accept either `X-API-Key` or `Authorization: Bearer ...`.

```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: $ENTERPRISE_RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"What does AUTH-429 affect?","top_k":3}'
```

## Configuration

Runtime retrieval defaults can live in a JSON config file instead of being hardcoded in CLI calls.

```bash
enterprise-rag query "What does AUTH-429 affect?" --config config/default.json
enterprise-rag query "What does AUTH-429 affect?" --config config/default.json --enable-graph --top-k 8
```

CLI flags override config values. This mirrors enterprise deployments where each environment can tune
retrieval depth, graph expansion, and default access groups without changing application code.

`config/production.example.json` shows a production-style setup with Qdrant, Redis cache, Redis leases,
API key auth, audit logging, ingestion source allowlists, and latency/cost/human-review guardrails.
It is intentionally an example file: replace placeholder hashes and tune budgets before using it.
For local production-style demos, `ENTERPRISE_RAG_API_KEYS` can provide comma-separated raw API keys.
For stricter tenant-scoped production auth, replace `api_security.api_keys[].key_hash` with a SHA-256 hash
and configure its `allowed_tenants`.

Generate a SHA-256 API key hash:

```bash
python -c 'import hashlib; print(hashlib.sha256(b"replace-with-real-api-key").hexdigest())'
```

Run a readiness check against the production-style config:

```bash
enterprise-rag readiness-report \
  --config config/production.example.json \
  --index data/processed/chunks.json \
  --eval data/eval/retrieval_eval.json \
  --query-log data/logs/query_log.jsonl \
  --self-healing-dir data/eval/self_healing \
  --k 5
```

Or through the production-style API:

```bash
curl http://localhost:8000/readiness \
  -H "X-API-Key: $ENTERPRISE_RAG_API_KEY" \
  -H "X-Tenant-ID: acme"
```

When API key auth is enabled, protected API calls also require `X-Tenant-ID`:

```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: $ENTERPRISE_RAG_API_KEY" \
  -H "X-Tenant-ID: acme" \
  -H "Content-Type: application/json" \
  -d '{"query":"What does AUTH-429 affect?","top_k":3}'
```

## Current Architecture

```text
Raw Documents
  -> Document Loader
  -> Dirty Data Cleaner
  -> Structure Parser
  -> Semantic / Structure-Aware Chunker
  -> Local Hybrid Index

User Query
  -> Query Analyzer
  -> Query Rewrite / Correction
  -> Metadata Filter Extraction
  -> BM25 Retrieval + Vector Retrieval + optional Graph Retrieval
  -> Reciprocal Rank Fusion
  -> Rerank
  -> Context Compression
  -> Evidence Sufficiency Check
  -> Grounded Answer Generation
  -> Formatted Citations
  -> Trace / Query Log

Query Logs
  -> Log Analysis
  -> Eval Draft Generation
  -> Evidence Suggestions
  -> Human Approval
  -> Regression Eval Promotion
  -> Readiness Report
```

## Main Components

Ingestion:

- `ingestion/loaders.py` loads local `.txt` and `.md` documents.
- `processing/cleaning.py` filters low-quality or duplicated text.
- `processing/parser.py` preserves headings, paragraphs, and tables as document blocks.
- `processing/chunking.py` creates structure-aware chunks with source metadata.

Query planning:

- `query/engine.py` normalizes the query, corrects simple typos, rewrites questions, detects ambiguity, and extracts metadata filters such as `extension:.md`.

Retrieval:

- `retrieval/bm25.py` handles exact keywords, product names, and error codes.
- `retrieval/vector.py` handles semantic similarity with a deterministic local embedding model.
- `retrieval/graph.py` expands entity relationships through a lightweight knowledge graph.
- `retrieval/hybrid.py` combines retrievers with reciprocal rank fusion.

Answering:

- `retrieval/rerank.py` reranks broad recall candidates.
- `rag/compression.py` keeps compact evidence context.
- `rag/answer_generation.py` checks evidence sufficiency before generating grounded answers.
- `rag/citations.py` formats answer citations.

Observability and evaluation:

- `observability/tracing.py` records retrieved, reranked, and final context hits for one query.
- `observability/query_logging.py` writes JSONL query summaries for long-term analysis.
- `evaluation/retrieval_eval.py` runs Recall@K, Precision@K, and MRR.
- `evaluation/readiness.py` summarizes index, eval, logs, self-healing artifacts, and recommendations.

## Self-Healing Workflow

The project includes a human-in-the-loop self-healing loop. The system can find failed production-like queries, generate eval drafts, suggest candidate evidence, and stop for human approval before promoting anything into the official benchmark.

```bash
enterprise-rag self-healing-report data/logs/query_log.jsonl \
  --index data/processed/chunks.json \
  --workdir data/eval/self_healing

enterprise-rag approve-suggested-evidence \
  data/eval/self_healing/generated_with_suggestions.json \
  --case-id log_1_missing_escalation_path \
  --suggestion-index 0 \
  --output data/eval/self_healing/reviewed_eval.json

enterprise-rag promote-eval-draft \
  data/eval/self_healing/reviewed_eval.json \
  --output data/eval/regression_eval.json

enterprise-rag eval data/eval/regression_eval.json
```

The approval step is intentionally explicit. Suggested evidence helps reviewers move faster, but only human-reviewed evidence becomes part of the regression benchmark.

## Readiness Report

Use the readiness report before a demo or interview walkthrough.

```bash
enterprise-rag readiness-report \
  --index data/processed/chunks.json \
  --eval data/eval/retrieval_eval.json \
  --query-log data/logs/query_log.jsonl \
  --self-healing-dir data/eval/self_healing \
  --k 5
```

It reports index presence, chunk count, eval metrics, query-log health, self-healing artifacts, and recommendations.

## LLM Adapter

The default answer generator is deterministic, so tests and local demos do not require network access.
The project also includes an `OpenAIClient` adapter stub at `src/enterprise_rag/llm/openai_client.py`.
It implements the same `complete(prompt: str) -> str` shape expected by `LLMAnswerGenerator`, but intentionally raises a configuration error until a real SDK-backed implementation is added.

```python
from enterprise_rag.llm import OpenAIClient
from enterprise_rag.rag.answer_generation import LLMAnswerGenerator
from enterprise_rag.rag.pipeline import RagPipeline

pipeline = RagPipeline(
    chunks,
    answer_generator=LLMAnswerGenerator(OpenAIClient()),
)
```

## Roadmap

1. Ingestion
   - PDF, DOCX, PPTX, HTML, Markdown, Excel
   - OCR for scanned pages and images
   - table extraction and image captions
   - source metadata and access control metadata

2. Chunking
   - heading-aware chunks
   - table-preserving chunks
   - parent-child chunks
   - multimodal chunks with OCR/caption text

3. Retrieval
   - production vector database adapter
   - persisted BM25/vector indexes
   - cross-encoder or LLM reranking

4. Query Engine
   - HyDE-style hypothetical answer generation
   - LLM-based query decomposition
   - richer metadata filter grammar

5. Knowledge Graph RAG
   - stronger entity extraction
   - richer relationship extraction
   - graph persistence
   - graph-grounded citations

6. RAG Self-Healing
   - reviewer UI
   - Slack or ticket-based approval workflow
   - index refresh suggestions
   - automatic regression trend reporting

## Interview Story

The project is designed to support this positioning:

> I built an enterprise-grade RAG pipeline that starts with structure-aware ingestion instead of naive text splitting, then uses query planning, hybrid retrieval, graph expansion, reranking, context compression, grounded answer generation, citations, tracing, evaluation, and a human-in-the-loop self-healing workflow.

Useful talking points:

- I treated ingestion quality as a retrieval problem, preserving document structure and metadata before indexing.
- I combined BM25, vector retrieval, and graph retrieval because exact terms, semantic similarity, and entity relationships fail in different ways.
- I added trace and query logs so retrieval failures can be explained and studied over time.
- I built eval generation and promotion as a human-in-the-loop process so failed queries improve the benchmark without polluting it.
- I added readiness reporting so the system can be evaluated as an engineering artifact, not just a demo.
