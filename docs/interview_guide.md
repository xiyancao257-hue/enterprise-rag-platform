# Enterprise RAG Interview Guide

## 30-Second Pitch

I built an enterprise-style RAG system focused on reliability, not just answer generation. It starts with structure-aware ingestion, then uses query planning, hybrid retrieval, knowledge graph expansion, reranking, context compression, citations, tracing, retrieval evaluation, and a human-in-the-loop self-healing workflow. The goal is to improve recall, grounding, observability, and benchmark quality.

## 2-Minute Architecture Walkthrough

The system has four main layers.

First is ingestion. Documents are loaded, cleaned, parsed into structure-aware blocks, and chunked with headings and metadata preserved. This avoids the common failure mode where naive fixed-size chunks lose tables, section titles, and source context.

Second is query planning. The query engine normalizes input, applies simple typo correction, rewrites questions, detects ambiguity, and extracts metadata filters such as `extension:.md`. This gives retrieval a cleaner and more intentional search plan.

Third is retrieval and answering. The retriever combines BM25 for exact terms, vector search for semantic similarity, and optional graph retrieval for entity relationship questions. Results are fused with reciprocal rank fusion, reranked, compressed, checked for evidence sufficiency, and then used to produce a grounded answer with citations.

Fourth is evaluation and self-healing. The system logs query outcomes, analyzes failed queries, generates eval drafts, suggests candidate evidence, requires human approval, promotes reviewed cases into regression evals, and reports readiness metrics.

## Ingestion Deep Dive

The ingestion design treats data quality as part of retrieval quality. Instead of splitting raw text immediately, the system:

- loads source documents through a connector abstraction
- preserves source metadata such as source system, URI, version, and update time
- writes a source sync manifest so active and deleted source documents can be audited
- bumps an explicit index version so query cache entries, logs, and eval runs can be tied to the exact indexed state
- filters dirty or low-information text
- parses headings, paragraphs, and tables
- creates chunks that preserve document structure
- stores chunks as JSON for repeatable local demos

The main tradeoff is simplicity versus document coverage. The current implementation supports Markdown, text, CSV tables, text-based PDFs, image OCR through Tesseract, and scanned PDF OCR by rendering PDF pages with Poppler before OCR. OCR is still disabled by default so local demos stay deterministic until a provider is explicitly configured. DOCX and richer multimodal extraction are clear adapter extensions.

## Query Planning Deep Dive

Query planning turns a raw user query into a `QueryPlan`.

It handles:

- normalization
- typo correction
- query rewrite
- ambiguity detection
- metadata filter extraction

For example, a query like `extension:.md hybrid retrival` becomes a clean search query for `hybrid retrieval` plus a metadata filter for Markdown files. This prevents metadata constraints from polluting semantic retrieval text.

## Retrieval Deep Dive

The retrieval layer uses multiple retrievers because enterprise questions fail in different ways.

BM25 is strong for exact tokens such as product names, error codes, and policy IDs.

Vector retrieval is better when the user uses different wording from the document.

Graph retrieval helps with relationship and multi-hop questions, such as `AUTH-429 -> Rate Limit Policy -> Auth Service -> Product Atlas`.

The system combines candidates with reciprocal rank fusion, then reranks and compresses context before answer generation.

## Evaluation Story

The evaluation layer measures retrieval quality with:

- Recall@K
- Precision@K
- MRR
- diagnostics
- top-k experiments

The key point is that the system does not rely on vibes. It compares retrieved chunks against expected evidence and produces measurable retrieval quality.

## Self-Healing Story

The self-healing loop is human-in-the-loop.

The workflow is:

```text
query log
-> failed query analysis
-> eval draft generation
-> evidence suggestion
-> human approval
-> regression eval promotion
-> eval run
-> retrieval or ingestion improvement
```

The system can suggest evidence, but it does not automatically promote suggestions into the benchmark. A human must approve the expected evidence. This protects benchmark quality and prevents noisy generated cases from becoming official tests.

## Readiness Story

The readiness report summarizes whether the local RAG system is ready for a demo or release-style review.

It checks:

- index presence
- chunk count
- eval file presence
- eval case count
- Recall@K, Precision@K, and MRR
- query log health
- self-healing artifacts
- recommendations

This makes the project feel like an engineering system, not just a notebook demo.

## Tradeoffs

The local vector model is deterministic and lightweight, which makes tests fast and reproducible. For production-style runs, the embedding provider can be switched to OpenAI while keeping hashing embeddings as the test fallback.

The graph extractor is rule-based, which is transparent and testable. In production, I would add stronger entity and relationship extraction, likely with a model-assisted extraction pipeline plus human validation for critical domains.

The answer generator is deterministic by default. That keeps local demos and tests network-free. For production-style runs, the LLM provider can be switched to OpenAI without changing retrieval, compression, guardrails, or citation code.

The self-healing workflow uses file-based review. That is simple and auditable. In production, I would move this into a reviewer UI or ticket workflow.

## Common Interview Questions

### Why not just use vector search?

Vector search is good for semantic similarity, but it can miss exact identifiers such as error codes, product names, and policy IDs. BM25 handles exact lexical matches better. Graph retrieval adds another layer for relationship-based questions. Combining them improves recall across different query types.

### Why do you need query planning?

Raw user queries often contain typos, ambiguity, metadata constraints, or wording that does not match the document. Query planning gives retrieval a cleaner search input and separates filters from semantic text.

### How do you know retrieval is working?

I use retrieval eval cases with expected evidence and compute Recall@K, Precision@K, and MRR. I also added diagnostics and top-k experiments to identify whether failures come from missing data, poor recall, ranking, or compression.

### What makes this enterprise-oriented?

The project includes structure-aware ingestion, metadata filters, ACL-style filtering, hybrid retrieval, graph expansion, citations, tracing, query logs, regression evals, self-healing workflow, and readiness reporting. These are the operational pieces needed beyond a basic RAG demo.

### Where would you improve it next?

I would plug the OCR adapter into a production provider, add DOCX/PPTX loaders, Excel multi-sheet extraction, a persistent vector database, a stronger embedding model, external reranker integration, model-assisted graph extraction, and a reviewer UI for the self-healing loop.

## STAR Story

Situation: Enterprise RAG systems often fail because raw documents are noisy, chunking loses structure, single-retriever search misses relevant context, and failures are hard to diagnose.

Task: I wanted to build a portfolio project that demonstrates a production-style RAG architecture with reliable ingestion, better retrieval, observability, evaluation, and self-healing.

Action: I built a modular pipeline with structure-aware ingestion, query planning, hybrid retrieval, graph retrieval, reranking, compression, grounded answer generation, citations, tracing, query logging, retrieval evaluation, generated eval drafts, human approval, and readiness reporting.

Result: The project can ingest local knowledge files, answer with citations, explain retrieval behavior, measure retrieval quality, collect failed queries, turn them into reviewed regression evals, and summarize readiness for a demo or deployment-style review.
