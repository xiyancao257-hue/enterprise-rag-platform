# Production Enterprise RAG Roadmap

This document captures the production upgrade plan for the Enterprise RAG project. The current codebase demonstrates the core RAG architecture: ingestion, query planning, hybrid retrieval, graph retrieval, reranking, compression, citations, tracing, evaluation, self-healing, and readiness reporting.

The next phase is to evolve it into a production-style enterprise RAG platform.

## Production Goals

- Improve reliability under real user traffic.
- Control latency and cost.
- Defend against prompt injection and unsafe retrieved context.
- Support incremental index updates.
- Support multi-tenant isolation.
- Expose an OpenAPI service layer.
- Add Prometheus-style monitoring.
- Add CI/CD and Docker deployment.
- Expand evaluation beyond retrieval metrics.
- Support golden sets, LLM-as-judge, and A/B testing.
- Add optional production adapters such as Qdrant, Weaviate, LangChain, and LlamaIndex.

## 1. Latency And Cost

Production RAG must measure where time and money go.

Track:

- retrieval latency
- rerank latency
- compression latency
- LLM generation latency
- total request latency
- prompt tokens
- completion tokens
- estimated embedding cost
- estimated generation cost
- cache hit rate
- insufficient-evidence rate

Potential modules:

```text
src/enterprise_rag/observability/latency.py
src/enterprise_rag/observability/costing.py
```

Example metrics:

```text
retrieval_ms
rerank_ms
compression_ms
generation_ms
total_ms
estimated_prompt_tokens
estimated_completion_tokens
estimated_cost_usd
```

Key design question:

```text
How does changing top_k, reranking, graph retrieval, or compression affect latency, cost, and answer quality?
```

## 2. Prompt Injection And Security

Enterprise RAG must assume both user input and retrieved documents can be hostile.

Risks:

- user asks model to ignore system instructions
- retrieved document contains malicious prompt instructions
- context asks the model to reveal secrets
- document says to override developer policies
- cross-tenant content leakage

Potential modules:

```text
src/enterprise_rag/security/prompt_injection.py
src/enterprise_rag/security/policies.py
```

Initial guardrails:

- detect phrases such as `ignore previous instructions`
- detect attempts to reveal system prompts or secrets
- flag suspicious retrieved chunks
- separate retrieved context from trusted instructions
- require citations for grounded claims
- enforce metadata ACL and tenant filters before generation

Production stance:

```text
Retrieved documents are data, not instructions.
```

## 3. Incremental Index Updates

Enterprise document collections change constantly. Full reindexing is expensive and slow.

Support:

- document hash
- chunk hash
- changed-document detection
- deleted-document detection
- upsert chunks
- remove stale chunks
- index versioning
- rollback

Potential module:

```text
src/enterprise_rag/indexing/incremental.py
```

Core idea:

```text
Only reprocess documents whose content or metadata changed.
```

Future CLI:

```bash
enterprise-rag incremental-ingest data/raw --index data/processed/chunks.json
```

## 4. Multi-Tenant Isolation

Enterprise RAG usually serves many teams, departments, or customers.

Support:

- `tenant_id`
- `allowed_groups`
- metadata filters
- per-tenant query logging
- per-tenant eval reporting
- per-tenant index option
- shared index with mandatory tenant filter option

Current foundation:

- ACL-style filtering exists through `allowed_groups`.

Next upgrade:

```text
tenant_id filtering must be mandatory in retrieval.
```

Potential files:

```text
src/enterprise_rag/retrieval/filters.py
src/enterprise_rag/security/tenant.py
```

Design choice:

```text
Small systems can use one shared index with tenant filters.
Large or strict-isolation systems may use one index per tenant.
```

## 5. OpenAPI Service Layer

The project currently exposes a CLI. Production systems need an API layer.

Potential module:

```text
src/enterprise_rag/api/app.py
```

Potential endpoints:

```text
POST /query
POST /ingest
GET /readiness
GET /metrics
POST /eval/run
POST /self-healing/report
```

Example query request:

```json
{
  "query": "What does AUTH-429 affect?",
  "top_k": 5,
  "enable_graph": true,
  "tenant_id": "tenant_a",
  "user_groups": ["engineering"]
}
```

Response should include:

- answer
- citations
- query plan
- trace id
- insufficient evidence flag

## 6. Prometheus Monitoring

Production RAG needs operational metrics.

Potential module:

```text
src/enterprise_rag/observability/metrics.py
```

Initial metrics:

```text
rag_queries_total
rag_query_latency_seconds
rag_retrieval_hits_total
rag_provider_latency_ms{component,provider}
rag_insufficient_evidence_total
rag_eval_recall_at_k
rag_eval_precision_at_k
rag_self_healing_candidates_total
```

The API exports Prometheus text format.

API:

```text
GET /metrics
```

Implemented stack:

```text
Prometheus -> Grafana -> alerts
```

Project files:

```text
monitoring/prometheus.yml
monitoring/alerts.yml
docker-compose.prod.yml
```

The API also exports provider-level latency and call counters. These help separate slow embedding calls,
vector database searches, and LLM generation from the broader query pipeline timing.

## 7. CI/CD

CI protects the regression suite and makes the project more professional.

Add:

```text
.github/workflows/ci.yml
```

CI should run:

```bash
python -m pip install -e .
python -m pytest tests
```

Future CD:

```text
.github/workflows/docker.yml
Dockerfile
docker-compose.yml
```

CD stages:

- build Docker image
- run tests
- run readiness report
- optionally publish image
- deploy to staging

## 8. Docker And Local Infra

Add reproducible local deployment.

Files:

```text
Dockerfile
docker-compose.yml
```

Compose services:

```text
enterprise-rag
qdrant
prometheus
```

Initial commands:

```bash
docker compose up qdrant
docker compose run enterprise-rag enterprise-rag readiness-report
```

## 9. Qdrant Or Weaviate Adapter

The current vector index is local and deterministic. Production should support a real vector database.

Recommended first adapter:

```text
Qdrant
```

Why Qdrant first:

- simple Docker setup
- clear collection model
- good local development experience

Potential files:

```text
src/enterprise_rag/vector_index/qdrant.py
src/enterprise_rag/vector_index/weaviate.py
```

Design principle:

```text
Keep vector storage behind the VectorIndex interface.
```

Interview framing:

```text
Local tests use in-memory search; production can swap in Qdrant or Weaviate through an adapter.
```

## 10. LLM Engine Adapter

The current answer generator is deterministic by default, with an OpenAI client stub.

Upgrade plan:

```text
src/enterprise_rag/llm/base.py
src/enterprise_rag/llm/openai_client.py
src/enterprise_rag/llm/local_stub.py
```

Support:

- provider selection through config
- timeout handling
- retry policy
- token accounting
- structured errors
- model name tracking

Do not hardwire one provider into core RAG logic.

## 11. LangChain And LlamaIndex Integration

LangChain and LlamaIndex should be optional adapters, not the core architecture.

Potential files:

```text
src/enterprise_rag/integrations/langchain_adapter.py
src/enterprise_rag/integrations/llamaindex_adapter.py
```

Good interview framing:

```text
I built the core retrieval architecture myself to show I understand the system, then added optional framework adapters for ecosystem compatibility.
```

## 12. Evaluation Framework Expansion

The current evaluation focuses on retrieval quality.

Expand to:

- retrieval eval
- answer faithfulness eval
- citation accuracy eval
- latency eval
- cost eval
- security eval
- regression eval
- A/B eval
- LLM-as-judge eval

Potential files:

```text
src/enterprise_rag/evaluation/answer_eval.py
src/enterprise_rag/evaluation/citation_eval.py
src/enterprise_rag/evaluation/security_eval.py
src/enterprise_rag/evaluation/ab_testing.py
src/enterprise_rag/evaluation/judge.py
```

## 13. Golden Sets

Golden sets are human-reviewed benchmark cases.

They should include:

- query
- expected answer or answer rubric
- expected evidence text
- expected chunk ids
- allowed citations
- tenant or access metadata if relevant
- category tags

Potential file:

```text
data/eval/golden_set.json
```

Important rule:

```text
Generated eval drafts are not golden sets until reviewed by humans.
```

## 14. LLM-As-Judge With Bias Controls

LLM-as-judge is useful but can be biased.

Bias controls:

- use a rubric
- blind model names
- randomize answer order
- use pairwise comparison when useful
- separate answer model from judge model
- require citation grounding checks
- sample human audits
- track judge disagreement

Potential module:

```text
src/enterprise_rag/evaluation/judge.py
```

Example rubric dimensions:

- correctness
- faithfulness
- citation support
- completeness
- refusal quality when evidence is insufficient

## 15. A/B Testing

A/B testing compares pipeline variants.

Example variants:

- baseline: BM25 + vector
- variant: BM25 + vector + graph
- variant: different top_k
- variant: different reranker
- variant: different prompt template
- variant: different compression policy

Metrics:

- Recall@K
- Precision@K
- MRR
- answer faithfulness
- citation accuracy
- latency
- cost
- insufficient evidence rate
- user feedback

Potential module:

```text
src/enterprise_rag/evaluation/ab_testing.py
```

## Implementation Priority

Recommended order:

1. GitHub Actions CI
2. Dockerfile
3. docker-compose with app and Qdrant
4. Prometheus metrics exporter
5. latency and cost tracing
6. prompt injection guard
7. tenant filtering
8. incremental indexing
9. OpenAPI service layer
10. Qdrant adapter
11. expanded evaluation framework
12. golden set support
13. LLM-as-judge with bias controls
14. A/B testing
15. optional LangChain and LlamaIndex adapters

## Interview Positioning

Use this positioning:

> Beyond the core RAG pipeline, I designed a production readiness roadmap covering latency, cost, prompt injection defense, incremental indexing, multi-tenant isolation, OpenAPI serving, Prometheus monitoring, CI/CD, golden-set evaluation, LLM-as-judge with bias controls, and A/B testing. I treat RAG as a production platform, not just a prompt-and-vector-search demo.

## Production Architecture Summary

```text
Core RAG
  -> ingestion
  -> query planning
  -> hybrid retrieval
  -> graph retrieval
  -> rerank
  -> compression
  -> answer generation
  -> citations

Reliability Layer
  -> tracing
  -> query logging
  -> retrieval eval
  -> self-healing workflow
  -> readiness report

Production Platform Layer
  -> CI/CD
  -> Docker
  -> OpenAPI
  -> Prometheus metrics
  -> latency/cost tracking
  -> prompt injection defense
  -> tenant isolation
  -> incremental indexing
  -> vector DB adapters
  -> golden sets
  -> LLM-as-judge
  -> A/B testing
```
