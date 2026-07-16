# Prompting And RAG Evaluation Notes

This note maps common AI engineering interview topics to the current Enterprise RAG project.
The goal is to show what is implemented, what is intentionally handled by architecture instead of
prompt tricks, and what should stay on the roadmap until there is real training or evaluation data.

## What Is Already Implemented

| Interview topic | Project status | Where to look |
| --- | --- | --- |
| RAG overview | Implemented as an end-to-end pipeline: ingest, retrieve, compress, generate, cite, trace, evaluate. | `src/enterprise_rag/rag/pipeline.py`, `docs/architecture.md` |
| Retrieval | Implemented with BM25, vector retrieval, hybrid fusion, graph retrieval, rerank, and metadata filtering. | `src/enterprise_rag/retrieval/`, `src/enterprise_rag/vector_index/`, `src/enterprise_rag/graph/` |
| Document parsing and chunking | Implemented with loaders, OCR adapter path, cleaning, redaction, structure parsing, table-aware chunks, and configurable chunking. | `src/enterprise_rag/ingestion/`, `src/enterprise_rag/processing/` |
| Indexing | Implemented with keyword/BM25, vector indexing, Qdrant adapter, knowledge graph index, source sync manifest, and index versioning. | `src/enterprise_rag/retrieval/bm25.py`, `src/enterprise_rag/vector_index/`, `src/enterprise_rag/indexing/`, `src/enterprise_rag/storage/index_version.py` |
| Generation | Implemented with grounded answer generation and provider adapters for stub or OpenAI. | `src/enterprise_rag/rag/answer_generation.py`, `src/enterprise_rag/llm/` |
| Prompt engineering for RAG | Implemented as a grounded QA prompt template with evidence-only answering, insufficient-evidence behavior, and citation requirements. | `src/enterprise_rag/rag/prompts.py`, `tests/test_prompts.py` |
| Evaluation | Implemented with retrieval metrics, experiment comparison, readiness reports, LLM-as-judge support, and human-in-the-loop self-healing. | `src/enterprise_rag/evaluation/`, `data/eval/` |

## Prompt Engineering Choices

The production path uses a zero-shot grounded prompt. It gives the model the task, the user question,
the retrieved evidence, and citation rules without relying on examples. This is a good default for
RAG because the evidence changes on every query.

Few-shot prompting is useful when the answer format is difficult or highly domain-specific. For this
project, it should be added as a configurable prompt variant and evaluated through A/B testing before
becoming the default. Good few-shot examples should teach output shape, refusal behavior, and citation
style, not inject facts that could conflict with retrieved evidence.

Chain-of-thought prompting should not be exposed to users or stored as an answer artifact. In an
enterprise RAG system, use query plans, traces, citations, and evaluation reports for explainability.
The final answer should be concise and grounded, while internal reasoning remains private.

Role-specific and user-context prompting should be treated carefully. Tenant, user, group, and role
identity come from trusted API headers and access policy, not from natural-language query text. The
prompt can receive safe presentation context, but authorization decisions must happen before generation.

## Evaluation Mapping

Context relevance asks whether the retrieved evidence is useful for the query. In this project, it is
measured mostly by Recall@K, Precision@K, MRR, retrieval experiments, and diagnostics.

Faithfulness asks whether the answer is supported by the retrieved evidence. This project supports it
through grounded prompts, citations, prompt-injection filtering, and LLM-as-judge scaffolding. For
production, faithfulness should be checked on a golden set and sampled human reviews.

Answer correctness asks whether the final answer is actually correct for the business question. This
requires golden answers, human review for important cases, and blinded LLM-as-judge checks with bias
controls. Retrieval can be perfect while the answer is still incomplete, so answer correctness should
be evaluated separately from retrieval quality.

## RAFT Roadmap

RAFT, or Retrieval-Augmented Fine-Tuning, is a training technique where the model learns to answer
questions using retrieved documents and to ignore irrelevant distractor documents. It is not needed for
the current portfolio implementation because the project is focused on production RAG architecture,
not model training.

Add RAFT only when you have:

- a stable domain-specific golden set
- reviewed answers and evidence
- representative distractor documents
- enough repeated failure patterns to justify training
- a deployment plan for evaluating the fine-tuned model against the base model

Until then, the better production investment is better ingestion, retrieval, evaluation, observability,
access control, latency, and cost controls.

## Interview Summary

The project already covers the most important RAG engineering concepts. Prompting is represented by a
grounded zero-shot template and can be extended with few-shot variants. Evaluation covers retrieval
quality today and has the right hooks for faithfulness and answer correctness. RAFT is best presented
as a future supervised-training upgrade, not as a required feature for this architecture.
