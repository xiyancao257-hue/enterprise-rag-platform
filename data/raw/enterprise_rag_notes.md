# Enterprise RAG Notes

Enterprise RAG systems often fail because raw documents are parsed as plain text and then split with fixed-size chunks.
This loses headings, tables, figures, page references, and business metadata.

## Data Cleaning

Dirty data filtering removes empty pages, repeated headers and footers, broken OCR output, duplicated documents, and low-information text.
Cleaning at ingestion time improves retrieval quality before embeddings or reranking are involved.

## Structure-Aware Chunking

Structure-aware chunking preserves headings, paragraphs, tables, and image OCR text.
Tables should stay with their column names and surrounding explanation.
Images should be represented with OCR text, captions, and nearby context.

## Hybrid Retrieval

Hybrid retrieval combines BM25 keyword search with vector search.
BM25 is useful for exact terms, codes, product names, error messages, and compliance language.
Vector search is useful for semantic similarity and natural-language questions.

## Reranking And Compression

Reranking improves precision after broad recall.
Context compression selects only the evidence relevant to the user question before answer generation.

## Query Engine

A robust query engine can detect ambiguity, correct typos, rewrite questions, and expand short queries.
This helps reduce missed recall when users ask incomplete or messy questions.

## RAG Self-Healing

RAG self-healing logs failed queries, diagnoses whether the problem came from ingestion, chunking, retrieval, reranking, or answer generation, and suggests index or data fixes.

