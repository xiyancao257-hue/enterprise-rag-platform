from __future__ import annotations

from enterprise_rag.cache.base import CacheStore
from enterprise_rag.graph.knowledge_graph import KnowledgeGraphBuilder
from enterprise_rag.models import Chunk, RagAnswer
from enterprise_rag.observability.tracing import QueryTrace, trace_hits
from enterprise_rag.query.engine import QueryEngine
from enterprise_rag.rag.answer_generation import AnswerGenerator, DeterministicAnswerGenerator
from enterprise_rag.rag.compression import ContextCompressor
from enterprise_rag.rag.prompt_security import PromptInjectionDetector
from enterprise_rag.reranking.base import Reranker
from enterprise_rag.retrieval.graph import GraphRetriever
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.rerank import LightweightReranker
from enterprise_rag.text import tokenize
from enterprise_rag.vector_index.base import VectorIndex


class RagPipeline:
    def __init__(
        self,
        chunks: list[Chunk],
        answer_generator: AnswerGenerator | None = None,
        reranker: Reranker | None = None,
        prompt_injection_detector: PromptInjectionDetector | None = None,
        enable_graph: bool = False,
        graph_max_hops: int = 2,
        vector_index: VectorIndex | None = None,
        embedding_cache: CacheStore | None = None,
    ) -> None:
        vocabulary = {token for chunk in chunks for token in tokenize(chunk.text)}
        extra_retrievers = []
        if enable_graph:
            graph = KnowledgeGraphBuilder().build(chunks)
            extra_retrievers.append(GraphRetriever(graph, max_hops=graph_max_hops))
        self.query_engine = QueryEngine(vocabulary=vocabulary)
        self.retriever = HybridRetriever(
            chunks,
            extra_retrievers=extra_retrievers,
            vector_index=vector_index,
            embedding_cache=embedding_cache,
        )
        self.reranker = reranker or LightweightReranker()
        self.prompt_injection_detector = prompt_injection_detector or PromptInjectionDetector()
        self.compressor = ContextCompressor()
        self.answer_generator = answer_generator or DeterministicAnswerGenerator()

    def answer(self, query: str, top_k: int = 5) -> RagAnswer:
        return self.answer_for_user(query, top_k=top_k)

    def answer_for_user(self, query: str, top_k: int = 5, user_groups: set[str] | None = None) -> RagAnswer:
        answer, _trace = self.answer_for_user_with_trace(query, top_k=top_k, user_groups=user_groups)
        return answer

    def answer_for_user_with_trace(
        self,
        query: str,
        top_k: int = 5,
        user_groups: set[str] | None = None,
        mandatory_metadata_filters: dict[str, str] | None = None,
    ) -> tuple[RagAnswer, QueryTrace]:
        plan = self.query_engine.plan(query)
        metadata_filters = {
            **plan.metadata_filters,
            **(mandatory_metadata_filters or {}),
        }
        retrieved = self.retriever.search(
            list(plan.rewritten_queries),
            top_k=max(top_k * 2, 8),
            metadata_filters=metadata_filters,
            user_groups=user_groups,
        )
        reranked = self.reranker.rerank(plan.normalized_query, retrieved, top_k=top_k)
        prompt_security = self.prompt_injection_detector.filter_hits(reranked)
        compressed = self.compressor.compress(plan.normalized_query, prompt_security.safe_hits)
        answer_text = self.answer_generator.generate(plan.normalized_query, compressed)
        answer = RagAnswer(query=query, answer=answer_text, citations=tuple(compressed), query_plan=plan)
        trace = QueryTrace(
            original_query=plan.original_query,
            normalized_query=plan.normalized_query,
            rewritten_queries=plan.rewritten_queries,
            metadata_filters=metadata_filters,
            retrieved=trace_hits(retrieved),
            reranked=trace_hits(reranked),
            blocked_context=trace_hits(prompt_security.blocked_hits),
            final_context=trace_hits(compressed),
        )
        return answer, trace
