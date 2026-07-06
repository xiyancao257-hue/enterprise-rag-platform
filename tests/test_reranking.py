import pytest

from enterprise_rag.models import Chunk, SearchHit
from enterprise_rag.rag.pipeline import RagPipeline
from enterprise_rag.reranking.external import ExternalReranker, ExternalRerankerNotConfiguredError


class ReverseReranker:
    def rerank(self, query: str, hits: list[SearchHit], top_k: int = 5) -> list[SearchHit]:
        reversed_hits = list(reversed(hits))[:top_k]
        return [
            SearchHit(chunk=hit.chunk, score=hit.score, retriever=f"{hit.retriever}+reverse", rank=rank)
            for rank, hit in enumerate(reversed_hits, start=1)
        ]


def test_pipeline_accepts_reranker_interface() -> None:
    chunks = [
        Chunk(id="first", document_id="doc1", text="Hybrid retrieval uses BM25."),
        Chunk(id="second", document_id="doc1", text="Hybrid retrieval uses vector search."),
    ]

    answer = RagPipeline(chunks, reranker=ReverseReranker()).answer("hybrid retrieval", top_k=2)

    assert answer.citations
    assert all("reverse" in hit.retriever for hit in answer.citations)


def test_external_reranker_stub_raises_clear_error() -> None:
    reranker = ExternalReranker(provider="example")

    assert reranker.provider == "example"
    with pytest.raises(ExternalRerankerNotConfiguredError, match="adapter stub"):
        reranker.rerank("query", [])

