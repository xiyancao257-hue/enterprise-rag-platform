from __future__ import annotations


def recall_at_k(expected_chunk_ids: set[str], retrieved_chunk_ids: list[str], k: int) -> float:
    if not expected_chunk_ids:
        return 0.0
    return len(expected_chunk_ids & set(retrieved_chunk_ids[:k])) / len(expected_chunk_ids)


def precision_at_k(expected_chunk_ids: set[str], retrieved_chunk_ids: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(expected_chunk_ids & set(retrieved_chunk_ids[:k])) / k


def reciprocal_rank(expected_chunk_ids: set[str], retrieved_chunk_ids: list[str]) -> float:
    for index, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in expected_chunk_ids:
            return 1 / index
    return 0.0
