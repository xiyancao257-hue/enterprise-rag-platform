from enterprise_rag.vector_index.in_memory import InMemoryVectorIndex


def test_in_memory_vector_index_adds_and_searches_vectors() -> None:
    index = InMemoryVectorIndex()
    index.add("hybrid", [1.0, 0.0])
    index.add("cleaning", [0.0, 1.0])

    results = index.search([1.0, 0.0], top_k=2)

    assert [result.id for result in results] == ["hybrid"]
    assert results[0].score == 1.0
    assert results[0].rank == 1


def test_in_memory_vector_index_replaces_existing_vector() -> None:
    index = InMemoryVectorIndex()
    index.add("chunk", [0.0, 1.0])
    index.add("chunk", [1.0, 0.0])

    results = index.search([1.0, 0.0], top_k=1)

    assert results[0].id == "chunk"
    assert results[0].score == 1.0

