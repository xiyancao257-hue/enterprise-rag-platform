from enterprise_rag.config import AppConfig, load_config, parse_config


def test_load_config_without_path_uses_defaults() -> None:
    config = load_config()

    assert config == AppConfig()
    assert config.retrieval.top_k == 5
    assert config.retrieval.enable_graph is False
    assert config.retrieval.graph_max_hops == 2
    assert config.security.default_user_groups == ()
    assert config.vector_index.provider == "memory"


def test_parse_config_loads_retrieval_and_security_settings() -> None:
    config = parse_config(
        {
            "retrieval": {
                "top_k": 8,
                "enable_graph": True,
                "graph_max_hops": 3,
                "experiment_k_values": [2, 4, 6],
            },
            "security": {
                "default_user_groups": ["engineering", "support"],
            },
            "vector_index": {
                "provider": "qdrant",
                "collection_name": "chunks",
                "url": "http://qdrant:6333",
            },
        }
    )

    assert config.retrieval.top_k == 8
    assert config.retrieval.enable_graph is True
    assert config.retrieval.graph_max_hops == 3
    assert config.retrieval.experiment_k_values == (2, 4, 6)
    assert config.security.default_user_groups == ("engineering", "support")
    assert config.vector_index.provider == "qdrant"
    assert config.vector_index.collection_name == "chunks"
    assert config.vector_index.url == "http://qdrant:6333"


def test_load_config_from_json_file(tmp_path) -> None:
    config_path = tmp_path / "enterprise-rag.json"
    config_path.write_text(
        """
        {
          "retrieval": {
            "top_k": 3,
            "enable_graph": true
          },
          "security": {
            "default_user_groups": ["admin"]
          }
        }
        """,
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.retrieval.top_k == 3
    assert config.retrieval.enable_graph is True
    assert config.retrieval.graph_max_hops == 2
    assert config.security.default_user_groups == ("admin",)
