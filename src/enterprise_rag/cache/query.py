from __future__ import annotations

import hashlib
import json
from pathlib import Path

from enterprise_rag.storage.index_version import fallback_index_version


def build_query_cache_key(
    *,
    query: str,
    tenant_id: str | None,
    user_groups: set[str],
    user_id: str | None = None,
    user_roles: set[str] | None = None,
    metadata_filters: dict[str, str],
    top_k: int,
    index_path: Path,
    index_version_id: str | None = None,
    retrieval_profile: dict[str, object] | None = None,
) -> str:
    payload = {
        "query": query,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "user_groups": sorted(user_groups),
        "user_roles": sorted(user_roles or set()),
        "metadata_filters": dict(sorted(metadata_filters.items())),
        "top_k": top_k,
        "index_version": index_version_id or index_version(index_path),
        "retrieval_profile": retrieval_profile or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"query:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def index_version(index_path: Path) -> str:
    return fallback_index_version(index_path)
