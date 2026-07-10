from __future__ import annotations

import hashlib
import json
from pathlib import Path


def build_query_cache_key(
    *,
    query: str,
    tenant_id: str | None,
    user_groups: set[str],
    metadata_filters: dict[str, str],
    top_k: int,
    index_path: Path,
    retrieval_profile: dict[str, object] | None = None,
) -> str:
    payload = {
        "query": query,
        "tenant_id": tenant_id,
        "user_groups": sorted(user_groups),
        "metadata_filters": dict(sorted(metadata_filters.items())),
        "top_k": top_k,
        "index_version": index_version(index_path),
        "retrieval_profile": retrieval_profile or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"query:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def index_version(index_path: Path) -> str:
    if not index_path.exists():
        return "missing"
    stat = index_path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"
