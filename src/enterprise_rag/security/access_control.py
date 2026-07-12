from __future__ import annotations

from dataclasses import dataclass, field

from enterprise_rag.models import Chunk


@dataclass(frozen=True)
class AccessContext:
    tenant_id: str | None = None
    user_id: str | None = None
    groups: set[str] = field(default_factory=set)
    roles: set[str] = field(default_factory=set)


class AccessPolicy:
    def can_read(self, chunk: Chunk, context: AccessContext) -> bool:
        metadata = chunk.metadata
        chunk_tenant = metadata.get("tenant_id")
        if context.tenant_id is not None and chunk_tenant != context.tenant_id:
            return False

        if self._matches_denies(metadata, context):
            return False

        allow_users = _parse_csv(metadata.get("allowed_users", ""))
        allow_groups = _parse_csv(metadata.get("allowed_groups", ""))
        allow_roles = _parse_csv(metadata.get("allowed_roles", ""))
        if not allow_users and not allow_groups and not allow_roles:
            return True

        return (
            (context.user_id is not None and context.user_id in allow_users)
            or bool(context.groups & allow_groups)
            or bool(context.roles & allow_roles)
        )

    def _matches_denies(self, metadata: dict[str, str], context: AccessContext) -> bool:
        denied_users = _parse_csv(metadata.get("denied_users", ""))
        denied_groups = _parse_csv(metadata.get("denied_groups", ""))
        denied_roles = _parse_csv(metadata.get("denied_roles", ""))
        return (
            (context.user_id is not None and context.user_id in denied_users)
            or bool(context.groups & denied_groups)
            or bool(context.roles & denied_roles)
        )


def _parse_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}
