from enterprise_rag.models import Chunk
from enterprise_rag.security.access_control import AccessContext, AccessPolicy


def test_access_policy_allows_public_chunk() -> None:
    chunk = Chunk(id="public", document_id="doc1", text="Public doc.")

    assert AccessPolicy().can_read(chunk, AccessContext()) is True


def test_access_policy_requires_allowed_group() -> None:
    chunk = Chunk(
        id="security",
        document_id="doc1",
        text="Security doc.",
        metadata={"allowed_groups": "security"},
    )

    assert AccessPolicy().can_read(chunk, AccessContext(groups={"security"})) is True
    assert AccessPolicy().can_read(chunk, AccessContext(groups={"finance"})) is False


def test_access_policy_supports_user_and_role_allows() -> None:
    user_chunk = Chunk(
        id="user",
        document_id="doc1",
        text="User doc.",
        metadata={"allowed_users": "alice"},
    )
    role_chunk = Chunk(
        id="role",
        document_id="doc1",
        text="Role doc.",
        metadata={"allowed_roles": "auditor"},
    )

    assert AccessPolicy().can_read(user_chunk, AccessContext(user_id="alice")) is True
    assert AccessPolicy().can_read(role_chunk, AccessContext(roles={"auditor"})) is True


def test_access_policy_denies_take_precedence() -> None:
    chunk = Chunk(
        id="deny",
        document_id="doc1",
        text="Sensitive doc.",
        metadata={"allowed_groups": "security", "denied_users": "alice", "denied_roles": "contractor"},
    )

    assert AccessPolicy().can_read(chunk, AccessContext(user_id="alice", groups={"security"})) is False
    assert AccessPolicy().can_read(chunk, AccessContext(groups={"security"}, roles={"contractor"})) is False
    assert AccessPolicy().can_read(chunk, AccessContext(user_id="bob", groups={"security"})) is True


def test_access_policy_enforces_tenant_boundary() -> None:
    chunk = Chunk(
        id="tenant",
        document_id="doc1",
        text="Tenant doc.",
        metadata={"tenant_id": "acme"},
    )

    assert AccessPolicy().can_read(chunk, AccessContext(tenant_id="acme")) is True
    assert AccessPolicy().can_read(chunk, AccessContext(tenant_id="globex")) is False
