from __future__ import annotations

import hashlib

from enterprise_rag.models import BlockType, Chunk, DocumentBlock
from enterprise_rag.text import tokenize


class StructureAwareChunker:
    def __init__(self, target_tokens: int = 220, max_tokens: int = 360) -> None:
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens

    def chunk(self, blocks: list[DocumentBlock]) -> list[Chunk]:
        chunks: list[Chunk] = []
        current: list[DocumentBlock] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current_tokens
            if not current:
                return
            chunks.append(self._make_chunk(current))
            current.clear()
            current_tokens = 0

        for block in blocks:
            block_tokens = len(tokenize(block.text))
            if block.block_type in {BlockType.TITLE, BlockType.HEADING}:
                flush()
                continue

            if block.block_type == BlockType.TABLE:
                flush()
                chunks.append(self._make_chunk([block]))
                continue

            if current_tokens + block_tokens > self.max_tokens:
                flush()

            current.append(block)
            current_tokens += block_tokens

            if current_tokens >= self.target_tokens:
                flush()

        flush()
        return chunks

    def _make_chunk(self, blocks: list[DocumentBlock]) -> Chunk:
        text_parts = []
        heading_path = blocks[-1].heading_path
        if heading_path:
            text_parts.append(" > ".join(heading_path))
        text_parts.extend(block.text for block in blocks)
        text = "\n\n".join(text_parts)
        fingerprint = "|".join(block.id for block in blocks)
        chunk_id = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
        return Chunk(
            id=chunk_id,
            document_id=blocks[0].document_id,
            text=text,
            heading_path=heading_path,
            source_blocks=tuple(block.id for block in blocks),
            metadata={
                **blocks[0].metadata,
                "chunking_strategy": "structure_aware",
                "chunk_target_tokens": str(self.target_tokens),
                "chunk_max_tokens": str(self.max_tokens),
            },
        )
