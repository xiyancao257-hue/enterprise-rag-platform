from __future__ import annotations

import hashlib
import re

from enterprise_rag.models import BlockType, Document, DocumentBlock

MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")


class StructureParser:
    def parse(self, document: Document) -> list[DocumentBlock]:
        blocks: list[DocumentBlock] = []
        heading_stack: list[str] = []
        buffer: list[str] = []
        table_buffer: list[str] = []

        def flush_paragraph() -> None:
            if not buffer:
                return
            text = "\n".join(buffer).strip()
            buffer.clear()
            if text:
                blocks.append(self._block(document, BlockType.PARAGRAPH, text, heading_stack))

        def flush_table() -> None:
            if not table_buffer:
                return
            text = "\n".join(table_buffer).strip()
            table_buffer.clear()
            if text:
                blocks.append(self._block(document, BlockType.TABLE, text, heading_stack))

        for line in document.text.splitlines():
            heading = MARKDOWN_HEADING_RE.match(line)
            if heading:
                flush_paragraph()
                flush_table()
                level = len(heading.group(1))
                title = heading.group(2).strip()
                heading_stack = heading_stack[: level - 1] + [title]
                block_type = BlockType.TITLE if level == 1 and not blocks else BlockType.HEADING
                blocks.append(self._block(document, block_type, title, tuple(heading_stack)))
                continue

            if TABLE_LINE_RE.match(line):
                flush_paragraph()
                table_buffer.append(line)
                continue

            if not line.strip():
                flush_paragraph()
                flush_table()
                continue

            flush_table()
            buffer.append(line)

        flush_paragraph()
        flush_table()
        return blocks

    def _block(
        self,
        document: Document,
        block_type: BlockType,
        text: str,
        heading_path: list[str] | tuple[str, ...],
    ) -> DocumentBlock:
        fingerprint = f"{document.id}:{block_type}:{text}:{len(text)}"
        block_id = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
        return DocumentBlock(
            id=block_id,
            document_id=document.id,
            block_type=block_type,
            text=text,
            heading_path=tuple(heading_path),
            metadata={**document.metadata, "source_path": document.source_path},
        )
