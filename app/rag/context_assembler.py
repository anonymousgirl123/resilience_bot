from __future__ import annotations

from typing import Any, Dict, List


def assemble_context(
    chunks:    List[Dict[str, Any]],
    max_chars: int   = 1200,
    min_score: float = 0.20,
) -> str:
    """
    Convert a list of retrieved chunks into a structured context block.

    Each chunk is labelled with its rank, source document, similarity
    score, and text. Chunks below min_score are excluded as noise.
    The total assembled text is capped at max_chars to avoid overflowing
    the model's context window.
    """
    if not chunks:
        return "(No relevant context retrieved.)"

    filtered = [c for c in chunks if c.get("score", 0.0) >= min_score]
    if not filtered:
        return "(No relevant context above confidence threshold.)"

    lines: list[str] = ["Retrieved Context:"]
    total_chars = 0

    for rank, chunk in enumerate(filtered, start=1):
        source = chunk.get("source", "unknown")
        score  = chunk.get("score",  0.0)
        text   = (chunk.get("text", "") or "").strip()

        header = f"[{rank}] Source: {source} (relevance: {score:.2f})"
        entry  = f"{header}\n{text}"

        if total_chars + len(entry) > max_chars:
            remaining = max_chars - total_chars - len(header) - 5
            if remaining > 40:
                lines.append(header)
                lines.append(text[:remaining] + "...")
            break

        lines.append(entry)
        lines.append("")   # blank separator between chunks
        total_chars += len(entry) + 1

    return "\n".join(lines).strip()


def summarise_context(chunks: List[Dict[str, Any]]) -> str:
    """One-line retrieval summary for printing and debugging."""
    if not chunks:
        return "retrieval=empty"
    top = chunks[0]
    return (
        f"retrieval=ok  top_source={top.get('source', '?')}  "
        f"top_score={top.get('score', 0.0):.3f}  total_chunks={len(chunks)}"
    )
