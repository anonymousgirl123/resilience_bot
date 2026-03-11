from __future__ import annotations

import re
from typing import Any, Dict, Optional

from app.core.schemas import BotResponse
from app.rag.query_reformulator import reformulate
from app.rag.context_assembler import assemble_context, summarise_context


def _clean_answer(text: str) -> str:
    text = text.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    items: list[str] = []
    seen: set[str] = set()
    for line in lines:
        match = re.match(r"^(?:\d+[\.\)]\s*|-+\s*)(.+)$", line)
        candidate = match.group(1).strip() if match else line.strip()
        lowered = candidate.lower().rstrip(":")
        if not candidate:
            continue
        if lowered.startswith(("answer", "question", "checklist", "bullet point",
                                "here are", "context", "relevant", "based on")):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        items.append(candidate[0].upper() + candidate[1:])
    if len(items) < 3:
        chunks = re.split(r"\s(?=\d+[\.\)])", text)
        reparsed: list[str] = []
        seen = set()
        for chunk in chunks:
            match = re.match(r"^\d+[\.\)]\s*(.+)$", chunk.strip())
            if match:
                candidate = match.group(1).strip()
                lowered = candidate.lower().rstrip(":")
                if candidate and lowered not in seen:
                    seen.add(lowered)
                    reparsed.append(candidate[0].upper() + candidate[1:])
        if len(reparsed) >= 3:
            items = reparsed
    items = items[:5]
    return "\n".join(f"- {item}" for item in items)


def _detect_filters(question: str) -> Optional[Dict[str, Any]]:
    """
    Lightweight heuristic filter detection from Day 2 Lab 3.
    Detects explicit intent signals in the question text.
    Returns a filters dict for the retriever, or None for no filtering.
    """
    q = question.lower()
    if any(kw in q for kw in ["runbook", "procedure", "step-by-step", "how do i"]):
        return {"doc_type": "runbook"}
    if any(kw in q for kw in ["incident", "postmortem", "what happened", "timeline"]):
        return {"doc_type": "incident"}
    if any(kw in q for kw in ["database", "db", "query", "postgres", "sql",
                               "slow query", "connection pool"]):
        return {"tags": ["db"]}
    if any(kw in q for kw in ["tls", "ssl", "certificate", "dns", "handshake"]):
        return {"tags": ["tls"]}
    if any(kw in q for kw in ["retry storm", "retry", "backoff", "jitter",
                               "circuit breaker"]):
        return {"tags": ["retries"]}
    return None


class ResilienceBot:

    def __init__(self, client, retriever=None) -> None:
        self.client    = client
        self.retriever = retriever

    def ask(self, question: str) -> BotResponse:
        # ── 1. Query reformulation ──────────────────────────────────
        retrieval_query = reformulate(question)
        print(f"[INFO] Reformulated query: {retrieval_query[:120]}")

        # ── 2. Retrieval with metadata filtering ────────────────────
        # _detect_filters and filters= are preserved from Day 2 Lab 3.
        # The only change: we pass retrieval_query instead of question.
        # LocalRetriever.retrieve(query, filters=None) — signature unchanged.
        context_block = ""
        if self.retriever is not None:
            filters = _detect_filters(question)
            if filters:
                print(f"[INFO] Detected filters: {filters}")
            chunks = self.retriever.retrieve(retrieval_query, filters=filters)
            print(summarise_context(chunks))
            # ── 3. Structured context assembly ──────────────────────
            # Replaces the raw text dump from Day 2 Lab 3 with a
            # ranked, labelled context block.
            context_block = assemble_context(chunks)

        # ── 4. Build prompt ─────────────────────────────────────────
        # Prompt structure preserved from Day 2 Lab 3.
        # context_block is now structured instead of a raw text dump.
        prompt = (
            "Task: Write a troubleshooting checklist for a reliability engineering question.\n"
            "Return exactly 5 short numbered items.\n"
            "Each item must be practical and action-oriented.\n"
            "Focus on API failures, 504 timeouts, retries, retry storms, outages, "
            "upstream dependencies, recent deployments, configuration changes, "
            "logs, metrics, and traces.\n"
            "Do not write headings. Do not write explanations. Do not write paragraphs.\n\n"
            + context_block
            + "\n\nExample:\n"
            "Question: Our service latency increased after a deployment. What should I check?\n"
            "Checklist:\n"
            "1. Check recent deployments and configuration changes.\n"
            "2. Review service and dependency latency metrics.\n"
            "3. Inspect logs for new errors or timeout spikes.\n"
            "4. Verify retry and timeout settings between services.\n"
            "5. Use traces to identify the slowest downstream component.\n\n"
            f"Question: {question}\n"
            "Checklist:\n"
        )

        # ── 5. Generate ─────────────────────────────────────────────
        # HFPipelineLocalClient.generate() signature (unchanged from Day 2):
        #   generate(self, prompt, system_prompt=None, temperature=0.2,
        #            max_new_tokens=180, top_p=0.9) -> str
        answer = self.client.generate(
            prompt=prompt,
            temperature=0.2,
            max_new_tokens=180,
            top_p=0.9,
        )

        cleaned = _clean_answer(answer)
        final_text = cleaned if cleaned.strip() else answer.strip()
        return BotResponse(text=final_text)
