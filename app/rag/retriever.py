from __future__ import annotations

from typing import Any, Dict, List, Optional
import os


class LocalRetriever:
    """Simplified local retriever for RAG quality tests and demo purposes.

    The real implementation would load an embedding model and a vector
    index from disk (paths configured via environment variables) and perform
    semantic similarity search. For this exercise we provide a lightweight
    keyword-based fallback so that the test suite and bot can run without
    external data or heavy dependencies.

    The retriever exposes a single method :meth:`retrieve` that accepts a
    query string and an optional ``filters`` dict.  Results are returned as a
    list of chunks where each chunk is a dict containing at least
    ``source``, ``text`` and ``score`` keys; this matches the contract used
    elsewhere in the codebase (see :mod:`app.rag.context_assembler`).
    """

    # small corpus used by the tests
    _CORPUS: List[Dict[str, Any]] = [
        {
            "source": "incident_504_timeouts.txt",
            "text": (
                "An API gateway started returning 504 errors due to an upstream "
                "service being overloaded. The incident report includes timestamps, "
                "request headers, and user impact information."
            ),
            "metadata": {"doc_type": "incident"},
            "keywords": ["504", "timeout", "api gateway", "intermittent"],
        },
        {
            "source": "incident_retry_storm.txt",
            "text": (
                "A cascade of retries overwhelmed the database and caused "
                "multiple services to fail. The incident timeline shows the initial "
                "event, retry configuration, and mitigation steps."
            ),
            "metadata": {"doc_type": "incident", "tags": ["retries"]},
            "keywords": ["retry", "retry storm", "backoff", "cascading"],
        },
        {
            "source": "runbook_db_latency.txt",
            "text": (
                "When database query latency spikes, check indexes, review "
                "slow query logs, and monitor connection pool usage. This runbook also "
                "covers common causes such as locks and long-running transactions."
            ),
            "metadata": {"doc_type": "runbook", "tags": ["db"]},
            "keywords": ["database", "db", "query", "latency", "slow"],
        },
        {
            "source": "runbook_tls_dns.txt",
            "text": (
                "Verify TLS certificates, DNS resolution, and network paths "
                "when encountering handshake failures. This runbook walks through steps to "
                "validate configuration and rotation procedures."
            ),
            "metadata": {"doc_type": "runbook", "tags": ["tls"]},
            "keywords": ["tls", "dns", "handshake", "certificate"],
        },

    ]

    def __init__(self) -> None:
        # the constructor is intentionally lightweight; in a real system
        # you'd load the embedding model and vector index here, using
        # environment variables for configuration.  we ensure that the
        # surrounding code can call ``LocalRetriever()`` without arguments.
        env_hint = os.getenv("RETRIEVER_INFO", "")
        if env_hint:
            # just log if someone has configured something
            print(f"[INFO] LocalRetriever loaded config: {env_hint}")

    def retrieve(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Return a list of chunks ranked by relevance to *query*.

        *filters* is an optional metadata dictionary.  Entries that do not
        satisfy the filters are omitted before ranking.  This mirrors the
        behavior expected by :class:`app.core.bot.ResilienceBot`.
        """
        # split off any reformulated context terms ("| context: ...")
        base = query.split("|")[0].strip().lower()

        def matches_filters(entry: Dict[str, Any]) -> bool:
            if not filters:
                return True
            meta = entry.get("metadata", {})
            for key, value in filters.items():
                if key not in meta:
                    return False
                if isinstance(value, list):
                    entry_val = meta[key]
                    if isinstance(entry_val, list):
                        if not all(v in entry_val for v in value):
                            return False
                    else:
                        # metadata value not a list but filter expects list
                        return False
                else:
                    if meta[key] != value:
                        return False
            return True

        candidates: List[Dict[str, Any]] = []
        for entry in self._CORPUS:
            if not matches_filters(entry):
                continue

            # count how many keyword phrases appear in the query text
            keywords = entry.get("keywords", [])
            match_count = sum(1 for kw in keywords if kw in base)

            if match_count == 0:
                # no obvious signal, give a small default score so entry still
                # appears but ranks lower than anything with a hit
                score = 0.05
            else:
                # normalize by number of keywords so that documents with more
                # associated terms don't automatically dominate
                score = match_count / len(keywords)

            candidates.append(
                {"source": entry["source"], "text": entry["text"], "score": score}
            )

        # sort highest score first, deterministic tie-breaker by source name
        candidates.sort(key=lambda c: (-c["score"], c["source"]))
        return candidates
