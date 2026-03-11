from __future__ import annotations

# Keyword sets for each SRE investigation layer.
# These are injected into the query to broaden its semantic coverage
# without changing the user's original intent.
_LAYER_KEYWORDS: dict[str, list[str]] = {
    "gateway":  ["api gateway", "load balancer", "upstream timeout", "502", "504"],
    "upstream": ["upstream service", "latency", "error rate", "service dependency"],
    "app":      ["connection pool", "thread starvation", "timeout", "queue depth"],
    "db":       ["database", "slow query", "index", "lock wait", "connection limit"],
    "network":  ["dns", "tls", "certificate", "handshake", "packet loss"],
    "config":   ["deploy", "config change", "rollback", "feature flag", "env var"],
    "retry":    ["retry storm", "backoff", "jitter", "circuit breaker", "cascading"],
}


def _detect_layers(question: str) -> list[str]:
    """Return layer keys whose trigger words appear in the question."""
    q = question.lower()
    matched = []
    for layer, keywords in _LAYER_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            matched.append(layer)
    return matched


def reformulate(question: str, max_extra_terms: int = 12) -> str:
    """
    Expand a short engineer question into a richer retrieval query.

    Strategy:
    1. Keep the original question as the anchor.
    2. Detect which SRE investigation layers are implied.
    3. Append relevant layer keywords as additional context terms.
    4. Cap extra terms to avoid over-diluting the original signal.
    """
    layers = _detect_layers(question)
    if not layers:
        layers = ["gateway", "upstream", "app"]

    extra_terms: list[str] = []
    for layer in layers:
        extra_terms.extend(_LAYER_KEYWORDS[layer])

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_terms: list[str] = []
    for term in extra_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)

    unique_terms = unique_terms[:max_extra_terms]

    if unique_terms:
        return f"{question} | context: {', '.join(unique_terms)}"
    return question
