from __future__ import annotations

from dotenv import load_dotenv

from app.rag.retriever import LocalRetriever
from app.rag.query_reformulator import reformulate
from app.rag.context_assembler import assemble_context, summarise_context

TEST_QUESTIONS = [
    "My API has intermittent 504 timeouts. Give a troubleshooting checklist.",
    "We are seeing a retry storm causing cascading failures. What should I check?",
    "Database query latency is spiking. How do I diagnose this?",
    "TLS handshake failures are appearing intermittently. What do I verify?",
]

EXPECTED_SOURCES = [
    "incident_504_timeouts.txt",
    "incident_retry_storm.txt",
    "runbook_db_latency.txt",
    "runbook_tls_dns.txt",
]

SEP  = "=" * 70
LINE = "-" * 70


def run_comparison(retriever: LocalRetriever) -> None:
    print("\n" + SEP)
    print("RAG OUTPUT QUALITY COMPARISON")
    print(SEP)

    for question, expected in zip(TEST_QUESTIONS, EXPECTED_SOURCES):
        print("\n" + LINE)
        print(f"Question : {question}")
        print(f"Expected : {expected}")

        # ── Baseline: raw question, no reformulation ───────────────
        baseline_chunks = retriever.retrieve(question)
        baseline_top    = baseline_chunks[0] if baseline_chunks else {}
        baseline_hit    = baseline_top.get("source", "") == expected
        print("\n  BASELINE (raw query)")
        print(f"    {summarise_context(baseline_chunks)}")
        print(f"    Top source match : {'PASS' if baseline_hit else 'FAIL'}")
        print(f"    Context preview  : {baseline_top.get('text', '')[:120]}...")

        # ── Improved: reformulated query + structured context ──────
        reformed_query  = reformulate(question)
        improved_chunks = retriever.retrieve(reformed_query)
        improved_top    = improved_chunks[0] if improved_chunks else {}
        improved_hit    = improved_top.get("source", "") == expected
        structured_ctx  = assemble_context(improved_chunks)
        print("\n  IMPROVED (reformulated query + structured context)")
        print(f"    Reformulated query : {reformed_query[:100]}")
        print(f"    {summarise_context(improved_chunks)}")
        print(f"    Top source match   : {'PASS' if improved_hit else 'FAIL'}")
        print("    Structured context :")
        for line in structured_ctx.splitlines()[:8]:
            print(f"      {line}")

    print("\n" + SEP)


def main() -> None:
    load_dotenv()
    print("Loading embedding model and index...")
    # LocalRetriever() loads embedder and index from .env — no args needed
    retriever = LocalRetriever()
    run_comparison(retriever)


if __name__ == "__main__":
    main()
