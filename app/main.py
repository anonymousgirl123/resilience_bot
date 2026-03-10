from __future__ import annotations

import sys
import time
from dotenv import load_dotenv

from app.clients.hf_pipeline_local import HFPipelineLocalClient
from app.core.bot import ResilienceBot
from app.llm.safe_wrapper import safe_generate


def main():
    load_dotenv()

    if len(sys.argv) < 2:
        print('Usage: python -m app.main "your question"')
        return

    question = " ".join(sys.argv[1:])
    app_start = time.perf_counter()

    client = HFPipelineLocalClient()
    bot = ResilienceBot(client)

    def bot_generate(
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_new_tokens: int = 120,
        top_p: float = 0.9,
    ) -> str:
        resp = bot.ask(prompt)
        return resp.text

    result = safe_generate(
        generate_fn=bot_generate,
        prompt=question,
        system_prompt=None,
        temperature=0.2,
        max_new_tokens=120,
        top_p=0.9,
    )

    total_latency_ms = int((time.perf_counter() - app_start) * 1000)

    print("\nResilienceBot:\n")
    print(result.answer)
    print("\n---")
    print(f"Success: {result.success}")
    print(f"Used fallback: {result.used_fallback}")
    print(f"Attempts: {result.attempts}")
    print(f"Latency: {total_latency_ms} ms")

    if result.error_type:
        print(f"Error type: {result.error_type}")
    if result.error_message:
        print(f"Error message: {result.error_message}")


if __name__ == "__main__":
    main()
