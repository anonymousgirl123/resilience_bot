from __future__ import annotations

import os
import time
from dotenv import load_dotenv
from app.llm.safe_wrapper import safe_generate


class FakeLLMClient:
    def __init__(self, mode: str):
        self.mode = mode
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_new_tokens: int = 200,
        top_p: float = 0.9,
    ) -> str:
        self.call_count += 1

        if self.mode == "success":
            return f"[OK] Reliable answer for: {prompt}"

        if self.mode == "retry_then_success":
            if self.call_count == 1:
                raise RuntimeError("Transient model backend failure")
            return f"[OK after retry] Reliable answer for: {prompt}"

        if self.mode == "timeout":
            time.sleep(8)
            return "This should not return before timeout."

        if self.mode == "always_fail":
            raise RuntimeError("Persistent backend failure")

        return "Unknown mode"


def run_case(mode: str):
    print(f"\n===== TEST CASE: {mode} =====")
    client = FakeLLMClient(mode)

    result = safe_generate(
        generate_fn=client.generate,
        prompt="My API has intermittent 504 timeouts. Give a troubleshooting checklist.",
        system_prompt="You are ResilienceBot.",
        temperature=0.2,
        max_new_tokens=120,
        top_p=0.9,
    )

    print("Answer:", result.answer)
    print("Success:", result.success)
    print("Used fallback:", result.used_fallback)
    print("Attempts:", result.attempts)
    print("Latency ms:", result.latency_ms)
    print("Error type:", result.error_type)
    print("Error message:", result.error_message)


def main():
    load_dotenv()

    original_timeout = os.getenv("RESBOT_LLM_TIMEOUT_SEC")
    try:
        os.environ["RESBOT_LLM_TIMEOUT_SEC"] = "5"

        run_case("success")
        run_case("retry_then_success")
        run_case("timeout")
        run_case("always_fail")
    finally:
        if original_timeout is None:
            os.environ.pop("RESBOT_LLM_TIMEOUT_SEC", None)
        else:
            os.environ["RESBOT_LLM_TIMEOUT_SEC"] = original_timeout


if __name__ == "__main__":
    main()
