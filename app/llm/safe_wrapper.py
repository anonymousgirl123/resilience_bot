from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from app.llm.schemas import LLMCallResult


def _run_with_timeout(func, timeout_sec: float, *args, **kwargs):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout_sec)
    except FuturesTimeoutError:
        future.cancel()
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def safe_generate(
    generate_fn,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.2,
    max_new_tokens: int = 200,
    top_p: float = 0.9,
) -> LLMCallResult:
    timeout_sec = float(os.getenv("RESBOT_LLM_TIMEOUT_SEC", "75"))
    max_retries = int(os.getenv("RESBOT_LLM_MAX_RETRIES", "2"))
    backoff_base = float(os.getenv("RESBOT_LLM_BACKOFF_BASE_SEC", "1.5"))
    fallback_message = os.getenv(
        "RESBOT_LLM_FALLBACK_MESSAGE",
        "ResilienceBot could not generate a reliable response right now. Please retry."
    )

    total_attempts = max_retries + 1
    start_total = time.perf_counter()

    last_error_type = None
    last_error_message = None

    for attempt in range(1, total_attempts + 1):
        try:
            start_attempt = time.perf_counter()

            answer = _run_with_timeout(
                generate_fn,
                timeout_sec,
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
            )

            total_latency_ms = int((time.perf_counter() - start_total) * 1000)
            attempt_latency_ms = int((time.perf_counter() - start_attempt) * 1000)

            print(f"[INFO] Attempt {attempt}/{total_attempts} succeeded in {attempt_latency_ms} ms")

            return LLMCallResult(
                answer=answer,
                success=True,
                used_fallback=False,
                attempts=attempt,
                latency_ms=total_latency_ms,
                error_type=None,
                error_message=None,
            )

        except FuturesTimeoutError:
            last_error_type = "TimeoutError"
            last_error_message = f"Model call exceeded {timeout_sec} seconds"
            print(f"[WARN] Attempt {attempt}/{total_attempts} timed out")

        except Exception as ex:
            last_error_type = type(ex).__name__
            last_error_message = str(ex)
            print(f"[WARN] Attempt {attempt}/{total_attempts} failed: {last_error_type}: {last_error_message}")

        if attempt < total_attempts:
            sleep_sec = backoff_base * (2 ** (attempt - 1))
            print(f"[INFO] Waiting {sleep_sec:.1f}s before retry...")
            time.sleep(sleep_sec)

    total_latency_ms = int((time.perf_counter() - start_total) * 1000)

    return LLMCallResult(
        answer=fallback_message,
        success=False,
        used_fallback=True,
        attempts=total_attempts,
        latency_ms=total_latency_ms,
        error_type=last_error_type,
        error_message=last_error_message,
    )
