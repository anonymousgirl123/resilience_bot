from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMCallResult:
    answer: str
    success: bool
    used_fallback: bool
    attempts: int
    latency_ms: int
    error_type: Optional[str] = None
    error_message: Optional[str] = None
