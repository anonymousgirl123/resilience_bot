from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

# transformers (and its torch dependency) can be expensive to import and
# may fail in environments without a compatible GPU or runtime.  we delay the
# import and capture any errors so that merely importing this module doesn't
# crash the interpreter.
_transformers_import_error: Optional[Exception] = None

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
except Exception as exc:  # covers ImportError, OSError from torch, etc.
    _transformers_import_error = exc
    AutoModelForSeq2SeqLM = AutoTokenizer = pipeline = None  # type: ignore


class HFPipelineLocalClient:
    def __init__(self) -> None:
        self.model_name = os.getenv("RESBOT_MODEL_NAME", "google/flan-t5-large")
        self._pipeline = None
        # if transformers failed to import at module load, we fall back to a
        # lightweight stub client so that unit tests and other simple scripts
        # can still execute without installing the full stack.
        self._unavailable = _transformers_import_error is not None
        if self._unavailable:
            print(
                "[WARN] transformers/torch unavailable; HFPipelineLocalClient "
                "will operate in stub mode."
            )

    def _load_pipeline(self):
        if _transformers_import_error is not None:
            # raise at usage time with a helpful message instead of during import
            raise RuntimeError(
                "Unable to import the transformers/torch stack; \"HFPipelineLocalClient\" "
                "cannot be used. Original error: "
                f"{_transformers_import_error!r}"
            )

        if self._pipeline is None:
            start = time.perf_counter()

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

            self._pipeline = pipeline(
                task="text2text-generation",
                model=model,
                tokenizer=tokenizer,
            )

            load_ms = int((time.perf_counter() - start) * 1000)
            print(f"[INFO] HFPipelineLocalClient loaded model: {self.model_name} in {load_ms} ms")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_new_tokens: int = 180,
        top_p: float = 0.9,
    ) -> str:
        # if the real transformer stack wasn't available, return a stubbed
        # response so callers can exercise the higher layers without crashing.
        if self._unavailable:
            final_prompt = prompt
            if system_prompt:
                final_prompt = f"{system_prompt}\n\n{prompt}"
            return f"[stubbed-response] {final_prompt}"  # simple placeholder

        self._load_pipeline()

        final_prompt = prompt
        if system_prompt:
            final_prompt = f"{system_prompt}\n\n{prompt}"

        result = self._pipeline(
            final_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=6,
            early_stopping=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

        return result[0]["generated_text"].strip()
