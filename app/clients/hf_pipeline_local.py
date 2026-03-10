from __future__ import annotations

import os
import time

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


class HFPipelineLocalClient:
    def __init__(self) -> None:
        self.model_name = os.getenv("RESBOT_MODEL_NAME", "google/flan-t5-large")
        self._pipeline = None

    def _load_pipeline(self):
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
