from __future__ import annotations

from dotenv import load_dotenv

# hf_pipeline_local brings in transformers/torch; on some Windows setups the
# torch DLLs fail to load (WinError 1114).  catch import errors so that the
# test script can fail gracefully instead of blowing up immediately.
try:
    from app.clients.hf_pipeline_local import HFPipelineLocalClient
except Exception as exc:  # keep broad so OSError from torch is caught too
    HFPipelineLocalClient = None  # type: ignore
    import sys
    print(f"[ERROR] could not import HFPipelineLocalClient: {exc}")

from app.core.bot import ResilienceBot


def main():
    load_dotenv()

    if HFPipelineLocalClient is None:
        print("Exiting because HFPipelineLocalClient is unavailable.\n" \
              "Ensure torch/transformers are properly installed or use a CPU-only build.")
        return

    client = HFPipelineLocalClient()
    bot = ResilienceBot(client)

    response = bot.ask(
        "My API has intermittent 504 timeouts. Give a troubleshooting checklist."
    )

    print("\n===== PIPELINE CLIENT TEST =====")
    print(response.text)


if __name__ == "__main__":
    main()
