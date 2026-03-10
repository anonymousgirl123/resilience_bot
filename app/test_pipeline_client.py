from __future__ import annotations

from dotenv import load_dotenv
from app.clients.hf_pipeline_local import HFPipelineLocalClient
from app.core.bot import ResilienceBot


def main():
    load_dotenv()

    client = HFPipelineLocalClient()
    bot = ResilienceBot(client)

    response = bot.ask(
        "My API has intermittent 504 timeouts. Give a troubleshooting checklist."
    )

    print("\n===== PIPELINE CLIENT TEST =====")
    print(response.text)


if __name__ == "__main__":
    main()
