"""KnowledgeBase tap class."""

from __future__ import annotations

from singer_sdk import Tap
from singer_sdk import typing as th  # JSON schema typing helpers

from map_gpt_embeddings import streams


class TapOpenAIDocuments(Tap):
    """KnowledgeBase tap class."""

    name = "tap-openai-documents"

    config_jsonschema = th.PropertiesList(
        th.Property("source_name", th.StringType, required=True, default="sdk-docs"),
        th.Property(
            "source_url",
            th.StringType,
            required=True,
            default="/Users/aj/Source/tap-nlp-kb-vectors/sdk.meltano.com/en/latest",
        ),
        th.Property(
            "loader",
            th.StringType,
            required=True,
            default="readthedocs",
            allowed_values=["readthedocs"],
        ),
    ).to_dict()

    def discover_streams(self) -> list[streams.Stream]:
        """Return a list of discovered streams.

        Returns:
            A list of discovered streams.
        """
        return [
            streams.KBStream(
                self,
                name=self.config["source_name"],
                source_url=self.config["source_url"],
            )
        ]


if __name__ == "__main__":
    TapOpenAIDocuments.cli()
