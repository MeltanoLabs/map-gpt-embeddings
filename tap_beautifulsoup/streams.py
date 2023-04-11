"""Custom client handling, including KnowledgeBaseStream base class."""

from __future__ import annotations

import typing as t

from langchain.docstore.document import Document
from singer_sdk import typing as th  # JSON schema typing helpers
from singer_sdk.streams import Stream

from map_gpt_embeddings.documents_parser import CustomReadTheDocsLoader

if t.TYPE_CHECKING:
    from singer_sdk.plugin_base import PluginBase as TapBaseClass


class KBStream(Stream):
    """Stream class for KnowledgeBase embeddings."""

    name = "kb_documents"

    def __init__(
        self,
        tap: TapBaseClass,
        name: str,
        source_url: str,
    ) -> None:
        """Initialize the stream.

        Args:
            tap: The tap instance.
            source_url: The source URL to import from.
        """
        super().__init__(tap, schema=None, name=name)
        self.source_url = source_url

    def get_records(self, context: dict | None) -> t.Iterable[dict]:
        """Return a generator of record-type dictionary objects.

        The optional `context` argument is used to identify a specific slice of the
        stream if partitioning is required for the stream. Most implementations do not
        require partitioning and should ignore the `context` argument.

        Args:
            context: Stream partition or context dictionary.
        """
        loader = CustomReadTheDocsLoader(
            self.source_url,
            parser="html.parser",
            top_attrs={"role": "main"},
            logger=self.logger,
        )
        documents = loader.load()
        assert len(documents), f"No documents found in {self.source_url}"
        self.logger.info("Found len(raw_documents): %s", len(documents))

        for doc in documents:
            result = {
                "site_url": self.source_url,
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            yield result

    @property
    def schema(self) -> dict:
        return th.PropertiesList(
            th.Property("site_url", th.StringType, required=True),
            th.Property("page_content", th.StringType),
            th.Property(
                "metadata",
                th.ObjectType(
                    th.Property("source", th.StringType),
                ),
            ),
        ).to_dict()

    @property
    def openai_api_key(self) -> str:
        """Return the OpenAI API key."""
        result = self.config["openai_api_key"]
        if result is None:
            raise Exception("No OpenAI API key found in config.")

        return result
