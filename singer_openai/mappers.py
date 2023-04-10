import typing as t

import openai
import pendulum
import singer_sdk._singerlib as singer
from singer_openai.openai_helpers import get_embeddings
from singer_sdk import exceptions
from singer_sdk import typing as th
from singer_sdk._singerlib.messages import RecordMessage as _RecordMessage
from singer_sdk.mapper_base import InlineMapper

if t.TYPE_CHECKING:
    import singer_sdk._singerlib as singer
    from singer_sdk._singerlib.messages import (
        ActivateVersionMessage,
    )


class RecordMessage(_RecordMessage):
    @staticmethod
    def from_dict(data: dict) -> _RecordMessage:
        if "time_extracted" in data:
            data["time_extracted"] = pendulum.parse(data["time_extracted"])
        return t.cast(_RecordMessage, _RecordMessage.from_dict(data))


from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class BasicPassthroughMapper(InlineMapper):
    """A mapper to split documents into document segments."""

    def map_schema_message(self, message_dict: dict) -> t.Iterable[singer.Message]:
        """Map a schema message to zero or more new messages.

        Args:
            message_dict: A SCHEMA message JSON dictionary.
        """
        yield singer.SchemaMessage.from_dict(message_dict)

    def map_record_message(
        self, message_dict: dict
    ) -> t.Iterable[singer.RecordMessage]:
        """Map a record message to zero or more new messages.

        Args:
            message_dict: A RECORD message JSON dictionary.
        """
        yield RecordMessage.from_dict(message_dict)

    def map_state_message(self, message_dict: dict) -> t.Iterable[singer.Message]:
        """Map a state message to zero or more new messages.

        Args:
            message_dict: A STATE message JSON dictionary.
        """
        yield singer.StateMessage.from_dict(message_dict)

    def map_activate_version_message(
        self,
        message_dict: dict,
    ) -> t.Iterable[singer.Message]:
        """Map a version message to zero or more new messages.

        Args:
            message_dict: An ACTIVATE_VERSION message JSON dictionary.
        """
        yield ActivateVersionMessage.from_dict(message_dict)


class DocumentSplitMapper(BasicPassthroughMapper):
    """Split documents into segments, then vectorize."""

    config_jsonschema = th.PropertiesList(
        th.Property("document_text_property", th.StringType, default="page_content"),
        th.Property("document_metadata_property", th.StringType, default="metadata"),
    ).to_dict()

    name = "map-openai-doc-splitter"

    def transform_record(self, record: dict) -> t.Iterable[dict]:
        """Map a record message to zero or more new messages.

        Args:
            message_dict: A RECORD message JSON dictionary.
        """
        raw_document_text = record[self.config["document_text_property"]]
        metadata_dict = record[self.config["document_metadata_property"]]

        if self.config.get("split_documents", True):
            splitter_config = self.config.get("splitter_config", {})
            if "chunk_size" not in splitter_config:
                splitter_config["chunk_size"] = 1000
            if "chunk_overlap" not in splitter_config:
                splitter_config["chunk_overlap"] = 200
            text_splitter = RecursiveCharacterTextSplitter(**splitter_config)

            document = Document(page_content=raw_document_text, metadata=metadata_dict)

            document_segments = text_splitter.split_documents([document])

        # assert document_segments and len(
        #     document_segments
        # ), "No documents output from split."
        if len(document_segments) > 1:
            self.logger.debug("Document split into %s segments", len(document_segments))
        elif len(document_segments) == 1:
            self.logger.debug("Document not split", len(document_segments))

        for doc_segment in document_segments:
            new_record = record.copy()
            new_record[self.config["document_text_property"]] = doc_segment.page_content
            new_record[self.config["document_metadata_property"]] = doc_segment.metadata
            yield new_record

    def map_record_message(
        self, message_dict: dict
    ) -> t.Iterable[singer.RecordMessage]:
        """_summary_

        Args:
            message_dict: _description_

        Returns:
            _description_
        """
        for record in self.transform_record(message_dict["record"]):
            new_msg = message_dict.copy()
            yield RecordMessage.from_dict(new_msg)


class DocumentEmbeddingMapper(BasicPassthroughMapper):
    """Split documents into segments, then vectorize."""

    name = "map-openai-embeddings"

    config_jsonschema = th.PropertiesList(
        th.Property("document_text_property", th.StringType, default="page_content"),
        th.Property("document_metadata_property", th.StringType, default="metadata"),
        th.Property(
            "openai_api_key",
            th.StringType,
            required=True,
            secret=True,
            description="OpenAI API key",
        ),
    ).to_dict()

    def map_record_message(
        self, message_dict: dict
    ) -> t.Iterable[singer.RecordMessage]:
        # Superclass splits document into segments
        doc_segment_messages = super().map_record_message(message_dict)

        for segment_message in doc_segment_messages:
            try:
                segment_message.record["embeddings"] = get_embeddings(
                    text=segment_message.record[self.config["document_text_property"]],
                    model="text-embedding-ada-002",
                    api_key=self.config["openai_api_key"],
                )
            except openai.error.RateLimitError as ex:
                raise exceptions.AbortedSyncFailedException(
                    "Sync aborted due to OpenAI rate limit reached. Error message:\n"
                    + str(ex)
                ) from ex

            yield segment_message

    def map_schema_message(self, message_dict: dict) -> t.Iterable[singer.Message]:
        for result in t.cast(
            t.Iterable[singer.SchemaMessage], super().map_schema_message(message_dict)
        ):
            result.schema["properties"]["encodings"] = th.ArrayType(
                th.NumberType
            ).to_dict()
            yield result
