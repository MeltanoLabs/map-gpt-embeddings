import asyncio
import atexit
import json
import logging
import os
import tempfile
import typing as t

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from singer_sdk import exceptions
from singer_sdk import typing as th
from singer_sdk._singerlib.messages import Message, RecordMessage, SchemaMessage

from map_gpt_embeddings.cookbook import process_api_requests_from_file
from map_gpt_embeddings.sdk_fixes.mapper_base import BasicPassthroughMapper


class GPTEmbeddingMapper(BasicPassthroughMapper):
    """Split documents into segments, then vectorize."""

    name = "map-gpt-embeddings"

    def __init__(self, *args, **kwargs):
        """Initialize the mapper.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.stream = None
        self.requests_filepath = self._create_temp_file()
        self.save_filepath = self._create_temp_file()
        self.cursor_position = 0

    def _create_temp_file(self) -> tempfile.NamedTemporaryFile:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_filename = temp_file.name
        self.logger.info(f"Temporary file created: {temp_filename}")
        atexit.register(self._delete_temp_file, temp_filename)
        return temp_file

    def _delete_temp_file(self, temp_filename) -> None:
        try:
            self.logger.info(f"Cleaning up: {temp_filename}")
        finally:
            os.remove(temp_filename)
            self.logger.info(f"Temporary file deleted: {temp_filename}")

    def _clear_file(self, file_path):
        with open(file_path, 'w'):
            pass

    def map_schema_message(self, message_dict: dict) -> t.Iterable[Message]:
        for result in t.cast(
            t.Iterable[SchemaMessage], super().map_schema_message(message_dict)
        ):
            # Add an "embeddings" property to the schema
            result.schema["properties"]["embeddings"] = th.ArrayType(
                th.NumberType
            ).to_dict()
            yield result

    config_jsonschema = th.PropertiesList(
        th.Property(
            "document_text_property",
            th.StringType,
            default="page_content",
            description="The name of the property containing the document text."
        ),
        th.Property(
            "document_metadata_property",
            th.StringType,
            default="metadata",
            description="The name of the property containing the document metadata."
        ),
        th.Property(
            "openai_api_key",
            th.StringType,
            secret=True,
            description="OpenAI API key. Optional if `OPENAI_API_KEY` env var is set.",
        ),
        th.Property(
            "splitter_config",
            th.ObjectType(),
            description="Configuration for the text splitter.",
            default={
                "chunk_size": 1000,
                "chunk_overlap": 200,
            }
        ),
        th.Property(
            "split_documents",
            th.BooleanType,
            description="Whether to split document into chunks.",
            default=True,
        ),
        th.Property(
            "embedding_model",
            th.StringType,
            description="The embedding model to use.",
            default="text-embedding-ada-002",
        ),
        th.Property(
            "max_requests_per_minute",
            th.NumberType,
            description="The embedding model to use.",
            default=3_000 * 0.5,
        ),
        th.Property(
            "max_tokens_per_minute",
            th.NumberType,
            description="The embedding model to use.",
            default=1_000_000 * 0.5,
        ),
    ).to_dict()

    def _validate_config(self, *, raise_errors: bool = True) -> list[str]:
        """Validate configuration input against the plugin configuration JSON schema.

        Args:
            raise_errors: Flag to throw an exception if any validation errors are found.

        Returns:
            A list of validation errors.

        Raises:
            ConfigValidationError: If raise_errors is True and validation fails.
        """
        errors = super()._validate_config(raise_errors=raise_errors)
        if (
            raise_errors
            and self.config.get("openai_api_key", None) is None
            and "OPENAI_API_KEY" not in os.environ
        ):
            raise exceptions.ConfigValidationError(
                "Must set at least one of the following: `openai_api_key` setting, "
                f"`{self.name.upper().replace('-', '_')}_OPEN_API_KEY` env var, or "
                " `OPENAI_API_KEY` env var."
            )

        return errors

    def split_record(self, record: dict) -> t.Iterable[dict]:
        """Split a record dict to zero or more record dicts.

        Args:
            record: The record object to split.

        Yields:
            A generator of record dicts.
        """
        if not self.config["split_documents"]:
            yield record
            return

        raw_document_text = record[self.config["document_text_property"]]
        metadata_dict = record[self.config["document_metadata_property"]]

        text_splitter = RecursiveCharacterTextSplitter(
            **self.config["splitter_config"]
        )

        document = Document(page_content=raw_document_text, metadata=metadata_dict)

        document_segments = text_splitter.split_documents([document])

        if len(document_segments) > 1:
            self.logger.debug("Document split into %s segments", len(document_segments))
        elif len(document_segments) == 1:
            self.logger.debug("Document not split", len(document_segments))

        for doc_segment in document_segments:
            new_record = record.copy()
            new_record[self.config["document_text_property"]] = doc_segment.page_content
            new_record[self.config["document_metadata_property"]] = doc_segment.metadata
            yield new_record

    def map_record_message(self, message_dict: dict) -> t.Iterable[RecordMessage]:
        # Add to async batch file
        for split_record in self.split_record(message_dict["record"]):
            with open(self.requests_filepath.name, "a") as file:
                text = message_dict["record"][self.config["document_text_property"]]
                request = {
                    "input": text.replace("\n", " "),
                    "model": self.config["embedding_model"],
                    "metadata": message_dict,
                }
                file.write(
                    json.dumps(request)+ "\n"
                )
                self.cursor_position += 1
        # Run async process and output batch results
        if self.cursor_position >= 50:
            self.cursor_position = 0
            asyncio.run(
                process_api_requests_from_file(
                    self.requests_filepath.name,
                    self.save_filepath.name,
                    request_url="https://api.openai.com/v1/embeddings",
                    api_key=self.config.get("openai_api_key", os.environ.get("OPENAI_API_KEY")),
                    max_requests_per_minute=self.config["max_requests_per_minute"],
                    max_tokens_per_minute=self.config["max_tokens_per_minute"],
                    token_encoding_name="cl100k_base",
                    max_attempts=5,
                    logging_level=logging.DEBUG,
                )
            )
            with open(self.save_filepath.name, "r") as file:
                for response in file.readlines():
                    response = json.loads(response)
                    orig_message = response[2]
                    orig_message["record"]["embeddings"] = response[1]["data"][0]["embedding"]
                    yield t.cast(RecordMessage, RecordMessage.from_dict(orig_message))
            self._clear_file(self.save_filepath.name)
            self._clear_file(self.requests_filepath.name)

if __name__ == "__main__":
    GPTEmbeddingMapper.cli()
