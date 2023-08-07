"""Adds a no-op base implementation for mappers classes."""

from __future__ import annotations

import typing as t

from singer_sdk._singerlib.messages import (
    ActivateVersionMessage,
    Message,
    SchemaMessage,
    StateMessage,
)
from singer_sdk.mapper_base import InlineMapper

from map_gpt_embeddings.sdk_fixes.messages import RecordMessage


class BasicPassthroughMapper(InlineMapper):
    """A mapper to split documents into document segments."""

    def map_schema_message(self, message_dict: dict) -> t.Iterable[Message]:
        """Map a schema message to zero or more new messages.

        Args:
            message_dict: A SCHEMA message JSON dictionary.

        Yields:
            A new SCHEMA message.
        """
        yield SchemaMessage.from_dict(message_dict)

    def map_record_message(self, message_dict: dict) -> t.Iterable[RecordMessage]:
        """Map a record message to zero or more new messages.

        Args:
            message_dict: A RECORD message JSON dictionary.

        Yields:
            A new RECORD message.
        """
        yield t.cast(RecordMessage, RecordMessage.from_dict(message_dict))

    def map_state_message(self, message_dict: dict) -> t.Iterable[Message]:
        """Map a state message to zero or more new messages.

        Args:
            message_dict: A STATE message JSON dictionary.

        Yields:
            A new STATE message.
        """
        yield StateMessage.from_dict(message_dict)

    def map_activate_version_message(
        self,
        message_dict: dict,
    ) -> t.Iterable[Message]:
        """Map a version message to zero or more new messages.

        Args:
            message_dict: An ACTIVATE_VERSION message JSON dictionary.

        Yields:
            A new ACTIVATE_VERSION message.
        """
        yield ActivateVersionMessage.from_dict(message_dict)
