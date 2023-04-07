"""Custom client handling, including KnowledgeBaseStream base class."""

from __future__ import annotations

from typing import Iterable

from singer_sdk.streams import Stream


class KnowledgeBaseStream(Stream):
    """Stream class for KnowledgeBase streams."""

    def get_records(self, context: dict | None) -> Iterable[dict]:
        """Return a generator of record-type dictionary objects.

        The optional `context` argument is used to identify a specific slice of the
        stream if partitioning is required for the stream. Most implementations do not
        require partitioning and should ignore the `context` argument.

        Args:
            context: Stream partition or context dictionary.

        Raises:
            NotImplementedError: If the implementation is TODO
        """
        # TODO: Write logic to extract data from the upstream source.
        # records = mysource.getall()
        # for record in records:
        #     yield record.to_dict()
        raise NotImplementedError("The method is not yet implemented (TODO)")
