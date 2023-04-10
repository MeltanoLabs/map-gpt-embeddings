"""Stream class to deduplicate records in a stream."""

from __future__ import annotations

import hashlib
import json
import typing as t

from singer_sdk import Stream

Breadcrumb = t.Tuple[str, ...]


class DedupeStream(Stream):
    """Stream class to deduplicate records in a stream."""

    def check_if_seen(
        self, record: dict, context: dict | None, ignored_breadcrumbs: set[Breadcrumb]
    ) -> bool:
        """Return a boolean indicating whether the record hash has been seen.

        Calling this method will also update the state of the stream to indicate that
        the checked record hash has now been seen.

        Args:
            record_hash: A string record hash.
            context: Stream partition or context dictionary.

        Returns:
            A boolean indicating whether the record hash has been seen.
        """
        record_hash = self.get_record_hash(record)
        if record_hash in self.seen_record_hashes(context):
            return True
        else:
            self.get_context_state(context)["records_seen"].append(record_hash)
            return False

    def get_record_hash(self, record: dict) -> str:
        """Return a string record hash.

        Args:
            record: A dictionary representing a record.

        Returns:
            A string record hash.
        """
        # Return a SHA-1 hash of the record
        return hashlib.sha1(json.dumps(record).encode("utf-8")).hexdigest()

    def seen_record_hashes(self, context: dict | None) -> t.Iterable[str]:
        """Return a generator of string record hashes.

        Args:
            context: Stream partition or context dictionary.
        """
        state = self.get_context_state(context)
        if "records_seen" not in state:
            state["records_seen"] = set()
            return

        yield from state["records_seen"]
