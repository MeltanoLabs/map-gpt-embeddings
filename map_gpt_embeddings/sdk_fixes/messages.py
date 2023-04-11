from __future__ import annotations

import typing as t

import pendulum
from singer_sdk._singerlib.messages import RecordMessage as _RecordMessage


class RecordMessage(_RecordMessage):
    """Fixes the RecordMessage.from_dict() method to parse the time_extracted field."""

    @staticmethod
    def from_dict(data: dict) -> _RecordMessage:
        if "time_extracted" in data:
            data["time_extracted"] = pendulum.parse(data["time_extracted"])
        return t.cast(_RecordMessage, _RecordMessage.from_dict(data))
