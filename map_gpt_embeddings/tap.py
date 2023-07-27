from singer_sdk.tap_base import Tap

class TapOpenAI(Tap):
    """OpenAI tap class."""

    name = "tap-openai"
    def discover_streams(self):
        return []