import os

from singer_sdk.authenticators import BearerTokenAuthenticator
from singer_sdk.streams import RESTStream


class OpenAIStream(RESTStream):
    name = "openai"
    path = "/v1/embeddings"
    rest_method = "POST"

    @property
    def http_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
        }
    @property
    def authenticator(self):
        return BearerTokenAuthenticator(
            stream=self,
            token=self.config.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))
        )
    
    @property
    def url_base(self) -> str:
        base_url = "https://api.openai.com"
        return base_url
    
    def prepare_request_payload(
        self,
        context,
        next_page_token,
    ):
        return {
            "input": context["text"].replace("\n", " "),
            "model": context["model"],
        }
