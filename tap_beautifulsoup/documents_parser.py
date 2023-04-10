from __future__ import annotations

import logging
from pathlib import Path

from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.document_loaders import ReadTheDocsLoader


class CustomReadTheDocsLoader(ReadTheDocsLoader):
    """Overrides parsing logic for sites that are not parsed by the base class."""

    def __init__(
        self,
        path: str,
        top_name_tag: str | None = None,
        top_attrs: dict | None = None,
        parser: str = "html.parser",
        logger: logging.Logger | None = None,
    ):
        kwargs = {
            "features": parser,
        }
        self.path = path
        self.top_name_tag = top_name_tag
        self.top_attrs = top_attrs
        self.logger = logger or logging.getLogger(__name__)
        super().__init__(path, encoding=None, errors=None, **kwargs)

    def clean_data(self, data: str) -> str:
        soup = BeautifulSoup(data, **self.bs_kwargs)
        text = soup.find_all(name=self.top_name_tag, attrs=self.top_attrs)
        if len(text) != 0:
            text = text[0].get_text()
        else:
            text = ""
        return "\n".join([t for t in text.split("\n") if t])

    def load(self) -> list[Document]:
        """Load documents."""
        docs = []
        for p in Path(self.path).rglob("*"):
            if p.is_dir():
                continue
            with open(p, encoding=self.encoding, errors=self.errors) as f:
                text = self.clean_data(f.read())
                if not text:
                    self.logger.warning(f"Could not find {self.top_attrs} in file {p}.")

            metadata = {"source": str(p)}
            docs.append(Document(page_content=text, metadata=metadata))

        assert len(docs), f"No documents found in {self.path}"

        return docs
