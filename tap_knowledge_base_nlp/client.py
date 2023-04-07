"""Custom client handling, including KnowledgeBaseStream base class."""

from __future__ import annotations

import pickle
from typing import Iterable

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.faiss import FAISS
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
        for something in self.process_kb():
            yield something

    def process_kb(self) -> VectorStore:
        """Process the KnowledgeBase."""
        loader = ReadTheDocsLoader("sdk.meltano.com")
        raw_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        documents = text_splitter.split_documents(raw_documents)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)

        # # Save vectorstore
        # with open("vectorstore.pkl", "wb") as f:
        #     pickle.dump(vectorstore, f)
        return vectorstore
