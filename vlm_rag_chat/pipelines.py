import abc
from pathlib import Path

import numpy as np
import pydantic
from haystack import Document, Pipeline, component
from haystack.components.converters import (
    MarkdownToDocument,
    PyPDFToDocument,
    TextFileToDocument,
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from termcolor import cprint


@component
class DummyEmbedder:
    def __init__(self, name: str, embedding_dim: int = 512):
        self.name = name
        self.embedding_dim = embedding_dim

    def warm_up(self):
        pass

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        for doc in documents:
            doc.embedding = np.random.rand(self.embedding_dim).tolist()
        return {"documents": documents}


class DocumentStoreParams(pydantic.BaseModel):
    location: str = ":memory:"
    index: str = "documents"
    embedding_dim: int = 512
    recreate_index: bool = True
    hnsw_config: dict = {"m": 16, "ef_construct": 64}


class DocumentSplitterParams(pydantic.BaseModel):
    split_by: str = "word"
    split_length: int = 250
    split_overlap: int = 25


class BasePipeline(pydantic.BaseModel, abc.ABC):
    text_embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    store_params: DocumentStoreParams = DocumentStoreParams()

    class Config:
        arbitrary_types_allowed = True

    @property
    def text_embedder(self):
        # embedder = SentenceTransformersDocumentEmbedder(
        #     model=self.text_embedder_name)
        embedder = DummyEmbedder(self.text_embedder_name)
        embedder.warm_up()
        return embedder

    @property
    def document_store(self):
        return QdrantDocumentStore(**dict(self.store_params))

    @abc.abstractmethod
    def create_pipeline(self):
        ...

    @abc.abstractmethod
    def run(self):
        ...


class IndexingPipeline(BasePipeline):
    splitter_params: DocumentSplitterParams = DocumentSplitterParams()

    @property
    def document_splitter(self):
        return DocumentSplitter(**dict(self.splitter_params))

    @property
    def document_writer(self):
        return DocumentWriter(self.document_store)

    def create_pipeline(self):
        # indexing components
        file_type_router = FileTypeRouter(
            mime_types=["text/plain", "application/pdf", "text/markdown"]
        )
        text_file_converter = TextFileToDocument()
        markdown_converter = MarkdownToDocument()
        pdf_converter = PyPDFToDocument()
        document_joiner = DocumentJoiner()
        document_cleaner = DocumentCleaner()

        # indexing Pipeline
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component(
            instance=file_type_router, name="file_type_router"
        )
        indexing_pipeline.add_component(
            instance=text_file_converter, name="text_file_converter"
        )
        indexing_pipeline.add_component(
            instance=markdown_converter, name="markdown_converter"
        )
        indexing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
        indexing_pipeline.add_component(
            instance=document_joiner, name="document_joiner"
        )
        indexing_pipeline.add_component(
            instance=document_cleaner, name="document_cleaner"
        )
        indexing_pipeline.add_component(
            instance=self.document_splitter, name="document_splitter"
        )
        indexing_pipeline.add_component(
            instance=self.text_embedder, name="text_embedder"
        )
        indexing_pipeline.add_component(
            instance=self.document_writer, name="document_writer"
        )

        # connect components
        indexing_pipeline.connect(
            "file_type_router.text/plain", "text_file_converter.sources"
        )
        indexing_pipeline.connect(
            "file_type_router.application/pdf", "pypdf_converter.sources"
        )
        indexing_pipeline.connect(
            "file_type_router.text/markdown", "markdown_converter.sources"
        )
        indexing_pipeline.connect("text_file_converter", "document_joiner")
        indexing_pipeline.connect("pypdf_converter", "document_joiner")
        indexing_pipeline.connect("markdown_converter", "document_joiner")
        indexing_pipeline.connect("document_joiner", "document_cleaner")
        indexing_pipeline.connect("document_cleaner", "document_splitter")
        indexing_pipeline.connect("document_splitter", "text_embedder")
        indexing_pipeline.connect("text_embedder", "document_writer")

        cprint("Created an indexing pipeline", "yellow")
        return indexing_pipeline

    def run(self, path: str):
        pipeline = self.create_pipeline()
        files = list(Path(path).glob("**/*"))
        cprint(f"Detected {len(files)} files @ {path}", "yellow")
        res = pipeline.run({"file_type_router": {"sources": files}})
        n_written = res["document_writer"]["documents_written"]
        cprint(
            f"{n_written} documents have been written to {self.store_params.location!r}",
            "green",
        )
        return res


class RAG(BasePipeline):
    ...
