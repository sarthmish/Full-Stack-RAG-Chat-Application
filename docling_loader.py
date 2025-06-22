"""
Docling LangChain loader module for processing and chunking documents.
Copyright IBM Corp. 2025 - 2025
SPDX-License-Identifier: MIT
Modified: @sarthmish
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, List

from docling.chunking import BaseChunk
from docling.datamodel.document import DoclingDocument
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownPictureSerializer,
    MarkdownTableSerializer,
)
from docling_core.types.doc.document import (
    PictureClassificationData,
    PictureDescriptionData,
    PictureItem,
    PictureMiscData,
)
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from transformers import AutoTokenizer
from typing_extensions import override
import logging

from doc_ingest import DocIngest

logger = logging.getLogger("root")
logger.info("Imported Docling Loader")


class AnnotationPictureSerializer(MarkdownPictureSerializer):
    """Serializer for handling picture annotations with classifications and descriptions."""

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """
        Serialize picture annotations into a formatted text result.

        Args:
            item: Picture item to serialize
            doc_serializer: Document serializer instance
            doc: Docling document
            kwargs: Additional keyword arguments

        Returns:
            Serialized result containing formatted text
        """
        text_parts = []
        classes = []
        captions = []
        descriptions = []

        for annotation in item.annotations:
            if isinstance(annotation, PictureClassificationData):
                if annotation.predicted_classes:
                    predicted_class = annotation.predicted_classes[0].class_name
                    classes.append(predicted_class)
            elif isinstance(annotation, PictureMiscData):
                if 'captions' in annotation.content:
                    captions.append(annotation.content['captions'])
            elif isinstance(annotation, PictureDescriptionData):
                descriptions.append(annotation.text)

        for predicted_class in classes:
            text_parts.append(f"Picture type: {predicted_class}")
        for caption in captions:
            text_parts.append(f"Picture caption: {caption}")
        for description in descriptions:
            text_parts.append(f"Picture description: {description}")

        processed_text = doc_serializer.post_process(text="\n".join(text_parts))
        return create_ser_result(text=processed_text, span_source=item)


class TableAndImgAnnotationSerializerProvider(ChunkingSerializerProvider):
    """Provider for table and image annotation serialization."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        """Get appropriate serializer for the document."""
        return ChunkingDocSerializer(
            doc=doc,
            picture_serializer=AnnotationPictureSerializer(),
            table_serializer=MarkdownTableSerializer(),
        )


class BaseMetaExtractor(ABC):
    """Abstract base class for metadata extraction."""

    @abstractmethod
    def extract_chunk_meta(self, file_path: str, chunk: BaseChunk) -> dict[str, Any]:
        """Extract metadata from a chunk."""
        pass

    @abstractmethod
    def extract_dl_doc_meta(
        self, file_path: str, dl_doc: DoclingDocument
    ) -> dict[str, Any]:
        """Extract metadata from a Docling document."""
        pass


class MetaExtractor(BaseMetaExtractor):
    """Concrete implementation of metadata extractor."""

    def extract_chunk_meta(self, file_path: str, chunk: BaseChunk) -> dict[str, Any]:
        return {
            "source": file_path,
            "dl_meta": chunk.meta.export_json_dict(),
        }

    def extract_dl_doc_meta(
        self, file_path: str, dl_doc: DoclingDocument
    ) -> dict[str, Any]:
        return {"source": file_path}


class DoclingLoader(BaseLoader):
    """Loader for processing Docling documents with chunking support."""

    def __init__(self, conv_res_list: list, embed_model_id: str):
        """
        Initialize the DoclingLoader.

        Args:
            conv_res_list: List of conversion results from DocIngest
            embed_model_id: Embedding model identifier
        """
        self.conv_res_list = conv_res_list
        self.embed_model_id = embed_model_id
        self.tokenizer: BaseTokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(embed_model_id), max_tokens=512
        )
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            serializer_provider=TableAndImgAnnotationSerializerProvider(),
        )
        self._meta_extractor = MetaExtractor()

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily load and process documents.

        Yields:
            Processed document chunks with metadata
        """
        for conv_res in self.conv_res_list:
            if conv_res is None or conv_res.result is None:
                continue

            dl_doc = conv_res.result
            for chunk in self.chunker.chunk(dl_doc=dl_doc):
                yield Document(
                    page_content=self.chunker.contextualize(chunk=chunk),
                    metadata=self._meta_extractor.extract_chunk_meta(
                        file_path=dl_doc.name,
                        chunk=chunk,
                    ),
                )