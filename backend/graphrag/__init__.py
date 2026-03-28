"""GraphRAG package for Layer 1 and Layer 2 ingestion."""

from .canonicalization import EntityCanonicalizer
from .chunking import chunk_article
from .edges import Layer3CorpusRecord, Layer3EdgeRecord, build_layer3
from .embeddings import HashingEmbedder
from .entities import Layer2DocumentRecord, Layer2EntityRecord
from .extraction import extract_layer2
from .graph_retrieval import GraphRetrieval
from .indexing import GraphIndexManager
from .models import ChunkRecord, PaperRecord, SearchHit, SectionRecord, TableRecord
from .parser import parse_article
from .rag import QuerySynthesizer
from .retrieval import LocalVectorIndex
from .search_service import GraphRAGSearchService, build_search_bundle
from .tracing import TracingManager, get_tracing_manager

__all__ = [
    "ChunkRecord",
    "EntityCanonicalizer",
    "GraphRetrieval",
    "GraphRAGSearchService",
    "GraphIndexManager",
    "HashingEmbedder",
    "Layer2DocumentRecord",
    "Layer2EntityRecord",
    "Layer3CorpusRecord",
    "Layer3EdgeRecord",
    "LocalVectorIndex",
    "PaperRecord",
    "QuerySynthesizer",
    "SearchHit",
    "SectionRecord",
    "TableRecord",
    "TracingManager",
    "build_layer3",
    "build_search_bundle",
    "chunk_article",
    "extract_layer2",
    "get_tracing_manager",
    "parse_article",
]
