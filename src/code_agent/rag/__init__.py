"""RAG (Retrieval-Augmented Generation) components."""

from code_agent.rag.ingest import (
    RAGIndex,
    chunk_documents,
    chunk_text,
    load_documents,
)
from code_agent.rag.retrieve import (
    assemble_context,
    format_rag_prompt,
    get_sources,
    search_with_threshold,
)
from code_agent.rag.chat import (
    RAGChat,
    create_rag_chat,
)

__all__ = [
    # Ingestion
    "RAGIndex",
    "chunk_text",
    "chunk_documents",
    "load_documents",
    # Retrieval
    "search_with_threshold",
    "assemble_context",
    "format_rag_prompt",
    "get_sources",
    # Chat
    "RAGChat",
    "create_rag_chat",
]
