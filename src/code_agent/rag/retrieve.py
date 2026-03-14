"""Retrieval utilities for RAG.

This module provides functions for:
1. Searching the vector index
2. Reranking results (optional)
3. Assembling context for the LLM

Interview talking points:
- Retrieval quality factors:
  - Embedding model quality
  - Chunking strategy
  - Top-K selection (usually 3-5 for chat, 10-20 for search)
  - Similarity threshold filtering

- Advanced techniques:
  - Hybrid search: BM25 (keyword) + vector (semantic)
  - Reranking: Use cross-encoder to reorder top-K
  - Query expansion: Add synonyms/related terms
  - Metadata filtering: Narrow search space first

- Production gotchas:
  - Empty results handling
  - Very long queries (truncate first)
  - Out-of-domain queries (detect and decline)
"""

from typing import Any

from code_agent.rag.ingest import RAGIndex


def search_with_threshold(
    index: RAGIndex,
    query: str,
    k: int = 5,
    min_score: float = 0.3,
) -> list[dict[str, Any]]:
    """Search index and filter by minimum similarity score.

    Args:
        index: RAGIndex to search
        query: Query text
        k: Maximum results to return
        min_score: Minimum similarity score (0-1 for normalized vectors)

    Returns:
        List of results with score >= min_score
    """
    results = index.search(query, k=k)
    return [r for r in results if r.get("score", 0) >= min_score]


def assemble_context(
    results: list[dict[str, Any]],
    max_chars: int = 8000,
    include_source: bool = True,
) -> str:
    """Assemble retrieved chunks into context string for LLM.

    Args:
        results: Search results from RAGIndex.search()
        max_chars: Maximum characters for context
        include_source: Whether to include source file info

    Returns:
        Formatted context string
    """
    if not results:
        return ""

    context_parts = []
    total_chars = 0

    for i, result in enumerate(results, 1):
        content = result["content"]
        source = result.get("source", "unknown")

        if include_source:
            part = f"[Source {i}: {source}]\n{content}"
        else:
            part = content

        # Check if adding this would exceed limit
        if total_chars + len(part) > max_chars:
            # Add truncated version if we have room
            remaining = max_chars - total_chars - 50
            if remaining > 200:
                context_parts.append(part[:remaining] + "...[truncated]")
            break

        context_parts.append(part)
        total_chars += len(part) + 10  # 10 for separator

    return "\n\n---\n\n".join(context_parts)


def format_rag_prompt(
    query: str,
    context: str,
    system_instruction: str | None = None,
) -> list[dict[str, str]]:
    """Format a RAG prompt with retrieved context.

    Args:
        query: User's question
        context: Retrieved context from assemble_context()
        system_instruction: Optional custom system prompt

    Returns:
        List of messages ready for LLM
    """
    default_system = """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Only answer based on the provided context
2. If the context doesn't contain the answer, say "I don't have information about that in my knowledge base."
3. Cite which source you used when answering
4. Be concise and accurate"""

    system = system_instruction or default_system

    if context:
        user_content = f"""Context:
{context}

---

Question: {query}

Please answer based on the context above."""
    else:
        user_content = f"""Question: {query}

Note: No relevant context was found in the knowledge base."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def get_sources(results: list[dict[str, Any]]) -> list[str]:
    """Extract unique source files from results.

    Args:
        results: Search results

    Returns:
        List of unique source file paths
    """
    sources = []
    seen = set()

    for r in results:
        source = r.get("source", "")
        if source and source not in seen:
            sources.append(source)
            seen.add(source)

    return sources
