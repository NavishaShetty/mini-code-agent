"""Document ingestion for RAG: chunking, embedding, and indexing.

This module handles the "write" side of RAG:
1. Load documents from files
2. Split into chunks (using token-based chunking)
3. Generate embeddings with sentence-transformers
4. Store in FAISS index

Interview talking points:
- Chunking strategies:
  - Too small: loses context, increases noise
  - Too large: dilutes relevance, wastes context window
  - OpenShift Lightspeed uses 380 tokens, 0 overlap

- Embedding models:
  - all-MiniLM-L6-v2: fast, 384 dimensions, good quality
  - text-embedding-3-small: OpenAI, better but costs money
  - Trade-off: quality vs speed vs cost

- Vector stores:
  - FAISS: fast, in-memory, good for <1M docs (OpenShift choice)
  - pgvector: if already using Postgres
  - Pinecone/Weaviate: managed, scales, costs money

- Index types in FAISS:
  - IndexFlatIP: exact search, O(n), good for <100k docs
  - IndexIVFFlat: approximate, faster for >100k docs
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


# Default configuration (matches OpenShift Lightspeed patterns)
DEFAULT_CHUNK_SIZE = 380  # tokens
DEFAULT_CHUNK_OVERLAP = 0  # OpenShift uses zero overlap
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality


def estimate_tokens(text: str) -> int:
    """Rough token estimation (~4 chars per token).

    For production, use tiktoken for accuracy.
    This is fast and good enough for chunking.
    """
    return len(text) // 4


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into chunks of approximately chunk_size tokens.

    Uses a simple approach:
    1. Split by paragraphs first
    2. Combine paragraphs until chunk_size is reached
    3. If a single paragraph exceeds chunk_size, split by sentences

    Args:
        text: The text to chunk
        chunk_size: Target size in tokens (default 380)
        chunk_overlap: Overlap between chunks in tokens (default 0)

    Returns:
        List of text chunks
    """
    # If text is small enough, return as single chunk
    if estimate_tokens(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Split by paragraphs (double newline)
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Check if adding this paragraph exceeds chunk size
        potential = current_chunk + "\n\n" + para if current_chunk else para

        if estimate_tokens(potential) <= chunk_size:
            # Fits - add to current chunk
            current_chunk = potential
        else:
            # Doesn't fit - save current chunk and start new one
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # If paragraph itself is too big, split by sentences
            if estimate_tokens(para) > chunk_size:
                sentence_chunks = _split_by_sentences(para, chunk_size)
                chunks.extend(sentence_chunks[:-1])  # Add all but last
                current_chunk = sentence_chunks[-1] if sentence_chunks else ""
            else:
                current_chunk = para

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _split_by_sentences(text: str, chunk_size: int) -> list[str]:
    """Split text by sentences when paragraph is too large."""
    import re

    # Simple sentence splitting (handles . ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""

    for sentence in sentences:
        potential = current + " " + sentence if current else sentence

        if estimate_tokens(potential) <= chunk_size:
            current = potential
        else:
            if current.strip():
                chunks.append(current.strip())
            # If single sentence is too long, force split by characters
            if estimate_tokens(sentence) > chunk_size:
                char_limit = chunk_size * 4
                while sentence:
                    chunks.append(sentence[:char_limit])
                    sentence = sentence[char_limit:]
                current = ""
            else:
                current = sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


def load_documents(
    path: str | Path,
    extensions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load documents from a file or directory.

    Args:
        path: File or directory path
        extensions: File extensions to include (default: .md, .txt, .py)

    Returns:
        List of dicts with 'content', 'source', 'metadata'
    """
    extensions = extensions or [".md", ".txt", ".py", ".rst", ".json"]
    path = Path(path)
    documents = []

    if path.is_file():
        files = [path]
    else:
        files = [f for f in path.rglob("*") if f.is_file() and f.suffix in extensions]

    for file_path in files:
        try:
            content = file_path.read_text(errors="ignore")
            if content.strip():  # Skip empty files
                documents.append({
                    "content": content,
                    "source": str(file_path),
                    "metadata": {
                        "filename": file_path.name,
                        "extension": file_path.suffix,
                        "size_chars": len(content),
                    },
                })
        except (PermissionError, UnicodeDecodeError) as e:
            print(f"Skipping {file_path}: {e}")

    return documents


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Chunk documents into smaller pieces.

    Args:
        documents: List of document dicts from load_documents()
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunk dicts with 'content', 'source', 'chunk_index', 'metadata'
    """
    all_chunks = []

    for doc in documents:
        doc_chunks = chunk_text(doc["content"], chunk_size, chunk_overlap)

        for i, chunk_content in enumerate(doc_chunks):
            all_chunks.append({
                "content": chunk_content,
                "source": doc["source"],
                "chunk_index": i,
                "total_chunks": len(doc_chunks),
                "metadata": doc.get("metadata", {}),
            })

    return all_chunks


class RAGIndex:
    """FAISS-based vector index for RAG retrieval.

    This uses the same pattern as OpenShift Lightspeed:
    - FAISS IndexFlatIP (Inner Product for cosine similarity)
    - Sentence Transformers for embeddings
    - Normalized embeddings for cosine similarity

    Example usage:
        # Create and populate index
        index = RAGIndex()
        index.add_documents("/path/to/docs")
        index.save("./my_index")

        # Load and search
        index = RAGIndex.load("./my_index")
        results = index.search("How do I configure X?", k=5)
        for r in results:
            print(f"{r['source']}: {r['content'][:100]}...")
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        """Initialize the RAG index.

        Args:
            embedding_model: Name of sentence-transformers model
            chunk_size: Token size for chunking documents
        """
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self._model = None  # Lazy loaded
        self._index = None  # FAISS index
        self.chunks: list[dict[str, Any]] = []  # Store chunk metadata

    @property
    def model(self):
        """Lazy load the embedding model (expensive to load)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.embedding_model}")
            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    def add_documents(self, path: str | Path) -> int:
        """Load, chunk, embed, and index documents.

        Args:
            path: File or directory to index

        Returns:
            Number of chunks added to index
        """
        import faiss

        # Step 1: Load documents
        print(f"Loading documents from {path}...")
        documents = load_documents(path)
        print(f"  Loaded {len(documents)} documents")

        if not documents:
            return 0

        # Step 2: Chunk documents
        print(f"Chunking with size={self.chunk_size} tokens...")
        new_chunks = chunk_documents(documents, self.chunk_size)
        print(f"  Created {len(new_chunks)} chunks")

        if not new_chunks:
            return 0

        # Step 3: Generate embeddings
        print("Generating embeddings...")
        texts = [chunk["content"] for chunk in new_chunks]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # For cosine similarity
            show_progress_bar=True,
        )
        embeddings = embeddings.astype("float32")
        print(f"  Embedding shape: {embeddings.shape}")

        # Step 4: Create or extend FAISS index
        if self._index is None:
            dimension = embeddings.shape[1]
            # IndexFlatIP = Inner Product (cosine sim with normalized vectors)
            self._index = faiss.IndexFlatIP(dimension)
            print(f"  Created new FAISS index (dim={dimension})")

        self._index.add(embeddings)
        self.chunks.extend(new_chunks)
        print(f"  Index now contains {len(self.chunks)} chunks")

        return len(new_chunks)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search for relevant chunks.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of chunk dicts with added 'score' field, sorted by relevance
        """
        if self._index is None or not self.chunks:
            return []

        # Embed the query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        query_embedding = query_embedding.astype("float32")

        # Search FAISS index
        k = min(k, len(self.chunks))  # Can't return more than we have
        scores, indices = self._index.search(query_embedding, k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def save(self, path: str | Path) -> None:
        """Save index to disk.

        Creates a directory with:
        - index.faiss: The FAISS index
        - chunks.pkl: Chunk metadata
        - config.json: Index configuration
        """
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self._index is not None:
            faiss.write_index(self._index, str(path / "index.faiss"))

        # Save chunks
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        # Save config
        config = {
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "num_chunks": len(self.chunks),
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved index to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "RAGIndex":
        """Load index from disk."""
        import faiss

        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config = json.load(f)

        # Create instance
        index = cls(
            embedding_model=config["embedding_model"],
            chunk_size=config["chunk_size"],
        )

        # Load FAISS index
        index_path = path / "index.faiss"
        if index_path.exists():
            index._index = faiss.read_index(str(index_path))

        # Load chunks
        with open(path / "chunks.pkl", "rb") as f:
            index.chunks = pickle.load(f)

        print(f"Loaded index with {len(index.chunks)} chunks")
        return index

    def __len__(self) -> int:
        """Return number of chunks in index."""
        return len(self.chunks)

    def __repr__(self) -> str:
        return f"RAGIndex(chunks={len(self.chunks)}, model='{self.embedding_model}')"
