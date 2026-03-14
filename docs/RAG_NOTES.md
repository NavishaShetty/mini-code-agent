# RAG (Retrieval-Augmented Generation): Complete Guide

> Comprehensive notes on building RAG systems for AI applications.

---

## Table of Contents

1. [What is RAG?](#1-what-is-rag)
2. [Document Processing](#2-document-processing)
3. [Embeddings](#3-embeddings)
4. [Vector Stores](#4-vector-stores)
5. [Retrieval](#5-retrieval)
6. [Context Assembly](#6-context-assembly)
7. [Generation](#7-generation)
8. [Evaluation](#8-evaluation)
9. [Production Concerns](#9-production-concerns)
10. [RAG vs Alternatives](#10-rag-vs-alternatives)
11. [Our Implementation](#11-our-implementation)
12. [Interview Preparation](#12-interview-preparation)

---

## 1. What is RAG?

### The Problem

LLMs have a knowledge cutoff and don't know about:
- Your private documents
- Recent information
- Company-specific knowledge
- Your codebase

### The Solution: RAG

**RAG = Retrieval-Augmented Generation**

Instead of training the model on your data, you:
1. **Retrieve** relevant documents at query time
2. **Inject** them into the prompt
3. **Generate** answer based on retrieved context

### Visual Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INDEXING (offline, once)                                       │
│  ────────────────────────                                       │
│                                                                  │
│  Documents ──► Chunk ──► Embed ──► Store                        │
│                                      │                          │
│                                      ▼                          │
│                               ┌────────────┐                    │
│                               │   Vector   │                    │
│                               │   Store    │                    │
│                               └────────────┘                    │
│                                      │                          │
│  QUERYING (online, every request)    │                          │
│  ────────────────────────────────    │                          │
│                                      │                          │
│  Question ──► Embed ──► Search ◄─────┘                          │
│                            │                                    │
│                            ▼                                    │
│                       Top-K Chunks                              │
│                            │                                    │
│                            ▼                                    │
│              ┌─────────────────────────┐                        │
│              │ Prompt:                 │                        │
│              │ Context: [chunks]       │ ──► LLM ──► Answer    │
│              │ Question: [user query]  │                        │
│              └─────────────────────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why RAG Works

| Benefit | Explanation |
|---------|-------------|
| **No training needed** | Just index documents, no GPU required |
| **Always up-to-date** | Re-index when docs change |
| **Citations** | Can show which docs were used |
| **Controllable** | You decide what knowledge is available |
| **Cost-effective** | Cheaper than fine-tuning |

---

## 2. Document Processing

### 2.1 Document Loading

**Supported Formats:**

| Format | Complexity | Tools |
|--------|------------|-------|
| `.txt` | Easy | Built-in Python |
| `.md` | Easy | Built-in Python |
| `.py` | Easy | Built-in Python |
| `.pdf` | Medium | PyPDF2, pdfplumber |
| `.docx` | Medium | python-docx |
| `.html` | Medium | BeautifulSoup |

**Our Implementation:**

```python
def load_documents(path: str, extensions: list[str] = None):
    extensions = extensions or [".md", ".txt", ".py"]
    path = Path(path)

    if path.is_file():
        files = [path]
    else:
        files = [f for f in path.rglob("*") if f.suffix in extensions]

    documents = []
    for file_path in files:
        content = file_path.read_text()
        documents.append({
            "content": content,
            "source": str(file_path),
            "metadata": {"filename": file_path.name}
        })

    return documents
```

### 2.2 Chunking Strategies

**Why Chunk?**
- Documents are too long for embedding models (usually 512 token limit)
- Smaller chunks = more precise retrieval
- Each chunk should be self-contained and meaningful

**Chunking Methods:**

| Method | How It Works | Pros | Cons |
|--------|--------------|------|------|
| **Fixed size** | Split every N chars | Simple | Breaks mid-sentence |
| **Paragraph** | Split on `\n\n` | Natural boundaries | Uneven sizes |
| **Sentence** | Split on `.!?` | Complete thoughts | May be too small |
| **Recursive** | Try paragraph, then sentence, then char | Best balance | More complex |
| **Semantic** | Split on topic changes | Most meaningful | Requires ML model |

**Our Implementation (Recursive):**

```python
def chunk_text(text: str, chunk_size: int = 380) -> list[str]:
    # If small enough, return as-is
    if estimate_tokens(text) <= chunk_size:
        return [text.strip()]

    # Try splitting by paragraphs
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if estimate_tokens(current + para) <= chunk_size:
            current += "\n\n" + para
        else:
            if current:
                chunks.append(current.strip())
            # If paragraph too big, split by sentences
            if estimate_tokens(para) > chunk_size:
                chunks.extend(split_by_sentences(para, chunk_size))
            else:
                current = para

    return chunks
```

### 2.3 Chunk Size Selection

**OpenShift Lightspeed uses 380 tokens with 0 overlap.**

| Chunk Size | Use Case | Trade-off |
|------------|----------|-----------|
| 100-200 | Q&A with specific facts | High precision, low context |
| **300-500** | General RAG (recommended) | Good balance |
| 500-1000 | Document summarization | More context, lower precision |

**Visual: Chunk Size Trade-off**

```
SMALL CHUNKS (100 tokens):
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│  1  │ │  2  │ │  3  │ │  4  │ │  5  │  More chunks
└─────┘ └─────┘ └─────┘ └─────┘ └─────┘  + Precise matching
                                          - Loses context

LARGE CHUNKS (1000 tokens):
┌───────────────────────────────────────────────┐
│                      1                         │  Fewer chunks
└───────────────────────────────────────────────┘  + More context
                                                   - Dilutes relevance
```

### 2.4 Chunk Overlap

**What is Overlap?**

```
WITHOUT OVERLAP:
[Chunk 1: "The quick brown fox"] [Chunk 2: "jumps over the lazy dog"]

WITH OVERLAP (50%):
[Chunk 1: "The quick brown fox jumps"]
              [Chunk 2: "brown fox jumps over the"]
                            [Chunk 3: "jumps over the lazy dog"]
```

| Overlap | Pros | Cons |
|---------|------|------|
| 0% | Less storage, faster | May miss context at boundaries |
| 10-20% | Good balance | Slight redundancy |
| 50% | Never misses context | 2x storage, slower |

**OpenShift Lightspeed uses 0% overlap** - simpler and works well in practice.

### 2.5 Metadata

Store metadata with each chunk for filtering and citation:

```python
chunk = {
    "content": "Authentication is configured via...",
    "source": "docs/auth.md",
    "chunk_index": 3,
    "total_chunks": 10,
    "metadata": {
        "filename": "auth.md",
        "section": "Configuration",
        "last_updated": "2024-01-15"
    }
}
```

**Use Cases:**
- Filter by file type before search
- Show source in citations
- Filter by date for freshness

---

## 3. Embeddings

### 3.1 What Are Embeddings?

**Embeddings** convert text into vectors (lists of numbers) where:
- Similar meanings → vectors close together
- Different meanings → vectors far apart

```
"How do I configure auth?"  ──►  [0.12, -0.45, 0.78, ..., 0.33]  (384 numbers)
"Setting up authentication" ──►  [0.11, -0.43, 0.76, ..., 0.31]  (similar!)
"The weather is nice"       ──►  [0.89, 0.12, -0.54, ..., -0.21] (different)
```

### 3.2 How Embeddings Work

```
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING PROCESS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Text: "How do I configure authentication?"                     │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────────────────────┐                    │
│  │         Embedding Model                  │                    │
│  │   (Transformer neural network)           │                    │
│  │                                          │                    │
│  │  1. Tokenize: ["How", "do", "I", ...]   │                    │
│  │  2. Encode: Token → vector              │                    │
│  │  3. Pool: Combine into single vector    │                    │
│  └─────────────────────────────────────────┘                    │
│           │                                                      │
│           ▼                                                      │
│  Vector: [0.12, -0.45, 0.78, 0.23, ..., 0.33]                  │
│          └─────────── 384 dimensions ───────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Embedding Models

| Model | Dimensions | Speed | Quality | Cost |
|-------|------------|-------|---------|------|
| **all-MiniLM-L6-v2** | 384 | Fast | Good | Free |
| all-mpnet-base-v2 | 768 | Medium | Better | Free |
| bge-large-en | 1024 | Slow | Great | Free |
| text-embedding-3-small | 1536 | API | Excellent | $0.02/1M tokens |
| text-embedding-3-large | 3072 | API | Best | $0.13/1M tokens |

**Our Choice: all-MiniLM-L6-v2**
- Fast (good for development)
- Free (no API costs)
- Good quality (sufficient for most use cases)
- Small vectors (less storage)

### 3.4 Normalization

**Why Normalize?**

For cosine similarity, vectors should have length 1:

```python
# Normalized embedding
embedding = model.encode(text, normalize_embeddings=True)
# Now: ||embedding|| = 1

# Cosine similarity = dot product (when normalized)
similarity = np.dot(embedding1, embedding2)  # Range: -1 to 1
```

**FAISS IndexFlatIP** assumes normalized vectors for cosine similarity.

### 3.5 Embedding Code

```python
from sentence_transformers import SentenceTransformer

# Load model (downloads ~90MB first time)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed single text
embedding = model.encode("Hello world", normalize_embeddings=True)
# Shape: (384,)

# Embed batch (faster)
texts = ["Hello", "World", "Foo", "Bar"]
embeddings = model.encode(texts, normalize_embeddings=True)
# Shape: (4, 384)
```

---

## 4. Vector Stores

### 4.1 What is a Vector Store?

A **vector store** (or vector database) is optimized for:
- Storing high-dimensional vectors
- Finding similar vectors quickly
- Scaling to millions of vectors

### 4.2 Vector Store Options

| Store | Type | Best For | Pros | Cons |
|-------|------|----------|------|------|
| **FAISS** | Library | <1M docs | Fast, free, in-memory | No persistence built-in |
| Pinecone | Managed | Production | Fully managed, scales | Costs money |
| Weaviate | Self-hosted | Flexibility | Open source, GraphQL | More complex |
| pgvector | Postgres ext | Existing Postgres | Familiar, ACID | Slower than specialized |
| Chroma | Library | Prototyping | Easy to use | Not production-proven |
| Qdrant | Self-hosted | Production | Fast, feature-rich | Self-manage |

**OpenShift Lightspeed uses FAISS** - fast, free, and sufficient for their scale.

### 4.3 FAISS Deep Dive

**FAISS = Facebook AI Similarity Search**

```python
import faiss
import numpy as np

# Create index (384 dimensions, Inner Product similarity)
dimension = 384
index = faiss.IndexFlatIP(dimension)

# Add vectors
vectors = np.random.rand(1000, 384).astype('float32')
faiss.normalize_L2(vectors)  # Normalize for cosine similarity
index.add(vectors)

# Search
query = np.random.rand(1, 384).astype('float32')
faiss.normalize_L2(query)
scores, indices = index.search(query, k=5)
# scores: similarity scores
# indices: positions in the index
```

### 4.4 FAISS Index Types

| Index | Algorithm | Speed | Accuracy | Memory | Use Case |
|-------|-----------|-------|----------|--------|----------|
| **IndexFlatIP** | Brute force | O(n) | 100% | High | <100k vectors |
| IndexIVFFlat | Inverted file | O(√n) | ~95% | High | 100k-1M vectors |
| IndexHNSW | Graph-based | O(log n) | ~98% | Higher | 1M+ vectors |
| IndexPQ | Product quantization | O(n) | ~90% | Low | Memory constrained |

**Visual: How IndexIVFFlat Works**

```
BRUTE FORCE (IndexFlatIP):
Query compares against ALL vectors
┌─────────────────────────────────────┐
│ ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● │  Compare all 1000
│ ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● │  = 1000 comparisons
└─────────────────────────────────────┘

INVERTED FILE (IndexIVFFlat):
First find closest cluster, then search within
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Cluster │ │ Cluster │ │ Cluster │ │ Cluster │
│    1    │ │    2    │ │    3    │ │    4    │
│  ● ● ●  │ │  ● ● ●  │ │  ● ● ●  │ │  ● ● ●  │
│   ● ●   │ │   ● ●   │ │  ● ●●   │ │   ● ●   │
└─────────┘ └─────────┘ └────▲────┘ └─────────┘
                             │
                        Query lands here
                        Only compare ~250
```

### 4.5 Similarity Metrics

| Metric | Formula | Range | Use Case |
|--------|---------|-------|----------|
| **Cosine** | A·B / (‖A‖‖B‖) | -1 to 1 | Text (direction matters) |
| Dot Product | A·B | -∞ to ∞ | Normalized vectors |
| L2 (Euclidean) | √Σ(A-B)² | 0 to ∞ | Images, general |

**For text embeddings, use cosine similarity (or dot product with normalized vectors).**

---

## 5. Retrieval

### 5.1 Basic Vector Search

```python
def search(index, query_embedding, k=5):
    scores, indices = index.search(query_embedding, k)
    return [(idx, score) for idx, score in zip(indices[0], scores[0])]
```

### 5.2 Keyword Search (BM25)

**BM25** is the traditional keyword matching algorithm:

```python
from rank_bm25 import BM25Okapi

# Index
corpus = ["doc 1 text", "doc 2 text", ...]
tokenized = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized)

# Search
query = "authentication config"
scores = bm25.get_scores(query.split())
```

**When BM25 Beats Vectors:**
- Exact keyword matches ("error code 404")
- Acronyms and technical terms
- Names and identifiers

### 5.3 Hybrid Search

**Combine vector (semantic) + keyword (BM25) search:**

```python
def hybrid_search(query, vector_index, bm25_index, alpha=0.5):
    # Vector search (semantic)
    vector_scores = vector_search(query)

    # BM25 search (keyword)
    bm25_scores = bm25_search(query)

    # Combine scores
    combined = {}
    for doc_id, score in vector_scores:
        combined[doc_id] = alpha * score
    for doc_id, score in bm25_scores:
        combined[doc_id] = combined.get(doc_id, 0) + (1 - alpha) * score

    # Sort by combined score
    return sorted(combined.items(), key=lambda x: -x[1])
```

**Visual: Hybrid Search**

```
Query: "kubernetes pod restart CrashLoopBackOff"

VECTOR SEARCH (semantic):                 BM25 SEARCH (keyword):
"container restart policies" ✓            "CrashLoopBackOff error" ✓
"pod lifecycle management" ✓              "pod restart count" ✓
"deployment strategies"                    "kubernetes troubleshooting" ✓

HYBRID (combines both):
"CrashLoopBackOff troubleshooting" ✓✓  ← Appears in both!
"container restart policies" ✓
"pod restart count" ✓
```

### 5.4 Top-K Selection

**How many results to retrieve?**

| Use Case | Recommended K | Reason |
|----------|---------------|--------|
| Chat/Q&A | 3-5 | Focused, less noise |
| Document search | 10-20 | More comprehensive |
| Code search | 5-10 | Balance precision/recall |

### 5.5 Similarity Threshold

**Filter out low-confidence results:**

```python
def search_with_threshold(index, query, k=5, min_score=0.3):
    results = index.search(query, k=k)
    return [r for r in results if r["score"] >= min_score]
```

**Typical Thresholds (cosine similarity):**

| Threshold | Meaning |
|-----------|---------|
| > 0.8 | Very similar (almost identical) |
| 0.5 - 0.8 | Similar (related topic) |
| 0.3 - 0.5 | Somewhat related |
| < 0.3 | Probably not relevant |

### 5.6 Reranking

**Problem:** Vector search returns top-K, but ranking may not be optimal.

**Solution:** Use a more powerful model to re-order results.

```python
from sentence_transformers import CrossEncoder

# Cross-encoder is more accurate but slower
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, results, top_n=3):
    # Score each result against query
    pairs = [(query, r["content"]) for r in results]
    scores = reranker.predict(pairs)

    # Sort by reranker score
    reranked = sorted(zip(results, scores), key=lambda x: -x[1])
    return [r for r, s in reranked[:top_n]]
```

**Visual: Reranking**

```
INITIAL RETRIEVAL (vector search, fast):
1. Doc A (score: 0.75)
2. Doc B (score: 0.72)
3. Doc C (score: 0.70)
4. Doc D (score: 0.68)
5. Doc E (score: 0.65)

AFTER RERANKING (cross-encoder, accurate):
1. Doc C (rerank: 0.92)  ← Was #3, now #1
2. Doc A (rerank: 0.85)
3. Doc D (rerank: 0.78)  ← Was #4, now #3
```

---

## 6. Context Assembly

### 6.1 Formatting Retrieved Chunks

**Simple Format:**

```
Context:
Authentication is configured in config.yaml. Set the AUTH_TOKEN
environment variable to enable token-based auth.

---

To reset your password, run the reset-password command with the
--user flag.

---

Question: How do I set up authentication?
```

**With Sources:**

```
Context:
[Source: docs/auth.md]
Authentication is configured in config.yaml...

[Source: docs/commands.md]
To reset your password, run the reset-password command...

Question: How do I set up authentication?
Answer based on the context above. Cite your sources.
```

### 6.2 Context Window Limits

**Problem:** Retrieved context + prompt must fit in LLM's context window.

```
┌─────────────────────────────────────────────────────────────────┐
│  LLM CONTEXT WINDOW (e.g., 16K tokens)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐  System prompt: ~500 tokens                 │
│  ├────────────────┤                                              │
│  │                │  Retrieved context: ~4000 tokens             │
│  │   Retrieved    │  (10 chunks × 400 tokens)                   │
│  │   Context      │                                              │
│  │                │                                              │
│  ├────────────────┤                                              │
│  │  Question      │  User question: ~100 tokens                 │
│  ├────────────────┤                                              │
│  │                │  Reserved for answer: ~4000 tokens          │
│  │  (Response)    │                                              │
│  │                │                                              │
│  └────────────────┘                                              │
│                                                                  │
│  Buffer: ~7400 tokens                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 RAG Prompt Template

```python
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. ONLY answer based on the provided context
2. If the context doesn't contain the answer, say "I don't have information about that."
3. Cite which source you used: [Source: filename]
4. Be concise and accurate
5. Do not make up information"""

def format_rag_prompt(query: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"""Context:
{context}

---

Question: {query}

Answer based on the context above."""}
    ]
```

---

## 7. Generation

### 7.1 Grounding

**Grounding** = LLM only uses information from the provided context.

**Good (Grounded):**
```
Context: "The API rate limit is 100 requests per minute."
Question: "What's the rate limit?"
Answer: "The rate limit is 100 requests per minute. [Source: api-docs.md]"
```

**Bad (Not Grounded):**
```
Context: "The API rate limit is 100 requests per minute."
Question: "What's the rate limit?"
Answer: "The rate limit is typically 1000 requests per minute for most APIs."
         ↑ Made up! Not from context.
```

### 7.2 Hallucination Prevention

**Techniques:**

| Technique | How It Works |
|-----------|--------------|
| Strong system prompt | "ONLY answer from context" |
| Low temperature | Less creative, more factual |
| Citation requirement | Forces reference to source |
| "I don't know" fallback | Explicit handling of no-match |
| Confidence scoring | LLM rates its own confidence |

### 7.3 Handling "I Don't Know"

```python
NO_CONTEXT_RESPONSE = """I don't have information about that in my knowledge base.

This could mean:
1. The topic isn't covered in the indexed documents
2. Try rephrasing your question
3. The relevant documents may need to be added"""

def generate_answer(query, context):
    if not context:
        return NO_CONTEXT_RESPONSE

    # Generate with context
    response = llm.generate(format_rag_prompt(query, context))
    return response
```

### 7.4 Citation in Answers

**Prompt for Citations:**

```
Answer the question and cite your sources using [Source: filename] format.

Example:
"Authentication uses JWT tokens [Source: auth.md]. The tokens expire
after 24 hours [Source: security.md]."
```

---

## 8. Evaluation

### 8.1 Retrieval Metrics

**Precision@K:** What fraction of retrieved docs are relevant?

```
Retrieved: [A, B, C, D, E]  (K=5)
Relevant:  [A, C, F, G]

Precision@5 = (A, C are relevant) / 5 = 2/5 = 0.4
```

**Recall@K:** What fraction of relevant docs were retrieved?

```
Retrieved: [A, B, C, D, E]  (K=5)
Relevant:  [A, C, F, G]

Recall@5 = (A, C retrieved) / 4 relevant = 2/4 = 0.5
```

**MRR (Mean Reciprocal Rank):** How high is the first relevant result?

```
Query 1: First relevant at position 1 → RR = 1/1 = 1.0
Query 2: First relevant at position 3 → RR = 1/3 = 0.33
Query 3: First relevant at position 2 → RR = 1/2 = 0.5

MRR = (1.0 + 0.33 + 0.5) / 3 = 0.61
```

### 8.2 Generation Metrics

| Metric | What It Measures | How to Evaluate |
|--------|------------------|-----------------|
| **Faithfulness** | Does answer match context? | LLM judge or human |
| **Relevance** | Does answer address question? | LLM judge or human |
| **Completeness** | Is the answer complete? | Human evaluation |
| **Conciseness** | Is it appropriately brief? | Human evaluation |

### 8.3 End-to-End Evaluation

**Create a test set:**

```python
test_cases = [
    {
        "question": "How do I configure auth?",
        "expected_sources": ["docs/auth.md"],
        "expected_answer_contains": ["config.yaml", "AUTH_TOKEN"]
    },
    {
        "question": "What's the rate limit?",
        "expected_sources": ["docs/api.md"],
        "expected_answer_contains": ["100 requests", "per minute"]
    }
]
```

**Evaluate:**

```python
def evaluate_rag(test_cases, rag_system):
    results = []
    for test in test_cases:
        response = rag_system.ask(test["question"])

        # Check retrieval
        retrieved_sources = response["sources"]
        source_match = any(s in retrieved_sources for s in test["expected_sources"])

        # Check answer
        answer = response["answer"]
        content_match = all(term in answer for term in test["expected_answer_contains"])

        results.append({
            "question": test["question"],
            "source_match": source_match,
            "content_match": content_match
        })

    return results
```

---

## 9. Production Concerns

### 9.1 Scaling

| Scale | Approach |
|-------|----------|
| < 10k docs | FAISS IndexFlatIP (exact) |
| 10k - 1M docs | FAISS IndexIVFFlat (approximate) |
| 1M+ docs | Managed service (Pinecone) or distributed FAISS |

### 9.2 Caching

**Cache frequent queries:**

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_search(query: str) -> tuple:
    return tuple(index.search(query, k=5))
```

**Cache embeddings:**

```python
embedding_cache = {}

def get_embedding(text: str):
    if text not in embedding_cache:
        embedding_cache[text] = model.encode(text)
    return embedding_cache[text]
```

### 9.3 Latency Optimization

| Component | Typical Latency | Optimization |
|-----------|-----------------|--------------|
| Embedding | 10-50ms | Batch, cache |
| Vector search | 1-10ms | Approximate index |
| LLM generation | 500-2000ms | Streaming, smaller model |
| **Total** | 500-2000ms | Mostly LLM-bound |

### 9.4 Index Updates

**Adding new documents:**

```python
# Incremental add (fast)
new_chunks = process_new_document(new_doc)
new_embeddings = model.encode([c["content"] for c in new_chunks])
index.add(new_embeddings)
chunks.extend(new_chunks)

# Full reindex (when structure changes)
index = rebuild_index_from_scratch(all_documents)
```

### 9.5 Monitoring

**Track these metrics:**

| Metric | Why Important |
|--------|---------------|
| Query latency | User experience |
| Retrieval precision | Are we finding relevant docs? |
| Empty results rate | Index coverage |
| LLM cost per query | Budget management |
| User feedback | Ground truth quality |

---

## 10. RAG vs Alternatives

### 10.1 RAG vs Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Knowledge update** | Instant (re-index) | Requires retraining |
| **Cost** | Retrieval + generation | Training + generation |
| **Citations** | Yes (show sources) | No |
| **Hallucination** | Lower (grounded) | Higher |
| **Consistency** | Variable | Very consistent |
| **Latency** | Higher (retrieval step) | Lower |
| **Best for** | Dynamic knowledge | Consistent style/format |

### 10.2 RAG vs Long Context

| Aspect | RAG | Long Context |
|--------|-----|--------------|
| **Scalability** | Millions of docs | Limited by context window |
| **Cost** | Pay for retrieved only | Pay for all context |
| **Precision** | High (targeted retrieval) | Variable (needle in haystack) |
| **Simplicity** | More complex | Simpler |
| **Best for** | Large knowledge bases | Small doc sets |

### 10.3 When to Use Each

```
┌─────────────────────────────────────────────────────────────────┐
│                   DECISION TREE                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Is your knowledge base > 100k tokens?                          │
│  ├── YES → Use RAG                                              │
│  └── NO → Consider long context                                 │
│                                                                  │
│  Does knowledge change frequently?                              │
│  ├── YES → Use RAG                                              │
│  └── NO → Consider fine-tuning                                  │
│                                                                  │
│  Do you need citations?                                         │
│  ├── YES → Use RAG                                              │
│  └── NO → Either works                                          │
│                                                                  │
│  Is consistent style critical?                                  │
│  ├── YES → Consider fine-tuning                                 │
│  └── NO → Use RAG                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 10.4 InstructLab Context (Red Hat)

**InstructLab** is Red Hat's approach to fine-tuning:
- Uses synthetic data generation
- Taxonomy-based knowledge organization
- Cheaper than pure human annotation
- Used for customizing Granite models

**When Red Hat uses each:**

| Product | Approach | Why |
|---------|----------|-----|
| OpenShift Lightspeed | RAG | Dynamic docs, need citations |
| Granite models | Fine-tuning (InstructLab) | Consistent code style |
| Ansible Lightspeed | Both | RAG for docs + fine-tuned for Ansible syntax |

---

## 11. Our Implementation

### 11.1 Files We Built

```
src/code_agent/rag/
├── __init__.py      # Exports
├── ingest.py        # Document processing + FAISS index
├── retrieve.py      # Search + context assembly
└── chat.py          # Full RAG chatbot
```

### 11.2 Key Classes

**RAGIndex (ingest.py):**

```python
index = RAGIndex(embedding_model="all-MiniLM-L6-v2", chunk_size=380)
index.add_documents("./docs")
index.save("./my_index")

# Later
index = RAGIndex.load("./my_index")
results = index.search("How to configure X?", k=5)
```

**RAGChat (chat.py):**

```python
chat = RAGChat(index, model, top_k=5, min_score=0.3)
response = chat.ask("How do I configure auth?")
print(response["answer"])
print(response["sources"])
```

### 11.3 Usage Example

```python
from code_agent.rag import RAGIndex, create_rag_chat

# Step 1: Index documents
index = RAGIndex()
index.add_documents("./docs")
index.save("./my_index")

# Step 2: Create chatbot
chat = create_rag_chat("./my_index", model_name="claude-sonnet-4-20250514")

# Step 3: Ask questions
response = chat.ask("What is the ReAct pattern?")
print(f"Answer: {response['answer']}")
print(f"Sources: {response['sources']}")
```

---

## 12. Interview Preparation

### 12.1 Key Concepts Checklist

| Concept | Can You Explain? |
|---------|------------------|
| [ ] What is RAG and why use it? | |
| [ ] Chunking strategies and trade-offs | |
| [ ] What are embeddings? | |
| [ ] Vector stores (FAISS vs alternatives) | |
| [ ] Similarity metrics (cosine vs L2) | |
| [ ] Hybrid search (vector + BM25) | |
| [ ] Reranking | |
| [ ] RAG vs Fine-tuning | |
| [ ] Evaluation metrics (Precision, Recall, MRR) | |

### 12.2 Common Interview Questions

**Q: What chunk size should I use?**
> "300-500 tokens is typical. OpenShift Lightspeed uses 380 with 0 overlap. Smaller chunks give more precise retrieval but lose context. Larger chunks preserve context but dilute relevance."

**Q: FAISS vs Pinecone?**
> "FAISS is fast, free, and good for <1M docs. It's what OpenShift Lightspeed uses. Pinecone is managed, scales better, but costs money. Use FAISS for development, consider managed for production at scale."

**Q: RAG vs Fine-tuning?**
> "RAG is better for dynamic knowledge that changes frequently - you just re-index. Fine-tuning is better for consistent style and domain language. OpenShift Lightspeed uses RAG for docs; InstructLab uses fine-tuning for code style."

**Q: How do you prevent hallucination in RAG?**
> "Strong system prompt saying 'only answer from context', require citations, add 'I don't know' fallback, use low temperature. The context provides grounding that pure LLMs lack."

### 12.3 Red Hat Specific

| Product | RAG Usage |
|---------|-----------|
| **OpenShift Lightspeed** | FAISS index, 380-token chunks, BYOK docs |
| **Ansible Lightspeed** | RAG for docs + fine-tuned model for syntax |
| **InstructLab** | Fine-tuning approach, not RAG |

### 12.4 Quick Reference

```
RAG Pipeline:
Documents → Chunk (380 tokens) → Embed (MiniLM) → FAISS Index
                                                        ↓
Query → Embed → Search → Top-K → Assemble Context → LLM → Answer

Key Numbers:
- Chunk size: 380 tokens (OpenShift standard)
- Top-K: 3-5 for chat, 10-20 for search
- Threshold: 0.3-0.5 minimum similarity
- Embedding dim: 384 (MiniLM), 1536 (OpenAI)
```

---

## See Also

- `src/code_agent/rag/ingest.py` - Our indexing implementation
- `src/code_agent/rag/retrieve.py` - Our retrieval implementation
- `src/code_agent/rag/chat.py` - Our chatbot implementation
- `docs/CONTEXT_MANAGEMENT_NOTES.md` - Related context handling
