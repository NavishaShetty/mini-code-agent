"""RAG Chatbot interface.

This module provides a simple chatbot that uses RAG to answer questions
based on indexed documents. It demonstrates the full RAG pipeline:
Query → Retrieve → Generate

Interview talking points:
- RAG vs Fine-tuning:
  - RAG: Dynamic knowledge, citations, no training needed
  - Fine-tuning: Consistent style, domain language, no retrieval latency

- RAG quality factors:
  - Retrieval precision (did we get relevant docs?)
  - Generation faithfulness (did LLM stick to context?)
  - Answer completeness (did we answer the full question?)

- Production considerations:
  - Caching frequent queries
  - Fallback when retrieval fails
  - Confidence scoring
  - Feedback loops for improvement
"""

from typing import Any

from code_agent.model.litellm import LiteLLMModel
from code_agent.rag.ingest import RAGIndex
from code_agent.rag.retrieve import (
    assemble_context,
    format_rag_prompt,
    get_sources,
    search_with_threshold,
)


class RAGChat:
    """RAG-powered chatbot.

    Example usage:
        # Initialize
        index = RAGIndex.load("./my_index")
        model = LiteLLMModel(model_name="claude-sonnet-4-20250514")
        chat = RAGChat(index, model)

        # Ask questions
        response = chat.ask("How do I configure authentication?")
        print(response["answer"])
        print(f"Sources: {response['sources']}")
    """

    def __init__(
        self,
        index: RAGIndex,
        model: LiteLLMModel,
        top_k: int = 5,
        min_score: float = 0.3,
        max_context_chars: int = 8000,
    ):
        """Initialize RAG chatbot.

        Args:
            index: RAGIndex with indexed documents
            model: LLM model for generation
            top_k: Number of chunks to retrieve
            min_score: Minimum similarity score for results
            max_context_chars: Maximum chars for retrieved context
        """
        self.index = index
        self.model = model
        self.top_k = top_k
        self.min_score = min_score
        self.max_context_chars = max_context_chars
        self.conversation_history: list[dict[str, str]] = []

    def ask(
        self,
        question: str,
        include_history: bool = False,
    ) -> dict[str, Any]:
        """Ask a question and get a RAG-powered answer.

        Args:
            question: User's question
            include_history: Whether to include conversation history

        Returns:
            Dict with 'answer', 'sources', 'num_results', 'context_used'
        """
        # Step 1: Retrieve relevant chunks
        results = search_with_threshold(
            self.index,
            question,
            k=self.top_k,
            min_score=self.min_score,
        )

        # Step 2: Assemble context
        context = assemble_context(
            results,
            max_chars=self.max_context_chars,
            include_source=True,
        )

        # Step 3: Format prompt
        messages = format_rag_prompt(question, context)

        # Optionally add conversation history
        if include_history and self.conversation_history:
            # Insert history between system and user message
            system_msg = messages[0]
            user_msg = messages[1]
            messages = [system_msg] + self.conversation_history[-6:] + [user_msg]

        # Step 4: Generate answer
        response = self.model.query(messages)
        answer = response["content"]

        # Step 5: Update history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        # Keep history bounded
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return {
            "answer": answer,
            "sources": get_sources(results),
            "num_results": len(results),
            "context_used": bool(context),
            "model_stats": self.model.get_stats(),
        }

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def get_similar_docs(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Get similar documents without generating an answer.

        Useful for debugging retrieval quality.
        """
        return search_with_threshold(self.index, query, k=k, min_score=0.0)


def create_rag_chat(
    index_path: str,
    model_name: str = "claude-sonnet-4-20250514",
    **kwargs,
) -> RAGChat:
    """Convenience function to create a RAG chatbot.

    Args:
        index_path: Path to saved RAGIndex
        model_name: LLM model name
        **kwargs: Additional args for RAGChat

    Returns:
        Configured RAGChat instance
    """
    index = RAGIndex.load(index_path)
    model = LiteLLMModel(model_name=model_name)
    return RAGChat(index, model, **kwargs)


# Simple CLI for testing
def run_chat_cli(index_path: str, model_name: str = "claude-sonnet-4-20250514"):
    """Run an interactive chat session.

    Args:
        index_path: Path to saved RAGIndex
        model_name: LLM model name
    """
    print("Loading RAG index...")
    chat = create_rag_chat(index_path, model_name)
    print(f"Loaded index with {len(chat.index)} chunks")
    print("\nRAG Chatbot ready! Type 'quit' to exit, 'clear' to clear history.\n")

    while True:
        try:
            question = input("You: ").strip()

            if not question:
                continue
            if question.lower() == "quit":
                print("Goodbye!")
                break
            if question.lower() == "clear":
                chat.clear_history()
                print("History cleared.\n")
                continue

            print("\nSearching...")
            result = chat.ask(question, include_history=True)

            print(f"\nAssistant: {result['answer']}")
            if result["sources"]:
                print(f"\nSources ({result['num_results']} chunks):")
                for src in result["sources"][:3]:
                    print(f"  - {src}")
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m code_agent.rag.chat <index_path> [model_name]")
        sys.exit(1)

    index_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "claude-sonnet-4-20250514"
    run_chat_cli(index_path, model_name)
