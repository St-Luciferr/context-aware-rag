# graph.py
"""
LangGraph-based Context-Aware Chatbot
Implements a stateful conversation graph with RAG capabilities
"""

from pprint import pprint
from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from datetime import datetime
from dataclasses import field, dataclass
from src.config import settings
from src.chat_storage import ChatStorage

from src.history_manager import (
    HistoryStrategy,
    HistoryConfig,
    create_history_manager,
    SlidingWindowStrategy,
)

import logging

logger = logging.getLogger(__name__)

DEBUG = False
PREVIEW_CHARS = 200       # how many chars for content_preview in the citations


class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    context: str
    current_query: str
    citations: list[dict]


@dataclass
class SessionMetadata:
    """Metadata for a chat session."""
    created_at: datetime = field(default_factory=datetime.now)
    last_message_at: datetime = field(default_factory=datetime.now)
    messages: list[BaseMessage] = field(default_factory=list)

    def update(self):
        """Update last message timestamp."""
        self.last_message_at = datetime.now()


class RAGChatbot:
    """
    RAG Chatbot with persistent storage, citations, and optimized history.

    """

    def __init__(self,
                 history_strategy: str = "sliding_window",
                 history_config: Optional[HistoryConfig] = None
                 ):
        self.embeddings: HuggingFaceEmbeddings = self._init_embeddings()
        self.vector_store = self._load_vector_store()
        self.llm = self._init_llm()
        self.storage = ChatStorage()

        self.history_config = history_config or HistoryConfig(
            max_messages=10,
            max_tokens=4000,
            summarize_after=8,
            always_keep_last=2,
        )

        self.history_manager = self._init_history_manager(history_strategy)

        #  Initialize BM25 retriever once during initialization
        self.bm25_retriever = self._init_bm25_retriever()
        self.ensemble_retriever = self._init_ensemble_retriever()

        self.graph: CompiledStateGraph = self._build_graph()

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize embeddings for semantic filtering."""
        return HuggingFaceEmbeddings(
            model_name=settings.embedding.model_name,
            model_kwargs={"device": settings.embedding.device},
            encode_kwargs={
                "normalize_embeddings": settings.embedding.normalize}
        )

    def _init_history_manager(self, strategy: str) -> HistoryStrategy:
        """Initialize the history management strategy."""
        logger.info(f"Initializing history manager with strategy: {strategy}")

        return create_history_manager(
            strategy=strategy,
            llm=self.llm,
            embeddings=self.embeddings,
            config=self.history_config
        )

    def _init_llm(self) -> ChatOllama:
        """Initialize the Ollama LLM with settings."""
        return ChatOllama(
            model=settings.ollama.model,
            base_url=settings.ollama.base_url,
            temperature=settings.ollama.temperature
        )

    def _load_vector_store(self) -> Chroma:
        """Load existing ChromaDB vector store."""
        return Chroma(
            collection_name=settings.chroma.collection_name,
            persist_directory=settings.chroma.persist_dir,
            embedding_function=self.embeddings
        )

    def _init_bm25_retriever(self, k: int = 5) -> BM25Retriever:
        """Initialize BM25 retriever once with all documents."""
        # Get all documents from ChromaDB once
        raw_docs = self.vector_store.get(include=["documents", "metadatas"])

        # Convert to Document objects
        documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(raw_docs["documents"], raw_docs["metadatas"])
        ]

        # BM25Retriever
        return BM25Retriever.from_documents(documents=documents, k=k)

    def _init_ensemble_retriever(self, k: int = 5) -> EnsembleRetriever:
        """Initialize ensemble retriever combining vector and BM25 search."""
        # Create vector search retriever
        similarity_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': k}
        )

        # Combine retrievers with weights
        # You can adjust weights: higher weight = more influence
        return EnsembleRetriever(
            retrievers=[similarity_retriever, self.bm25_retriever],
            weights=[0.5, 0.5]  # Equal weighting
        )

    def hybrid_search(self, query: str, k: int = 5) -> list[Document]:
        """
        Perform hybrid search using pre-initialized ensemble retriever.

        Args:
            query: Search query string
            k: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        # If k is different from initialization, recreate retriever
        if k != 5:
            similarity_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={'k': k}
            )
            self.bm25_retriever.k = k

            ensemble_retriever = EnsembleRetriever(
                retrievers=[similarity_retriever, self.bm25_retriever],
                weights=[0.5, 0.5]
            )
            return ensemble_retriever.invoke(query)

        # Use pre-initialized retriever for default k
        return self.ensemble_retriever.invoke(query)

    def refresh_bm25_index(self):
        """
        Refresh BM25 index when documents are added/updated.
        Call this after adding new documents to ChromaDB.
        """
        self.bm25_retriever = self._init_bm25_retriever()
        self.ensemble_retriever = self._init_ensemble_retriever()

    def _retrieve_context(self, state: ChatState) -> ChatState:
        """Retrieve relevant context using hybrid search with citation tracking."""
        query = state["current_query"]
        docs = self.hybrid_search(query, k=settings.rag.retrieval_k)

        if not docs:
            return {
                "context": "",
                "citations": [],
                "messages": [],
                "current_query": query
            }

        context_parts: List[str] = []
        citations: List[Dict[str, Any]] = []

        for idx, doc in enumerate(docs, 1):
            metadata = getattr(doc, "metadata", {}) or {}
            source_type = metadata.get("source", "unknown")
            title = metadata.get("title") or metadata.get("name") or "Unknown"
            url = metadata.get("url")
            page = metadata.get("page")
            chunk_id = metadata.get("chunk_id")

            raw_text = getattr(doc, "page_content", "") or ""

            display_title = title
            if source_type == "wikipedia":
                display_title = f"{title} (Wikipedia)"
            elif page is not None:
                display_title = f"{title}, Page {page}"

            context_block = f"[{idx}] {display_title}\n{raw_text}"
            context_parts.append(context_block)

            preview = raw_text[:PREVIEW_CHARS] + \
                ("..." if len(raw_text) > PREVIEW_CHARS else "")
            citation: Dict[str, Any] = {
                "number": idx,
                "title": title,
                "display": display_title,
                "source_type": source_type,
                "content_preview": preview,
            }

            if url:
                citation["url"] = url
            if page is not None:
                citation["page"] = page
            if chunk_id is not None:
                citation["chunk_id"] = chunk_id

            citations.append(citation)

        context = "\n\n".join(context_parts)

        return {
            "context": context,
            "citations": citations,
            "messages": state["messages"],
            "current_query": query
        }

    def _generate_response(self, state: ChatState) -> ChatState:
        """Generate response using LLM with retrieved context and citations."""
        context = state.get("context", "")
        citations = state.get("citations", [])
        query = state.get("current_query", "")

        system_prompt = (
            "You are a helpful AI assistant with access to a knowledge base. "
            "Use the provided context to answer questions accurately and conversationally.\n\n"
            "IMPORTANT: When referencing information from the context, ALWAYS cite your sources "
            "using the citation numbers provided in square brackets [1], [2], etc.\n\n"
            "If the context doesn't contain relevant information, say so honestly.\n"
            "Keep responses concise but informative, and always include citations where relevant.\n\n"
            "Context from knowledge base:\n"
            f"{context}\n\n"
            f"User query: {query}"
        )

        messages: List[Any] = [SystemMessage(content=system_prompt)]
        previous_messages = state.get("messages") or []
        messages.extend(previous_messages if isinstance(
            previous_messages, list) else [])

        try:
            response = self.llm.invoke(messages)
        except Exception as exc:
            logger.exception(f"LLM invocation failed: {exc}")
            failure_content = (
                "I'm sorry — I couldn't generate a response due to an internal error."
            )
            return {
                "messages": [AIMessage(content=failure_content)],
                "context": context,
                "current_query": query,
                "citations": citations
            }

        resp_text = getattr(response, "content", str(response)) or ""
        final_content = resp_text.strip()
        response_message = AIMessage(content=final_content)

        return {
            "messages": [response_message],
            "context": context,
            "current_query": query,
            "citations": citations
        }

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(ChatState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("generate", self._generate_response)

        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def chat(self, session_id: str, user_message: str) -> dict:
        """Process a chat message and return response with citations."""
        # Create session if it doesn't exist
        if not self.storage.session_exists(session_id):
            self.storage.create_session(session_id)

        # Load conversation history
        full_history = self._load_history_as_messages(session_id)
        human_msg = HumanMessage(content=user_message)
        full_history.append(human_msg)

        optimized_history = self.history_manager.filter_history(
            messages=full_history,
            current_query=user_message
        )

        # Prepare initial state
        initial_state: ChatState = {
            "messages": optimized_history,
            "context": "",
            "current_query": user_message,
            "citations": []
        }

        # Run the graph
        result = self.graph.invoke(initial_state)

        # Extract AI response
        ai_messages = [m for m in result["messages"]
                       if isinstance(m, AIMessage)]

        if ai_messages:
            response = ai_messages[-1].content
            citations = result.get("citations", [])

            # Save FULL messages to database (not optimized)
            self.storage.save_message(session_id, "user", user_message)
            self.storage.save_message(
                session_id, "assistant", response, citations)
            self.storage.update_session(session_id)

            return {
                "response": response,
                "citations": citations,
                "session_id": session_id
            }

        return {
            "response": "I'm sorry, I couldn't generate a response.",
            "citations": [],
            "session_id": session_id
        }

    def _load_history_as_messages(self, session_id: str) -> list[BaseMessage]:
        """Load conversation history as LangChain messages."""
        messages = self.storage.load_messages(session_id)

        langchain_messages = []
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        return langchain_messages

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self.storage.delete_session(session_id)

    def get_history(self, session_id: str) -> list[dict]:
        """Get conversation history for a session."""
        return self.storage.load_messages(session_id)

    def get_all_sessions(self) -> list[dict]:
        """Get information about all active sessions."""
        return self.storage.get_all_sessions()

    def format_citation(self, citation: dict) -> str:
        """Format a single citation for display."""
        if citation["source_type"] == "wikipedia" and "url" in citation:
            return f"[{citation['number']}] {citation['title']} (Wikipedia) - {citation['url']}"
        elif "page" in citation:
            return f"[{citation['number']}] {citation['title']}, Page {citation['page']}"
        else:
            return f"[{citation['number']}] {citation['title']}"

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return settings.ollama.model


# Singleton instance
_chatbot_instance = None


def get_chatbot(
    history_strategy: str = "sliding_window",
    history_config: Optional[HistoryConfig] = None
) -> RAGChatbot:
    """Get or create the chatbot singleton."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = RAGChatbot(
            history_strategy=history_strategy,
            history_config=history_config
        )
    return _chatbot_instance


# Test the chatbot
if __name__ == "__main__":
    print("Testing RAG Chatbot with Persistent Storage and Citations...")
    print(f"Model: {settings.ollama.model}")
    print(f"Base URL: {settings.ollama.base_url}")
    print(f"Retrieval K: {settings.rag.retrieval_k}")
    print("-" * 40)

    bot = get_chatbot()

    test_questions = [
        "What is artificial intelligence?",
        "Can you explain neural networks?",
        "How does that relate to deep learning?"
    ]

    session = "test_session"
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"User: {q}")
        print(f"{'='*60}")

        result = bot.chat(session, q)
        print(f"\nAssistant: {result['response']}")

        # Display formatted citations
        if result['citations']:
            print(f"\n{'─'*60}")
            print("📚 Retrieved Sources:")
            print(f"{'─'*60}")
            for cite in result['citations']:
                print(f"  {bot.format_citation(cite)}")
                if cite.get('chunk_id'):
                    print(f"     (Chunk ID: {cite['chunk_id']})")

    # Test persistence - load history
    print(f"\n\n{'='*60}")
    print("💾 Testing Persistence - Loading Conversation History")
    print(f"{'='*60}")
    history = bot.get_history(session)
    print(f"Total messages stored: {len(history)}")

    # Show last exchange
    if len(history) >= 2:
        print(f"\nLast exchange:")
        print(f"  User: {history[-2]['content'][:100]}...")
        print(f"  Assistant: {history[-1]['content'][:100]}...")
        if history[-1].get('citations'):
            print(f"  Citations: {len(history[-1]['citations'])} sources")

    # Test sessions list
    print(f"\n{'='*60}")
    print("📋 Active Sessions")
    print(f"{'='*60}")
    for sess in bot.get_all_sessions():
        print(f"  Session: {sess['session_id']}")
        print(f"    Messages: {sess['message_count']}")
        print(f"    Created: {sess['created_at']}")
        print(f"    Last active: {sess['last_message_at']}")
        if sess['preview']:
            print(f"    Preview: {sess['preview']}")
        print()
