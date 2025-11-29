"""
LangGraph-based Context-Aware Chatbot
Implements a stateful conversation graph with RAG capabilities
"""

from typing import TypedDict, Annotated, Sequence
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from datetime import datetime
from dataclasses import field, dataclass
from src.config import settings


# State definition
class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    context: str
    current_query: str


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
    def __init__(self):
        self.vector_store = self._load_vector_store()
        self.llm = self._init_llm()
        self.graph: CompiledStateGraph = self._build_graph()
        self.sessions: dict[str, SessionMetadata] = {}

    def _init_llm(self) -> ChatOllama:
        """Initialize the Ollama LLM with settings."""
        return ChatOllama(
            model=settings.ollama.model,
            base_url=settings.ollama.base_url,
            temperature=settings.ollama.temperature
        )

    def _load_vector_store(self) -> Chroma:
        """Load existing ChromaDB vector store."""
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding.model_name,
            model_kwargs={"device": settings.embedding.device},
            encode_kwargs={
                "normalize_embeddings": settings.embedding.normalize}
        )
        return Chroma(
            collection_name=settings.chroma.collection_name,
            persist_directory=settings.chroma.persist_dir,
            embedding_function=embeddings
        )

    def _retrieve_context(self, state: ChatState) -> ChatState:
        """Retrieve relevant context from vector store."""
        query = state["current_query"]
        docs = self.vector_store.similarity_search(
            query, k=settings.rag.retrieval_k)

        context_parts = []
        for doc in docs:
            source = doc.metadata.get("title", "Unknown")
            context_parts.append(f"[Source: {source}]\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)
        return {"context": context, "messages": [], "current_query": query}

    def _generate_response(self, state: ChatState) -> ChatState:
        """Generate response using LLM with retrieved context."""
        system_prompt = f"""You are a helpful AI assistant with access to a knowledge base. 
                        Use the provided context to answer questions accurately and conversationally.
                        If the context doesn't contain relevant information, say so honestly.
                        Keep responses concise but informative.

                        Context from knowledge base:
                        {state["context"]}"""

        messages = [SystemMessage(content=system_prompt)
                    ] + list(state["messages"])
        response = self.llm.invoke(messages)

        return {
            "messages": [response],
            "context": state["context"],
            "current_query": state["current_query"]
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

    def chat(self, session_id: str, user_message: str) -> str:
        """Process a chat message and return response."""
        # Get or create conversation history
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionMetadata()

        session = self.sessions[session_id]

        # Add user message to history
        human_msg = HumanMessage(content=user_message)
        session.messages.append(human_msg)
        session.update()

        # Prepare initial state
        initial_state: ChatState = {
            "messages": list(session.messages),
            "context": "",
            "current_query": user_message
        }

        # Run the graph
        result = self.graph.invoke(initial_state)

        # Extract AI response
        ai_messages = [m for m in result["messages"]
                       if isinstance(m, AIMessage)]
        if ai_messages:
            response = ai_messages[-1].content
            session.messages.append(AIMessage(content=response))
            session.update()
            return response

        return "I'm sorry, I couldn't generate a response."

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_history(self, session_id: str) -> list[dict]:
        """Get conversation history for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return []
        return [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content
            }
            for m in session.messages
        ]

    def get_all_sessions(self) -> list[dict]:
        """Get information about all active sessions."""
        sessions_info = []
        for session_id, session in self.sessions.items():
            # Get preview from last message
            preview = None
            if session.messages:
                last_msg = session.messages[-1]
                preview = last_msg.content[:100] + "..." if len(
                    last_msg.content) > 100 else last_msg.content

            sessions_info.append({
                "session_id": session_id,
                "message_count": len(session.messages),
                "created_at": session.created_at,
                "last_message_at": session.last_message_at,
                "preview": preview
            })

        # Sort by last_message_at descending (most recent first)
        sessions_info.sort(key=lambda x: x["last_message_at"], reverse=True)
        return sessions_info

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return settings.ollama.model


# Singleton instance
_chatbot_instance = None


def get_chatbot() -> RAGChatbot:
    """Get or create the chatbot singleton."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = RAGChatbot()
    return _chatbot_instance


# Test the chatbot
if __name__ == "__main__":
    print("Testing RAG Chatbot...")
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
        print(f"\nUser: {q}")
        response = bot.chat(session, q)
        print(f"Bot: {response}")
