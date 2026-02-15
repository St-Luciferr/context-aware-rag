# graph.py
"""
LangGraph-based Context-Aware Chatbot
Implements a stateful conversation graph with RAG capabilities
"""

from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from datetime import datetime, timezone
from dataclasses import field, dataclass
from src.config import settings
from src.chat_storage import ChatStorage
from src.autocut import AutoCut
import time
from src.history_manager import (
    HistoryStrategy,
    HistoryConfig,
    create_history_manager
)

import logging

# LangSmith tracing support
try:
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def get_current_run_tree():
        return None

logger = logging.getLogger(__name__)

DEBUG = False
PREVIEW_CHARS = 200       # how many chars for content_preview in the citations


class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    context: str
    current_query: str
    citations: list[dict]
    # Tool calling state
    tool_calls: list[dict]  # Pending tool calls from LLM
    tool_results: list[str]  # Results from executed tools
    tool_iterations: int  # Number of tool calling iterations


@dataclass
class SessionMetadata:
    """Metadata for a chat session."""
    created_at: datetime = field(default_factory=datetime.now)
    last_message_at: datetime = field(default_factory=datetime.now)
    messages: list[BaseMessage] = field(default_factory=list)

    def update(self):
        """Update last message timestamp."""
        self.last_message_at = datetime.now(timezone.utc)


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
        self.autocut = AutoCut()
        self.history_config = history_config or HistoryConfig(
            max_messages=10,
            max_tokens=4000,
            summarize_after=8
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
        # support for cloud models
        client_kwargs = {
            "headers": {'Authorization': 'Bearer ' + settings.ollama.api_key}
        } if settings.ollama.api_key else {}
        llm = ChatOllama(
            model=settings.ollama.model,
            base_url=settings.ollama.base_url,
            temperature=settings.ollama.temperature,
            keep_alive=-1,
            client_kwargs=client_kwargs
        )

        # Bind tools if tool calling is enabled
        if settings.tools.enabled:
            llm = self._bind_tools(llm)

        return llm

    def _bind_tools(self, llm: ChatOllama) -> ChatOllama:
        """Bind enabled tools to the LLM."""
        try:
            from src.tools import get_enabled_tools
            tools = get_enabled_tools()
            if tools:
                logger.info(f"Binding {len(tools)} tools to LLM: {[t.name for t in tools]}")
                return llm.bind_tools(tools)
            else:
                logger.warning("Tool calling enabled but no tools found")
                return llm
        except ImportError as e:
            logger.error(f"Could not import tools: {e}")
            return llm
        except Exception as e:
            logger.error(f"Error binding tools: {e}")
            return llm

    def _load_vector_store(self) -> Chroma:
        """Load existing ChromaDB vector store."""
        return Chroma(
            collection_name=settings.chroma.collection_name,
            persist_directory=settings.chroma.persist_dir,
            embedding_function=self.embeddings
        )

    def _init_bm25_retriever(self, k: int = settings.rag.retrieval_k) -> BM25Retriever:
        """Initialize BM25 retriever once with all documents."""
        # Get all documents from ChromaDB once
        raw_docs = self.vector_store.get(include=["documents", "metadatas"])

        # Convert to Document objects
        documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(raw_docs["documents"], raw_docs["metadatas"])
        ]
        if len(documents) > 0:
            # BM25Retriever
            return BM25Retriever.from_documents(documents=documents, k=k)
        else:
            return None

    def _init_ensemble_retriever(self, k: int = settings.rag.retrieval_k) -> EnsembleRetriever:
        """Initialize ensemble retriever combining vector and BM25 search."""
        # Create vector search retriever
        similarity_retriever = self.vector_store.as_retriever(
            search_type="mmr",
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
        if k != settings.rag.retrieval_k:
            print(f"Recreating Ensemble: K = {k}")
            similarity_retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={'k': k}
            )
            self.bm25_retriever.k = k

            ensemble_retriever = EnsembleRetriever(
                retrievers=[similarity_retriever, self.bm25_retriever],
                weights=[0.5, 0.5]
            )
            docs = ensemble_retriever.invoke(query)
        else:
            docs = self.ensemble_retriever.invoke(query)
        return docs

    def refresh_bm25_index(self):
        """
        Refresh all retrieval components when documents are added/updated.
        Call this after adding new documents to ChromaDB.
        """
        # Refresh vector store connection to ensure we see new documents
        self.vector_store = self._load_vector_store()
        self.bm25_retriever = self._init_bm25_retriever()
        self.ensemble_retriever = self._init_ensemble_retriever()

    def _format_docs_for_trace(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Format documents for LangSmith trace output."""
        return [
            {
                "title": doc.metadata.get("title", "Unknown"),
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "relevance_score": doc.metadata.get("relevance_score"),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            for doc in docs
        ]

    def _retrieve_context(self, state: ChatState) -> ChatState:
        """Retrieve relevant context using hybrid search with citation tracking."""
        query = state["current_query"]
        start_time = time.time()

        # Hybrid search retrieval
        raw_docs = self.hybrid_search(query, k=settings.rag.retrieval_k)

        # Apply autocut distillation if enabled (may be redundant with MMR)
        if settings.rag.autocut_enabled:
            # Log autocut input for tracing
            autocut_input = {
                "query": query,
                "input_doc_count": len(raw_docs),
                "input_docs": self._format_docs_for_trace(raw_docs)
            }

            docs = self.autocut.distill(query, raw_docs)

            # Log autocut output for tracing
            autocut_output = {
                "output_doc_count": len(docs),
                "docs_removed": len(raw_docs) - len(docs),
                "output_docs": self._format_docs_for_trace(docs)
            }
        else:
            # Skip AutoCut - MMR already provides diversity
            docs = raw_docs[:settings.rag.distilled_retrieval_k]  # Just take top-k
            autocut_input = {}
            autocut_output = {"skipped": True, "reason": "AUTOCUT_ENABLED=false"}

        # Add trace metadata if LangSmith is available
        if LANGSMITH_AVAILABLE:
            try:
                run_tree = get_current_run_tree()
                if run_tree:
                    run_tree.metadata = run_tree.metadata or {}
                    run_tree.metadata["autocut_input"] = autocut_input
                    run_tree.metadata["autocut_output"] = autocut_output
            except Exception:
                pass  # Silently ignore tracing errors

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

        autocut_status = f"-> {len(docs)}" if settings.rag.autocut_enabled else "(no autocut)"
        print(f"[Retrieval] {len(raw_docs)} docs {autocut_status} ({time.time()-start_time:.2f}s)")

        return {
            "context": context,
            "citations": citations,
            "messages": state["messages"],
            "current_query": query
        }

    def _execute_tools(self, state: ChatState) -> ChatState:
        """Execute tool calls and return results."""
        tool_calls = state.get("tool_calls", [])

        if not tool_calls:
            return state

        print(f"[Tools] Executing {len(tool_calls)} tool(s)")

        tool_results = []
        new_messages = []  # Only new ToolMessages (state uses 'add' operator)

        try:
            from src.tools import get_all_tools
            tools_dict = get_all_tools()
        except ImportError:
            logger.error("Could not import tools for execution")
            return state

        for i, tool_call in enumerate(tool_calls, 1):
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", "")

            print(f"  - {tool_name}({tool_args})")

            tool_func = tools_dict.get(tool_name)
            if tool_func:
                try:
                    result = tool_func.invoke(tool_args)
                    result_len = len(str(result))
                    print(f"    OK: {result_len} chars")
                    tool_results.append(result)
                    new_messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_id
                    ))
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    print(f"    FAILED: {error_msg}")
                    logger.error(f"Tool {tool_name} failed: {e}")
                    tool_results.append(error_msg)
                    new_messages.append(ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_id
                    ))
            else:
                print(f"    FAILED: Tool not found")
                logger.warning(f"Tool '{tool_name}' not found")
                tool_results.append(f"Tool '{tool_name}' not found")
                new_messages.append(ToolMessage(
                    content=f"Tool '{tool_name}' not found",
                    tool_call_id=tool_id
                ))

        return {
            **state,
            "messages": new_messages,
            "tool_calls": [],
            "tool_results": tool_results,
            "tool_iterations": state.get("tool_iterations", 0) + 1
        }

    def _should_continue_tools(self, state: ChatState) -> str:
        """Determine if tool loop should continue or end."""
        messages = state.get("messages", [])
        tool_iterations = state.get("tool_iterations", 0)
        max_iterations = settings.tools.max_iterations

        # Check iteration limit
        if tool_iterations >= max_iterations:
            logger.info(f"Tool iteration limit reached ({max_iterations})")
            return "end"

        # Check if last message has tool calls
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                return "execute_tools"

        return "end"

    def _generate_with_tools(self, state: ChatState) -> ChatState:
        """Generate response that may include tool calls."""
        start_time = time.time()
        context = state.get("context", "")
        citations = state.get("citations", [])
        query = state.get("current_query", "")

        # Build system prompt with tool usage guidance
        system_prompt = f"""## Role
You are an AI assistant that provides accurate, factual information. You MUST NOT make up or assume information.

## Critical Rules
1. NEVER fabricate, assume, or hallucinate information
2. ONLY use information from the provided context OR from tool results
3. If the context does not contain the answer, you MUST use a tool to find it
4. If you cannot find information even after using tools, say "I don't have information about that"

## Available Tools - USE THEM
- retrieve_documents: Search the knowledge base for more documents
- web_search: Search the web for current information (USE THIS for topics not in context)
- clarify_query: Ask the user for clarification if the question is unclear

## When to Use Tools
- Topic NOT in the provided context -> USE web_search or retrieve_documents
- Question about recent events/news -> USE web_search
- Technical term or concept you're unsure about -> USE web_search
- Ambiguous question -> USE clarify_query

## Response Rules
- Cite sources using [1], [2], etc. when using context
- Be concise (2-4 paragraphs)
- If unsure, USE A TOOL rather than guessing

## Knowledge Base Context
{context}

## User Question
{query}
"""

        messages: List[Any] = [SystemMessage(content=system_prompt)]
        previous_messages = state.get("messages") or []

        # Filter and fix messages to ensure they're valid for Ollama
        for msg in (previous_messages if isinstance(previous_messages, list) else []):
            if isinstance(msg, AIMessage):
                # Ensure AIMessage has content or valid tool_calls
                content = getattr(msg, 'content', '') or ''
                has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls

                if not content and not has_tool_calls:
                    # Skip empty AIMessages
                    logger.warning("Skipping empty AIMessage without content or tool_calls")
                    logger.debug(f"Invalid AIMessage: {msg}")
                    continue

                # If AIMessage has tool_calls but empty content, add placeholder
                if has_tool_calls and not content:
                    msg = AIMessage(
                        content="[Calling tools...]",
                        tool_calls=msg.tool_calls
                    )

            messages.append(msg)

        try:
            response = self.llm.invoke(messages)
        except Exception as exc:
            logger.exception(f"LLM invocation failed: {exc}")
            return {
                **state,
                "messages": [AIMessage(content="I'm sorry — I couldn't generate a response.")],
                "tool_calls": []
            }

        # Check for tool calls in response
        tool_calls = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tc in response.tool_calls:
                # Handle both dict and object formats
                if isinstance(tc, dict):
                    tool_calls.append({
                        "id": tc.get("id", ""),
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {})
                    })
                else:
                    # Object format (e.g., ToolCall namedtuple)
                    tool_calls.append({
                        "id": getattr(tc, "id", ""),
                        "name": getattr(tc, "name", ""),
                        "args": getattr(tc, "args", {})
                    })
            tool_names = [t['name'] for t in tool_calls]
            print(f"[LLM] Calling tools: {tool_names}")
            logger.info(f"LLM requested tools: {tool_names}")

        print(f"[LLM] Generation: {time.time()-start_time:.2f}s")

        return {
            **state,
            "messages": [response],
            "tool_calls": tool_calls,
            "citations": citations
        }

    def _generate_response(self, state: ChatState) -> ChatState:
        """Generate response using LLM with retrieved context and citations."""
        start_time = time.time()
        context = state.get("context", "")
        citations = state.get("citations", [])
        query = state.get("current_query", "")

        # Structured prompt following best practices: Role, Instructions, Output Format
        system_prompt = f"""## Role
You are an AI research assistant specialized in Artificial Intelligence and Machine Learning topics. You provide accurate, well-sourced information from a curated knowledge base.

## Instructions
1. Answer questions using ONLY the provided context and conversation history
2. ALWAYS cite sources using [1], [2], etc. when referencing context information
3. If the context lacks relevant information, clearly state: "I don't have specific information about that in my knowledge base"
4. Distinguish between: (a) new information from context, (b) follow-up from conversation history, (c) combination of both
5. Keep responses concise (2-4 paragraphs maximum) unless detailed explanation is requested

## Output Format
- Start with a direct answer to the question
- Support with relevant details from sources
- Include citations [1], [2], etc. inline where information is referenced

## Knowledge Base Context
{context}

## User Question
{query}
"""

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
            print(f"Generation Complete in {time.time()-start_time} Seconds")
            return {
                "messages": [AIMessage(content=failure_content)],
                "context": context,
                "current_query": query,
                "citations": citations
            }

        resp_text = getattr(response, "content", str(response)) or ""
        final_content = resp_text.strip()
        response_message = AIMessage(content=final_content)
        print(f"Generation Complete in {time.time()-start_time} Seconds")
        return {
            "messages": [response_message],
            "context": context,
            "current_query": query,
            "citations": citations
        }

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph state machine with optional tool calling."""
        workflow = StateGraph(ChatState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_context)

        if settings.tools.enabled:
            # Tool-enabled flow: retrieve -> generate_with_tools -> (conditional) -> execute_tools | END
            workflow.add_node("generate_with_tools", self._generate_with_tools)
            workflow.add_node("execute_tools", self._execute_tools)

            # Add edges
            workflow.set_entry_point("retrieve")
            workflow.add_edge("retrieve", "generate_with_tools")

            # Conditional edge after generate_with_tools
            # If tool calls exist -> execute_tools, otherwise -> END (response is complete)
            workflow.add_conditional_edges(
                "generate_with_tools",
                self._should_continue_tools,
                {
                    "execute_tools": "execute_tools",
                    "end": END
                }
            )

            # After executing tools, go back to generate_with_tools for the response
            workflow.add_edge("execute_tools", "generate_with_tools")
        else:
            # Standard flow without tools: retrieve -> generate -> END
            workflow.add_node("generate", self._generate_response)
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
        start_time = time.time()
        optimized_history = self.history_manager.filter_history(
            messages=full_history,
            current_query=user_message
        )
        print(f"[History] {len(full_history)} msgs -> {len(optimized_history)} optimized ({time.time()-start_time:.2f}s)")

        # Prepare initial state
        initial_state: ChatState = {
            "messages": optimized_history,
            "context": "",
            "current_query": user_message,
            "citations": [],
            "tool_calls": [],
            "tool_results": [],
            "tool_iterations": 0
        }

        # Run the graph with tracing metadata for LangSmith
        start_time = time.time()
        result = self.graph.invoke(
            initial_state,
            config={
                "run_name": f"RAG Chat - {session_id[:8]}",
                "metadata": {
                    "session_id": session_id,
                    "query_length": len(user_message),
                    "history_strategy": self.get_current_strategy(),
                    "model": settings.ollama.model,
                },
                "tags": ["rag", "chat", self.get_current_strategy()],
            }
        )
        print(f"[Done] Response generated in {time.time()-start_time:.2f}s")

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

    def get_history_paginated(
        self,
        session_id: str,
        limit: int = 20,
        before_timestamp: str = None
    ) -> dict:
        """Get paginated conversation history for a session.

        Args:
            session_id: The session ID
            limit: Maximum number of messages to return
            before_timestamp: Only return messages before this timestamp

        Returns:
            Dict with messages, has_more, total_count, oldest_timestamp
        """
        return self.storage.load_messages_paginated(
            session_id,
            limit=limit,
            before_timestamp=before_timestamp
        )

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

    def set_history_strategy(self, strategy: str) -> None:
        """Change the history management strategy at runtime."""
        valid_strategies = ["sliding_window", "token_budget", "summarization"]
        if strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy: {strategy}. Choose from {valid_strategies}")

        self.history_manager = create_history_manager(
            strategy=strategy,
            llm=self.llm,
            embeddings=self.embeddings,
            config=self.history_config
        )
        logger.info(f"History strategy changed to: {strategy}")

    def get_current_strategy(self) -> str:
        """Get the current history strategy name."""
        class_name = self.history_manager.__class__.__name__
        # Map class names to strategy IDs
        mapping = {
            "SlidingWindowStrategy": "sliding_window",
            "TokenBudgetStrategy": "token_budget",
            "SummarizationStrategy": "summarization",
            "SemanticFilterStrategy": "semantic",
            "HybridStrategy": "hybrid",
        }
        return mapping.get(class_name, "sliding_window")

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
