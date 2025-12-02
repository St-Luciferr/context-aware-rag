import traceback
from fastapi import APIRouter, HTTPException
import uuid
from src.schemas import (
    SessionInfo,
    StatusResponse,
    ChatRequest,
    ChatResponse,
    ConfigResponse,
    HistoryResponse,
    SessionListResponse,
    Citation,
    MessageResponse,
    StrategiesResponse,
    ChangeStrategyRequest,
    ChangeStrategyResponse
)

from src.schemas import STRATEGY_INFO

from src.graph import get_chatbot
from src.config import settings
from src.ingest import run_ingestion
from src.history_manager import HistoryConfig

router = APIRouter(prefix="", tags=["default"])

history_config = HistoryConfig(
    max_messages=settings.history.max_messages,
    max_tokens=settings.history.max_tokens,
    model_name=settings.ollama.model,
    summarize_after=settings.history.summarize_after,
    summary_max_tokens=settings.history.summary_max_tokens
)


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Check API and component status."""
    try:
        _ = get_chatbot(
            history_strategy=settings.history.strategy,
            history_config=history_config
        )
        return StatusResponse(
            status="online",
            model=settings.ollama.model,
            llm_url=settings.ollama.base_url,
            vector_store="ChromaDB",
            collection=settings.chroma.collection_name
        )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/sessions", response_model=SessionListResponse)
async def get_all_sessions():
    """Get all active chat sessions."""
    try:

        bot = get_chatbot(
            history_strategy=settings.history.strategy,
            history_config=history_config
        )
        sessions = bot.get_all_sessions()
        return SessionListResponse(
            total_sessions=len(sessions),
            sessions=[SessionInfo(**s) for s in sessions]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration (non-sensitive)."""
    return ConfigResponse(
        ollama_model=settings.ollama.model,
        ollama_base_url=settings.ollama.base_url,
        chroma_collection=settings.chroma.collection_name,
        embedding_model=settings.embedding.model_name,
        retrieval_k=settings.rag.retrieval_k,
        history_strategy=getattr(
            settings.history, 'strategy', 'sliding_window'),
        history_max_messages=getattr(settings.history, 'max_messages', 6),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response."""
    try:
        bot = get_chatbot(
            history_strategy=settings.history.strategy,
            history_config=history_config
        )

        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Get response from chatbot
        result = bot.chat(session_id, request.message)
        citations = [
            Citation(**cite) for cite in result.get("citations", [])
        ]

        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            citations=citations
        )

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Get conversation history for a session."""
    try:
        bot = get_chatbot(
            history_strategy=settings.history.strategy,
            history_config=history_config
        )
        history = bot.get_history(session_id)
        messages = []
        for msg in history:
            message = MessageResponse(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg.get("timestamp"),
                citations=[Citation(
                    **c) for c in msg.get("citations", [])] if msg.get("citations") else None
            )
            messages.append(message)
        return HistoryResponse(session_id=session_id, messages=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ingest")
async def ingest_documents():
    """
    Ingest the documents
    """
    resp = run_ingestion()
    return resp


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    try:
        bot = get_chatbot(
            history_strategy=settings.history.strategy,
            history_config=history_config
        )
        bot.clear_session(session_id)
        return {"message": f"Session {session_id} cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/new")
async def new_session():
    """Create a new chat session."""
    return {"session_id": str(uuid.uuid4())}


@router.get("/strategies", response_model=StrategiesResponse)
async def get_strategies():
    """Get available history management strategies."""
    try:
        bot = get_chatbot(
            history_strategy=settings.history.strategy,
            history_config=history_config
        )
        current_id = bot.get_current_strategy()

        return StrategiesResponse(
            current=current_id,
            strategies=list(STRATEGY_INFO.values())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies", response_model=ChangeStrategyResponse)
async def change_strategy(request: ChangeStrategyRequest):
    """Change the history management strategy."""
    try:
        if request.strategy not in STRATEGY_INFO:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy. Choose from: {list(STRATEGY_INFO.keys())}"
            )

        bot = get_chatbot()
        bot.set_history_strategy(request.strategy)

        return ChangeStrategyResponse(
            success=True,
            current_strategy=request.strategy,
            message=f"Strategy changed to {STRATEGY_INFO[request.strategy].name}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
