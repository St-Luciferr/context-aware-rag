import traceback
from fastapi import APIRouter, HTTPException
import uuid
from src.schemas import StatusResponse, ChatRequest, ChatResponse, ConfigResponse, HistoryResponse, SessionListResponse
from src.graph import get_chatbot
from src.config import settings

router = APIRouter(prefix="", tags=["default"])


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Check API and component status."""
    try:
        bot = get_chatbot()
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
        bot = get_chatbot()
        sessions = bot.get_all_sessions()
        return SessionListResponse(
            total_sessions=len(sessions),
            sessions=sessions
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
        retrieval_k=settings.rag.retrieval_k
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response."""
    try:
        bot = get_chatbot()

        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Get response from chatbot
        response = bot.chat(session_id, request.message)

        return ChatResponse(response=response, session_id=session_id)

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Get conversation history for a session."""
    try:
        bot = get_chatbot()
        history = bot.get_history(session_id)
        return HistoryResponse(session_id=session_id, messages=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    try:
        bot = get_chatbot()
        bot.clear_session(session_id)
        return {"message": f"Session {session_id} cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/new")
async def new_session():
    """Create a new chat session."""
    return {"session_id": str(uuid.uuid4())}
