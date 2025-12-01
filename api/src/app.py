"""
FastAPI Server for RAG Chatbot
Exposes REST API endpoints for chat interaction
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.router import router
from src.config import settings
from fastapi.responses import FileResponse


def get_application() -> FastAPI:

    # Initialize FastAPI app with config
    app = FastAPI(
        title=settings.api.title,
        description="Context-aware chatbot powered by LangGraph and Ollama",
        version=settings.api.version
    )

    # Enable CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = get_application()
app.include_router(router=router, prefix="/api/v1")


@app.get("/", response_class=FileResponse)
async def serve_ui():
    """Serve the web UI."""
    return FileResponse("index.html")


