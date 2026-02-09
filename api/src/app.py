"""
FastAPI Server for RAG Chatbot
Exposes REST API endpoints for chat interaction
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.router import router
from src.config import settings
from fastapi.responses import JSONResponse
from src.graph import get_chatbot
from src.ingest import run_ingestion
from contextlib import asynccontextmanager


def configure_langsmith() -> bool:
    """Configure LangSmith environment variables for tracing.

    Returns:
        True if LangSmith tracing is enabled, False otherwise.
    """
    if settings.langsmith.is_configured:
        # Set environment variables that LangChain/LangGraph read automatically
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith.api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith.project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith.endpoint
        print(f"LangSmith tracing enabled for project: {settings.langsmith.project}")
        return True
    else:
        # Explicitly disable to avoid accidental tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        print("LangSmith tracing disabled (no API key or not enabled)")
        return False


# ----------------------------
# Lifespan: startup + shutdown
# ----------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure LangSmith BEFORE initializing chatbot
    configure_langsmith()

    # Initialize chatbot at startup
    _ = get_chatbot()
    print("Chatbot initialized and ready!")

    result = run_ingestion()
    if result.get("status") == "success":
        print(f"{result.get('message')} document_count: {result.get('document_count')}")
    elif result.get("status") == "skipped":
        print(f"{result.get('message')}")

    else:
        print(f"Error in Ingestion: {result.get('message')}")

    yield

    print("🛑 Shutting down chatbot...")


def get_application() -> FastAPI:

    # Initialize FastAPI app with config
    app = FastAPI(
        title=settings.api.title,
        description="Context-aware chatbot powered by LangGraph and Ollama",
        version=settings.api.version,
        lifespan=lifespan
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


@app.get("/", response_class=JSONResponse)
async def serve_ui():
    return JSONResponse(
        status_code=200,
        content={
            "status": "Running"
        }
    )
