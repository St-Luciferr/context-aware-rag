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
    ChangeStrategyResponse,
    TopicsResponse,
    AddTopicRequest,
    AddTopicResponse,
    RemoveTopicResponse,
    IngestResponse,
    DatasetGenerateRequest,
    DatasetListResponse,
    DatasetGenerateResponse,
    DatasetInfo,
    EvalRunRequest,
    EvalRunResponse,
    EvalResultsListResponse,
    EvalResultInfo,
    EvalSummaryResponse,
    EvalReportResponse,
)

from src.schemas import STRATEGY_INFO

from src.graph import get_chatbot
from src.config import settings
from src.ingest import run_ingestion, add_topics_to_existing
from src.history_manager import HistoryConfig
from src.topics import (
    get_topics_status,
    get_pending_topics,
    add_topic,
    remove_topic,
    reset_additional_topics,
)

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
async def get_history(
    session_id: str,
    limit: int = 20,
    before: str = None
):
    """Get conversation history for a session with pagination.

    Args:
        session_id: The session ID
        limit: Maximum number of messages to return (default 20)
        before: Only return messages before this timestamp (ISO format) for pagination
    """
    try:
        bot = get_chatbot(
            history_strategy=settings.history.strategy,
            history_config=history_config
        )
        result = bot.get_history_paginated(session_id, limit=limit, before_timestamp=before)

        return HistoryResponse(
            session_id=session_id,
            messages=result["messages"],
            has_more=result["has_more"],
            total_count=result["total_count"],
            oldest_timestamp=result["oldest_timestamp"]
        )
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


# ==================== Topics Endpoints ====================

@router.get("/topics", response_model=TopicsResponse)
async def get_topics():
    """Get current Wikipedia topics with ingestion status."""
    try:
        status = get_topics_status()
        return TopicsResponse(
            topics=status["topics"],
            total=status["total"],
            ingested_count=status["ingested_count"],
            pending_count=status["pending_count"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/topics", response_model=AddTopicResponse)
async def add_new_topic(request: AddTopicRequest):
    """Add a new Wikipedia topic (does not ingest immediately)."""
    try:
        success, message = add_topic(request.topic)

        if not success:
            raise HTTPException(status_code=400, detail=message)

        return AddTopicResponse(success=success, message=message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/topics/{topic}", response_model=RemoveTopicResponse)
async def delete_topic(topic: str):
    """Remove a Wikipedia topic (only additional topics can be removed)."""
    try:
        success, message = remove_topic(topic)

        if not success:
            raise HTTPException(status_code=400, detail=message)

        return RemoveTopicResponse(success=success, message=message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/topics/reset", response_model=AddTopicResponse)
async def reset_topics_to_default():
    """Remove all additional topics, keeping only defaults."""
    try:
        success, message = reset_additional_topics()
        return AddTopicResponse(success=success, message=message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/topics/ingest", response_model=IngestResponse)
async def ingest_pending_topics():
    """Ingest only pending topics (incremental - adds to existing collection)."""
    try:
        pending = get_pending_topics()

        if not pending:
            return IngestResponse(
                status="skipped",
                message="No pending topics to ingest. All topics are already in the knowledge base.",
                document_count=0
            )

        result = add_topics_to_existing(pending)

        return IngestResponse(
            status=result.get("status", "error"),
            message=result.get("message", "Unknown error"),
            document_count=result.get("document_count"),
            chunks_added=result.get("chunks_added"),
            topics_added=result.get("topics_added")
        )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/topics/ingest/full", response_model=IngestResponse)
async def full_reingest():
    """Full re-ingestion - deletes existing collection and re-ingests all topics."""
    try:
        result = run_ingestion(force=True)

        return IngestResponse(
            status=result.get("status", "error"),
            message=result.get("message", "Unknown error"),
            document_count=result.get("document_count")
        )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Evaluation Endpoints ====================

@router.post("/eval/dataset/generate", response_model=DatasetGenerateResponse)
async def generate_eval_dataset(request: DatasetGenerateRequest):
    """Generate an evaluation dataset from indexed topics."""
    try:
        from src.evaluation.dataset import DatasetGenerator

        generator = DatasetGenerator()
        dataset = generator.generate_dataset(
            name=request.name,
            questions_per_topic=request.questions_per_topic,
            question_types=request.question_types
        )

        return DatasetGenerateResponse(
            success=True,
            name=dataset.name,
            question_count=len(dataset.questions),
            topics_covered=dataset.topics_covered,
            message=f"Generated {len(dataset.questions)} questions across {len(dataset.topics_covered)} topics"
        )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eval/datasets", response_model=DatasetListResponse)
async def list_eval_datasets():
    """List available evaluation datasets."""
    try:
        from src.evaluation.dataset import DatasetGenerator

        generator = DatasetGenerator()
        datasets = generator.list_datasets()

        return DatasetListResponse(
            datasets=[DatasetInfo(**d) for d in datasets],
            total=len(datasets)
        )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/eval/datasets/{name}")
async def delete_eval_dataset(name: str):
    """Delete an evaluation dataset."""
    try:
        from src.evaluation.dataset import DatasetGenerator

        generator = DatasetGenerator()
        success = generator.delete_dataset(name)

        if not success:
            raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

        return {"success": True, "message": f"Dataset '{name}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/eval/run", response_model=EvalRunResponse)
async def run_evaluation(request: EvalRunRequest):
    """Run evaluation on a dataset."""
    try:
        from src.evaluation.dataset import DatasetGenerator
        from src.evaluation.runner import EvaluationRunner

        # Load dataset
        generator = DatasetGenerator()
        dataset = generator.load_dataset(request.dataset_name)

        if not dataset:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{request.dataset_name}' not found"
            )

        # Run evaluation
        runner = EvaluationRunner()
        results = runner.run_evaluation(
            dataset=dataset,
            experiment_name=request.experiment_name
        )

        return EvalRunResponse(
            success=True,
            run_id=results.run_id,
            dataset_name=results.dataset_name,
            total_questions=results.total_questions,
            successful_questions=results.successful_questions,
            failed_questions=results.failed_questions,
            total_time_seconds=results.total_time_seconds,
            message=f"Evaluation complete. {results.successful_questions}/{results.total_questions} questions successful."
        )
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eval/results", response_model=EvalResultsListResponse)
async def list_eval_results():
    """List available evaluation results."""
    try:
        from src.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner()
        results = runner.list_results()

        return EvalResultsListResponse(
            results=[EvalResultInfo(**r) for r in results],
            total=len(results)
        )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eval/results/{run_id}", response_model=EvalSummaryResponse)
async def get_eval_results(run_id: str):
    """Get summary of evaluation results."""
    try:
        from src.evaluation.runner import EvaluationRunner, get_evaluation_summary

        runner = EvaluationRunner()
        results = runner.load_results(run_id)

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"Results for run '{run_id}' not found"
            )

        summary = get_evaluation_summary(results)
        return EvalSummaryResponse(**summary)
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eval/results/{run_id}/report", response_model=EvalReportResponse)
async def generate_eval_report(run_id: str):
    """Generate HTML report for evaluation results."""
    try:
        from src.evaluation.runner import EvaluationRunner
        from src.evaluation.dashboard import ReportGenerator

        runner = EvaluationRunner()
        results = runner.load_results(run_id)

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"Results for run '{run_id}' not found"
            )

        report_gen = ReportGenerator()
        report_path = report_gen.generate_html_report(results)

        return EvalReportResponse(
            success=True,
            run_id=run_id,
            report_path=report_path,
            message=f"Report generated at {report_path}"
        )
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/eval/results/{run_id}")
async def delete_eval_results(run_id: str):
    """Delete evaluation results."""
    try:
        from src.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner()
        success = runner.delete_results(run_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Results for run '{run_id}' not found"
            )

        return {"success": True, "message": f"Results '{run_id}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
