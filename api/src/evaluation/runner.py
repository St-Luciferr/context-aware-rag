"""
Evaluation Runner Module

Orchestrates the full RAG evaluation pipeline.
"""

import json
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
import logging

from src.config import settings
from src.evaluation.dataset import EvalDataset, EvalQuestion, DatasetGenerator
from src.evaluation.retrieval import RetrievalMetrics, aggregate_retrieval_metrics
from src.evaluation.generation import GenerationMetrics, aggregate_generation_metrics
from src.graph import get_chatbot, RAGChatbot

logger = logging.getLogger(__name__)


class QuestionResult(BaseModel):
    """Results for a single evaluation question."""
    question_id: str
    question: str
    question_type: str
    topic: str

    # Retrieved data
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    retrieved_contexts: list[str] = Field(default_factory=list)

    # Generated data
    generated_answer: str = ""

    # Retrieval metrics
    retrieval_metrics: dict = Field(default_factory=dict)

    # Generation metrics
    generation_metrics: dict = Field(default_factory=dict)

    # Timing
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Ground truth
    ground_truth_answer: str = ""
    ground_truth_chunk_ids: list[str] = Field(default_factory=list)


class EvalResults(BaseModel):
    """Complete evaluation results for a dataset."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    dataset_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str = ""

    # Configuration
    config: dict = Field(default_factory=dict)

    # Per-question results
    question_results: list[QuestionResult] = Field(default_factory=list)

    # Aggregated metrics
    aggregated_retrieval: dict = Field(default_factory=dict)
    aggregated_generation: dict = Field(default_factory=dict)

    # Summary statistics
    total_questions: int = 0
    successful_questions: int = 0
    failed_questions: int = 0
    total_time_seconds: float = 0.0


class EvaluationRunner:
    """Orchestrates the full RAG evaluation pipeline."""

    def __init__(
        self,
        chatbot: Optional[RAGChatbot] = None
    ):
        """Initialize the evaluation runner.

        Args:
            chatbot: RAGChatbot instance. If None, uses singleton.
        """
        self.chatbot = chatbot or get_chatbot()
        self.retrieval_metrics = RetrievalMetrics(self.chatbot.embeddings)
        self.generation_metrics = GenerationMetrics()
        self.dataset_generator = DatasetGenerator()

    def _retrieve_for_question(
        self,
        question: str
    ) -> tuple[list[str], list[str], float]:
        """Run retrieval for a question and return results.

        Returns:
            Tuple of (doc_ids, contexts, time_ms)
        """
        start_time = time.time()

        # Use the chatbot's hybrid search directly
        docs = self.chatbot.hybrid_search(question)

        elapsed_ms = (time.time() - start_time) * 1000

        doc_ids = []
        contexts = []
        for doc in docs:
            doc_id = doc.metadata.get("chunk_id") or doc.metadata.get("id", "")
            doc_ids.append(doc_id)
            contexts.append(doc.page_content)

        return doc_ids, contexts, elapsed_ms

    def _generate_for_question(
        self,
        question: str,
        session_id: str
    ) -> tuple[str, float]:
        """Run generation for a question and return results.

        Returns:
            Tuple of (answer, time_ms)
        """
        start_time = time.time()

        result = self.chatbot.chat(
            session_id=session_id,
            user_message=question
        )

        elapsed_ms = (time.time() - start_time) * 1000

        return result.get("response", ""), elapsed_ms

    def evaluate_single_question(
        self,
        eval_question: EvalQuestion,
        session_prefix: str = "eval"
    ) -> QuestionResult:
        """Evaluate a single question.

        Args:
            eval_question: The question to evaluate
            session_prefix: Prefix for the session ID

        Returns:
            QuestionResult with all metrics
        """
        question_id = eval_question.id
        question = eval_question.question
        session_id = f"{session_prefix}-{question_id}"

        logger.info(f"Evaluating question: {question[:50]}...")

        result = QuestionResult(
            question_id=question_id,
            question=question,
            question_type=eval_question.question_type,
            topic=eval_question.topic,
            ground_truth_answer=eval_question.ground_truth_answer,
            ground_truth_chunk_ids=eval_question.ground_truth_chunk_ids
        )

        start_time = time.time()

        try:
            # Step 1: Retrieval
            doc_ids, contexts, retrieval_time = self._retrieve_for_question(question)
            result.retrieved_doc_ids = doc_ids
            result.retrieved_contexts = contexts
            result.retrieval_time_ms = retrieval_time

            # Step 2: Compute retrieval metrics
            ground_truth_ids = set(eval_question.ground_truth_chunk_ids)
            from langchain_core.documents import Document
            retrieved_docs = [
                Document(page_content=ctx, metadata={"chunk_id": doc_id})
                for doc_id, ctx in zip(doc_ids, contexts)
            ]
            result.retrieval_metrics = self.retrieval_metrics.compute_all_metrics(
                query=question,
                retrieved_docs=retrieved_docs,
                ground_truth_ids=ground_truth_ids
            )

            # Step 3: Generation
            answer, generation_time = self._generate_for_question(
                question, session_id
            )
            result.generated_answer = answer
            result.generation_time_ms = generation_time

            # Step 4: Compute generation metrics
            result.generation_metrics = self.generation_metrics.compute_all_metrics(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=eval_question.ground_truth_answer
            )

        except Exception as e:
            logger.error(f"Error evaluating question {question_id}: {e}")
            result.generation_metrics = {"error": str(e)}

        result.total_time_ms = (time.time() - start_time) * 1000
        return result

    def run_evaluation(
        self,
        dataset: EvalDataset,
        experiment_name: Optional[str] = None
    ) -> EvalResults:
        """Run full evaluation on a dataset.

        Args:
            dataset: The evaluation dataset
            experiment_name: Optional name for this evaluation run

        Returns:
            EvalResults with all metrics
        """
        logger.info(f"Starting evaluation of dataset '{dataset.name}' "
                   f"with {len(dataset.questions)} questions")

        start_time = time.time()

        results = EvalResults(
            dataset_name=dataset.name,
            model=settings.ollama.model,
            config={
                "retrieval_k": settings.rag.retrieval_k,
                "distilled_k": settings.rag.distilled_retrieval_k,
                "embedding_model": settings.embedding.model_name,
                "experiment_name": experiment_name
            },
            total_questions=len(dataset.questions)
        )

        session_prefix = f"eval-{results.run_id}"

        # Evaluate each question
        all_retrieval_metrics = []
        all_generation_metrics = []

        for i, question in enumerate(dataset.questions):
            logger.info(f"Evaluating question {i+1}/{len(dataset.questions)}")

            try:
                q_result = self.evaluate_single_question(
                    question, session_prefix
                )
                results.question_results.append(q_result)
                results.successful_questions += 1

                all_retrieval_metrics.append(q_result.retrieval_metrics)
                all_generation_metrics.append(q_result.generation_metrics)

            except Exception as e:
                logger.error(f"Failed to evaluate question {question.id}: {e}")
                results.failed_questions += 1

        # Aggregate metrics
        results.aggregated_retrieval = aggregate_retrieval_metrics(
            all_retrieval_metrics
        )
        results.aggregated_generation = aggregate_generation_metrics(
            all_generation_metrics
        )

        results.total_time_seconds = time.time() - start_time

        # Save results
        self.save_results(results)

        logger.info(f"Evaluation complete. Run ID: {results.run_id}")
        logger.info(f"Successful: {results.successful_questions}, "
                   f"Failed: {results.failed_questions}")

        return results

    def save_results(self, results: EvalResults) -> Path:
        """Save evaluation results to JSON file."""
        results_dir = Path(settings.evaluation.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        filepath = results_dir / f"{results.run_id}.json"
        with open(filepath, "w") as f:
            f.write(results.model_dump_json(indent=2))

        logger.info(f"Results saved to {filepath}")
        return filepath

    def load_results(self, run_id: str) -> Optional[EvalResults]:
        """Load evaluation results from JSON file."""
        filepath = Path(settings.evaluation.results_dir) / f"{run_id}.json"
        if not filepath.exists():
            logger.warning(f"Results file not found: {filepath}")
            return None

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return EvalResults(**data)
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return None

    def list_results(self) -> list[dict]:
        """List all available evaluation results."""
        results_dir = Path(settings.evaluation.results_dir)
        if not results_dir.exists():
            return []

        results = []
        for filepath in results_dir.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                results.append({
                    "run_id": data.get("run_id", filepath.stem),
                    "dataset_name": data.get("dataset_name"),
                    "timestamp": data.get("timestamp"),
                    "total_questions": data.get("total_questions", 0),
                    "successful_questions": data.get("successful_questions", 0),
                    "model": data.get("model")
                })
            except Exception as e:
                logger.warning(f"Error reading results {filepath}: {e}")

        # Sort by timestamp descending
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results

    def delete_results(self, run_id: str) -> bool:
        """Delete evaluation results file."""
        filepath = Path(settings.evaluation.results_dir) / f"{run_id}.json"
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted results: {run_id}")
            return True
        return False


def get_evaluation_summary(results: EvalResults) -> dict:
    """Get a summary of evaluation results for display.

    Args:
        results: The evaluation results

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "run_id": results.run_id,
        "dataset": results.dataset_name,
        "model": results.model,
        "timestamp": results.timestamp.isoformat() if results.timestamp else None,
        "questions": {
            "total": results.total_questions,
            "successful": results.successful_questions,
            "failed": results.failed_questions
        },
        "timing": {
            "total_seconds": round(results.total_time_seconds, 2),
            "avg_per_question_ms": round(
                results.total_time_seconds * 1000 / max(1, results.total_questions),
                2
            )
        }
    }

    # Add retrieval metrics summary
    if results.aggregated_retrieval:
        ret = results.aggregated_retrieval
        summary["retrieval"] = {
            "mrr": round(ret.get("mrr", {}).get("mean", 0), 3),
            "hit_rate": round(ret.get("hit_rate", {}).get("mean", 0), 3),
            "context_relevance": round(
                ret.get("context_relevance", {}).get("mean", 0), 3
            )
        }
        # Add precision/recall at common K values
        for k in [1, 3, 5]:
            p_at_k = ret.get("precision_at_k", {}).get(k, {})
            r_at_k = ret.get("recall_at_k", {}).get(k, {})
            if p_at_k:
                summary["retrieval"][f"precision@{k}"] = round(
                    p_at_k.get("mean", 0), 3
                )
            if r_at_k:
                summary["retrieval"][f"recall@{k}"] = round(
                    r_at_k.get("mean", 0), 3
                )

    # Add generation metrics summary
    if results.aggregated_generation:
        gen = results.aggregated_generation
        summary["generation"] = {
            "faithfulness": round(
                gen.get("faithfulness", {}).get("mean", 0), 3
            ),
            "answer_relevance": round(
                gen.get("answer_relevance", {}).get("mean", 0), 3
            ),
            "context_precision": round(
                gen.get("context_precision", {}).get("mean", 0), 3
            ),
            "answer_correctness": round(
                gen.get("answer_correctness", {}).get("mean", 0), 3
            )
        }

    return summary
