"""
LangSmith Evaluation Module for RAG Pipeline

Provides utilities for creating evaluation datasets and running evaluations
to measure RAG pipeline quality.

This module provides LangSmith integration. For comprehensive RAG evaluation
with retrieval and generation metrics, see the evaluation package:
    from src.evaluation import quick_evaluate, generate_report

Quick start:
    # CLI
    python -m src.evaluation.cli generate --name my_dataset --questions 5
    python -m src.evaluation.cli run --dataset my_dataset --report

    # API
    POST /api/v1/eval/dataset/generate
    POST /api/v1/eval/run
    GET /api/v1/eval/results/{run_id}/report
"""

from typing import Optional
import logging

from src.config import settings

logger = logging.getLogger(__name__)


def get_langsmith_client():
    """Get LangSmith client if configured.

    Returns:
        LangSmith Client instance or None if not configured.
    """
    if not settings.langsmith.is_configured:
        logger.warning("LangSmith not configured - evaluation unavailable")
        return None

    try:
        from langsmith import Client
        return Client()
    except ImportError:
        logger.error("langsmith package not installed")
        return None


def create_evaluation_dataset(
    dataset_name: str,
    examples: list[dict],
) -> Optional[str]:
    """Create a dataset for evaluation.

    Args:
        dataset_name: Name for the dataset
        examples: List of dicts with 'input' and optional 'expected_output' keys
            Example: [{"input": "What is AI?", "expected_output": "AI is..."}]

    Returns:
        Dataset ID if successful, None otherwise.
    """
    client = get_langsmith_client()
    if not client:
        raise RuntimeError("LangSmith not configured")

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=f"Evaluation dataset for {settings.langsmith.project}"
    )

    for example in examples:
        client.create_example(
            inputs={"question": example["input"]},
            outputs={"answer": example.get("expected_output", "")},
            dataset_id=dataset.id,
        )

    logger.info(f"Created dataset '{dataset_name}' with {len(examples)} examples")
    return dataset.id


def run_rag_evaluation(
    dataset_name: str,
    experiment_prefix: str = "rag-eval",
) -> dict:
    """Run evaluation on the RAG pipeline.

    Args:
        dataset_name: Name of the dataset to evaluate against
        experiment_prefix: Prefix for the experiment name

    Returns:
        Evaluation results summary.
    """
    client = get_langsmith_client()
    if not client:
        raise RuntimeError("LangSmith not configured")

    from langsmith.evaluation import evaluate, LangChainStringEvaluator
    from src.graph import get_chatbot

    bot = get_chatbot()

    def predict(inputs: dict) -> dict:
        """Prediction function for evaluation."""
        result = bot.chat(
            session_id=f"eval-{hash(inputs.get('question', '')) % 10000}",
            user_message=inputs["question"]
        )
        return {
            "answer": result["response"],
            "citations": result.get("citations", [])
        }

    # Define evaluators for different quality dimensions
    evaluators = [
        LangChainStringEvaluator("qa"),  # QA correctness
        LangChainStringEvaluator("criteria", config={"criteria": "helpfulness"}),
        LangChainStringEvaluator("criteria", config={"criteria": "relevance"}),
    ]

    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
    )

    return results


# Sample evaluation questions for RAG testing on AI/ML topics
SAMPLE_EVALUATION_QUESTIONS = [
    {
        "input": "What is artificial intelligence?",
        "expected_output": "Artificial intelligence (AI) is the simulation of human intelligence by machines, particularly computer systems."
    },
    {
        "input": "Explain neural networks.",
        "expected_output": "Neural networks are computing systems inspired by biological neural networks in the brain, consisting of interconnected nodes that process information."
    },
    {
        "input": "What is deep learning?",
        "expected_output": "Deep learning is a subset of machine learning that uses neural networks with many layers (deep neural networks) to learn representations of data."
    },
    {
        "input": "How does machine learning differ from traditional programming?",
        "expected_output": "Machine learning learns patterns from data rather than following explicit programmed instructions, allowing systems to improve from experience."
    },
    {
        "input": "What is natural language processing used for?",
        "expected_output": "Natural language processing (NLP) is used for text analysis, machine translation, chatbots, sentiment analysis, and understanding human language by computers."
    },
    {
        "input": "What are the main types of machine learning?",
        "expected_output": "The main types are supervised learning (labeled data), unsupervised learning (unlabeled data), and reinforcement learning (learning from rewards)."
    },
    {
        "input": "What is a transformer model?",
        "expected_output": "A transformer is a deep learning architecture that uses self-attention mechanisms to process sequential data, widely used in NLP tasks like translation and text generation."
    },
    {
        "input": "What is the difference between AI and ML?",
        "expected_output": "AI is the broader concept of machines performing tasks intelligently, while ML is a subset of AI where machines learn from data without explicit programming."
    },
]


def get_langsmith_status() -> dict:
    """Get the current LangSmith configuration status.

    Returns:
        Dictionary with status information.
    """
    return {
        "enabled": settings.langsmith.is_configured,
        "project": settings.langsmith.project if settings.langsmith.is_configured else None,
        "endpoint": settings.langsmith.endpoint if settings.langsmith.is_configured else None,
    }


# ==================== Comprehensive Evaluation Functions ====================

def quick_evaluate(
    dataset_name: str = "quick_eval",
    questions_per_topic: int = 3
) -> dict:
    """Run a quick end-to-end evaluation and return summary.

    This is a convenience function that:
    1. Generates an evaluation dataset (if not exists)
    2. Runs the full evaluation pipeline
    3. Returns a summary of results

    Args:
        dataset_name: Name for the evaluation dataset
        questions_per_topic: Questions to generate per topic

    Returns:
        Dictionary with evaluation summary including retrieval and generation metrics
    """
    from src.evaluation.dataset import DatasetGenerator
    from src.evaluation.runner import EvaluationRunner, get_evaluation_summary

    generator = DatasetGenerator()
    dataset = generator.load_dataset(dataset_name)

    if not dataset:
        logger.info(f"Generating dataset '{dataset_name}'...")
        dataset = generator.generate_dataset(
            name=dataset_name,
            questions_per_topic=questions_per_topic
        )

    logger.info("Running evaluation...")
    runner = EvaluationRunner()
    results = runner.run_evaluation(dataset)

    return get_evaluation_summary(results)


def generate_evaluation_report(run_id: str) -> str:
    """Generate an HTML report for evaluation results.

    Args:
        run_id: The evaluation run ID

    Returns:
        Path to the generated HTML report
    """
    from src.evaluation.runner import EvaluationRunner
    from src.evaluation.dashboard import ReportGenerator

    runner = EvaluationRunner()
    results = runner.load_results(run_id)

    if not results:
        raise ValueError(f"Results for run '{run_id}' not found")

    report_gen = ReportGenerator()
    return report_gen.generate_html_report(results)
