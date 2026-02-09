"""
LangSmith Evaluation Module for RAG Pipeline

Provides utilities for creating evaluation datasets and running evaluations
to measure RAG pipeline quality.
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
