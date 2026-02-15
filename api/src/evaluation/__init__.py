"""
RAG Evaluation Package

Comprehensive evaluation pipeline for measuring RAG system quality.

Modules:
- dataset: Generate evaluation Q&A pairs from indexed content
- retrieval: Retrieval quality metrics (Precision, Recall, MRR, NDCG)
- generation: Generation quality metrics (Faithfulness, Relevance, etc.)
- runner: Orchestrate full evaluation pipeline
- dashboard: Generate visual HTML reports
- cli: Command-line interface

Quick Start:
    # CLI
    python -m src.evaluation.cli generate --name my_dataset --questions 5
    python -m src.evaluation.cli run --dataset my_dataset --report

    # API
    POST /api/v1/eval/dataset/generate
    POST /api/v1/eval/run
    GET /api/v1/eval/results/{run_id}/report

    # Python
    from src.evaluation import quick_evaluate, generate_report
    summary = quick_evaluate("my_dataset")
    report_path = generate_report(summary["run_id"])
"""

from src.evaluation.dataset import (
    DatasetGenerator,
    EvalDataset,
    EvalQuestion,
)
from src.evaluation.retrieval import (
    RetrievalMetrics,
    aggregate_retrieval_metrics,
)
from src.evaluation.generation import (
    GenerationMetrics,
    aggregate_generation_metrics,
)
from src.evaluation.runner import (
    EvaluationRunner,
    EvalResults,
    QuestionResult,
    get_evaluation_summary,
)
from src.evaluation.dashboard import ReportGenerator

__all__ = [
    # Dataset
    "DatasetGenerator",
    "EvalDataset",
    "EvalQuestion",
    # Retrieval
    "RetrievalMetrics",
    "aggregate_retrieval_metrics",
    # Generation
    "GenerationMetrics",
    "aggregate_generation_metrics",
    # Runner
    "EvaluationRunner",
    "EvalResults",
    "QuestionResult",
    "get_evaluation_summary",
    # Dashboard
    "ReportGenerator",
]


def quick_evaluate(
    dataset_name: str = "quick_eval",
    questions_per_topic: int = 3
) -> dict:
    """Run a quick end-to-end evaluation and return summary.

    Args:
        dataset_name: Name for the evaluation dataset
        questions_per_topic: Questions to generate per topic

    Returns:
        Dictionary with evaluation summary
    """
    generator = DatasetGenerator()
    dataset = generator.load_dataset(dataset_name)

    if not dataset:
        dataset = generator.generate_dataset(
            name=dataset_name,
            questions_per_topic=questions_per_topic
        )

    runner = EvaluationRunner()
    results = runner.run_evaluation(dataset)
    return get_evaluation_summary(results)


def generate_report(run_id: str) -> str:
    """Generate an HTML report for evaluation results.

    Args:
        run_id: The evaluation run ID

    Returns:
        Path to the generated HTML report
    """
    runner = EvaluationRunner()
    results = runner.load_results(run_id)

    if not results:
        raise ValueError(f"Results for run '{run_id}' not found")

    report_gen = ReportGenerator()
    return report_gen.generate_html_report(results)
