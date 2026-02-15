#!/usr/bin/env python3
"""
RAG Evaluation CLI

Command-line interface for running RAG evaluations.

Usage:
    python -m src.evaluation.cli generate --name my_dataset --questions 5
    python -m src.evaluation.cli run --dataset my_dataset
    python -m src.evaluation.cli report --run-id abc123
    python -m src.evaluation.cli list-datasets
    python -m src.evaluation.cli list-results
"""

import argparse
import sys
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_generate(args):
    """Generate an evaluation dataset."""
    from src.evaluation.dataset import DatasetGenerator

    print(f"\nGenerating dataset '{args.name}'...")
    print(f"  Questions per topic: {args.questions}")
    print(f"  Question types: {args.types}")

    generator = DatasetGenerator()
    dataset = generator.generate_dataset(
        name=args.name,
        questions_per_topic=args.questions,
        question_types=args.types
    )

    print(f"\nDataset generated successfully!")
    print(f"  Total questions: {len(dataset.questions)}")
    print(f"  Topics covered: {', '.join(dataset.topics_covered)}")
    print(f"  Saved to: data/eval_datasets/{args.name}.json")


def cmd_run(args):
    """Run evaluation on a dataset."""
    from src.evaluation.dataset import DatasetGenerator
    from src.evaluation.runner import EvaluationRunner, get_evaluation_summary

    print(f"\nLoading dataset '{args.dataset}'...")

    generator = DatasetGenerator()
    dataset = generator.load_dataset(args.dataset)

    if not dataset:
        print(f"Error: Dataset '{args.dataset}' not found")
        sys.exit(1)

    print(f"  Questions: {len(dataset.questions)}")
    print(f"  Topics: {', '.join(dataset.topics_covered)}")

    print(f"\nRunning evaluation...")
    runner = EvaluationRunner()
    results = runner.run_evaluation(
        dataset=dataset,
        experiment_name=args.experiment
    )

    summary = get_evaluation_summary(results)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Run ID: {results.run_id}")
    print(f"Dataset: {results.dataset_name}")
    print(f"Model: {results.model}")
    print(f"\nQuestions: {summary['questions']['successful']}/{summary['questions']['total']} successful")
    print(f"Total time: {summary['timing']['total_seconds']:.2f}s")

    if 'retrieval' in summary:
        print(f"\nRetrieval Metrics:")
        for k, v in summary['retrieval'].items():
            print(f"  {k}: {v:.3f}")

    if 'generation' in summary:
        print(f"\nGeneration Metrics:")
        for k, v in summary['generation'].items():
            print(f"  {k}: {v:.3f}")

    print(f"\nResults saved to: data/eval_results/{results.run_id}.json")

    # Generate report if requested
    if args.report:
        from src.evaluation.dashboard import ReportGenerator
        report_gen = ReportGenerator()
        report_path = report_gen.generate_html_report(results)
        print(f"Report saved to: {report_path}")


def cmd_report(args):
    """Generate HTML report for evaluation results."""
    from src.evaluation.runner import EvaluationRunner
    from src.evaluation.dashboard import ReportGenerator

    print(f"\nLoading results '{args.run_id}'...")

    runner = EvaluationRunner()
    results = runner.load_results(args.run_id)

    if not results:
        print(f"Error: Results for run '{args.run_id}' not found")
        sys.exit(1)

    print(f"Generating report...")
    report_gen = ReportGenerator()
    report_path = report_gen.generate_html_report(
        results,
        output_path=args.output
    )

    print(f"\nReport generated: {report_path}")


def cmd_list_datasets(args):
    """List available evaluation datasets."""
    from src.evaluation.dataset import DatasetGenerator

    generator = DatasetGenerator()
    datasets = generator.list_datasets()

    if not datasets:
        print("\nNo evaluation datasets found.")
        print("Generate one with: python -m src.eval_cli generate --name my_dataset")
        return

    print(f"\n{'='*60}")
    print(f"AVAILABLE DATASETS ({len(datasets)})")
    print(f"{'='*60}")

    for d in datasets:
        print(f"\n  {d['name']}")
        print(f"    Questions: {d['question_count']}")
        print(f"    Topics: {', '.join(d.get('topics', []))}")
        if d.get('created_at'):
            print(f"    Created: {d['created_at'][:19]}")


def cmd_list_results(args):
    """List available evaluation results."""
    from src.evaluation.runner import EvaluationRunner

    runner = EvaluationRunner()
    results = runner.list_results()

    if not results:
        print("\nNo evaluation results found.")
        print("Run an evaluation with: python -m src.eval_cli run --dataset <name>")
        return

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({len(results)})")
    print(f"{'='*60}")

    for r in results:
        print(f"\n  Run ID: {r['run_id']}")
        print(f"    Dataset: {r.get('dataset_name', 'N/A')}")
        print(f"    Model: {r.get('model', 'N/A')}")
        print(f"    Questions: {r.get('successful_questions', 0)}/{r.get('total_questions', 0)}")
        if r.get('timestamp'):
            print(f"    Timestamp: {r['timestamp'][:19]}")


def cmd_show(args):
    """Show detailed results for a run."""
    from src.evaluation.runner import EvaluationRunner, get_evaluation_summary

    runner = EvaluationRunner()
    results = runner.load_results(args.run_id)

    if not results:
        print(f"Error: Results for run '{args.run_id}' not found")
        sys.exit(1)

    summary = get_evaluation_summary(results)

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS: {args.run_id}")
        print(f"{'='*60}")
        print(json.dumps(summary, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a dataset with 5 questions per topic
  python -m src.eval_cli generate --name ai_eval --questions 5

  # Run evaluation and generate report
  python -m src.eval_cli run --dataset ai_eval --report

  # Generate report for existing results
  python -m src.eval_cli report --run-id abc123

  # List available datasets and results
  python -m src.eval_cli list-datasets
  python -m src.eval_cli list-results
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate an evaluation dataset from indexed topics"
    )
    gen_parser.add_argument(
        "--name", "-n",
        required=True,
        help="Name for the dataset"
    )
    gen_parser.add_argument(
        "--questions", "-q",
        type=int,
        default=5,
        help="Questions to generate per topic (default: 5)"
    )
    gen_parser.add_argument(
        "--types", "-t",
        nargs="+",
        default=["factual", "comparative", "explanatory"],
        help="Question types (default: factual comparative explanatory)"
    )

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run evaluation on a dataset"
    )
    run_parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Name of the dataset to evaluate"
    )
    run_parser.add_argument(
        "--experiment", "-e",
        help="Optional experiment name"
    )
    run_parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Generate HTML report after evaluation"
    )

    # Report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate HTML report for evaluation results"
    )
    report_parser.add_argument(
        "--run-id", "-r",
        required=True,
        help="Run ID of the evaluation"
    )
    report_parser.add_argument(
        "--output", "-o",
        help="Output path for the report (default: data/eval_reports/<run_id>.html)"
    )

    # List datasets command
    subparsers.add_parser(
        "list-datasets",
        help="List available evaluation datasets"
    )

    # List results command
    subparsers.add_parser(
        "list-results",
        help="List available evaluation results"
    )

    # Show command
    show_parser = subparsers.add_parser(
        "show",
        help="Show detailed results for a run"
    )
    show_parser.add_argument(
        "--run-id", "-r",
        required=True,
        help="Run ID of the evaluation"
    )
    show_parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to command handlers
    commands = {
        "generate": cmd_generate,
        "run": cmd_run,
        "report": cmd_report,
        "list-datasets": cmd_list_datasets,
        "list-results": cmd_list_results,
        "show": cmd_show,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
