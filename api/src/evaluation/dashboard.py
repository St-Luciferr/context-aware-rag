"""
Evaluation Dashboard and Visualization Module

Generates visual reports and HTML dashboards for evaluation results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from src.config import settings
from src.evaluation.runner import EvalResults, get_evaluation_summary

logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Install with: pip install plotly")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Install with: pip install matplotlib")


class ReportGenerator:
    """Generates HTML reports and visualizations for evaluation results."""

    def __init__(self):
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Ensure report directories exist."""
        Path(settings.evaluation.reports_dir).mkdir(parents=True, exist_ok=True)

    def generate_radar_chart(self, results: EvalResults) -> Optional[str]:
        """Generate a radar chart showing all key metrics.

        Returns:
            HTML string with embedded chart, or None if plotly unavailable
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Extract metrics
        ret = results.aggregated_retrieval
        gen = results.aggregated_generation

        categories = []
        values = []

        # Retrieval metrics
        if "mrr" in ret:
            categories.append("MRR")
            values.append(ret["mrr"].get("mean", 0))
        if "hit_rate" in ret:
            categories.append("Hit Rate")
            values.append(ret["hit_rate"].get("mean", 0))
        if "context_relevance" in ret:
            categories.append("Context Relevance")
            values.append(ret["context_relevance"].get("mean", 0))

        # Generation metrics
        if "faithfulness" in gen:
            categories.append("Faithfulness")
            values.append(gen["faithfulness"].get("mean", 0))
        if "answer_relevance" in gen:
            categories.append("Answer Relevance")
            values.append(gen["answer_relevance"].get("mean", 0))
        if "context_precision" in gen:
            categories.append("Context Precision")
            values.append(gen["context_precision"].get("mean", 0))
        if "answer_correctness" in gen:
            categories.append("Answer Correctness")
            values.append(gen["answer_correctness"].get("mean", 0))

        if not categories:
            return None

        # Close the radar chart
        categories.append(categories[0])
        values.append(values[0])

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Metrics',
            line_color='rgb(99, 110, 250)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="RAG Evaluation Metrics Overview",
            height=500
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def generate_bar_chart(self, results: EvalResults) -> Optional[str]:
        """Generate bar charts for per-question scores.

        Returns:
            HTML string with embedded chart, or None if plotly unavailable
        """
        if not PLOTLY_AVAILABLE:
            return None

        questions = results.question_results
        if not questions:
            return None

        # Extract data
        q_labels = [f"Q{i+1}" for i in range(len(questions))]
        faithfulness = []
        relevance = []
        correctness = []

        for q in questions:
            gen = q.generation_metrics
            faithfulness.append(gen.get("faithfulness", {}).get("score", 0))
            relevance.append(gen.get("answer_relevance", {}).get("score", 0))
            correctness.append(gen.get("answer_correctness", {}).get("score", 0))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Faithfulness',
            x=q_labels,
            y=faithfulness,
            marker_color='rgb(55, 83, 109)'
        ))
        fig.add_trace(go.Bar(
            name='Relevance',
            x=q_labels,
            y=relevance,
            marker_color='rgb(26, 118, 255)'
        ))
        fig.add_trace(go.Bar(
            name='Correctness',
            x=q_labels,
            y=correctness,
            marker_color='rgb(50, 171, 96)'
        ))

        fig.update_layout(
            barmode='group',
            title="Per-Question Generation Metrics",
            xaxis_title="Question",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            height=400
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def generate_precision_recall_chart(self, results: EvalResults) -> Optional[str]:
        """Generate Precision@K and Recall@K line chart.

        Returns:
            HTML string with embedded chart, or None if plotly unavailable
        """
        if not PLOTLY_AVAILABLE:
            return None

        ret = results.aggregated_retrieval
        if "precision_at_k" not in ret:
            return None

        precision = ret.get("precision_at_k", {})
        recall = ret.get("recall_at_k", {})

        k_values = sorted([int(k) for k in precision.keys()])
        p_values = [precision.get(k, {}).get("mean", 0) for k in k_values]
        r_values = [recall.get(k, {}).get("mean", 0) for k in k_values]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values,
            y=p_values,
            mode='lines+markers',
            name='Precision@K',
            line=dict(color='rgb(55, 83, 109)')
        ))
        fig.add_trace(go.Scatter(
            x=k_values,
            y=r_values,
            mode='lines+markers',
            name='Recall@K',
            line=dict(color='rgb(26, 118, 255)')
        ))

        fig.update_layout(
            title="Precision and Recall at Different K Values",
            xaxis_title="K",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            height=400
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def generate_question_details_table(self, results: EvalResults) -> str:
        """Generate HTML table with per-question details."""
        questions = results.question_results
        if not questions:
            return "<p>No question results available.</p>"

        rows = []
        for i, q in enumerate(questions):
            gen = q.generation_metrics
            ret = q.retrieval_metrics

            faithfulness = gen.get("faithfulness", {}).get("score", "N/A")
            relevance = gen.get("answer_relevance", {}).get("score", "N/A")
            correctness = gen.get("answer_correctness", {}).get("score", "N/A")
            mrr = ret.get("mrr", "N/A")

            # Format scores
            if isinstance(faithfulness, float):
                faithfulness = f"{faithfulness:.2f}"
            if isinstance(relevance, float):
                relevance = f"{relevance:.2f}"
            if isinstance(correctness, float):
                correctness = f"{correctness:.2f}"
            if isinstance(mrr, float):
                mrr = f"{mrr:.2f}"

            rows.append(f"""
            <tr>
                <td>{i+1}</td>
                <td title="{q.question}">{q.question[:50]}...</td>
                <td><span class="badge badge-{q.question_type}">{q.question_type}</span></td>
                <td>{faithfulness}</td>
                <td>{relevance}</td>
                <td>{correctness}</td>
                <td>{mrr}</td>
                <td>{q.total_time_ms:.0f}ms</td>
            </tr>
            """)

        return f"""
        <table class="results-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Question</th>
                    <th>Type</th>
                    <th>Faithfulness</th>
                    <th>Relevance</th>
                    <th>Correctness</th>
                    <th>MRR</th>
                    <th>Time</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        """

    def generate_html_report(
        self,
        results: EvalResults,
        output_path: Optional[str] = None
    ) -> str:
        """Generate a complete HTML report.

        Args:
            results: Evaluation results to visualize
            output_path: Optional path to save the report

        Returns:
            Path to the generated report
        """
        summary = get_evaluation_summary(results)

        # Generate charts
        radar_chart = self.generate_radar_chart(results) or ""
        bar_chart = self.generate_bar_chart(results) or ""
        pr_chart = self.generate_precision_recall_chart(results) or ""
        details_table = self.generate_question_details_table(results)

        # Build HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Evaluation Report - {results.run_id}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {{
            --primary-color: #4f46e5;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --bg-color: #f9fafb;
            --card-bg: #ffffff;
            --text-color: #1f2937;
            --text-muted: #6b7280;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}

        header {{
            background: linear-gradient(135deg, var(--primary-color), #7c3aed);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }}

        header h1 {{
            font-size: 1.75rem;
            margin-bottom: 0.5rem;
        }}

        header .meta {{
            opacity: 0.9;
            font-size: 0.9rem;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .card h3 {{
            font-size: 0.875rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}

        .card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary-color);
        }}

        .card .value.success {{
            color: var(--success-color);
        }}

        .card .value.warning {{
            color: var(--warning-color);
        }}

        .card .value.danger {{
            color: var(--danger-color);
        }}

        .section {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .section h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e5e7eb;
        }}

        .results-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }}

        .results-table th,
        .results-table td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}

        .results-table th {{
            background: #f3f4f6;
            font-weight: 600;
        }}

        .results-table tr:hover {{
            background: #f9fafb;
        }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }}

        .badge-factual {{
            background: #dbeafe;
            color: #1e40af;
        }}

        .badge-comparative {{
            background: #fef3c7;
            color: #92400e;
        }}

        .badge-explanatory {{
            background: #d1fae5;
            color: #065f46;
        }}

        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 1.5rem;
        }}

        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>RAG Evaluation Report</h1>
            <div class="meta">
                Run ID: {results.run_id} |
                Dataset: {results.dataset_name} |
                Model: {results.model} |
                Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
            </div>
        </header>

        <div class="grid">
            <div class="card">
                <h3>Total Questions</h3>
                <div class="value">{summary['questions']['total']}</div>
            </div>
            <div class="card">
                <h3>Successful</h3>
                <div class="value success">{summary['questions']['successful']}</div>
            </div>
            <div class="card">
                <h3>MRR Score</h3>
                <div class="value">{summary.get('retrieval', {}).get('mrr', 'N/A')}</div>
            </div>
            <div class="card">
                <h3>Faithfulness</h3>
                <div class="value">{summary.get('generation', {}).get('faithfulness', 'N/A')}</div>
            </div>
            <div class="card">
                <h3>Answer Relevance</h3>
                <div class="value">{summary.get('generation', {}).get('answer_relevance', 'N/A')}</div>
            </div>
            <div class="card">
                <h3>Total Time</h3>
                <div class="value">{summary['timing']['total_seconds']}s</div>
            </div>
        </div>

        <div class="section">
            <h2>Metrics Overview</h2>
            <div class="charts-grid">
                <div>{radar_chart}</div>
                <div>{pr_chart}</div>
            </div>
        </div>

        <div class="section">
            <h2>Per-Question Results</h2>
            {bar_chart}
        </div>

        <div class="section">
            <h2>Question Details</h2>
            {details_table}
        </div>

        <footer>
            Generated by Context-Aware RAG Evaluation Pipeline
        </footer>
    </div>
</body>
</html>
        """

        # Save report
        if output_path is None:
            output_path = Path(settings.evaluation.reports_dir) / f"{results.run_id}.html"
        else:
            output_path = Path(output_path)

        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"Report saved to {output_path}")
        return str(output_path)

    def generate_json_summary(self, results: EvalResults) -> dict:
        """Generate a JSON-serializable summary for API responses."""
        return get_evaluation_summary(results)
