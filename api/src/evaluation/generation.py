"""
Generation Evaluation Metrics Module (RAGAS-style)

Measures answer quality using LLM-as-judge approach.
"""

import json
import re
import numpy as np
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

import logging

logger = logging.getLogger(__name__)


class GenerationMetrics:
    """Computes generation quality metrics for RAG evaluation using LLM-as-judge."""

    FAITHFULNESS_PROMPT = """You are evaluating the faithfulness of an AI-generated answer to the provided context.

Context:
{context}

Answer:
{answer}

Task: Determine what fraction of the claims in the answer are supported by the context.

1. First, extract all factual claims from the answer.
2. For each claim, check if it is explicitly supported by the context.
3. Calculate the fraction of supported claims.

Respond in exactly this JSON format:
{{"claims": ["claim1", "claim2", ...], "supported": [true, false, ...], "score": 0.X}}

Where score is the fraction of claims that are supported (between 0 and 1)."""

    ANSWER_RELEVANCE_PROMPT = """You are evaluating how relevant an answer is to the given question.

Question: {question}

Answer: {answer}

Task: Rate how well the answer addresses the question on a scale from 0 to 1.

Consider:
- Does the answer directly address what was asked?
- Is the answer complete and informative?
- Does it stay on topic?

A score of 1.0 means the answer perfectly addresses the question.
A score of 0.0 means the answer is completely irrelevant.

Respond in exactly this JSON format:
{{"reasoning": "brief explanation", "score": 0.X}}"""

    CONTEXT_PRECISION_PROMPT = """You are evaluating whether the retrieved context chunks were useful for answering the question.

Question: {question}

Answer: {answer}

Context chunks:
{contexts}

Task: For each context chunk, determine if it was useful for generating the answer.

Respond in exactly this JSON format:
{{"useful": [true, false, ...], "score": 0.X}}

Where score is the weighted precision (earlier chunks weighted more)."""

    ANSWER_CORRECTNESS_PROMPT = """You are evaluating the correctness of an AI-generated answer compared to a ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {answer}

Task: Evaluate how correct the generated answer is compared to the ground truth.

Consider:
- Factual accuracy compared to ground truth
- Completeness of information
- Any contradictions

A score of 1.0 means the answers convey the same information.
A score of 0.0 means the generated answer is completely wrong.

Respond in exactly this JSON format:
{{"reasoning": "brief explanation", "score": 0.X}}"""

    def __init__(
        self,
        llm: Optional[ChatOllama] = None,
        embeddings: Optional[HuggingFaceEmbeddings] = None
    ):
        """Initialize with optional LLM and embeddings.

        Args:
            llm: ChatOllama instance for LLM-as-judge. If None, uses settings.
            embeddings: HuggingFace embeddings for semantic similarity.
        """
        if llm:
            self.llm = llm
        else:
            # Support for cloud models with API key
            client_kwargs = {
                "headers": {'Authorization': 'Bearer ' + settings.ollama.api_key}
            } if settings.ollama.api_key else {}
            self.llm = ChatOllama(
                model=settings.ollama.model,
                base_url=settings.ollama.base_url,
                temperature=0.1,  # Low temperature for consistent evaluation
                client_kwargs=client_kwargs
            )
        self.embeddings = embeddings or self._init_embeddings()

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize embeddings model."""
        return HuggingFaceEmbeddings(
            model_name=settings.embedding.model_name,
            model_kwargs={"device": settings.embedding.device},
            encode_kwargs={"normalize_embeddings": settings.embedding.normalize}
        )

    def _parse_llm_json(self, response: str) -> Optional[dict]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        content = response.strip()

        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON-like content
            match = re.search(r'\{[^{}]*"score"[^{}]*\}', content)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Failed to parse LLM JSON response: {content[:200]}")
            return None

    def faithfulness_score(
        self,
        answer: str,
        context: str
    ) -> dict:
        """Calculate faithfulness score - is the answer grounded in context?

        Args:
            answer: The generated answer
            context: The retrieved context used to generate the answer

        Returns:
            Dictionary with score (0-1) and details
        """
        try:
            prompt = self.FAITHFULNESS_PROMPT.format(
                context=context[:4000],  # Limit context length
                answer=answer
            )
            response = self.llm.invoke(prompt)
            result = self._parse_llm_json(response.content)

            if result and "score" in result:
                return {
                    "score": float(result["score"]),
                    "claims": result.get("claims", []),
                    "supported": result.get("supported", [])
                }
        except Exception as e:
            logger.error(f"Error calculating faithfulness: {e}")

        return {"score": 0.0, "error": "evaluation_failed"}

    def answer_relevance_score(
        self,
        question: str,
        answer: str
    ) -> dict:
        """Calculate answer relevance - does the answer address the question?

        Args:
            question: The user's question
            answer: The generated answer

        Returns:
            Dictionary with score (0-1) and reasoning
        """
        try:
            prompt = self.ANSWER_RELEVANCE_PROMPT.format(
                question=question,
                answer=answer
            )
            response = self.llm.invoke(prompt)
            result = self._parse_llm_json(response.content)

            if result and "score" in result:
                return {
                    "score": float(result["score"]),
                    "reasoning": result.get("reasoning", "")
                }
        except Exception as e:
            logger.error(f"Error calculating answer relevance: {e}")

        return {"score": 0.0, "error": "evaluation_failed"}

    def context_precision_score(
        self,
        question: str,
        answer: str,
        contexts: list[str]
    ) -> dict:
        """Calculate context precision - were the retrieved chunks useful?

        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context strings

        Returns:
            Dictionary with score (0-1) and per-chunk usefulness
        """
        if not contexts:
            return {"score": 0.0, "useful": [], "error": "no_contexts"}

        try:
            # Format contexts with numbers
            contexts_str = "\n\n".join(
                f"[Chunk {i+1}]: {ctx[:1000]}"
                for i, ctx in enumerate(contexts)
            )

            prompt = self.CONTEXT_PRECISION_PROMPT.format(
                question=question,
                answer=answer,
                contexts=contexts_str
            )
            response = self.llm.invoke(prompt)
            result = self._parse_llm_json(response.content)

            if result and "score" in result:
                return {
                    "score": float(result["score"]),
                    "useful": result.get("useful", [])
                }
        except Exception as e:
            logger.error(f"Error calculating context precision: {e}")

        return {"score": 0.0, "error": "evaluation_failed"}

    def answer_correctness_score(
        self,
        question: str,
        answer: str,
        ground_truth: str
    ) -> dict:
        """Calculate answer correctness - how correct compared to ground truth?

        Args:
            question: The user's question
            answer: The generated answer
            ground_truth: The expected correct answer

        Returns:
            Dictionary with score (0-1) and reasoning
        """
        try:
            prompt = self.ANSWER_CORRECTNESS_PROMPT.format(
                question=question,
                ground_truth=ground_truth,
                answer=answer
            )
            response = self.llm.invoke(prompt)
            result = self._parse_llm_json(response.content)

            if result and "score" in result:
                return {
                    "score": float(result["score"]),
                    "reasoning": result.get("reasoning", "")
                }
        except Exception as e:
            logger.error(f"Error calculating answer correctness: {e}")

        return {"score": 0.0, "error": "evaluation_failed"}

    def semantic_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Calculate semantic similarity between two texts using embeddings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between 0 and 1
        """
        try:
            emb1 = np.array(self.embeddings.embed_query(text1))
            emb2 = np.array(self.embeddings.embed_query(text2))

            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0

    def compute_all_metrics(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None
    ) -> dict:
        """Compute all generation metrics for a single Q&A pair.

        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context strings
            ground_truth: Optional expected answer for correctness evaluation

        Returns:
            Dictionary with all computed metrics
        """
        combined_context = "\n\n".join(contexts)

        metrics = {
            "faithfulness": self.faithfulness_score(answer, combined_context),
            "answer_relevance": self.answer_relevance_score(question, answer),
            "context_precision": self.context_precision_score(
                question, answer, contexts
            )
        }

        # Add correctness if ground truth provided
        if ground_truth:
            metrics["answer_correctness"] = self.answer_correctness_score(
                question, answer, ground_truth
            )
            metrics["semantic_similarity"] = self.semantic_similarity(
                answer, ground_truth
            )

        return metrics


def aggregate_generation_metrics(all_metrics: list[dict]) -> dict:
    """Aggregate generation metrics across multiple Q&A pairs.

    Args:
        all_metrics: List of metric dictionaries from compute_all_metrics

    Returns:
        Aggregated metrics with mean and std for each metric
    """
    if not all_metrics:
        return {}

    aggregated = {}

    # Metrics with sub-scores
    metric_names = [
        "faithfulness", "answer_relevance",
        "context_precision", "answer_correctness"
    ]

    for metric in metric_names:
        scores = []
        for m in all_metrics:
            if metric in m and "score" in m[metric]:
                scores.append(m[metric]["score"])

        if scores:
            aggregated[metric] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            }

    # Semantic similarity (simple float)
    sim_scores = [m.get("semantic_similarity", 0) for m in all_metrics
                  if "semantic_similarity" in m]
    if sim_scores:
        aggregated["semantic_similarity"] = {
            "mean": float(np.mean(sim_scores)),
            "std": float(np.std(sim_scores)),
            "min": float(np.min(sim_scores)),
            "max": float(np.max(sim_scores))
        }

    return aggregated
