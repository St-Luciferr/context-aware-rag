"""
Retrieval Evaluation Metrics Module

Measures how well the RAG system retrieves relevant documents.
"""

import numpy as np
from typing import Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

import logging

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """Computes retrieval quality metrics for RAG evaluation."""

    def __init__(self, embeddings: Optional[HuggingFaceEmbeddings] = None):
        """Initialize with optional embeddings for semantic similarity.

        Args:
            embeddings: HuggingFace embeddings model. If None, will initialize
                       using settings.
        """
        self.embeddings = embeddings or self._init_embeddings()

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize embeddings model."""
        return HuggingFaceEmbeddings(
            model_name=settings.embedding.model_name,
            model_kwargs={"device": settings.embedding.device},
            encode_kwargs={"normalize_embeddings": settings.embedding.normalize}
        )

    def precision_at_k(
        self,
        retrieved_ids: list[str],
        ground_truth_ids: set[str],
        k: int
    ) -> float:
        """Calculate Precision@K.

        Precision@K = (# relevant docs in top-K) / K

        Args:
            retrieved_ids: List of retrieved document IDs in rank order
            ground_truth_ids: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            Precision score between 0 and 1
        """
        if k <= 0:
            return 0.0

        top_k = retrieved_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in ground_truth_ids)
        return relevant_in_top_k / k

    def recall_at_k(
        self,
        retrieved_ids: list[str],
        ground_truth_ids: set[str],
        k: int
    ) -> float:
        """Calculate Recall@K.

        Recall@K = (# relevant docs in top-K) / (total # relevant docs)

        Args:
            retrieved_ids: List of retrieved document IDs in rank order
            ground_truth_ids: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            Recall score between 0 and 1
        """
        if not ground_truth_ids or k <= 0:
            return 0.0

        top_k = retrieved_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in ground_truth_ids)
        return relevant_in_top_k / len(ground_truth_ids)

    def mean_reciprocal_rank(
        self,
        retrieved_ids: list[str],
        ground_truth_ids: set[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR).

        MRR = 1 / (rank of first relevant document)

        Args:
            retrieved_ids: List of retrieved document IDs in rank order
            ground_truth_ids: Set of relevant document IDs

        Returns:
            MRR score between 0 and 1 (0 if no relevant docs found)
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in ground_truth_ids:
                return 1.0 / rank
        return 0.0

    def ndcg_at_k(
        self,
        retrieved_ids: list[str],
        ground_truth_ids: set[str],
        k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@K).

        NDCG considers both relevance and position in ranking.

        Args:
            retrieved_ids: List of retrieved document IDs in rank order
            ground_truth_ids: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            NDCG score between 0 and 1
        """
        if k <= 0 or not ground_truth_ids:
            return 0.0

        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in ground_truth_ids:
                # Using binary relevance (1 if relevant, 0 otherwise)
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0

        # Calculate ideal DCG (all relevant docs at top)
        ideal_dcg = 0.0
        num_relevant = min(len(ground_truth_ids), k)
        for i in range(num_relevant):
            ideal_dcg += 1.0 / np.log2(i + 2)

        if ideal_dcg == 0:
            return 0.0

        return dcg / ideal_dcg

    def context_relevance(
        self,
        query: str,
        retrieved_docs: list[Document]
    ) -> float:
        """Calculate semantic similarity between query and retrieved contexts.

        Uses embedding cosine similarity to measure how relevant the
        retrieved contexts are to the query.

        Args:
            query: The user's question
            retrieved_docs: List of retrieved documents

        Returns:
            Average cosine similarity score between 0 and 1
        """
        if not retrieved_docs:
            return 0.0

        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Get document embeddings
            doc_texts = [doc.page_content for doc in retrieved_docs]
            doc_embeddings = self.embeddings.embed_documents(doc_texts)

            # Calculate cosine similarities
            query_vec = np.array(query_embedding)
            similarities = []

            for doc_vec in doc_embeddings:
                doc_vec = np.array(doc_vec)
                # Cosine similarity
                similarity = np.dot(query_vec, doc_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
                )
                similarities.append(similarity)

            return float(np.mean(similarities))

        except Exception as e:
            logger.error(f"Error calculating context relevance: {e}")
            return 0.0

    def hit_rate(
        self,
        retrieved_ids: list[str],
        ground_truth_ids: set[str]
    ) -> float:
        """Calculate Hit Rate (binary relevance).

        Returns 1 if any relevant document is retrieved, 0 otherwise.

        Args:
            retrieved_ids: List of retrieved document IDs
            ground_truth_ids: Set of relevant document IDs

        Returns:
            1.0 if hit, 0.0 otherwise
        """
        for doc_id in retrieved_ids:
            if doc_id in ground_truth_ids:
                return 1.0
        return 0.0

    def compute_all_metrics(
        self,
        query: str,
        retrieved_docs: list[Document],
        ground_truth_ids: set[str],
        k_values: Optional[list[int]] = None
    ) -> dict:
        """Compute all retrieval metrics for a single query.

        Args:
            query: The user's question
            retrieved_docs: List of retrieved documents with metadata
            ground_truth_ids: Set of relevant document IDs
            k_values: List of K values for Precision@K, Recall@K, NDCG@K

        Returns:
            Dictionary with all computed metrics
        """
        if k_values is None:
            k_values = settings.evaluation.retrieval_k_values

        # Extract document IDs from metadata
        retrieved_ids = []
        for doc in retrieved_docs:
            doc_id = doc.metadata.get("chunk_id") or doc.metadata.get("id", "")
            retrieved_ids.append(doc_id)

        metrics = {
            "mrr": self.mean_reciprocal_rank(retrieved_ids, ground_truth_ids),
            "hit_rate": self.hit_rate(retrieved_ids, ground_truth_ids),
            "context_relevance": self.context_relevance(query, retrieved_docs),
            "precision_at_k": {},
            "recall_at_k": {},
            "ndcg_at_k": {}
        }

        for k in k_values:
            metrics["precision_at_k"][k] = self.precision_at_k(
                retrieved_ids, ground_truth_ids, k
            )
            metrics["recall_at_k"][k] = self.recall_at_k(
                retrieved_ids, ground_truth_ids, k
            )
            metrics["ndcg_at_k"][k] = self.ndcg_at_k(
                retrieved_ids, ground_truth_ids, k
            )

        return metrics


def aggregate_retrieval_metrics(all_metrics: list[dict]) -> dict:
    """Aggregate retrieval metrics across multiple queries.

    Args:
        all_metrics: List of metric dictionaries from compute_all_metrics

    Returns:
        Aggregated metrics with mean and std for each metric
    """
    if not all_metrics:
        return {}

    aggregated = {}

    # Simple metrics
    for metric in ["mrr", "hit_rate", "context_relevance"]:
        values = [m[metric] for m in all_metrics if metric in m]
        if values:
            aggregated[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }

    # K-dependent metrics
    for metric in ["precision_at_k", "recall_at_k", "ndcg_at_k"]:
        if metric not in all_metrics[0]:
            continue

        k_values = all_metrics[0][metric].keys()
        aggregated[metric] = {}

        for k in k_values:
            values = [m[metric][k] for m in all_metrics if k in m.get(metric, {})]
            if values:
                aggregated[metric][k] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }

    return aggregated
