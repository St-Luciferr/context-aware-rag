"""
AutoCut Context Distillation for RAG Systems
Removes redundant content from retrieved documents
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from src.config import settings

# LangSmith tracing support
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # Provide a no-op decorator if langsmith is not installed
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class AutoCut:
    """
    Implements context distillation to remove redundant content from RAG retrieval
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.80,
        max_chunks: int = settings.rag.distilled_retrieval_k
    ):
        """
        Implements context distillation to remove redundant content from RAG retrieval
        Args:
            model_name: Sentence transformer model for embeddings
            similarity_threshold: Threshold for redundancy (0-1, higher = more strict)
            max_chunks: Maximum number of chunks to keep
        """
        self.encoder = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_chunks = max_chunks

    @traceable(
        name="autocut_distill",
        run_type="chain",
        metadata={"component": "autocut", "method": "distill"}
    )
    def distill(
        self,
        query: str,
        retrieved_chunks: List[Any]
    ) -> List[Any]:
        """
        Main distillation method

        Args:
            query: User query
            retrieved_chunks: List of dicts with 'text' and 'score' keys

        Returns:
            Distilled list of chunks
        """
        if not retrieved_chunks:
            return []

        # Step 1: Encode query and all chunks
        query_embedding = self.encoder.encode([query])[0]
        chunk_texts = [self._get_text(chunk) for chunk in retrieved_chunks]
        chunk_embeddings = self.encoder.encode(chunk_texts)

        # Step 2: Calculate relevance scores (query-chunk similarity)
        relevance_scores = cosine_similarity(
            [query_embedding],
            chunk_embeddings
        )[0]

        # Step 3: Sort by relevance
        sorted_indices = np.argsort(relevance_scores)[::-1]

        # Step 4: Iteratively select non-redundant chunks
        selected_chunks = []
        selected_embeddings = []

        for idx in sorted_indices:
            if len(selected_chunks) >= self.max_chunks:
                break

            current_embedding = chunk_embeddings[idx]

            # Check if this chunk is redundant with already selected chunks
            if self._is_redundant(current_embedding, selected_embeddings):
                continue

            # Add to selected
            chunk = retrieved_chunks[idx]
            if hasattr(chunk, 'metadata'):
                chunk.metadata['relevance_score'] = float(
                    relevance_scores[idx])
            elif isinstance(chunk, dict):
                chunk = chunk.copy()
                chunk['relevance_score'] = float(relevance_scores[idx])

            selected_chunks.append(chunk)
            selected_embeddings.append(current_embedding)

        return selected_chunks

    def _get_text(self, chunk: any) -> str:
        """
        Extract text from chunk (handles both Document objects and dicts)
        """
        if hasattr(chunk, 'page_content'):
            # LangChain Document object
            return chunk.page_content
        elif isinstance(chunk, dict):
            # Dictionary with 'text' key
            return chunk.get('text', '') or chunk.get('page_content')
        else:
            # Assume it's already a string
            return str(chunk)

    def _is_redundant(
        self,
        candidate_embedding: np.ndarray,
        selected_embeddings: List[np.ndarray]
    ) -> bool:
        """
        Check if candidate is too similar to already selected chunks
        """
        if not selected_embeddings:
            return False

        similarities = cosine_similarity(
            [candidate_embedding],
            selected_embeddings
        )[0]

        max_similarity = np.max(similarities)
        return max_similarity > self.similarity_threshold

    @traceable(
        name="autocut_distill_mmr",
        run_type="chain",
        metadata={"component": "autocut", "method": "distill_with_mmr"}
    )
    def distill_with_mmr(
        self,
        query: str,
        retrieved_chunks: List[Any],
        lambda_param: float = 0.7
    ) -> List[Any]:
        """
        Distill using Maximal Marginal Relevance (MMR)
        Balances relevance and diversity

        Args:
            query: User query
            retrieved_chunks: List of Document objects or dicts
            lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
        """
        if not retrieved_chunks:
            return []

        # Encode
        query_embedding = self.encoder.encode([query])[0]
        chunk_texts = [self._get_text(chunk) for chunk in retrieved_chunks]
        chunk_embeddings = self.encoder.encode(chunk_texts)

        # Calculate relevance scores
        relevance_scores = cosine_similarity(
            [query_embedding],
            chunk_embeddings
        )[0]

        selected_chunks = []
        selected_embeddings = []
        remaining_indices = list(range(len(retrieved_chunks)))

        while remaining_indices and len(selected_chunks) < self.max_chunks:
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance component
                relevance = relevance_scores[idx]

                # Diversity component (max similarity to selected)
                if selected_embeddings:
                    similarities = cosine_similarity(
                        [chunk_embeddings[idx]],
                        selected_embeddings
                    )[0]
                    max_sim = np.max(similarities)
                else:
                    max_sim = 0

                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append(mmr)

            # Select chunk with highest MMR
            best_idx = remaining_indices[np.argmax(mmr_scores)]

            chunk = retrieved_chunks[best_idx]

            # Add scores to metadata if it's a Document object
            if hasattr(chunk, 'metadata'):
                chunk.metadata['relevance_score'] = float(
                    relevance_scores[best_idx])
                chunk.metadata['mmr_score'] = float(max(mmr_scores))
            elif isinstance(chunk, dict):
                chunk = chunk.copy()
                chunk['relevance_score'] = float(relevance_scores[best_idx])
                chunk['mmr_score'] = float(max(mmr_scores))

            selected_chunks.append(chunk)
            selected_embeddings.append(chunk_embeddings[best_idx])
            remaining_indices.remove(best_idx)

        return selected_chunks


# Example Usage
def demo_autocut():
    """Demonstrate AutoCut usage"""

    # Initialize AutoCut
    autocut = AutoCut(
        similarity_threshold=0.80,  # Chunks with >80% similarity are redundant
        max_chunks=3                # Keep max 3 chunks
    )

    # Simulated retrieved chunks (from vector DB)
    query = "What are the benefits of exercise?"

    retrieved_chunks = [
        {
            'text': 'Regular exercise improves cardiovascular health and reduces risk of heart disease.',
            'score': 0.92
        },
        {
            'text': 'Exercise strengthens the heart and improves blood circulation, reducing cardiovascular risk.',
            'score': 0.90  # Very similar to chunk 1 - should be filtered
        },
        {
            'text': 'Physical activity helps with weight management and boosts metabolism.',
            'score': 0.88
        },
        {
            'text': 'Exercise releases endorphins, improving mood and reducing stress and anxiety.',
            'score': 0.87
        },
        {
            'text': 'Working out regularly can help you maintain a healthy weight.',
            'score': 0.85  # Similar to chunk 3 - may be filtered
        }
    ]

    # Apply distillation
    print("=== Standard AutoCut ===")
    distilled = autocut.distill(query, retrieved_chunks)
    for i, chunk in enumerate(distilled, 1):
        print(f"\nChunk {i} (relevance: {chunk['relevance_score']:.3f}):")
        print(f"  {chunk['text'][:80]}...")

    # Apply MMR-based distillation
    print("\n\n=== MMR-Based AutoCut ===")
    distilled_mmr = autocut.distill_with_mmr(
        query, retrieved_chunks, lambda_param=0.7)
    for i, chunk in enumerate(distilled_mmr, 1):
        print(f"\nChunk {i} (MMR: {chunk['mmr_score']:.3f}):")
        print(f"  {chunk['text'][:80]}...")


if __name__ == "__main__":
    demo_autocut()
