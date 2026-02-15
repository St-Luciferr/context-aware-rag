"""
Document Retrieval Tool

Provides additional document retrieval capabilities for the LLM
to find more specific information from the vector store.
"""

from typing import Dict, Any, Optional
from langchain_core.tools import tool
import logging

from src.config import settings

logger = logging.getLogger(__name__)

# Cached vectorstore instance
_vectorstore = None


def _get_vectorstore():
    """Get or create the ChromaDB vectorstore instance."""
    global _vectorstore
    if _vectorstore is None:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding.model_name,
            model_kwargs={"device": settings.embedding.device}
        )
        _vectorstore = Chroma(
            collection_name=settings.chroma.collection_name,
            persist_directory=settings.chroma.persist_dir,
            embedding_function=embeddings
        )
    return _vectorstore


@tool
def retrieve_documents(
    query: str,
    num_results: int = 3,
    topic_filter: Optional[str] = None
) -> str:
    """Retrieve additional documents from the knowledge base.

    Use this tool when:
    - The initial context doesn't contain enough information
    - You need to search for specific facts or details
    - The user asks about a related but different topic
    - You want to verify information from multiple sources

    Args:
        query: The search query to find relevant documents
        num_results: Number of documents to retrieve (1-5, default 3)
        topic_filter: Optional topic name to filter results (e.g., "Machine learning")

    Returns:
        Retrieved document contents with source information
    """
    try:
        # Clamp num_results to valid range
        num_results = max(1, min(5, num_results))

        vectorstore = _get_vectorstore()

        # Build filter if topic specified
        where_filter = None
        if topic_filter:
            where_filter = {"topic": {"$eq": topic_filter}}

        # Perform similarity search
        results = vectorstore.similarity_search(
            query,
            k=num_results,
            filter=where_filter
        )

        if not results:
            return f"No documents found for query: '{query}'"

        # Format results for LLM consumption
        formatted_results = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            topic = doc.metadata.get("topic", "Unknown")
            content = doc.page_content[:1000]  # Limit content length

            formatted_results.append(
                f"[Document {i}]\n"
                f"Source: {source}\n"
                f"Topic: {topic}\n"
                f"Content: {content}\n"
            )

        return "\n---\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return f"Error retrieving documents: {str(e)}"


def get_retrieval_tool_info() -> Dict[str, Any]:
    """Get metadata about the retrieval tool."""
    return {
        "name": "retrieve_documents",
        "description": retrieve_documents.description,
        "parameters": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant documents",
                "required": True
            },
            "num_results": {
                "type": "integer",
                "description": "Number of documents to retrieve (1-5)",
                "default": 3
            },
            "topic_filter": {
                "type": "string",
                "description": "Optional topic name to filter results",
                "required": False
            }
        }
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add api directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    import argparse
    from src.config import settings

    parser = argparse.ArgumentParser(description="Test retrieve_documents tool")
    parser.add_argument("query", help="Search query")
    parser.add_argument("-n", "--num-results", type=int, default=3, help="Number of results (default: 3)")
    parser.add_argument("-t", "--topic", help="Filter by topic name")
    args = parser.parse_args()

    print(f"Collection: {settings.chroma.collection_name}")
    print(f"Query: {args.query}")
    print(f"Num results: {args.num_results}")
    if args.topic:
        print(f"Topic filter: {args.topic}")
    print("-" * 40)

    params = {"query": args.query, "num_results": args.num_results}
    if args.topic:
        params["topic_filter"] = args.topic

    result = retrieve_documents.invoke(params)
    print(result)
