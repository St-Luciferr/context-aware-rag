"""
Document Ingestion Script
Loads Wikipedia pages and stores embeddings in ChromaDB
"""

import traceback
import wikipedia
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from pathlib import Path
from src.config import settings


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize and return the embeddings model."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding.model_name,
        model_kwargs={"device": settings.embedding.device},
        encode_kwargs={"normalize_embeddings": settings.embedding.normalize}
    )


def check_collection_exists() -> tuple[bool, int]:
    """
    Check if the ChromaDB collection exists and has documents.

    Returns:
        tuple: (exists: bool, document_count: int)
    """
    persist_dir = settings.chroma.persist_dir
    collection_name = settings.chroma.collection_name

    # Check if persist directory exists
    if not Path(persist_dir).exists():
        return False, 0

    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=persist_dir)

        # Get list of collections
        collections = client.list_collections()
        collection_names = [c.name for c in collections]

        if collection_name not in collection_names:
            return False, 0

        # Get the collection and check document count
        collection = client.get_collection(collection_name)
        count = collection.count()

        return count > 0, count

    except Exception as e:
        print(f"Warning: Error checking collection: {e}")
        return False, 0


def delete_collection() -> bool:
    """
    Delete the existing ChromaDB collection.

    Returns:
        bool: True if deleted successfully, False otherwise
    """
    persist_dir = settings.chroma.persist_dir
    collection_name = settings.chroma.collection_name

    try:
        client = chromadb.PersistentClient(path=persist_dir)
        client.delete_collection(collection_name)
        print(f"✓ Deleted existing collection: {collection_name}")
        return True
    except Exception as e:
        print(f"Warning: Could not delete collection: {e}")
        return False


def fetch_wikipedia_content(topics: list[str]) -> list[dict]:
    """Fetch content from Wikipedia for given topics."""
    documents = []
    for topic in topics:
        try:
            print(f"Fetching: {topic}")
            page = wikipedia.page(topic, auto_suggest=False)
            documents.append({
                "content": page.content,
                "metadata": {
                    "title": page.title,
                    "url": page.url,
                    "source": "wikipedia"
                }
            })
            print(f"Loaded {len(page.content)} characters")
        except wikipedia.exceptions.DisambiguationError as e:
            print(
                f"Disambiguation error for '{topic}', trying first option...")
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                documents.append({
                    "content": page.content,
                    "metadata": {"title": page.title, "url": page.url, "source": "wikipedia"}
                })
            except Exception as inner_e:
                print(f"Failed: {inner_e}")
        except Exception as e:
            print(f"Error fetching '{topic}': {e}")
    return documents


def create_vector_store(documents: list[dict], embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Create ChromaDB vector store from documents using semantic chunking."""

    print(f"\nInitializing SemanticChunker:")
    print(f"  Breakpoint type: {settings.rag.breakpoint_threshold_type}")
    print(f"  Breakpoint amount: {settings.rag.breakpoint_threshold_amount}")

    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=settings.rag.breakpoint_threshold_type,
        breakpoint_threshold_amount=settings.rag.breakpoint_threshold_amount
    )

    # Split documents into semantic chunks
    texts, metadatas = [], []
    for doc in documents:
        print(f"Chunking: {doc['metadata']['title']}...", end=" ")
        chunks = text_splitter.split_text(doc["content"])
        print(f"{len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({**doc["metadata"], "chunk_id": i})

    print(f"\nCreated {len(texts)} semantic chunks total")

    # Calculate average chunk size for info
    avg_size = sum(len(t) for t in texts) / len(texts) if texts else 0
    print(f"Average chunk size: {avg_size:.0f} characters")

    # Create and persist vector store
    print(f"\nCreating vector store at: {settings.chroma.persist_dir}")
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=settings.chroma.collection_name,
        persist_directory=settings.chroma.persist_dir
    )

    print(
        f"Vector store created with collection: {settings.chroma.collection_name}")
    return vector_store


def run_ingestion(force: bool = False) -> dict:
    """
    Run the ingestion pipeline with all topics.

    Args:
        force: If True, delete existing collection and re-ingest

    Returns:
        dict: Status and message
    """
    try:
        from src.topics import get_all_topics
        topics = get_all_topics()

        return run_ingestion_with_topics(topics, force)
    except Exception as e:
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}


def run_ingestion_with_topics(topics: list[str], force: bool = False) -> dict:
    """
    Run the ingestion pipeline with specified topics (full re-ingestion).

    Args:
        topics: List of Wikipedia topics to ingest
        force: If True, delete existing collection and re-ingest

    Returns:
        dict: Status and message
    """
    try:
        # Check if collection already exists
        exists, doc_count = check_collection_exists()

        if exists:
            if force:
                print(
                    f"\nCollection '{settings.chroma.collection_name}' exists with {doc_count} documents.")
                print("  Force flag set - deleting and re-ingesting...")
                delete_collection()
            else:
                print(
                    f"\nCollection '{settings.chroma.collection_name}' already exists with {doc_count} documents.")
                print("Skipping ingestion. Use --force to re-ingest.")
                return {
                    "status": "skipped",
                    "message": f"Collection already exists with {doc_count} documents",
                    "document_count": doc_count
                }

        print(f"\nLoading embedding model: {settings.embedding.model_name}")
        embeddings = get_embeddings()

        print("\n[1/2] Fetching Wikipedia content...")
        print(f"Topics: {topics}")
        documents = fetch_wikipedia_content(topics)
        print(f"✓ Fetched {len(documents)} documents")

        if not documents:
            return {
                "status": "error",
                "message": "No documents fetched. Check if topics are valid Wikipedia articles.",
                "document_count": 0
            }

        print("\n[2/2] Creating vector store with semantic chunking...")
        _ = create_vector_store(documents, embeddings)

        # Get final count
        _, final_count = check_collection_exists()

        # Refresh BM25 index in the chatbot if it exists
        _refresh_bm25_index()

        return {
            "status": "success",
            "message": f"Vector index initialized with {len(topics)} topics",
            "document_count": final_count
        }

    except Exception as e:
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e)
        }


def add_topics_to_existing(topics: list[str]) -> dict:
    """
    Add new topics to the existing vector store (incremental ingestion).

    This adds documents to the existing ChromaDB collection without deleting existing data.
    Topics that already exist in ChromaDB are automatically skipped to prevent duplicates.

    Args:
        topics: List of Wikipedia topics to add

    Returns:
        dict: Status and message
    """
    try:
        if not topics:
            return {
                "status": "skipped",
                "message": "No topics to add",
                "document_count": 0,
                "topics_added": []
            }

        # Filter out topics that already exist in ChromaDB to prevent duplicates
        from src.topics import get_ingested_topics_from_db
        already_ingested = get_ingested_topics_from_db()
        already_ingested_lower = {t.lower() for t in already_ingested}

        new_topics = [t for t in topics if t.lower() not in already_ingested_lower]

        if not new_topics:
            return {
                "status": "skipped",
                "message": "All requested topics already exist in the knowledge base",
                "document_count": 0,
                "topics_added": []
            }

        skipped_count = len(topics) - len(new_topics)
        if skipped_count > 0:
            print(f"\nSkipping {skipped_count} topics that already exist in ChromaDB")

        print(f"\nIncremental ingestion: Adding {len(new_topics)} new topics")
        print(f"Topics: {new_topics}")

        # Check if collection exists
        exists, initial_count = check_collection_exists()
        if not exists:
            print("Collection doesn't exist, running full ingestion...")
            return run_ingestion(force=False)

        print(f"\nExisting collection has {initial_count} documents")
        print(f"\nLoading embedding model: {settings.embedding.model_name}")
        embeddings = get_embeddings()

        print("\n[1/2] Fetching Wikipedia content for new topics...")
        documents = fetch_wikipedia_content(new_topics)
        print(f"✓ Fetched {len(documents)} documents")

        if not documents:
            return {
                "status": "error",
                "message": "No documents fetched. Check if topics are valid Wikipedia articles.",
                "document_count": initial_count,
                "topics_added": []
            }

        print("\n[2/2] Adding to existing vector store...")
        added_count = add_documents_to_existing_store(documents, embeddings)

        # Get final count
        _, final_count = check_collection_exists()

        # Refresh BM25 index
        _refresh_bm25_index()

        return {
            "status": "success",
            "message": f"Added {added_count} chunks from {len(documents)} topics",
            "document_count": final_count,
            "topics_added": [doc["metadata"]["title"] for doc in documents],
            "chunks_added": added_count
        }

    except Exception as e:
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e)
        }


def add_documents_to_existing_store(documents: list[dict], embeddings: HuggingFaceEmbeddings) -> int:
    """Add documents to existing ChromaDB collection.

    Args:
        documents: List of document dicts with content and metadata
        embeddings: Embeddings model instance

    Returns:
        Number of chunks added
    """
    print(f"\nInitializing SemanticChunker:")
    print(f"  Breakpoint type: {settings.rag.breakpoint_threshold_type}")
    print(f"  Breakpoint amount: {settings.rag.breakpoint_threshold_amount}")

    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=settings.rag.breakpoint_threshold_type,
        breakpoint_threshold_amount=settings.rag.breakpoint_threshold_amount
    )

    # Split documents into semantic chunks
    texts, metadatas = [], []
    for doc in documents:
        print(f"Chunking: {doc['metadata']['title']}...", end=" ")
        chunks = text_splitter.split_text(doc["content"])
        print(f"{len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({**doc["metadata"], "chunk_id": i})

    print(f"\nCreated {len(texts)} semantic chunks total")

    if not texts:
        return 0

    # Add to existing collection
    print(f"\nAdding to existing vector store at: {settings.chroma.persist_dir}")

    # Load existing vector store and add texts
    vector_store = Chroma(
        collection_name=settings.chroma.collection_name,
        persist_directory=settings.chroma.persist_dir,
        embedding_function=embeddings
    )

    vector_store.add_texts(texts=texts, metadatas=metadatas)

    print(f"✓ Added {len(texts)} chunks to collection: {settings.chroma.collection_name}")
    return len(texts)


def _refresh_bm25_index():
    """Refresh the BM25 index in the chatbot if it exists."""
    try:
        from src.graph import get_chatbot
        bot = get_chatbot()
        bot.refresh_bm25_index()
        print("✓ BM25 index refreshed")
    except Exception as e:
        print(f"Warning: Could not refresh BM25 index: {e}")


def main():
    print("=" * 50)
    print("Document Ingestion Pipeline (Semantic Chunking)")
    print("=" * 50)

    # Initialize embeddings once (shared between chunker and vector store)
    print(f"\nLoading embedding model: {settings.embedding.model_name}")
    embeddings = get_embeddings()

    # Fetch documents using topics from config
    print("\n[1/2] Fetching Wikipedia content...")
    print(f"Topics: {settings.wiki_topics}")
    documents = fetch_wikipedia_content(settings.wiki_topics)
    print(f"Fetched {len(documents)} documents")

    # Create vector store with semantic chunking
    print("\n[2/2] Creating vector store with semantic chunking...")
    vector_store = create_vector_store(documents, embeddings)

    # Test retrieval
    print("\n" + "=" * 50)
    print("Testing retrieval...")
    test_query = "What is machine learning?"
    results = vector_store.similarity_search(
        test_query, k=settings.rag.retrieval_k)
    print(f"Query: '{test_query}'")
    print(f"Found {len(results)} relevant chunks (k={settings.rag.retrieval_k})")

    for i, doc in enumerate(results):
        title = doc.metadata.get('title', 'Unknown')
        chunk_id = doc.metadata.get('chunk_id', '?')
        print(f"\n--- Result {i+1} ({title}, chunk {chunk_id}) ---")
        print(doc.page_content[:300] + "...")

    print("\nIngestion complete!")
    return {
        "status": "Complete"
    }


if __name__ == "__main__":
    main()
