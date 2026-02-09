"""
Wikipedia Topics Manager
Manages the list of Wikipedia topics with ingestion status from ChromaDB metadata.
"""

import json
from pathlib import Path
from src.config import settings

# File path for persisting additional topics only
TOPICS_FILE = Path("data/topics.json")


def _ensure_data_dir():
    """Ensure the data directory exists."""
    TOPICS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_additional_topics() -> list[str]:
    """Load additional topics from file."""
    _ensure_data_dir()

    if TOPICS_FILE.exists():
        try:
            with open(TOPICS_FILE, "r") as f:
                data = json.load(f)
                return data.get("additional_topics", [])
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load topics file: {e}")

    return []


def _save_additional_topics(topics: list[str]) -> bool:
    """Save additional topics to file."""
    _ensure_data_dir()

    try:
        with open(TOPICS_FILE, "w") as f:
            json.dump({"additional_topics": topics}, f, indent=2)
        return True
    except IOError as e:
        print(f"Error saving topics: {e}")
        return False


def get_ingested_topics_from_db() -> set[str]:
    """Query ChromaDB to get the list of ingested topics by metadata.

    Returns:
        Set of topic titles that exist in the vector database.
    """
    try:
        import chromadb
        from pathlib import Path

        persist_dir = settings.chroma.persist_dir
        collection_name = settings.chroma.collection_name

        if not Path(persist_dir).exists():
            return set()

        client = chromadb.PersistentClient(path=persist_dir)
        collections = client.list_collections()
        collection_names = [c.name for c in collections]

        if collection_name not in collection_names:
            return set()

        collection = client.get_collection(collection_name)

        # Get all unique titles from metadata
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        titles = set()
        for meta in metadatas:
            if meta and "title" in meta:
                titles.add(meta["title"])

        return titles

    except Exception as e:
        print(f"Warning: Could not query ChromaDB for topics: {e}")
        return set()


def get_default_topics() -> list[str]:
    """Get the default topics from environment/config."""
    return list(settings.wiki_topics)


def get_all_topics() -> list[str]:
    """Get all topics (default + additional).

    Returns:
        Combined list of default and additional topics (deduplicated).
    """
    default = get_default_topics()
    additional = _load_additional_topics()

    # Combine and deduplicate (case-insensitive)
    seen = set()
    all_topics = []
    for topic in default + additional:
        if topic.lower() not in seen:
            seen.add(topic.lower())
            all_topics.append(topic)

    return all_topics


def get_pending_topics() -> list[str]:
    """Get topics that haven't been ingested yet.

    Returns:
        List of topics that exist but aren't in the vector DB.
    """
    all_topics = get_all_topics()
    ingested = get_ingested_topics_from_db()
    ingested_lower = set(t.lower() for t in ingested)

    return [t for t in all_topics if t.lower() not in ingested_lower]


def get_topics_status() -> dict:
    """Get full status of all topics.

    Returns:
        Dict with categorized topics and counts.
    """
    default_topics = get_default_topics()
    additional = _load_additional_topics()
    ingested = get_ingested_topics_from_db()
    ingested_lower = set(t.lower() for t in ingested)

    all_topics = []
    seen = set()

    for topic in default_topics + additional:
        if topic.lower() not in seen:
            seen.add(topic.lower())
            all_topics.append({
                "name": topic,
                "is_default": topic in default_topics,
                "is_ingested": topic.lower() in ingested_lower
            })

    ingested_count = len([t for t in all_topics if t["is_ingested"]])

    return {
        "topics": all_topics,
        "total": len(all_topics),
        "ingested_count": ingested_count,
        "pending_count": len(all_topics) - ingested_count
    }


def add_topic(topic: str) -> tuple[bool, str]:
    """Add a new topic to the additional topics list.

    Args:
        topic: The Wikipedia topic to add.

    Returns:
        Tuple of (success: bool, message: str)
    """
    topic = topic.strip()
    if not topic:
        return False, "Topic cannot be empty"

    # Check if already exists in all topics
    all_topics = get_all_topics()
    if any(t.lower() == topic.lower() for t in all_topics):
        return False, f"Topic '{topic}' already exists"

    additional = _load_additional_topics()
    additional.append(topic)

    if _save_additional_topics(additional):
        return True, f"Topic '{topic}' added. Click 'Ingest New Topics' to add it to the knowledge base."
    return False, "Failed to save topic"


def remove_topic(topic: str) -> tuple[bool, str]:
    """Remove a topic from additional topics (cannot remove default topics).

    Args:
        topic: The Wikipedia topic to remove.

    Returns:
        Tuple of (success: bool, message: str)
    """
    default_topics = get_default_topics()

    # Check if it's a default topic
    if any(t.lower() == topic.lower() for t in default_topics):
        return False, f"Cannot remove default topic '{topic}'. Default topics are configured in environment variables."

    additional = _load_additional_topics()

    # Find and remove (case-insensitive)
    topic_to_remove = None
    for t in additional:
        if t.lower() == topic.lower():
            topic_to_remove = t
            break

    if topic_to_remove is None:
        return False, f"Topic '{topic}' not found in additional topics"

    additional.remove(topic_to_remove)

    if _save_additional_topics(additional):
        return True, f"Topic '{topic_to_remove}' removed. Note: Documents remain in the database until full re-ingestion."
    return False, "Failed to save changes"


def reset_additional_topics() -> tuple[bool, str]:
    """Remove all additional topics, keeping only defaults.

    Returns:
        Tuple of (success: bool, message: str)
    """
    if _save_additional_topics([]):
        return True, "Additional topics cleared. Only default topics remain."
    return False, "Failed to reset topics"
