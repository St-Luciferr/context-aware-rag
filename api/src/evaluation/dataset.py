"""
Evaluation Dataset Generation Module

Generates Q&A pairs from indexed Wikipedia content for RAG evaluation.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
import logging

import chromadb
from langchain_ollama import ChatOllama

from src.config import settings

logger = logging.getLogger(__name__)


class EvalQuestion(BaseModel):
    """A single evaluation question with ground truth."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str
    question_type: str  # factual, comparative, explanatory
    ground_truth_answer: str
    ground_truth_context: list[str]  # Source chunk contents
    ground_truth_chunk_ids: list[str]  # Chunk IDs for retrieval eval
    topic: str
    difficulty: str = "medium"  # easy, medium, hard


class EvalDataset(BaseModel):
    """A collection of evaluation questions."""
    name: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))
    questions: list[EvalQuestion] = Field(default_factory=list)
    topics_covered: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class DatasetGenerator:
    """Generates evaluation datasets from indexed ChromaDB content."""

    FACTUAL_PROMPT = """Based on the following context from a Wikipedia article about "{topic}", generate a factual question and its answer.

Context:
{context}

Generate a question that can be answered directly from this context. The question should be specific and test factual knowledge.

Respond in exactly this JSON format:
{{"question": "your question here", "answer": "the correct answer based on the context"}}"""

    COMPARATIVE_PROMPT = """Based on the following contexts from Wikipedia articles, generate a comparative question that relates the topics.

Topic 1: {topic1}
Context 1:
{context1}

Topic 2: {topic2}
Context 2:
{context2}

Generate a question that compares or contrasts these topics. The answer should draw from both contexts.

Respond in exactly this JSON format:
{{"question": "your comparative question here", "answer": "the answer comparing both topics"}}"""

    EXPLANATORY_PROMPT = """Based on the following context from a Wikipedia article about "{topic}", generate an explanatory question and its answer.

Context:
{context}

Generate a "why" or "how" question that requires explaining a concept from this context. The answer should provide an explanation.

Respond in exactly this JSON format:
{{"question": "your explanatory question here", "answer": "the explanation based on the context"}}"""

    def __init__(self):
        client_kwargs = {
            "headers": {'Authorization': 'Bearer ' + settings.ollama.api_key}
        } if settings.ollama.api_key else {}
        self.llm = ChatOllama(
            model=settings.ollama.model,
            base_url=settings.ollama.base_url,
            temperature=settings.ollama.temperature,
            keep_alive=-1,
            client_kwargs=client_kwargs
        )
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Ensure evaluation directories exist."""
        Path(settings.evaluation.datasets_dir).mkdir(
            parents=True, exist_ok=True)
        Path(settings.evaluation.results_dir).mkdir(
            parents=True, exist_ok=True)
        Path(settings.evaluation.reports_dir).mkdir(
            parents=True, exist_ok=True)

    def _get_chunks_by_topic(self) -> dict[str, list[dict]]:
        """Get all chunks from ChromaDB grouped by topic."""
        try:
            persist_dir = settings.chroma.persist_dir
            collection_name = settings.chroma.collection_name

            if not Path(persist_dir).exists():
                logger.warning("ChromaDB directory does not exist")
                return {}

            client = chromadb.PersistentClient(path=persist_dir)
            collections = client.list_collections()
            collection_names = [c.name for c in collections]

            if collection_name not in collection_names:
                logger.warning(f"Collection {collection_name} not found")
                return {}

            collection = client.get_collection(collection_name)
            results = collection.get(include=["documents", "metadatas"])

            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            # Group by topic
            chunks_by_topic: dict[str, list[dict]] = {}
            for doc, meta in zip(documents, metadatas):
                if meta and "title" in meta:
                    title = meta["title"]
                    if title not in chunks_by_topic:
                        chunks_by_topic[title] = []
                    # Use title:chunk_id for unique identifier matching retrieval path
                    meta_chunk_id = meta.get("chunk_id", len(chunks_by_topic[title]))
                    unique_id = f"{title}:{meta_chunk_id}"
                    chunks_by_topic[title].append({
                        "id": unique_id,
                        "content": doc,
                        "metadata": meta
                    })

            return chunks_by_topic

        except Exception as e:
            logger.error(f"Error getting chunks from ChromaDB: {e}")
            return {}

    def _generate_qa_from_llm(self, prompt: str) -> Optional[dict]:
        """Generate a Q&A pair using the LLM."""
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Try to parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating Q&A: {e}")
            return None

    def generate_factual_questions(
        self,
        topic: str,
        chunks: list[dict],
        count: int = 3
    ) -> list[EvalQuestion]:
        """Generate factual questions for a topic."""
        questions = []
        # Use different chunks to get variety
        selected_chunks = chunks[:min(count * 2, len(chunks))]

        for i, chunk in enumerate(selected_chunks):
            if len(questions) >= count:
                break

            prompt = self.FACTUAL_PROMPT.format(
                topic=topic,
                context=chunk["content"][:2000]  # Limit context length
            )

            qa = self._generate_qa_from_llm(prompt)
            if qa and "question" in qa and "answer" in qa:
                questions.append(EvalQuestion(
                    question=qa["question"],
                    question_type="factual",
                    ground_truth_answer=qa["answer"],
                    ground_truth_context=[chunk["content"]],
                    ground_truth_chunk_ids=[chunk["id"]],
                    topic=topic,
                    difficulty="easy"
                ))

        return questions

    def generate_comparative_questions(
        self,
        topics_chunks: dict[str, list[dict]],
        count: int = 2
    ) -> list[EvalQuestion]:
        """Generate comparative questions across topics."""
        questions = []
        topics = list(topics_chunks.keys())

        if len(topics) < 2:
            return questions

        # Generate comparisons between topic pairs
        for i in range(min(count, len(topics) - 1)):
            topic1 = topics[i]
            topic2 = topics[(i + 1) % len(topics)]

            chunks1 = topics_chunks[topic1]
            chunks2 = topics_chunks[topic2]

            if not chunks1 or not chunks2:
                continue

            prompt = self.COMPARATIVE_PROMPT.format(
                topic1=topic1,
                context1=chunks1[0]["content"][:1500],
                topic2=topic2,
                context2=chunks2[0]["content"][:1500]
            )

            qa = self._generate_qa_from_llm(prompt)
            if qa and "question" in qa and "answer" in qa:
                questions.append(EvalQuestion(
                    question=qa["question"],
                    question_type="comparative",
                    ground_truth_answer=qa["answer"],
                    ground_truth_context=[
                        chunks1[0]["content"],
                        chunks2[0]["content"]
                    ],
                    ground_truth_chunk_ids=[
                        chunks1[0]["id"],
                        chunks2[0]["id"]
                    ],
                    topic=f"{topic1} vs {topic2}",
                    difficulty="hard"
                ))

        return questions

    def generate_explanatory_questions(
        self,
        topic: str,
        chunks: list[dict],
        count: int = 2
    ) -> list[EvalQuestion]:
        """Generate explanatory (why/how) questions for a topic."""
        questions = []
        # Use chunks that likely have more explanatory content
        selected_chunks = chunks[1:min(count + 2, len(chunks))]

        for chunk in selected_chunks:
            if len(questions) >= count:
                break

            prompt = self.EXPLANATORY_PROMPT.format(
                topic=topic,
                context=chunk["content"][:2000]
            )

            qa = self._generate_qa_from_llm(prompt)
            if qa and "question" in qa and "answer" in qa:
                questions.append(EvalQuestion(
                    question=qa["question"],
                    question_type="explanatory",
                    ground_truth_answer=qa["answer"],
                    ground_truth_context=[chunk["content"]],
                    ground_truth_chunk_ids=[chunk["id"]],
                    topic=topic,
                    difficulty="medium"
                ))

        return questions

    def generate_dataset(
        self,
        name: str,
        questions_per_topic: int = 5,
        question_types: Optional[list[str]] = None
    ) -> EvalDataset:
        """Generate a complete evaluation dataset.

        Args:
            name: Name for the dataset
            questions_per_topic: Total questions to generate per topic
            question_types: Types of questions to generate
                           (factual, comparative, explanatory)

        Returns:
            EvalDataset with generated questions
        """
        if question_types is None:
            question_types = settings.evaluation.default_question_types

        logger.info(f"Generating dataset '{name}' with {questions_per_topic} "
                    f"questions per topic, types: {question_types}")

        chunks_by_topic = self._get_chunks_by_topic()
        if not chunks_by_topic:
            logger.warning("No chunks found in ChromaDB")
            return EvalDataset(name=name, metadata={"error": "No chunks found"})

        all_questions: list[EvalQuestion] = []

        # Calculate questions per type
        type_count = len(question_types)
        questions_per_type = max(1, questions_per_topic // type_count)

        for topic, chunks in chunks_by_topic.items():
            logger.info(f"Generating questions for topic: {topic}")

            if "factual" in question_types:
                factual_qs = self.generate_factual_questions(
                    topic, chunks, questions_per_type
                )
                all_questions.extend(factual_qs)
                logger.info(f"  Generated {len(factual_qs)} factual questions")

            if "explanatory" in question_types:
                explanatory_qs = self.generate_explanatory_questions(
                    topic, chunks, questions_per_type
                )
                all_questions.extend(explanatory_qs)
                logger.info(
                    f"  Generated {len(explanatory_qs)} explanatory questions")

        # Generate comparative questions across all topics
        if "comparative" in question_types and len(chunks_by_topic) >= 2:
            comparative_qs = self.generate_comparative_questions(
                chunks_by_topic, count=min(3, len(chunks_by_topic) - 1)
            )
            all_questions.extend(comparative_qs)
            logger.info(
                f"Generated {len(comparative_qs)} comparative questions")

        dataset = EvalDataset(
            name=name,
            questions=all_questions,
            topics_covered=list(chunks_by_topic.keys()),
            metadata={
                "questions_per_topic": questions_per_topic,
                "question_types": question_types,
                "total_questions": len(all_questions),
                "model": settings.ollama.model
            }
        )

        # Save dataset
        self.save_dataset(dataset)
        logger.info(
            f"Dataset '{name}' generated with {len(all_questions)} questions")

        return dataset

    def save_dataset(self, dataset: EvalDataset) -> Path:
        """Save dataset to JSON file."""
        filepath = Path(settings.evaluation.datasets_dir) / \
            f"{dataset.name}.json"
        with open(filepath, "w") as f:
            f.write(dataset.model_dump_json(indent=2))
        logger.info(f"Dataset saved to {filepath}")
        return filepath

    def load_dataset(self, name: str) -> Optional[EvalDataset]:
        """Load dataset from JSON file."""
        filepath = Path(settings.evaluation.datasets_dir) / f"{name}.json"
        if not filepath.exists():
            logger.warning(f"Dataset file not found: {filepath}")
            return None

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return EvalDataset(**data)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None

    def list_datasets(self) -> list[dict]:
        """List all available datasets."""
        datasets_dir = Path(settings.evaluation.datasets_dir)
        if not datasets_dir.exists():
            return []

        datasets = []
        for filepath in datasets_dir.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                datasets.append({
                    "name": data.get("name", filepath.stem),
                    "created_at": data.get("created_at"),
                    "question_count": len(data.get("questions", [])),
                    "topics": data.get("topics_covered", [])
                })
            except Exception as e:
                logger.warning(f"Error reading dataset {filepath}: {e}")

        return datasets

    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset file."""
        filepath = Path(settings.evaluation.datasets_dir) / f"{name}.json"
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted dataset: {name}")
            return True
        return False
