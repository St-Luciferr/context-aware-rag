# RAG Chatbot with LangGraph

A context-aware chatbot that retrieves information from Wikipedia articles about AI/ML topics, maintains conversational context, and provides responses via a FastAPI backend and web UI.

![Architecture](https://img.shields.io/badge/Architecture-RAG-blue)
![Search](https://img.shields.io/badge/Search-Hybrid-green)
![Chunking](https://img.shields.io/badge/Chunking-Semantic-orange)
![Framework](https://img.shields.io/badge/Framework-LangGraph-purple)


## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Hybrid Search** | Combines vector similarity + BM25 keyword search |
| 📄 **Semantic Chunking** | Intelligent document splitting based on meaning |
| 🧠 **LangGraph State Machine** | Structured conversation flow management |
| 📚 **Citation Tracking** | Every response cites its sources |
| 💾 **Persistent Storage** | SQLite-backed session and message history |
| ⚡ **History Optimization** | Multiple strategies to manage context length |
| 🎨 **Modern UI** | Next.js frontend with real-time updates |
| 🐳 **Docker Ready** | One-command deployment |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAG CHATBOT ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌──────────────────────────────┐   │
│  │   Next.js   │────▶│   FastAPI   │────▶│      LangGraph Engine        │   │
│  │   Frontend  │     │     API     │     │  ┌────────┐    ┌──────────┐  │   │
│  └─────────────┘     └─────────────┘     │  │Retrieve│───▶│ Generate │  │   │
│                                          │  └────────┘    └──────────┘  │   │
│                                          └──────────────────────────────┘   │
│                                                     │                       │
│                      ┌──────────────────────────────┼───────────────┐       │
│                      │                              │               │       │
│                      ▼                              ▼               ▼       │
│             ┌──────────────────┐          ┌──────────────┐   ┌───────────┐  │
│             │  Hybrid Search   │          │   History    │   │   Ollama  │  │
│             │ ┌──────┐ ┌─────┐ │          │   Manager    │   │    LLM    │  │
│             │ │Vector│+│BM25 │ │          │  (Strategy)  │   │           │  │
│             │ └──────┘ └─────┘ │          └──────────────┘   └───────────┘  │
│             └────────┬─────────┘                 │                          │
│                      │                           │                          │
│             ┌────────▼────────┐          ┌───────▼───────┐                  │
│             │    ChromaDB     │          │    SQLite     │                  │
│             │  (Embeddings)   │          │  (Sessions)   │                  │
│             └─────────────────┘          └───────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---


## 🔬 Advanced Techniques

### 1. Hybrid Search (Vector + BM25)

Combines the best of semantic and keyword search for superior retrieval:

```python
# Ensemble retriever with configurable weights
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # Balanced semantic + keyword
)
```

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **Vector Search** | Semantic understanding, synonyms | May miss exact matches |
| **BM25 Search** | Exact keyword matching, fast | No semantic understanding |
| **Hybrid** | ✅ Best of both worlds | Slightly more compute |

### 2. Semantic Chunking

Unlike fixed-size chunking, semantic chunking splits documents at natural topic boundaries:

```python
text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # or: standard_deviation, interquartile, gradient
    breakpoint_threshold_amount=95
)
```

**Breakpoint Types:**
| Type | Description | Best For |
|------|-------------|----------|
| `percentile` | Split at Nth percentile similarity drop | General use |
| `standard_deviation` | Split at N std deviations from mean | Technical docs |
| `interquartile` | IQR-based outlier detection | Mixed content |
| `gradient` | Rate of change in similarity | Narrative text |

### 3. LangGraph State Machine

Structured conversation flow with typed state:

```python
class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    context: str
    current_query: str
    citations: list[dict]

workflow = StateGraph(ChatState)
workflow.add_node("retrieve", self._retrieve_context)
workflow.add_node("generate", self._generate_response)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
```

```
┌────────────────────────────────────────────────────────────────────────┐
|        ┌─────────┐     ┌──────────┐     ┌──────────┐    ┌─────┐        |
|        │  START  │────▶│ Retrieve │────▶│ Generate │───▶│ END │        |
|        └─────────┘     └──────────┘     └──────────┘    └─────┘        |
|                            │                │                          |
|                            ▼                ▼                          |
|                      [Hybrid Search]   [LLM + Citations]               |
|                                                                        |
└────────────────────────────────────────────────────────────────────────┘
```

### 4. Intelligent History Management

Multiple strategies to optimize context sent to LLM:

```python
strategies = {
            "sliding_window": lambda: SlidingWindowStrategy(config.max_messages),
            "token_budget": lambda: TokenBudgetStrategy(
                config.max_tokens, config.model_name
            ),
            "summarization": lambda: SummarizationStrategy(
                llm, config.summarize_after, config.summary_max_tokens
            ) if llm else SlidingWindowStrategy(config.max_messages)
        }
```

| Strategy | How It Works | Best For |
|----------|--------------|----------|
| **Sliding Window** | Keep last N messages | Most conversations |
| **Token Budget** | Limit by token count | Cost-sensitive apps |
| **Summarization** | Summarize old messages | Long conversations |


**Flow:**
```
Full History (20 messages)
         │
         ▼
┌─────────────────────┐
│  History Manager    │
│  ┌───────────────┐  │
│  │   Strategy    │  │
│  └───────────────┘  │
│         │           │
│   20 → 6-10 msgs    │
└─────────────────────┘
         │
         ▼
   Optimized History
         │
         ▼
┌─────────────────────┐
│   LLM Generation    │
│  (faster, cheaper)  │
└─────────────────────┘
```

### 5. Citation Tracking System

Every response includes source citations with metadata:

```python
citation = {
    "number": 1,
    "title": "Neural Networks",
    "display": "Neural Networks (Wikipedia)",
    "source_type": "wikipedia",  # or: pdf, web, custom
    "content_preview": "A neural network is...",
    "url": "https://en.wikipedia.org/wiki/...",
    "page": 42,        # For PDFs
    "chunk_id": 3      # For traceability
}
```


### 6. Persistent Storage Architecture

SQLite-backed session management:

```sql
-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_message_at TEXT NOT NULL,
    metadata TEXT
);

-- Messages table with citations
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations TEXT,  -- JSON array
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
```

---
## Quick Start (Docker)

The easiest way to run the chatbot is with Docker Compose. Ollama runs on your host machine.

#### 1. Start Ollama on your host machine
- Install Ollama (Linux)
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

- Or download from https://ollama.com/download

- Run Ollama Server
  ```bash
  ollama serve
  ```

#### 2. Pull the model (if not already done)
```bash
ollama pull <model_name>
```

#### 3. Copy and configure `.env`
```bash
cp .env.example .env
```

#### 4. Build and run container
```bash
docker compose up
```

## Docker Commands

```bash
# Start API
docker compose up -d

# View logs
docker compose logs -f <container_name>

# Stop API
docker compose down

# Rebuild after code changes
docker compose build --no-cache

# Reset everything (including data)
docker compose down -v
```

## Manual Setup (Without Docker)

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and running locally

#### 1. Install Ollama and Pull Model

- Install Ollama (Linux)
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

- Or download from https://ollama.com/download

- Start Ollama service
  ```bash
  ollama serve
  ```

- Pull the model (in another terminal)
  ```bash
  ollama pull <model_name>
  ```

#### 2. Set Up Python Environment

- Create and activate virtual environment
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
- Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

#### 3. Configure Environment

- Copy example environment file
  ```bash
  cp .env.example .env
  ```
- Edit .env to customize settings (optional)

#### 4. Ingest Documents

```bash
python ingest.py
```

#### 5. Start the Server

```bash
python api.py
```

## Configuration

All configuration is managed via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2:1b` | Ollama model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_API_KEY` | `38a2258........` | Api Key for cloud Ollama model |
| `OLLAMA_TEMPERATURE` | `0.7` | LLM temperature (0.0-2.0) |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store directory |
| `CHROMA_COLLECTION_NAME` | `wiki_docs` | ChromaDB collection name |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `EMBEDDING_DEVICE` | `cpu` | Device for embeddings (cpu/cuda) |
| `RAG_RETRIEVAL_K` | `3` | Number of chunks to retrieve |
| `DISTILLED_RETRIEVAL_K` | `3` | Number of chunks after context distillation |
| `RAG_BREAKPOINT_TYPE` | `percentile` | Semantic chunker type* |
| `RAG_BREAKPOINT_AMOUNT` | `95` | Threshold for chunk boundaries |
| `HISTORY_STRATEGY`           | `sliding_window` | Strategy used for managing conversation history         |
| `HISTORY_MAX_MESSAGES`       | `10`             | Maximum number of past messages to retain               |
| `HISTORY_MAX_TOKENS`         | `4000`           | Maximum total tokens allowed in history before trimming |
| `HISTORY_SUMMARIZE_AFTER`    | `8`              | Number of messages after which summarization begins     |
| `HISTORY_SUMMARY_MAX_TOKENS` | `500`            | Maximum token limit for the generated summary           |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `WIKI_TOPICS` | *see .env.example* | Comma-separated Wikipedia topics |


*Breakpoint types: `percentile`, `standard_deviation`, `interquartile`, `gradient`





## 📡 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send message, get response with citations |
| `GET` | `/api/sessions` | List all sessions |
| `GET` | `/api/history/{id}` | Get session history |
| `DELETE` | `/api/session/{id}` | Delete session |
| `POST` | `/api/session/new` | Create new session |

### Strategy Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/strategies` | Get available strategies |
| `POST` | `/api/strategies` | Change active strategy |

### System Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | Health check |
| `GET` | `/api/config` | Current configuration |


## Switching Models

To use a different Ollama model:

```bash
# Pull new model
ollama pull mistral

# Update .env
OLLAMA_MODEL=mistral

# Restart the server
python main.py
```

Popular model options: `llama3.2`, `llama3.2:1b`, `mistral`, `phi3`, `gemma2`

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Ollama connection error** | Ensure `ollama serve` is running |
| **Model not found** | Run `ollama pull llama3.2:1b` |
| **ChromaDB error** | Delete `chroma_db/` and re-run `ingest.py` |
| **`Permission Error` for `data` directory** | Create `/api/data` directory before docker build |
| **Slow responses** | Use smaller model or reduce `RAG_RETRIEVAL_K` |
| **High memory usage** | Switch to `token_budget` history strategy |
| **Docker can't reach Ollama** | Use `host.docker.internal` or `network_mode: host` |

---

## 📚 Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM Framework** | LangChain, LangGraph |
| **Vector Store** | ChromaDB |
| **Embeddings** | HuggingFace (BGE) |
| **Search** | Ensemble (Vector + BM25) |
| **LLM** | Ollama (Llama 3.2, Mistral, etc.) |
| **Backend** | FastAPI, Pydantic |
| **Storage** | SQLite |
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS |
| **Deployment** | Docker, Docker Compose |

---

## 📄 License

MIT License - Feel free to use in your projects!