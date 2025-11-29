# RAG Chatbot with LangGraph

A context-aware chatbot that retrieves information from Wikipedia articles about AI/ML topics, maintains conversational context, and provides responses via a FastAPI backend and web UI.

## Quick Start (Docker)

The easiest way to run the chatbot is with Docker Compose. Ollama runs on your host machine.

### 1. Start Ollama on your host machine
```bash
ollama serve
```

#### 2. Pull the model (if not already done)
```bash
ollama pull llama3.2:1b
```

### 3. Copy and configure `.env`
```bash
cp .env.example .env
```

### 4. Build and run container
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

### 1. Install Ollama and Pull Model

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
  ollama pull llama3.2:1b
  ```

### 2. Set Up Python Environment

- Create and activate virtual environment
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
- Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

### 3. Configure Environment

- Copy example environment file
  ```bash
  cp .env.example .env
  ```
- Edit .env to customize settings (optional)

### 4. Ingest Documents

```bash
python ingest.py
```

### 5. Start the Server

```bash
python api.py
```

## Configuration

All configuration is managed via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2:1b` | Ollama model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_TEMPERATURE` | `0.7` | LLM temperature (0.0-2.0) |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store directory |
| `CHROMA_COLLECTION_NAME` | `wiki_docs` | ChromaDB collection name |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `EMBEDDING_DEVICE` | `cpu` | Device for embeddings (cpu/cuda) |
| `RAG_RETRIEVAL_K` | `3` | Number of chunks to retrieve |
| `RAG_BREAKPOINT_TYPE` | `percentile` | Semantic chunker type* |
| `RAG_BREAKPOINT_AMOUNT` | `95` | Threshold for chunk boundaries |

*Breakpoint types: `percentile`, `standard_deviation`, `interquartile`, `gradient`
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `WIKI_TOPICS` | *see .env.example* | Comma-separated Wikipedia topics |





## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve web UI |
| GET | `/api/status` | Check system status |
| GET | `/api/config` | Get current configuration |
| POST | `/api/chat` | Send message, get response |
| GET | `/api/sessions` | Get all active sessions |
| GET | `/api/history/{session_id}` | Get conversation history |
| DELETE | `/api/session/{session_id}` | Clear session |
| POST | `/api/session/new` | Create new session |

### Example API Usage

```bash
# Check status
curl http://localhost:8000/api/status

# View config
curl http://localhost:8000/api/config

# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is machine learning?"}'

# Get all sessions
curl http://localhost:8000/api/sessions

# Get session history
curl http://localhost:8000/api/history/{session_id}
```

## Switching Models

To use a different Ollama model:

```bash
# Pull new model
ollama pull mistral

# Update .env
OLLAMA_MODEL=mistral

# Restart the server
python api.py
```

Popular model options: `llama3.2`, `llama3.2:1b`, `mistral`, `phi3`, `gemma2`

## Troubleshooting

### Docker Issues

**Cannot connect to Ollama**: Ensure Ollama is running on your host
```bash
ollama serve
# Check it's working
curl http://localhost:11434/api/tags
```


### General Issues

**Ollama connection error**: Ensure Ollama is running (`ollama serve`)

**Model not found**: Pull the model first (`ollama pull llama3.2`)

**ChromaDB error**: Delete `chroma_db/` folder and re-run `ingest.py`

**Slow responses**: Use a smaller model (`OLLAMA_MODEL=llama3.2:1b`)

**Config not loading**: Ensure `.env` file exists in the project root
