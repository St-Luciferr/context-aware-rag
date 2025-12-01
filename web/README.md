# RAG Chatbot Frontend

A modern Next.js frontend for the RAG Chatbot API with session management.

## Features

- 🎨 Modern dark theme UI
- 💬 Real-time chat interface
- 📁 Session management (create, switch, delete)
- 📱 Responsive design (mobile-friendly)
- ⚡ Optimistic updates for fast UX
- 🔄 Auto-refresh status indicator

## Quick Start

### Development

```bash
# Install dependencies
npm install

# Copy environment file
cp .env.local.example .env.local

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Production

```bash
# Build
npm run build

# Start production server
npm start
```

### Docker

```bash
# Build and run
docker build -t rag-chatbot-frontend .
docker run -p 3000:3000 rag-chatbot-frontend
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API URL |


## Tech Stack

- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Lucide React** - Icons
- **date-fns** - Date formatting