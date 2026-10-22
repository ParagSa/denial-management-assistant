# Denial Management Assistant

A lightweight Revenue Cycle Management (RCM) assistant built to help billing teams understand insurance claim denials, search policy guidance, and draft appeal letters using local AI tooling.

This project combines:

- a LangGraph agent with tool calling
- a local Chroma vector database for policy document retrieval
- Ollama models for embeddings and LLM inference
- a FastAPI chat endpoint for app integration
- an MCP server for external tool access

## What it does

- Looks up common denial codes like `CO-97`, `CO-16`, `PR-1`, and `CO-50`
- Searches policy documentation for broader denial and appeal guidance
- Drafts concise appeal letters based on denial reason and claim details
- Exposes the same functionality through a local API and MCP-compatible server

## Project structure

```text
.
├── data/
│   └── policy_docs/
│       └── codes.txt
├── chroma_db/
│   └── local vector database files
├── src/
│   ├── agent.py             # interactive terminal agent
│   ├── api.py               # FastAPI chat endpoint
│   ├── ingest.py            # loads policy docs into Chroma
│   ├── mcp_server.py        # MCP tool server
│   ├── rag_chat.py          # direct RAG Q&A loop
│   └── tools.py             # denial lookup, policy search, appeal drafting tools
├── requirements.txt
├── README.md
└── .gitignore
```

## Prerequisites

- Python 3.10+
- Ollama installed and running locally
- Access to the following Ollama models:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

If you use a different model name in the code, make sure it matches the model available in your local Ollama installation.

## Installation

From the project root:

```bash
pip install -r requirements.txt
```

## Ingest policy documents

The vector database is stored in the `chroma_db/` folder and must be populated before searching policy documents:

```bash
python src/ingest.py
```

This loads the files in `data/policy_docs/` into Chroma using Ollama embeddings.

## Run the app

### 1) Interactive terminal agent

```bash
python src/agent.py
```

Example prompts:

- `What does CO-97 mean?`
- `How do I appeal a medical necessity denial?`
- `Draft an appeal letter for a denied claim with no prior authorization.`

### 2) Simple RAG chat loop

```bash
python src/rag_chat.py
```

This uses a retrieval-augmented generation flow grounded in the policy documents.

### 3) FastAPI backend

```bash
uvicorn src.api:app --reload --port 8000
```

Then call the API:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"What does CO-97 mean?"}'
```

### 4) MCP server

```bash
python src/mcp_server.py
```

This exposes the denial-code lookup and document search tools to MCP-compatible clients.

## Configuration notes

- The project stores the vector database locally in `chroma_db/`
- The LLM and embedding model names are configured in the Python files under `src/`
- The API is configured for local frontend development and currently allows origins from `http://localhost:4200`

## Use cases

This assistant is designed for:

- denial code interpretation
- policy-based appeal guidance
- assisting billing staff during claim review
- rapid drafting of short appeal letters

## Notes

This project is built for local development and experimentation. It is not a production-grade billing system, but it provides a practical foundation for RCM denial support workflows using LLMs and retrieval.

## Future improvements

- add persistent user/session storage
- expand the denial code lookup table
- improve prompt quality and template grounding
- add authentication and production deployment setup
- connect the backend to a real front-end application
