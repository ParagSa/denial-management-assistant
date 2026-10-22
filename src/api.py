"""
api.py
Thin FastAPI wrapper around the LangGraph agent, so the Angular frontend
can talk to it over REST.

Run with:
    uvicorn src.api:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from .tools import ALL_TOOLS

MODEL_NAME = "llama3.1:8b"

SYSTEM_PROMPT = """You are a Revenue Cycle Management assistant for a medical
billing team. You help staff understand insurance claim denials and draft
appeals.

Use your tools when appropriate:
- lookup_denial_code: for a specific code like CO-97
- search_policy_docs: for broader questions about appeal processes or policy
- draft_appeal_letter: when the user asks you to write/draft an appeal

Be concise and practical."""

app = FastAPI(title="RCM Denial Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

_llm = ChatOllama(model=MODEL_NAME, temperature=0)
_agent = create_react_agent(_llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)

# naive in-memory session store, keyed by session id from the client
_sessions: dict[str, list] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = _sessions.get(req.session_id, [])
    history.append({"role": "user", "content": req.message})

    result = _agent.invoke({"messages": history})
    _sessions[req.session_id] = result["messages"]

    reply = result["messages"][-1].content
    return ChatResponse(reply=reply)


@app.get("/health")
def health():
    return {"status": "ok"}