"""
mcp_server.py
Exposes denial-code lookup and policy doc search as an MCP server, so any
MCP-compatible client (Claude Desktop, another agent, etc.) can call them
without custom integration code.

Run with:
    python src/mcp_server.py

Then point an MCP client (e.g. Claude Desktop config) at this script.
"""

import os
from mcp.server.mcpserver import MCPServer
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Use an absolute path — Claude Desktop launches this script with an
# unpredictable working directory, so a relative path can fail or write
# to the wrong (sometimes restricted) location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "..", "chroma_db")
MODEL_NAME = "llama3.1:8b"

mcp = MCPServer("rcm-denial-assistant")

DENIAL_CODES = {
    "CO-97": "The benefit for this service is included in the payment/allowance for another service/procedure that has already been adjudicated.",
    "CO-16": "Claim/service lacks information needed for adjudication.",
    "PR-1": "Deductible amount. Not typically appealable; verify patient's deductible status.",
    "CO-50": "These are non-covered services because this is not deemed a medical necessity.",
}

_embeddings = OllamaEmbeddings(model="nomic-embed-text")
_vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings)
_retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})


@mcp.tool()
def lookup_denial_code(code: str) -> str:
    """Look up the exact meaning of a specific insurance claim denial code,
    e.g. 'CO-97'."""
    code = code.strip().upper()
    return DENIAL_CODES.get(code, f"No exact match found for {code}.")


@mcp.tool()
def search_policy_docs(query: str) -> str:
    """Search RCM policy documents for guidance on appeals, denial handling,
    and related billing policy questions."""
    docs = _retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)


if __name__ == "__main__":
    mcp.run(transport="stdio")