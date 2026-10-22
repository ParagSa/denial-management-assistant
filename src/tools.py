"""
tools.py
Tools the agent can call: structured denial code lookup, RAG search over
policy docs, and an appeal letter drafter.
"""

from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

PERSIST_DIR = "chroma_db"
MODEL_NAME = "phi3:mini"

# A small structured lookup table for exact-code queries (fast, no LLM needed)
DENIAL_CODES = {
    "CO-97": "The benefit for this service is included in the payment/allowance for another service/procedure that has already been adjudicated.",
    "CO-16": "Claim/service lacks information needed for adjudication.",
    "PR-1": "Deductible amount. Not typically appealable; verify patient's deductible status.",
    "CO-50": "These are non-covered services because this is not deemed a medical necessity.",
}

_embeddings = OllamaEmbeddings(model="nomic-embed-text")
_vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings)
_retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})
_llm = ChatOllama(model=MODEL_NAME, temperature=0)


@tool
def lookup_denial_code(code: str) -> str:
    """Look up the exact meaning of a specific denial code, e.g. 'CO-97'.
    Use this when the user asks what a specific code means."""
    code = code.strip().upper()
    return DENIAL_CODES.get(code, f"No exact match found for {code} in the lookup table.")


@tool
def search_policy_docs(query: str) -> str:
    """Search the policy documents for relevant guidance when the question
    is broader than a single code, e.g. 'how do I appeal a medical necessity denial'."""
    docs = _retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)


@tool
def draft_appeal_letter(denial_reason: str, claim_details: str) -> str:
    """Draft a short, professional appeal letter given a denial reason and
    relevant claim details (e.g. patient/service info, dates, notes)."""
    prompt = f"""Write a concise, professional insurance claim appeal letter.

Denial reason: {denial_reason}
Claim details: {claim_details}

The letter should be 3-4 short paragraphs, formal tone, and reference the
denial reason directly."""
    response = _llm.invoke(prompt)
    return response.content


ALL_TOOLS = [lookup_denial_code, search_policy_docs, draft_appeal_letter]