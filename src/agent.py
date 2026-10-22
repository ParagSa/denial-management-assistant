"""
agent.py
A LangGraph agent that picks between tools (denial code lookup, policy doc
search, appeal letter drafting) based on the user's question.
"""

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from tools import ALL_TOOLS

MODEL_NAME = "llama3.1:8b"

SYSTEM_PROMPT = """You are a Revenue Cycle Management assistant for a medical
billing team. You help staff understand insurance claim denials and draft
appeals.

Use your tools when appropriate:
- lookup_denial_code: for a specific code like CO-97
- search_policy_docs: for broader questions about appeal processes or policy
- draft_appeal_letter: when the user asks you to write/draft an appeal

Be concise and practical. If you don't have enough information to draft a
letter (e.g. missing claim details), ask a clarifying question instead of
guessing."""


def main():
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    agent = create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)

    print("RCM Agent (type 'exit' to quit)\n")
    history = []

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        history.append({"role": "user", "content": question})
        result = agent.invoke({"messages": history})

        # last message is the agent's final answer
        answer = result["messages"][-1].content
        print(f"\nAssistant: {answer}\n")

        history = result["messages"]


if __name__ == "__main__":
    main()