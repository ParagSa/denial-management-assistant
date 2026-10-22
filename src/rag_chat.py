"""
rag_chat.py
Simple RAG loop: ask a question, retrieve relevant chunks from Chroma,
and have Ollama answer grounded in that context.
"""

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PERSIST_DIR = "chroma_db"
MODEL_NAME = "phi3:mini"

PROMPT_TEMPLATE = """You are a Revenue Cycle Management assistant helping billing staff
understand and resolve insurance claim denials.

Answer the question using ONLY the context below. If the context doesn't contain
the answer, say you don't have that information rather than guessing.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Denial Management Assistant (type 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer = chain.invoke(question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()