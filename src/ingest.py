"""
ingest.py
Loads text files from data/policy_docs/ into a local Chroma vector store,
using Ollama for embeddings.
"""

import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "data/policy_docs"
PERSIST_DIR = "chroma_db"


def main():
    print("Loading documents...")
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunk(s).")

    print("Embedding and storing in Chroma (this calls your local Ollama model)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    print(f"Done. Vector store saved to '{PERSIST_DIR}/'.")


if __name__ == "__main__":
    main()