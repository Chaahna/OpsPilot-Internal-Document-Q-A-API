# app/data_loader.py

import os
from pathlib import Path
from langchain.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

# Constants
DATA_DIR = "data/internal_docs"
INDEX_DIR = "vectorstore_index"

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_and_split_docs():
    docs = []

    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)

        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(filepath)
        elif filename.endswith(".md"):
            loader = UnstructuredMarkdownLoader(filepath)
        else:
            continue

        # Load and chunk
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs.extend(splitter.split_documents(raw_docs))

    return docs

def create_faiss_index(docs):
    if not docs:
        print("No documents loaded.")
        return None

    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(INDEX_DIR)
    print(f"✅ FAISS index saved to {INDEX_DIR}")
    return db

if __name__ == "__main__":
    print("🔍 Loading and embedding documents...")
    docs = load_and_split_docs()
    create_faiss_index(docs)
