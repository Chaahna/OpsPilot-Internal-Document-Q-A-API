# 🔍 Internal Document Q&A API

This project is a **local retrieval-augmented question answering (RAG) system** built with LangChain, HuggingFace, and FastAPI. It allows users to query internal documents like IT policies, FAQs, and handbooks through a simple API.

---

## 🚀 Features

- 📂 Loads and indexes Markdown and PDF documents
- 🧠 Embeds documents using `all-MiniLM-L6-v2` via HuggingFace
- 🔎 Uses FAISS vector store for fast retrieval
- 💬 Answers questions using HuggingFace-hosted LLM (e.g., `mistralai/Mistral-7B-Instruct`)
- 🧪 Simple FastAPI interface at `POST /query`

---


## ⚙️ How It Works

1. Embeds all `.md` and `.pdf` files into vectors using HuggingFace embeddings.
2. Stores them in a FAISS index.
3. On a query, the top-k relevant chunks are retrieved.
4. A HuggingFace-hosted LLM generates an answer using those chunks.

---

## 🧪 Example Query

```bash
curl -X POST 'http://127.0.0.1:8000/query' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{ "query": "What is our leave policy?" }'

---

📦 Dependencies
langchain<br>

langchain-community

langchain-huggingface

transformers

sentence-transformers

faiss-cpu

fastapi

uvicorn
