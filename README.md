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

## 📁 Project Structure

