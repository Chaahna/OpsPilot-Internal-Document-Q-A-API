# OpsPilot-Internal-Document-Q&A-API
This project is a local retrieval-augmented question answering (RAG) system built with LangChain, HuggingFace, and FastAPI. It allows users to query internal documents like IT policies, FAQs, and handbooks through a simple API.


🚀 Features
📂 Loads and indexes Markdown and PDF documents

🧠 Embeds documents using all-MiniLM-L6-v2 via HuggingFace

🔎 Uses FAISS vector store for fast retrieval

💬 Answers questions using HuggingFace-hosted LLM (e.g., mistralai/Mistral-7B-Instruct)

🧪 Simple FastAPI interface at POST /query

📁 Project Structure
bash
Copy
Edit
├── app/
│   ├── main.py            # FastAPI app
│   ├── vectorstore/       # FAISS index
│   └── data/              # Internal .md and .pdf files
├── requirements.txt
└── README.md
⚙️ How It Works
Embeds all .md and .pdf files into vectors.

Stores them in a FAISS index.

On a query, the top-k relevant chunks are retrieved.

A HuggingFace-hosted model generates the answer using the retrieved content.

🧪 Example Query
bash
Copy
Edit
curl -X POST 'http://127.0.0.1:8000/query' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{ "query": "What is our leave policy?" }'
📦 Dependencies
langchain

langchain-community

langchain-huggingface

transformers

sentence-transformers

faiss-cpu

fastapi

uvicorn

Install with:

bash
Copy
Edit
pip install -r requirements.txt
🛠️ Notes
You must set your HuggingFace API key as an env variable:
export HF_API_KEY=your_token_here

The system runs on cpu by default and is intended for small-scale/local use.

🧑‍💻 Future Improvements
Agent integration with tool-calling (LangGraph)

Multi-turn chat history

Frontend UI
