# app/main.py

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from app.agent_pipeline import agent
from transformers import pipeline

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="OpsPilot – Internal Assistant API")

# Load FAISS vector index
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    folder_path="vectorstore_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever()

# Load local FLAN-T5 model via Transformers
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Input schema
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def ask_question(req: QueryRequest):
    try:
        result = qa_chain.invoke({"query": req.query})
        sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
        return {
            "answer": result["result"],
            "sources": list(set(sources))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))