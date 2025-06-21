# app/rag_pipeline.py

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import TextLoader

# 1. Load embeddings and vector index
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.load_local(
    folder_path="vectorstore_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# 2. Define your retriever
retriever = db.as_retriever()

# 3. Choose your LLM - OpenAI or HuggingFaceHub
# Option A: Using Hugging Face Hub (requires huggingface-cli login)
# from langchain_community.llms import HuggingFaceHub
# llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 256})

# Option B: Use any local LLM later — for now, simulate response
from langchain.llms.fake import FakeListLLM
llm = FakeListLLM(responses=["Simulated answer. Replace this with real model."])

# 4. Set up the QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 5. Ask a question
query = input("Ask your internal assistant a question: ")
result = qa({"query": query})

# 6. Show result
print("\n📣 Answer:", result["result"])
print("\n📄 Sources:")
for doc in result["source_documents"]:
    print("-", doc.metadata["source"])
