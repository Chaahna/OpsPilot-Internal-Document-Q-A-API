import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.agents.agent_toolkits.vectorstore.base import create_vectorstore_agent
from langchain.agents.agent_toolkits.vectorstore.toolkit import VectorStoreToolkit, VectorStoreInfo
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import RetrievalQA
# Set your HF token
hf_api_key = os.getenv("HF_API_KEY", "your_hf_token_here")

# Set up embedding model and load vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore_index", embedding_model, allow_dangerous_deserialization=True)

# Set up LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=hf_api_key,
    task="text-generation",
    temperature=0.5,
    max_new_tokens=512,
    return_full_text=False
)

# Create agent
vectorstore_info = VectorStoreInfo(
    name="InternalDocs",
    description="Access to internal FAQs, employee handbook, and IT policy documentation.",
    vectorstore=vectorstore
)

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)

retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",  
    return_source_documents=True
)

agent = retrieval_qa
