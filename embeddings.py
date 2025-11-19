# src/embeddings.py
from langchain.embeddings import SentenceTransformerEmbeddings

def get_embeddings_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    # LangChain's wrapper around sentence-transformers
    return SentenceTransformerEmbeddings(model_name=model_name)
