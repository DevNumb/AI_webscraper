# embeddings.py
from langchain.embeddings import HuggingFaceEmbeddings

def get_embeddings_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    # Utilisez HuggingFaceEmbeddings au lieu de SentenceTransformerEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)
