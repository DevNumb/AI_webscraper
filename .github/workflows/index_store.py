# src/index_store.py
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import faiss
import os
import pickle

INDEX_PATH = "faiss_index.pkl"  # small example persistence

def build_faiss_index(docs: list, embeddings, persist_path=INDEX_PATH):
    """
    docs: list of dicts with keys {'text','metadata'}
    embeddings: LangChain embeddings object
    """
    documents = [Document(page_content=d["text"], metadata=d.get("metadata", {})) for d in docs]
    vectorstore = FAISS.from_documents(documents, embeddings)
    # persist by saving object
    with open(persist_path, "wb") as f:
        pickle.dump(vectorstore, f)
    return vectorstore

def load_index(persist_path=INDEX_PATH):
    import pickle
    if not os.path.exists(persist_path):
        return None
    with open(persist_path, "rb") as f:
        return pickle.load(f)
