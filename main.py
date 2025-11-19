# main.py - COMPLETE WORKING VERSION (No LangChain dependencies)
import os
import re
import pandas as pd
import requests
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
import numpy as np
import faiss
import json

# ========== CONFIGURATION ==========
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_PRIMARY = os.getenv("OPENROUTER_MODEL", "deepseek-r1:free")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ========== LLM CLIENT ==========
def call_openrouter_model(model: str, messages: List[Dict[str,str]], extra: Optional[Dict]=None, timeout=60):
    payload = {
        "model": model,
        "messages": messages
    }
    if extra:
        payload["extra_body"] = extra
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()

# ========== SCRAPER ==========
def fetch_html(url, timeout=12):
    try:
        headers = {"User-Agent": "ai-websearch-agent/0.1"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print("Fetch error:", e)
        return None

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    title = soup.title.string.strip() if soup.title else ""
    paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    text = "\n\n".join([p for p in paragraphs if p])
    return title, text

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        end = i + chunk_size
        chunk = text[i:end]
        chunks.append(chunk)
        i = max(end - overlap, end)
    return chunks

# ========== EMBEDDINGS & SEARCH ==========
class SimpleVectorSearch:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def add_documents(self, documents):
        """Add documents to the search index"""
        self.documents = documents
        texts = [doc["text"] for doc in documents]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        # Create FAISS index
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.embeddings)
        
    def search(self, query, k=4):
        """Search for similar documents"""
        if self.index is None:
            return []
            
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(query_embedding, k)
        
        results = []
        for i, score in zip(I[0], D[0]):
            if i < len(self.documents):
                doc = self.documents[i]
                results.append({
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "score": float(score)
                })
        return results

# ========== SUPABASE ==========
def upload_dataframe(df: pd.DataFrame):
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            records = df.to_dict(orient="records")
            if records:
                result = supabase.table("web_search_results").insert(records).execute()
                print(f"Uploaded {len(records)} records to Supabase")
        except Exception as e:
            print(f"Supabase upload error: {e}")
    else:
        print("Supabase credentials not found, skipping upload")

# ========== MAIN PIPELINE ==========
def prompt_to_urls(query: str, model_name=OPENROUTER_PRIMARY):
    prompt = (
        "You are a web search assistant. Given this query, produce a list of up to 5 high-quality public URLs "
        "that are relevant, one URL per line. Do not include paywalled sites. Query:\n\n"
        f"{query}\n\nList URLs only:"
    )
    try:
        resp = call_openrouter_model(model_name, [{"role":"user","content":prompt}])
        content = resp["choices"][0]["message"]["content"]
        urls = re.findall(r'https?://[^\s,;]+', content)
        if not urls:
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("http"):
                    urls.append(line)
        return urls[:5]  # Limit to 5 URLs
    except Exception as e:
        print(f"Error getting URLs from LLM: {e}")
        # Fallback URLs
        return [
            "https://en.wikipedia.org/wiki/Web_scraping",
            "https://towardsdatascience.com/a-complete-guide-to-web-scraping-with-python-2024-2025-1234567890"
        ]

def get_answer_from_llm(query: str, context: str, model_name=OPENROUTER_PRIMARY):
    """Get answer from LLM using context"""
    prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
    
    try:
        resp = call_openrouter_model(model_name, [{"role":"user","content":prompt}])
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error getting answer from LLM: {e}"

def run_pipeline(query: str):
    print("Starting AI Web Scraper...")
    print("Query:", query)
    
    # Step 1: Get URLs from LLM
    urls = prompt_to_urls(query)
    print("URLs found:", urls)

    # Step 2: Scrape and process content
    docs = []
    for url in urls:
        print(f"Scraping: {url}")
        html = fetch_html(url)
        if not html:
            continue
        title, text = extract_text(html)
        if not text or len(text) < 100:
            continue
            
        print(f"Found content: {title} ({len(text)} chars)")
        chunks = chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            docs.append({
                "text": chunk,
                "metadata": {"source": url, "title": title, "chunk": i}
            })

    print(f"Chunks prepared: {len(docs)}")
    if not docs:
        print("No content found, exiting.")
        return None

    # Step 3: Create search index and find relevant content
    print("Creating search index...")
    search_engine = SimpleVectorSearch()
    search_engine.add_documents(docs)
    
    # Step 4: Search for relevant content
    relevant_docs = search_engine.search(query, k=3)
    print(f"Found {len(relevant_docs)} relevant documents")
    
    # Step 5: Prepare context for LLM
    context = "\n\n".join([doc["text"] for doc in relevant_docs])
    
    # Step 6: Get final answer from LLM
    print("Getting answer from LLM...")
    answer = get_answer_from_llm(query, context)
    print(f"LLM Answer: {answer}")

    # Step 7: Prepare data for Supabase
    rows = []
    for doc in relevant_docs:
        rows.append({
            "title": doc["metadata"].get("title", "No title"),
            "url": doc["metadata"].get("source", "No URL"),
            "snippet": doc["text"][:300] + "..." if len(doc["text"]) > 300 else doc["text"],
            "query": query,
            "score": doc["score"]
        })
    
    df = pd.DataFrame(rows)
    print("Data to upload:")
    print(df[['title', 'url', 'score']].to_string())
    
    # Step 8: Upload to Supabase
    print("Uploading to Supabase...")
    upload_dataframe(df)
    
    return {
        "answer": answer,
        "table": df,
        "sources_count": len(relevant_docs)
    }

if __name__ == "__main__":
    # Test with a simple query
    query = "what is web scraping"
    result = run_pipeline(query)
    
    if result:
        print(f"\n✅ Success! Processed {result['sources_count']} sources")
        print(f"Answer: {result['answer'][:200]}...")
    else:
        print("\n❌ Pipeline failed")
