# main.py (merged with llm_client)
import os
import re
import pandas as pd
import requests
from typing import Optional, List, Dict, Any
from scraper import fetch_html, extract_text, chunk_text, find_links_from_text
from embeddings import get_embeddings_model
from index_store import build_faiss_index, load_index
from uploader import upload_dataframe
from langchain.llms.base import LLM
from langchain.schema import LLMResult
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from tqdm.auto import tqdm

# OpenRouter configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_PRIMARY = os.getenv("OPENROUTER_MODEL", "deepseek-r1:free")
OPENROUTER_FALLBACK = os.getenv("OPENROUTER_FALLBACK", "nvidia/nemotron-nano-12b-v2-vl:free")

# A thin helper to call OpenRouter
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
    data = resp.json()
    # openrouter returns choices[0].message.content
    return data

# LangChain-compatible LLM wrapper for OpenRouter
class OpenRouterLLM(LLM):
    model_name: str = "deepseek-r1:free"
    temperature: float = 0.0

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Using chat format; we put prompt in a single user message
        msgs = [{"role": "user", "content": prompt}]
        extra = {"temperature": self.temperature}
        data = call_openrouter_model(self.model_name, msgs, extra=extra)
        content = data["choices"][0]["message"]["content"]
        return content

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # For simplicity not implemented (synchronous)
        raise NotImplementedError

def prompt_to_urls(query: str, model_name=OPENROUTER_PRIMARY):
    # ask LLM to propose URLs (simple prompt)
    prompt = (
        "You are a web search assistant. Given this query, produce a list of up to 8 high-quality public URLs "
        "that are relevant, one URL per line. Do not include paywalled sites. Query:\n\n"
        f"{query}\n\nList URLs only:"
    )
    resp = call_openrouter_model(model_name, [{"role":"user","content":prompt}])
    content = resp["choices"][0]["message"]["content"]
    # simple URL extraction
    urls = re.findall(r'https?://[^\s,;]+', content)
    # fallback: if none, try to extract lines that look like domains
    if not urls:
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("http"):
                urls.append(line)
    return urls

def run_pipeline(query: str):
    print("Query:", query)
    urls = prompt_to_urls(query)
    print("URLs found:", urls)

    docs = []
    for url in urls:
        html = fetch_html(url)
        if not html:
            continue
        title, text = extract_text(html)
        if not text or len(text) < 200:
            continue
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            docs.append({
                "text": chunk,
                "metadata": {"source": url, "title": title, "chunk": i}
            })

    print("Chunks prepared:", len(docs))
    if not docs:
        return None

    embeddings = get_embeddings_model()
    vs = build_faiss_index(docs, embeddings)

    # Build a LangChain retrieval QA chain using our OpenRouter LLM wrapper
    llm = OpenRouterLLM()
    llm.model_name = OPENROUTER_PRIMARY
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Simple prompt template
    prompt_template = """Use the context to answer succinctly. If unknown, say you don't know.
    Context:
    {context}

    Question: {question}
    Answer:"""
    prompt = PromptTemplate(input_variables=["context","question"], template=prompt_template)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    answer = qa_chain.run(query)

    # Prepare table rows for Supabase
    rows = []
    # We'll collect top-k docs and their similarity scores via retriever
    docs_retrieved = retriever.get_relevant_documents(query)
    for d in docs_retrieved:
        rows.append({
            "title": d.metadata.get("title"),
            "url": d.metadata.get("source"),
            "snippet": d.page_content[:500],
            "query": query
        })
    df = pd.DataFrame(rows)
    print(df.head())

    # upload
    upload_dataframe(df)
    return {"answer": str(answer), "table": df}

if __name__ == "__main__":
    q = "ai agents web scraping use cases and best practices"
    run_pipeline(q)
