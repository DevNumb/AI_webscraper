# src/main.py
import os
import re
import pandas as pd
from llm_client import OpenRouterLLM, call_openrouter_model
from scraper import fetch_html, extract_text, chunk_text, find_links_from_text
from embeddings import get_embeddings_model
from index_store import build_faiss_index, load_index
from uploader import upload_dataframe
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from tqdm.auto import tqdm

OPENROUTER_PRIMARY = os.getenv("OPENROUTER_MODEL", "deepseek-r1:free")
OPENROUTER_FALLBACK = os.getenv("OPENROUTER_FALLBACK", "nvidia/nemotron-nano-12b-v2-vl:free")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

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
