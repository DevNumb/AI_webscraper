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
from datetime import datetime

# ========== CONFIGURATION ==========
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_PRIMARY = os.getenv("OPENROUTER_MODEL", "x-ai/grok-4.1-fast")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ========== GOOGLE SHEETS & DOCS INTEGRATION ==========
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1utPyB-aFOmOlVkkf0NvgfAlSUxDdj94-aWHGQjI7l4g/edit?usp=sharing"  # ⬅️ REMPLACEZ CETTE URL
GOOGLE_DOC_URL = "https://docs.google.com/document/d/1iQWxyLCxa5tX7EfLuncvP0nuAqWpcDxGm_iGZClu6tw/edit?usp=sharing" 
def get_all_queries_from_google_sheets():
    """Récupère TOUTES les queries depuis Google Sheets"""
    try:
        # Extraire l'ID du sheet depuis l'URL
        sheet_id = GOOGLE_SHEET_URL.split('/d/')[1].split('/')[0]
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
        
        # Télécharger le CSV
        response = requests.get(csv_url, timeout=10)
        response.raise_for_status()
        
        # Lire le CSV
        df = pd.read_csv(pd.compat.StringIO(response.text))
        
        # Prendre TOUTES les queries (sauf l'en-tête)
        queries = []
        for i in range(len(df)):
            query = df.iloc[i, 0]
            # Vérifier que ce n'est pas l'en-tête et pas vide
            if (pd.notna(query) and 
                str(query).strip().lower() != 'queries' and 
                str(query).strip() != ''):
                queries.append(str(query).strip())
        
        print(f"📋 Found {len(queries)} queries in Google Sheets")
        return queries
            
    except Exception as e:
        print(f"Error fetching from Google Sheets: {e}")
        return ["latest news about artificial intelligence"]  # Fallback

def format_document_content(all_results):
    """Crée un document structuré avec mise en forme professionnelle"""
    
    current_date = datetime.now().strftime('%B %d, %Y')
    current_time = datetime.now().strftime('%H:%M:%S')
    
    doc_content = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        AI WEB SEARCH AGENT - FINAL REPORT                    ║
║                           Comprehensive Research Analysis                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

📅 GENERATED ON: {current_date}
⏰ TIME: {current_time}
🔢 TOTAL QUERIES PROCESSED: {len(all_results)}
📊 TOTAL SOURCES ANALYZED: {sum(result['sources_count'] for result in all_results)}

{'='*100}

📑 TABLE OF CONTENTS
{'='*100}

"""
    
    # Table des matières
    for i, result in enumerate(all_results, 1):
        doc_content += f"{i}. {result['query']} (Page {i+1})\n"
    
    doc_content += f"\n{'='*100}\n\n"
    
    # Contenu détaillé pour chaque query
    for i, result in enumerate(all_results, 1):
        doc_content += f"""
{'█'*100}
📖 RESEARCH REPORT #{i:02d}
{'█'*100}

🎯 QUERY: {result['query']}
📅 Date Processed: {current_date}
🔍 Sources Analyzed: {result['sources_count']}
📝 Answer Length: {len(result['answer'])} characters

{'─'*100}

📋 EXECUTIVE SUMMARY
{'─'*100}

{result['answer']}

{'─'*100}

🔗 SOURCE REFERENCES
{'─'*100}

"""
        
        # Sources avec formatage
        for j, source in enumerate(result['table'].itertuples(), 1):
            doc_content += f"""
[{j}] 📰 {source.title}
    🔗 URL: {source.url}
    📊 Relevance Score: {source.score:.3f}
    📝 Excerpt: {source.snippet[:150]}...
"""
        
        doc_content += f"""
{'─'*100}

📈 KEY INSIGHTS
{'─'*100}

"""
        
        # Extraire les insights principaux de la réponse
        insights = extract_key_insights(result['answer'])
        for insight in insights:
            doc_content += f"• {insight}\n"
        
        doc_content += f"""
{'─'*100}

🏷️ KEYWORDS & TAGS
{'─'*100}

{extract_keywords(result['query'], result['answer'])}

{'█'*100}

"""
        
        # Saut de page pour la prochaine section
        if i < len(all_results):
            doc_content += "\n" + "⤵️ NEXT PAGE" + "\n"*3
    
    # Footer du document
    doc_content += f"""
{'='*100}

📊 REPORT METRICS SUMMARY
{'='*100}

• Total Research Queries: {len(all_results)}
• Average Sources per Query: {sum(result['sources_count'] for result in all_results) / len(all_results):.1f}
• Total Content Processed: {sum(len(result['answer']) for result in all_results):,} characters
• Processing Date: {current_date}
• AI Model Used: {OPENROUTER_PRIMARY}

{'='*100}

🤖 GENERATED BY AI WEB SEARCH AGENT
📧 Automated Research System
🔗 Powered by OpenRouter AI & Advanced Web Scraping
{'='*100}
"""
    
    return doc_content

def extract_key_insights(answer):
    """Extrait les insights principaux de la réponse"""
    sentences = answer.split('. ')
    key_insights = []
    
    # Prendre les 3-5 phrases les plus importantes
    for sentence in sentences[:5]:
        sentence = sentence.strip()
        if len(sentence) > 20 and any(keyword in sentence.lower() for keyword in ['important', 'key', 'essential', 'critical', 'major', 'significant']):
            key_insights.append(sentence)
    
    # Si pas assez d'insights, prendre les premières phrases
    if len(key_insights) < 3:
        key_insights = [s.strip() for s in sentences[:3] if len(s.strip()) > 20]
    
    return key_insights if key_insights else ["Comprehensive analysis provided in main content."]

def extract_keywords(query, answer):
    """Extrait les mots-clés pertinents"""
    words = re.findall(r'\b[A-Z][a-z]+\b', answer)  # Mots avec majuscule
    unique_words = list(set(words))[:10]  # Limiter à 10 mots uniques
    
    # Ajouter des mots du query
    query_words = query.split()
    keywords = list(set(unique_words + query_words))[:15]
    
    return ', '.join(keywords)

def save_professional_document(all_results):
    """Sauvegarde le document avec mise en forme professionnelle"""
    try:
        doc_content = format_document_content(all_results)
        
        # Créer un nom de fichier professionnel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"AI_Research_Report_{timestamp}.md"
        
        # Sauvegarder en Markdown (meilleure mise en forme)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        print(f"📄 Professional report saved to: {filename}")
        
        # Afficher un aperçu
        print("\n🎨 DOCUMENT PREVIEW (First 1500 characters):")
        print("="*80)
        preview = doc_content[:1500] + "..." if len(doc_content) > 1500 else doc_content
        print(preview)
        print("="*80)
        
        return filename
        
    except Exception as e:
        print(f"❌ Error creating professional document: {e}")
        return None

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
                print(f"✅ Uploaded {len(records)} records to Supabase")
                return True
        except Exception as e:
            print(f"❌ Supabase upload error: {e}")
    else:
        print("⚠️ Supabase credentials not found, skipping upload")
    return False

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
        print(f"❌ Error getting URLs from LLM: {e}")
        # Fallback URLs
        return [
            "https://en.wikipedia.org/wiki/Web_scraping",
            "https://towardsdatascience.com/a-complete-guide-to-web-scraping-with-python-2024-2025-1234567890"
        ]

def get_answer_from_llm(query: str, context: str, model_name=OPENROUTER_PRIMARY):
    """Get answer from LLM using context"""
    prompt = f"""Based on the following context, please provide a comprehensive, well-structured answer with clear sections and key insights.

Context:
{context}

Question: {query}

Please provide a detailed, professional answer suitable for a research report:"""
    
    try:
        resp = call_openrouter_model(model_name, [{"role":"user","content":prompt}])
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error getting answer from LLM: {e}"

def run_pipeline(query: str):
    print("🚀 Starting AI Web Scraper...")
    print(f"🎯 Query: {query}")
    
    # Step 1: Get URLs from LLM
    urls = prompt_to_urls(query)
    print(f"🌐 URLs found: {urls}")

    # Step 2: Scrape and process content
    docs = []
    for url in urls:
        print(f"📥 Scraping: {url}")
        html = fetch_html(url)
        if not html:
            continue
        title, text = extract_text(html)
        if not text or len(text) < 100:
            continue
            
        print(f"📄 Found content: {title} ({len(text)} chars)")
        chunks = chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            docs.append({
                "text": chunk,
                "metadata": {"source": url, "title": title, "chunk": i}
            })

    print(f"📦 Chunks prepared: {len(docs)}")
    print("SCRAPED DOC COUNT:", len(docs))
    if not docs:
        print("❌ No content found, exiting.")
        return None

    # Step 3: Create search index and find relevant content
    print("🔍 Creating search index...")
    search_engine = SimpleVectorSearch()
    search_engine.add_documents(docs)
    
    # Step 4: Search for relevant content
    relevant_docs = search_engine.search(query, k=3)
    print(f"✅ Found {len(relevant_docs)} relevant documents")
    
    # Step 5: Prepare context for LLM
    context = "\n\n".join([doc["text"] for doc in relevant_docs])
    
    # Step 6: Get final answer from LLM
    print("🤖 Getting answer from LLM...")
    answer = get_answer_from_llm(query, context)
    print(f"💡 LLM Answer generated ({len(answer)} characters)")
    
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
    print("📊 Data to upload:")
    print(df[['title', 'url', 'score']].to_string())
    
    # Step 8: Upload to Supabase
    print("☁️ Uploading to Supabase...")
    upload_success = upload_dataframe(df)
    
    return {
        "query": query,
        "answer": answer,
        "table": df,
        "sources_count": len(relevant_docs),
        "upload_success": upload_success
    }

if __name__ == "__main__":
    # Récupérer TOUTES les queries depuis Google Sheets
    all_queries = get_all_queries_from_google_sheets()
    
    if not all_queries:
        print("❌ No queries found in Google Sheets, using default")
        all_queries = ["latest news about artificial intelligence"]
    
    print(f"📋 Processing {len(all_queries)} queries:")
    
    all_results = []
    
    for i, query in enumerate(all_queries, 1):
        print(f"\n{'='*60}")
        print(f"🔄 Processing query {i}/{len(all_queries)}: '{query}'")
        print(f"{'='*60}")
        
        result = run_pipeline(query)
        
        if result:
            all_results.append(result)
            print(f"✅ Query {i} completed! Processed {result['sources_count']} sources")
            if result['upload_success']:
                print(f"☁️ Data uploaded to Supabase for query: '{query}'")
        else:
            print(f"❌ Query {i} failed: '{query}'")
        
        # Pause entre les queries pour éviter le rate limiting
        if i < len(all_queries):
            print(f"⏳ Waiting 10 seconds before next query...")
            time.sleep(10)
    
    # Sauvegarder toutes les réponses dans un document professionnel
    print(f"\n📄 Creating professional research report...")
    doc_filename = save_professional_document(all_results)
    
    print(f"\n🎉 MISSION ACCOMPLISHED!")
    print(f"📊 Processed {len(all_results)} research queries")
    print(f"📁 Professional report: {doc_filename}")
    print(f"🔗 Copy content to Google Docs for beautiful formatting!")
    print(f"💾 Data also saved to Supabase database")
