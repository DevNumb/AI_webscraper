# src/scraper.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re

HEADERS = {
    "User-Agent": "ai-websearch-agent/0.1 (+https://example.com/contact)"
}

def fetch_html(url, timeout=12):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print("Fetch error:", e)
        return None

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts/styles
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    title = soup.title.string.strip() if soup.title else ""
    paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    text = "\n\n".join([p for p in paragraphs if p])
    return title, text

def chunk_text(text, chunk_size=800, overlap=100):
    # chunk size is characters; for production use token-based chunking
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        end = i + chunk_size
        chunk = text[i:end]
        chunks.append(chunk)
        i = max(end - overlap, end)  # overlap
    return chunks

def find_links_from_text(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#") or href.lower().startswith("mailto:"):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        # only http(s)
        if parsed.scheme in ("http", "https"):
            links.add(full)
    return list(links)
