from qdrant_client import QdrantClient
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import os


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "rag_docs"
VECTOR_NAME = "embedding"
VECTOR_SIZE = 768

urls = [
    "https://react.dev/",
]

def scrape_text(url):
    """Scrape main text from a webpage using requests and BeautifulSoup."""
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")

    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_into_chunks(text, min_words=200, max_words=300):
    """Split text into chunks of 200-300 words."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_words]
        if len(chunk) < min_words:
            break
        chunks.append(" ".join(chunk))
        i += max_words
    return chunks

all_chunks = []
all_metadatas = []

for url in urls:
    print(scrape_text(url)[:500])\

for url in urls:
    text = scrape_text(url)
    chunks = split_into_chunks(text)
    for chunk in chunks:
        all_chunks.append(chunk)
        all_metadatas.append({"source_url": url})



GEMINI_API_KEY =os.getenv("GEMINI_API_KEY")  
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key=" + GEMINI_API_KEY

def get_gemini_embedding(text):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text}]}
    }
    response = requests.post(GEMINI_EMBED_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["embedding"]["values"]

vectors = [get_gemini_embedding(chunk) for chunk in all_chunks]



client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

existing_collections = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in existing_collections:
    print(f"Collection '{COLLECTION_NAME}' exists. Deleting it to reset vector name...")
    client.delete_collection(collection_name=COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={VECTOR_NAME: {"size": VECTOR_SIZE, "distance": "Cosine"}},
)

points = [
    {
        "id": idx,
        "vector": {VECTOR_NAME: vectors[idx]},
        "payload": all_metadatas[idx] | {"text": all_chunks[idx]},
    }
    for idx in range(len(all_chunks))
]
if points:
    client.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"Uploaded {len(all_chunks)} chunks to Qdrant collection '{COLLECTION_NAME}'")
