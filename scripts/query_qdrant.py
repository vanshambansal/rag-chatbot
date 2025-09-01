from qdrant_client import QdrantClient
import requests
import os

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "rag_docs"
VECTOR_NAME = "embedding"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={GEMINI_API_KEY}"
GEMINI_CHAT_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def get_gemini_embedding(text):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text}]}
    }
    response = requests.post(GEMINI_EMBED_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["embedding"]["values"]

def ask_gemini(context, question):
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {
                "parts": [
                    {"text": f"Answer the question based on this context:\n\n{context}\n\nQuestion: {question}"}
                ]
            }
        ]
    }
    response = requests.post(GEMINI_CHAT_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]

def main():
    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        question_embedding = get_gemini_embedding(question)

        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=(VECTOR_NAME, question_embedding),
            limit=3
        )

        context = "\n".join([hit.payload["text"] for hit in search_results])

        answer = ask_gemini(context, question)
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()
