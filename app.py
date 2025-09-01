import streamlit as st
import requests
import time
from qdrant_client import QdrantClient
from qdrant_client.models import NamedVector
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
import os

from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

GEMINI_EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={GEMINI_API_KEY}"
GEMINI_CHAT_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"

st.set_page_config(page_title="RAG Web Scraper Chatbot", layout="wide")
st.title("🤖 Enhanced RAG Web Scraper Chatbot")
st.write("Now powered by LangChain! Add URLs to scrape content and ask questions about them!")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
COLLECTION_NAME = "rag_docs"

@st.cache_resource
def setup_langchain_components():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3
        )
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            vector_name="embedding"
        )
        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant that answers questions ONLY based on the provided context.

RULES:
1. If the context contains relevant information, provide a detailed answer
2. If the context does NOT contain enough information, respond with: "I don't have enough information in the provided sources to answer that question."
3. Always mention the source when possible
4. Do not use any external knowledge outside the provided context

Context:
{context}

Question: {question}

Answer:"""
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=True
        )
        return embeddings, vector_store, qa_chain, llm
    except Exception as e:
        st.error(f"Error setting up LangChain: {str(e)}")
        return None, None, None, None

embeddings, vector_store, qa_chain, llm = setup_langchain_components()

def safe_post(url, headers, json, retries=5):
    for i in range(retries):
        try:
            response = requests.post(url, headers=headers, json=json)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and i < retries - 1:
                wait = (2 ** i) + 1
                st.warning(f"Rate limit hit. Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                st.error(f"API Error after {retries} attempts: {str(e)}")
                raise

def scrape_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "advertisement"]):
            element.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r'\s+', ' ', text).strip()
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No Title"
        return text, title_text
    except Exception as e:
        st.error(f"Error scraping {url}: {str(e)}")
        return None, None

def split_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= 50:
            chunks.append(chunk)
    return chunks

def get_gemini_embedding(text):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "models/embedding-001",
        "content": {
            "parts": [{"text": text}]
        }
    }
    response = safe_post(GEMINI_EMBED_URL, headers=headers, json=data)
    return response.json()["embedding"]["values"]

st.sidebar.header("📝 Add Web Content")

new_url = st.sidebar.text_input("Enter URL to scrape:")
if st.sidebar.button("🔍 Scrape & Add URL"):
    if new_url:
        with st.sidebar:
            with st.spinner("Scraping content..."):
                scraped_text, title = scrape_text_from_url(new_url)
            if scraped_text:
                with st.spinner("Processing chunks..."):
                    chunks = split_into_chunks(scraped_text)
                    st.write(f"Created {len(chunks)} chunks from {title}")
                with st.spinner("Adding to vector store with LangChain..."):
                    try:
                        documents = []
                        for i, chunk in enumerate(chunks):
                            doc = Document(
                                page_content=chunk,
                                metadata={
                                    "source_url": new_url,
                                    "title": title,
                                    "chunk_index": i
                                }
                            )
                            documents.append(doc)
                        if vector_store:
                            vector_store.add_documents(documents)
                            st.success(f"✅ Successfully added {len(documents)} chunks from {title}")
                        else:
                            points = []
                            for i, chunk in enumerate(chunks):
                                try:
                                    embedding = get_gemini_embedding(chunk)
                                    point_id = hash(f"{new_url}_{i}") % (10**10)
                                    point = {
                                        "id": point_id,
                                        "vector": {"embedding": embedding},
                                        "payload": {
                                            "text": chunk,
                                            "source_url": new_url,
                                            "title": title,
                                            "chunk_index": i
                                        }
                                    }
                                    points.append(point)
                                    time.sleep(0.5)
                                except Exception as e:
                                    st.error(f"Error processing chunk {i}: {str(e)}")
                                    break
                            if points:
                                client.upsert(collection_name=COLLECTION_NAME, points=points)
                                st.success(f"✅ Successfully added {len(points)} chunks from {title}")
                    except Exception as e:
                        st.error(f"Error storing in database: {str(e)}")
    else:
        st.sidebar.error("Please enter a URL")

st.sidebar.header("📚 Stored Content")
try:
    collection_info = client.get_collection(COLLECTION_NAME)
    st.sidebar.write(f"Total chunks: {collection_info.points_count}")
    if st.sidebar.button("🔄 Refresh Stats"):
        st.sidebar.write("Collection updated!")
except Exception as e:
    st.sidebar.write("No content stored yet")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.header("💬 Ask Questions (Now with LangChain!)")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the scraped content:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                if qa_chain:
                    result = qa_chain.invoke({"query": prompt})
                    answer = result["result"]
                    source_docs = result["source_documents"]
                    st.markdown(answer)
                    if source_docs:
                        with st.expander("📖 Sources"):
                            for i, doc in enumerate(source_docs):
                                source_url = doc.metadata.get('source_url', 'Unknown')
                                title = doc.metadata.get('title', 'Unknown')
                                st.write(f"**Source {i+1}:** {title}")
                                st.write(f"**URL:** {source_url}")
                                st.write(f"**Content:** {doc.page_content[:200]}...")
                                st.write("---")
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.warning("Using fallback method...")
                    query_vector = get_gemini_embedding(prompt)
                    search_vector = NamedVector(name="embedding", vector=query_vector)
                    results = client.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=search_vector,
                        limit=3
                    )
                    if not results:
                        fallback_answer = "No relevant content found. Try adding more URLs or asking different questions."
                    else:
                        context_items = []
                        for item in results:
                            source_url = item.payload.get('source_url', 'Unknown')
                            title = item.payload.get('title', 'Unknown')
                            text = item.payload.get('text', '')
                            context_items.append(f"Source: {title} ({source_url})\nContent: {text}")
                        full_context = "\n\n---\n\n".join(context_items)
                        fallback_answer = f"Based on the retrieved content:\n\n{full_context}"
                    st.markdown(fallback_answer)
                    st.session_state.messages.append({"role": "assistant", "content": fallback_answer})
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

st.markdown("---")
st.markdown("**✨ Enhanced Features:** LangChain integration, conversation memory, better source handling!")
st.markdown("**How to use:** Add URLs in the sidebar, then ask questions about the scraped content!")