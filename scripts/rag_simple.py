import os
from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

pdf_file = Path("data/sample.pdf")
loader = PyPDFLoader(str(pdf_file))
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks")

chunks = [doc.page_content for doc in docs]

def ask_gemini(question):
    context = "\n\n".join(chunks[:3])
    prompt = f"Answer the question based on this context:\n{context}\n\nQuestion: {question}"
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return response.text

question = input("Ask me something about the PDF: ")
answer = ask_gemini(question)
print("\nAnswer:", answer)
chunks = [doc.page_content for doc in docs]


def ask_gemini(question):
    context = "\n\n".join(chunks[:3])
    prompt = f"Answer the question based on this context:\n{context}\n\nQuestion: {question}"
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return response.text


question = input("Ask me something about the PDF: ")
answer = ask_gemini(question)
print("\nAnswer:", answer)
