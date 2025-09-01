import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Configure Gemini
genai.configure(api_key=api_key)

# Pick a model
model = genai.GenerativeModel("gemini-1.5-flash")

# Test prompt
response = model.generate_content("Hello Gemini! Say hi in one sentence.")
print("Gemini response:", response.text)
