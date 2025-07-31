# backend/core/config.py
import os
from dotenv import load_dotenv

# This line finds the .env file in the project root and loads its variables.
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")