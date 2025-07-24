# backend/core/config.py
import os
from dotenv import load_dotenv

# This line finds the .env file in the project root and loads its variables.
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
