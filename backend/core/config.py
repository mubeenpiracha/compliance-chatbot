# backend/core/config.py
import os
from dotenv import load_dotenv

# Load environment variables from a .env file (for local development)
load_dotenv()

# Database connection URL
# Format: postgresql://<user>:<password>@<host>:<port>/<dbname>
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/compliance_db")

# You can add other configurations here later
# For example: API keys, secret keys, etc.
# SECRET_KEY = os.getenv("SECRET_KEY")
