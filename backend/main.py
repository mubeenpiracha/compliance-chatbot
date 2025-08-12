# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Import the CORS middleware
from backend.api.v1.api import api_router

# This function will be called on application startup
# Create the FastAPI app instance

app = FastAPI(title="Compliance Officer AI")
# --- Add CORS Middleware ---
# This is the list of origins that are allowed to make cross-origin requests.
# For development, we allow our React app's default server.
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)
# --- End CORS Middleware ---

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Compliance Officer AI API"}