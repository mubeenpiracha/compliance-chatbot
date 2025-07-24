# backend/main.py
from fastapi import FastAPI
from backend.api.v1.api import api_router

# This function will be called on application startup
# Create the FastAPI app instance

app = FastAPI(title="Compliance Officer AI")

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Hello, Compliance Officer! The database is connected."}