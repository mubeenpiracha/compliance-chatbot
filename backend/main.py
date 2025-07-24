# backend/main.py
from fastapi import FastAPI
from backend.db.session import engine
from backend.db.base import Base

# This function will be called on application startup
def create_tables():
    print("Creating tables...")
    # This line is not strictly necessary with Alembic, but good for verification
    # It ensures tables are created if they don't exist, but Alembic is the preferred way.
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully (if they didn't exist).")

# Create the FastAPI app instance
app = FastAPI(title="Compliance Officer AI")

# This is an event handler that runs when the application starts
@app.on_event("startup")
def on_startup():
    print("Application is starting up...")
    # A simple way to test the database connection
    try:
        # Try to connect to the database
        connection = engine.connect()
        connection.close()
        print("Database connection successful.")
    except Exception as e:
        print(f"Database connection failed: {e}")
    # We call our table creation function here.
    # create_tables() # We can comment this out as Alembic now handles table creation.

@app.get("/")
def read_root():
    return {"message": "Hello, Compliance Officer! The database is connected."}