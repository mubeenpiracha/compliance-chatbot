# backend/main.py

# 1. Import FastAPI
from fastapi import FastAPI

# 2. Create an instance of the FastAPI class
# This instance will be the main point of interaction for creating all your API.
app = FastAPI()

# 3. Define a path operation decorator
# @app is the decorator, .get() specifies the HTTP method,
# and "/" is the URL path.
# adding another comment to see what is changed
@app.get("/")
def read_root():
    """
    This is the root endpoint for the API.
    When a GET request is made to "/", this function is called.
    """
    # 4. Return a JSON response
    return {"message": "Hello, Compliance Officer!"}

