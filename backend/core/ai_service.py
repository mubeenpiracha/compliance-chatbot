# backend/core/ai_service.py
from langchain_openai import ChatOpenAI
from backend.core.config import OPENAI_API_KEY

def get_ai_response(user_message: str) -> str:
    """
    Gets a response from the OpenAI API.
    This is a simple implementation without RAG for now.
    """
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY is not configured."

    try:
        # Initialize the ChatOpenAI model.
        # We can specify the model name, temperature, etc.
        llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.7)

        # For now, we'll create a simple prompt.
        # In the future, this will be a more complex RAG prompt.
        prompt = f"""
        You are a helpful assistant. Answer the following user question.
        User Question: {user_message}
        """

        # Invoke the model with the prompt
        response = llm.invoke(prompt)

        # The response object has a 'content' attribute with the text
        return response.content

    except Exception as e:
        # Handle potential exceptions from the API call
        print(f"An error occurred while calling the OpenAI API: {e}")
        return "Sorry, I'm having trouble connecting to the AI service right now."