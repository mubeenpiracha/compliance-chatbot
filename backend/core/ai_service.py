# backend/core/ai_service.py
import os
import os
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import VectorStoreQuery, MetadataFilters, ExactMatchFilter
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

from backend.core.graph_agent import agent, AgentState
from langchain_core.messages import HumanMessage, AIMessage

from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import VectorStoreQuery, MetadataFilters, ExactMatchFilter
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from backend.core.config import PINECONE_API_KEY, OPENAI_API_KEY


# --- Global Setup ---
# The new agent is imported from graph_agent.py
# We still need the index for the retrieval step.
index = None 

def initialize_ai_service():
    """
    Initializes the connection to Pinecone and loads the vector store index.
    This function is called once when the application starts.
    """
    global index
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("compliance-bot-index")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store)
    print("AI Service Initialized (Index loaded and graph agent ready).")

def get_ai_response(user_message: str, history: list, jurisdiction: str) -> str:
    """
    Gets a response from the AI agent.

    This function now invokes the new LangGraph-based agent. It prepares the
    initial state and then calls the compiled graph.
    """
    if index is None:
        return "Error: AI Service Index is not initialized."

    print(f"--- Invoking Graph Agent for jurisdiction: {jurisdiction} ---")

    # 1. Convert the incoming history (a list of dicts) to a list of BaseMessages
    chat_history = []
    for msg in history:
        if msg['sender'] == 'user':
            chat_history.append(HumanMessage(content=msg['text']))
        else:
            chat_history.append(AIMessage(content=msg['text']))

    # 2. Prepare the initial state for the graph
    initial_state: AgentState = {
        "original_query": user_message,
        "jurisdiction": jurisdiction,
        "chat_history": chat_history,
        # The rest of the fields will be populated by the graph nodes
        "contextualized_query": "",
        "clarity_grade": "",
        "sub_questions": [],
        "retrieved_docs": [],
        "final_response": "",
        "route": "",
        "clarification_question": ""
    }

    # 3. Invoke the agent graph
    # The `agent.stream()` method can be used for streaming, but for now,
    # we'll use `invoke` to get the final state.
    try:
        final_state = agent.invoke(initial_state)

        # 4. Determine the response based on the final state
        if final_state.get("clarification_question"):
            return final_state["clarification_question"]
        elif final_state.get("final_response"):
            return final_state["final_response"]
        else:
            return "I'm sorry, I encountered an issue and couldn't process your request."

    except Exception as e:
        print(f"An error occurred during graph execution: {e}")
        # Provide a more detailed error for debugging if possible
        import traceback
        traceback.print_exc()
        return "Sorry, I encountered a critical error while processing your request."

initialize_ai_service()