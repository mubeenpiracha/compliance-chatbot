# backend/core/ai_service.py
import os
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI # Alias to avoid name conflict
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# LangChain imports for building the agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_openai import ChatOpenAI

from backend.core.config import PINECONE_API_KEY, OPENAI_API_KEY

# --- Global Setup ---
agent_executor = None

def initialize_ai_service():
    """
    Initializes the AI service by creating a RAG tool and building a ReAct agent.
    """
    global agent_executor

    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        print("Warning: API keys for Pinecone or OpenAI are not set.")
        return

    print("Initializing Agentic AI Service...")

    # --- 1. Configure LlamaIndex Components ---
    Settings.llm = LlamaOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.2)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY, dimensions=1536)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "compliance-bot-index"
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store)

    # --- 2. Create the RAG Query Engine (as before) ---
    query_engine = index.as_query_engine(similarity_top_k=5)

    # --- 3. Define the RAG Pipeline as a Tool ---
    rag_tool_llama = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="regulatory_document_retriever",
            description=(
                "Searches and retrieves information from the official DIFC and ADGM "
                "regulatory rulebooks. Use this tool for any questions about financial regulations, "
                "fund structures, licensing, or compliance obligations."
            ),
        ),
    )

    # --- THIS IS THE FIX ---
    # Convert the LlamaIndex tool into a LangChain compatible tool
    rag_tool_langchain = rag_tool_llama.to_langchain_tool()
    tools = [rag_tool_langchain]
    # --- END FIX ---

    # --- 4. Create the LangChain Agent ---
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("Agentic AI Service Initialized Successfully.")

def get_ai_response(user_message: str) -> str:
    """
    Gets a response from the agent executor.
    """
    if agent_executor is None:
        return "Error: AI Service is not initialized. Please check API keys."

    print(f"Agent processing message: {user_message}")
    try:
        response = agent_executor.invoke({"input": user_message})
        return response.get("output", "I'm sorry, I couldn't process that request.")
    except Exception as e:
        print(f"An error occurred during agent execution: {e}")
        return "Sorry, I encountered an error while processing your request."

# --- Application Startup ---
initialize_ai_service()