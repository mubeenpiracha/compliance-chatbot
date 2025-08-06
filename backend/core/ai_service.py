# backend/core/ai_service.py
import os
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import VectorStoreQuery, MetadataFilters, ExactMatchFilter
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

from backend.core.config import PINECONE_API_KEY, OPENAI_API_KEY

# --- Global Setup ---
agent_executor = None
index = None # Keep index global to create tools dynamically

def initialize_ai_service():
    global index
    # ... (Initialization code for Settings and Pinecone connection is the same) ...
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("compliance-bot-index")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store)
    print("AI Service Initialized (Index loaded).")

def get_ai_response(user_message: str, history: list, jurisdiction: str) -> str:
    if index is None:
        return "Error: AI Service Index is not initialized."

    print(f"Processing message for jurisdiction: {jurisdiction}")

    # --- 1. Create a jurisdiction-specific RAG tool ---
    query_engine_kwargs = {}
    if jurisdiction != "Both":
        query_engine_kwargs["vector_store_query_mode"] = "default"
        query_engine_kwargs["filters"] = MetadataFilters(
            filters=[ExactMatchFilter(key="jurisdiction", value=jurisdiction)]
        )

    query_engine = index.as_query_engine(**query_engine_kwargs)

    rag_tool = Tool(
        name=f"regulatory_document_retriever_{jurisdiction}",
        func=lambda q: query_engine.query(q).response,
        description=(
            f"Searches and retrieves information from the {jurisdiction} regulatory rulebooks. "
            "Use this for any questions about its financial regulations, funds, licensing, etc."
        ),
        coroutine=query_engine.aquery, # for async
    )
    tools = [rag_tool]

    # --- 2. Set up Conversation Memory ---
    memory = ConversationBufferWindowMemory(
        k=10, # Remember the last 10 messages
        memory_key="chat_history",
        input_key="input",
        output_key="output",
        return_messages=True
    )
    # Load history from the request
    for msg in history:
        if msg['sender'] == 'user':
            memory.chat_memory.add_user_message(msg['text'])
        else:
            memory.chat_memory.add_ai_message(msg['text'])

    # --- 3. Create and run the agent ---
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

    prompt_template = """
    You are an AI Regulatory Assistant. Your primary function is to provide accurate, factual information based on the documents provided to you.

    You have access to the following tool:
    {tools}

    To answer the user's question, you MUST follow these steps:
    1. Analyze the user's input and the conversation history to understand the core question.
    2. You MUST first use the `regulatory_document_retriever` tool to find relevant information from the official rulebooks.
    3. Analyze the information returned by the tool.
    4. If the tool provides a relevant answer, synthesize it into a clear, factual response, citing your sources.
    5. If the tool returns no relevant information, you MUST first state: "The provided documents do not contain specific information on this topic." Then, you may provide a general, tentative answer based on your broader knowledge. When you do this, you MUST preface your answer with the heading "General Analysis:" to clearly distinguish it from fact-based, retrieved information.

    Use the following format for your thought process:

    Question: the user's input question
    Thought: You should always think about what to do. Your primary thought should be to use the retriever tool.
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer.
    Final Answer: the final answer to the original input question.

    Begin!

    Conversation History:
    {chat_history}

    Question: {input}
    Thought:{agent_scratchpad}
    """
    prompt = PromptTemplate.from_template(prompt_template)

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, memory=memory, verbose=True
    )

     # Add handle_parsing_errors=True to make the agent more resilient
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory, 
        verbose=True,
        handle_parsing_errors=True # This will help the agent self-correct
    )

    try:
        response = agent_executor.invoke({"input": user_message})
        return response.get("output", "I'm sorry, I couldn't process that request.")
    except Exception as e:
        print(f"An error occurred during agent execution: {e}")
        return "Sorry, I encountered an error while processing your request."

initialize_ai_service()