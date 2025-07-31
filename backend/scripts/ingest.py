# backend/scripts/ingest.py
import os
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from backend.core.config import PINECONE_API_KEY, OPENAI_API_KEY

# Set OpenAI API key for LlamaIndex components
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def run_ingestion():
    """
    Connects to Pinecone, loads documents, creates an index, and upserts the data.
    """
    index_name = "compliance-bot-index"

    print("Connecting to Pinecone...")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY must be set.")

    # Initialize the Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    print(f"Checking if index '{index_name}' exists...")
    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' does not exist. Creating new index...")
        # Create a new index with the serverless spec
        pc.create_index(
            name=index_name,
            dimension=1536,  # Standard dimension for OpenAI's text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1' # Specify your region
            )
        )
        print("Index created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")

    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    print("Loading documents from './backend/data'...")
    reader = SimpleDirectoryReader("./backend/data")
    documents = reader.load_data()
    print(f"Loaded {len(documents)} document(s).")

    print("Configuring LlamaIndex settings...")
    # Configure the global settings for LlamaIndex
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    print("Creating vector store index and upserting vectors to Pinecone...")
    # This step now uses the global settings for parsing and embedding
    index = VectorStoreIndex.from_documents(
        documents, vector_store=vector_store, show_progress=True
    )
    print("Ingestion complete!")

if __name__ == "__main__":
    run_ingestion()
