# backend/scripts/ingest.py
import os
import sys
import time
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.readers.file import UnstructuredReader
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
    pinecone_dimension = 1536 # Explicitly define dimension

    print("Connecting to Pinecone...")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY must be set.")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    print(f"Checking if index '{index_name}' exists...")
    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' does not exist. Creating new index...")
        pc.create_index(
            name=index_name,
            dimension=pinecone_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print("Index created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")

    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # Use UnstructuredReader for parsing PDFs, which is better for tables and images
    unstructured_reader = UnstructuredReader()
    file_extractor = {".pdf": unstructured_reader}

    print("Loading documents from DIFC and ADGM directories...")
    difc_docs = SimpleDirectoryReader(
        "./backend/data/difc", file_extractor=file_extractor
    ).load_data()
    adgm_docs = SimpleDirectoryReader(
        "./backend/data/adgm", file_extractor=file_extractor
    ).load_data()

    # Add jurisdiction metadata to each document
    for doc in difc_docs:
        doc.metadata["jurisdiction"] = "DIFC"
    for doc in adgm_docs:
        doc.metadata["jurisdiction"] = "ADGM"

    # Combine documents from both jurisdictions
    documents = difc_docs + adgm_docs

    # Add a document ID to each document from its hash
    for doc in documents:
        doc.id_ = doc.hash
        doc.metadata["document_id"] = doc.hash

    if not documents:
        print("No documents found in the specified directories. Exiting.")
        return
    
    print(f"Loaded {len(documents)} documents from DIFC and ADGM directories.")


    print("Configuring LlamaIndex settings...")
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-large", 
        api_key=OPENAI_API_KEY,
        dimensions=pinecone_dimension 
    )

    print("Creating storage context and upserting vectors to Pinecone...")
    try:
        # This is a more robust way to handle the indexing process
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=True
        )
        print("Ingestion complete!")

        # Verification step
        stats = pinecone_index.describe_index_stats()
        print(f"Pinecone index stats: {stats}")
        if stats.total_vector_count > 0:
            print("Successfully verified that vectors were added to the index.")
        else:
            print("Warning: Ingestion complete but no vectors found in the index. Check for issues.")
    except Exception as e:
        print(f"An error occurred during indexing: {e}")


if __name__ == "__main__":
    run_ingestion()