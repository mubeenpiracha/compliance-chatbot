# Test ingestion script for a single file
import sys
sys.path.append('/home/mubeen/compliance-chatbot')
sys.path.append('/home/mubeen/compliance-chatbot/backend')

from backend.scripts.ingest import *
from pathlib import Path

def test_single_file():
    # Test with just one file
    test_file = Path("./backend/data/adgm/BENEFICIAL OWNERSHIP.pdf")
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
    
    print(f"Testing ingestion with: {test_file}")
    
    # Initialize components
    reader = UnstructuredReader()
    embedder = OpenAIEmbedding(
        model=EMBED_MODEL_NAME,
        api_key=OPENAI_API_KEY,
        dimensions=EMBED_DIM,
    )
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc, INDEX_NAME, EMBED_DIM)
    index = pc.Index(INDEX_NAME)
    
    try:
        # Load document
        print("Loading document...")
        docs = reader.load_data(file=str(test_file))
        print(f"Loaded {len(docs)} document(s)")
        
        if not docs:
            print("No documents loaded!")
            return
            
        doc = docs[0]  # Take first document
        doc.metadata.update({
            "file_name": test_file.name,
            "source_path": str(test_file),
            "jurisdiction": "ADGM",
        })
        
        # Chunk document
        print("Chunking document...")
        chunks = chunk_document(doc)
        print(f"Created {len(chunks)} chunks")
        
        if not chunks:
            print("No chunks created!")
            return
            
        # Take just first few chunks for testing
        test_chunks = chunks[:3]
        print(f"Testing with {len(test_chunks)} chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        texts = [c.text for c in test_chunks]
        vectors = embed_texts(embedder, texts)
        print(f"Generated {len(vectors)} embeddings")
        
        # Prepare for Pinecone
        print("Preparing metadata...")
        namespace = generate_namespace_for_file(test_file.name, str(test_file))
        print(f"Using namespace: {namespace}")
        
        items = []
        for i, (c, vec) in enumerate(zip(test_chunks, vectors)):
            node_id = deterministic_node_id(
                doc_id=sha1_hex(str(test_file)),
                page=c.page,
                start_char=c.start_char,
                text=c.text,
            )
            meta = project_metadata_for_chunk(
                file_name=test_file.name,
                source_path=str(test_file),
                jurisdiction="ADGM",
                page=c.page,
                section_path=c.section_path,
                element_type=c.element_type,
                chunk_index=i,
                chunking_strategy="token_aware_v1_test",
                text=c.text,
            )
            items.append({"id": node_id, "values": vec, "metadata": meta})
            
        # Upsert to Pinecone
        print("Upserting to Pinecone...")
        index.upsert(vectors=items, namespace=namespace)
        
        print("✅ Test completed successfully!")
        print(f"- Processed: {test_file.name}")
        print(f"- Chunks: {len(test_chunks)}")
        print(f"- Namespace: {namespace}")
        print(f"- Content store: {CONTENT_STORE}")
        
        # Check content store
        blob_dir = CONTENT_STORE / namespace
        if blob_dir.exists():
            content_files = list(blob_dir.glob("*.txt"))
            print(f"- Content files created: {len(content_files)}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_file()
