# backend/scripts/ingest.py
import os
import sys
import time
import re
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.readers.file import UnstructuredReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import (
    SentenceSplitter, 
    SemanticSplitterNodeParser,
    MarkdownNodeParser,
    HierarchicalNodeParser
)
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import BaseNode
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import PINECONE_API_KEY, OPENAI_API_KEY

# Set OpenAI API key for LlamaIndex components
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class RegulatoryDocumentChunker:
    """
    Advanced chunker for regulatory documents that preserves structure and context.
    """
    
    def __init__(self, embedding_model, chunk_size: int = 1024, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        # Initialize multiple chunking strategies
        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embedding_model
        )
        
        self.hierarchical_splitter = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 1024, 512],  # Increased minimum chunk size
            chunk_overlap=chunk_overlap
        )
        
        self.sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n",
            chunking_tokenizer_fn=self._custom_tokenizer
        )
    
    def _custom_tokenizer(self, text: str) -> List[str]:
        """Custom tokenizer that's more aware of regulatory document structure."""
        # Split on common regulatory patterns
        patterns = [
            r'\n(?=\d+\.\s)',  # Numbered sections
            r'\n(?=[A-Z][a-z]+:)',  # Section headers
            r'\n(?=\([a-z]\))',  # Subsections like (a), (b)
            r'\n(?=\([0-9]+\))',  # Numbered subsections like (1), (2)
            r'(?<=\.)\s+(?=[A-Z])',  # Sentence boundaries
        ]
        
        tokens = [text]
        for pattern in patterns:
            new_tokens = []
            for token in tokens:
                split_tokens = re.split(pattern, token)
                new_tokens.extend([t.strip() for t in split_tokens if t.strip()])
            tokens = new_tokens
        
        return tokens
    
    def extract_document_structure(self, text: str) -> dict:
        """Extract structural information from regulatory documents."""
        structure = {
            'sections': [],
            'subsections': [],
            'tables': [],
            'definitions': [],
            'references': []
        }
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Detect main sections (numbered sections)
            if re.match(r'^\d+\.\s+[A-Z]', line):
                structure['sections'].append({
                    'line_number': i,
                    'title': line,
                    'level': 1
                })
            
            # Detect subsections
            elif re.match(r'^\d+\.\d+\s+[A-Z]', line):
                structure['subsections'].append({
                    'line_number': i,
                    'title': line,
                    'level': 2
                })
            
            # Detect definitions
            elif 'definition' in line.lower() or 'means' in line.lower():
                structure['definitions'].append({
                    'line_number': i,
                    'text': line
                })
            
            # Detect table references
            elif re.search(r'table\s+\d+', line.lower()):
                structure['tables'].append({
                    'line_number': i,
                    'reference': line
                })
            
            # Detect cross-references
            elif re.search(r'section\s+\d+|paragraph\s+\d+|rule\s+\d+', line.lower()):
                structure['references'].append({
                    'line_number': i,
                    'reference': line
                })
        
        return structure
    
    def enhance_metadata(self, document: Document) -> Document:
        """Enhance document metadata with structural and content information."""
        text = document.text
        structure = self.extract_document_structure(text)
        
        # Add structural metadata
        document.metadata.update({
            'num_sections': len(structure['sections']),
            'num_subsections': len(structure['subsections']),
            'num_tables': len(structure['tables']),
            'num_definitions': len(structure['definitions']),
            'num_references': len(structure['references']),
            'document_length': len(text),
            'estimated_reading_time': len(text.split()) // 200,  # ~200 words per minute
        })
        
        # Extract document type from filename or content
        filename = document.metadata.get('file_name', '').lower()
        if 'regulation' in filename or 'regulation' in text[:1000].lower():
            document.metadata['document_type'] = 'regulation'
        elif 'rulebook' in filename or 'rules' in text[:1000].lower():
            document.metadata['document_type'] = 'rulebook'
        elif 'guidance' in filename or 'guide' in filename:
            document.metadata['document_type'] = 'guidance'
        else:
            document.metadata['document_type'] = 'other'
        
        # Extract key regulatory concepts
        regulatory_keywords = [
            'compliance', 'violation', 'penalty', 'requirement', 'obligation',
            'prohibited', 'permitted', 'license', 'authorization', 'approval',
            'reporting', 'disclosure', 'audit', 'inspection', 'enforcement'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        for keyword in regulatory_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        document.metadata['regulatory_concepts'] = found_keywords
        
        return document
    
    def chunk_document(self, document: Document) -> List[BaseNode]:
        """
        Apply intelligent chunking strategy based on document characteristics.
        """
        # Enhance metadata first
        document = self.enhance_metadata(document)
        
        # Choose chunking strategy based on document characteristics
        doc_length = len(document.text)
        doc_type = document.metadata.get('document_type', 'other')
        
        if doc_length > 50000:  # Large documents - use hierarchical
            print(f"Using hierarchical chunking for large document: {document.metadata.get('file_name', 'Unknown')}")
            nodes = self.hierarchical_splitter.get_nodes_from_documents([document])
        elif doc_type in ['regulation', 'rulebook']:  # Structured documents - use semantic
            print(f"Using semantic chunking for structured document: {document.metadata.get('file_name', 'Unknown')}")
            try:
                nodes = self.semantic_splitter.get_nodes_from_documents([document])
            except Exception as e:
                print(f"Semantic chunking failed, falling back to sentence splitting: {e}")
                nodes = self.sentence_splitter.get_nodes_from_documents([document])
        else:  # Default to enhanced sentence splitting
            print(f"Using sentence chunking for document: {document.metadata.get('file_name', 'Unknown')}")
            nodes = self.sentence_splitter.get_nodes_from_documents([document])
        
        # Add chunk-specific metadata
        for i, node in enumerate(nodes):
            node.metadata.update({
                'chunk_id': f"{document.metadata.get('document_id', 'unknown')}_{i}",
                'chunk_index': i,
                'total_chunks': len(nodes),
                'chunk_size': len(node.text),
                'chunk_type': self._classify_chunk_content(node.text)
            })
        
        return nodes
    
    def _classify_chunk_content(self, text: str) -> str:
        """Classify the type of content in a chunk."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['definition', 'means', 'refers to']):
            return 'definition'
        elif any(word in text_lower for word in ['table', 'schedule', 'appendix']):
            return 'table_or_schedule'
        elif any(word in text_lower for word in ['requirement', 'must', 'shall', 'obligation']):
            return 'requirement'
        elif any(word in text_lower for word in ['prohibition', 'prohibited', 'not permitted', 'forbidden']):
            return 'prohibition'
        elif any(word in text_lower for word in ['penalty', 'fine', 'sanction', 'enforcement']):
            return 'enforcement'
        elif re.match(r'^\d+\.\s+', text.strip()):
            return 'section_header'
        else:
            return 'general_content'

def run_ingestion():
    """
    Connects to Pinecone, loads documents, creates an index with improved chunking, and upserts the data.
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
    
    # Initialize embedding model for chunking and indexing
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-large", 
        api_key=OPENAI_API_KEY,
        dimensions=pinecone_dimension 
    )
    
    # Use UnstructuredReader for parsing PDFs, which is better for tables and images
    unstructured_reader = UnstructuredReader()
    file_extractor = {".pdf": unstructured_reader}

    print("Loading documents from DIFC and ADGM directories...")
    difc_docs = SimpleDirectoryReader(
        "./data/difc", file_extractor=file_extractor
    ).load_data()
    adgm_docs = SimpleDirectoryReader(
        "./data/adgm", file_extractor=file_extractor
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

    print("Configuring LlamaIndex settings with advanced chunking...")
    
    # Initialize the advanced chunker
    chunker = RegulatoryDocumentChunker(
        embedding_model=embed_model,
        chunk_size=1024,  # Increased chunk size for better context
        chunk_overlap=128  # Increased overlap for better continuity
    )
    
    # Process documents through advanced chunking
    all_nodes = []
    print("Processing documents with intelligent chunking...")
    
    for i, doc in enumerate(documents):
        print(f"Processing document {i+1}/{len(documents)}: {doc.metadata.get('file_name', 'Unknown')}")
        try:
            nodes = chunker.chunk_document(doc)
            all_nodes.extend(nodes)
            print(f"  Created {len(nodes)} chunks")
        except Exception as e:
            print(f"  Error processing document: {e}")
            # Fallback to basic chunking
            basic_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
            nodes = basic_splitter.get_nodes_from_documents([doc])
            all_nodes.extend(nodes)
            print(f"  Fallback: Created {len(nodes)} basic chunks")
    
    print(f"Total chunks created: {len(all_nodes)}")
    
    # Analyze chunk distribution
    chunk_types = {}
    chunk_sizes = []
    for node in all_nodes:
        chunk_type = node.metadata.get('chunk_type', 'unknown')
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        chunk_sizes.append(len(node.text))
    
    print("Chunk analysis:")
    print(f"  Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.0f} characters")
    print(f"  Chunk type distribution: {chunk_types}")

    # Set up LlamaIndex settings
    Settings.embed_model = embed_model

    print("Creating storage context and upserting vectors to Pinecone...")
    try:
        # Create index from nodes instead of documents for better control
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes=all_nodes, 
            storage_context=storage_context, 
            show_progress=True
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


def test_chunking_strategy(sample_docs: int = 3):
    """
    Test the chunking strategy on a sample of documents without uploading to Pinecone.
    """
    print(f"Testing chunking strategy on {sample_docs} sample documents...")
    
    # Initialize embedding model
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-large", 
        api_key=OPENAI_API_KEY,
        dimensions=1536 
    )
    
    # Load a small sample of documents
    unstructured_reader = UnstructuredReader()
    file_extractor = {".pdf": unstructured_reader}
    
    docs = []
    try:
        difc_docs = SimpleDirectoryReader(
            "./data/difc", file_extractor=file_extractor
        ).load_data()[:sample_docs//2]
        adgm_docs = SimpleDirectoryReader(
            "./data/adgm", file_extractor=file_extractor
        ).load_data()[:sample_docs//2]
        docs = difc_docs + adgm_docs
    except Exception as e:
        print(f"Error loading documents: {e}")
        return
    
    if not docs:
        print("No documents found for testing.")
        return
    
    # Initialize chunker
    chunker = RegulatoryDocumentChunker(
        embedding_model=embed_model,
        chunk_size=1024,
        chunk_overlap=128
    )
    
    print(f"\nTesting on {len(docs)} documents:")
    total_chunks = 0
    chunk_stats = {}
    
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1}: {doc.metadata.get('file_name', 'Unknown')} ---")
        print(f"Document length: {len(doc.text)} characters")
        
        try:
            nodes = chunker.chunk_document(doc)
            total_chunks += len(nodes)
            
            print(f"Generated {len(nodes)} chunks")
            print(f"Document metadata: {doc.metadata}")
            
            # Analyze chunk types
            for node in nodes:
                chunk_type = node.metadata.get('chunk_type', 'unknown')
                chunk_stats[chunk_type] = chunk_stats.get(chunk_type, 0) + 1
            
            # Show sample chunks
            print("Sample chunks:")
            for j, node in enumerate(nodes[:3]):  # Show first 3 chunks
                print(f"  Chunk {j+1} ({node.metadata.get('chunk_type', 'unknown')}): "
                      f"{node.text[:150]}...")
                
        except Exception as e:
            print(f"Error processing document: {e}")
    
    print(f"\n=== Summary ===")
    print(f"Total chunks generated: {total_chunks}")
    print(f"Average chunks per document: {total_chunks/len(docs):.1f}")
    print(f"Chunk type distribution: {chunk_stats}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_chunking_strategy()
    else:
        run_ingestion()