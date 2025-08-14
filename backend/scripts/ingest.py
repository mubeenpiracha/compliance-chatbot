# ingest.py — Unstructured → token-aware chunking → 1536-dim embeddings → Pinecone (per-file namespace)
# Constraints satisfied:
# - PDFs only
# - Hardcoded input paths from existing project: ./data/difc and ./data/adgm
# - Local content store for heavy artifacts (no S3)
# - Pinecone index dimension 1536
# - Namespace derived from each file name (per-file namespace)
# - Deterministic IDs, metadata size guardrails, batch upserts with retries

import os
import sys
import time
import re
import json
import math
import hashlib
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional, Tuple

# --- Third-party deps expected in your project ---
# pip install unstructured[all-docs] pinecone-client tiktoken pydantic openai
from pinecone import Pinecone, ServerlessSpec
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from llama_index.embeddings.openai import OpenAIEmbedding

try:
    import tiktoken
except Exception:  # fallback if not installed
    tiktoken = None

# -----------------
# CONFIG / CONSTANTS
# -----------------
MANIFEST_PATH = Path("./backend/data/Manifest.csv")
CONTENT_STORE = Path("./content_store")  # local path for heavy artifacts
CONTENT_STORE.mkdir(parents=True, exist_ok=True)

INDEX_NAME = "compliance-bot-index"
EMBED_MODEL_NAME = "text-embedding-3-large"  # will request 1536 dims
EMBED_DIM = 1536
EMBED_BATCH = 128
UPSERT_BATCH = 200

# Backoff settings
MAX_RETRIES = 5
BASE_SLEEP = 1.25

# Env vars must be set in your runtime
import sys
sys.path.append('/home/mubeen/compliance-chatbot')
sys.path.append('/home/mubeen/compliance-chatbot/backend')

from backend.core.config import OPENAI_API_KEY, PINECONE_API_KEY

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set")

# -----------------
# UTILS
# -----------------

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def byte_len(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return len(str(obj).encode("utf-8"))


# -----------------
# METADATA POLICY — keep Pinecone metadata tiny (<40KB total, prefer <4KB)
# -----------------
ALLOWED_META_KEYS = {
    "doc_id",
    "file_name",
    "source_path",
    "jurisdiction",
    "document_type",
    "document_title",
    "publication_date",
    "page",
    "element_type",
    "section_path",
    "chunk_index",
    "chunking_strategy",
    "content_uri",  # pointer to heavy blob on local disk
    "checksum",
}

MAX_FIELD_BYTES = 4096  # hard cap per field
MAX_METADATA_BYTES = 10_000  # conservative cap for the whole metadata dict



def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for k in list(meta.keys()):
        if k not in ALLOWED_META_KEYS:
            continue
        v = meta[k]
        
        # Skip null/None values - Pinecone doesn't accept them
        if v is None:
            continue
            
        # Convert to string if it's not a basic type
        if not isinstance(v, (str, int, float, bool)):
            v = str(v)
            
        # truncate any over-sized field
        if isinstance(v, str):
            while byte_len(v) > MAX_FIELD_BYTES:
                # keep last 100 chars context + hash prefix
                v = v[: max(0, len(v) - 500)] + f"…[trunc:{sha1_hex(v)[:8]}]"
        clean[k] = v
    # final safeguard on total
    if byte_len(clean) > MAX_METADATA_BYTES:
        # progressively drop least-critical keys
        for key in ["section_path", "element_type", "jurisdiction"]:
            if key in clean and byte_len(clean) > MAX_METADATA_BYTES:
                del clean[key]
    return clean


# -----------------
# TOKEN / CHUNKING
# -----------------

def get_encoder():
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def token_len(text: str, enc=None) -> int:
    if not text:
        return 0
    if enc is None:
        enc = get_encoder()
    if enc is None:
        # rough char->token fallback
        return max(1, len(text) // 4)
    return len(enc.encode(text))


# -----------------
# INGESTION
# -----------------

@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]

def load_documents_from_manifest() -> List[Document]:
    """Loads documents and their metadata from the CSV manifest."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest file not found at: {MANIFEST_PATH}")

    documents = []
    with open(MANIFEST_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = Path(row["file_path"])
            if not file_path.exists():
                print(f"WARN: File not found, skipping: {file_path}")
                continue
            
            doc_id = sha1_hex(str(file_path))
            # We will partition and chunk later, for now, just load the manifest data
            # The "text" will be populated from the chunks later on.
            doc = Document(
                id=doc_id,
                text="", # Placeholder, will be replaced by chunk text
                metadata={
                    "source_path": str(file_path),
                    "file_name": file_path.name,
                    "jurisdiction": row.get("jurisdiction"),
                    "document_type": row.get("document_type"),
                    "document_title": row.get("document_title"),
                    "publication_date": row.get("publication_date"),
                }
            )
            documents.append(doc)
    print(f"Loaded {len(documents)} document references from manifest.")
    return documents



def generate_namespace_for_file(file_name: str) -> str:
    stem = Path(file_name).stem
    return slugify(stem)[:64] or "default"


def project_metadata_for_chunk(
    chunk_text: str,
    chunk_index: int,
    doc_metadata: Dict[str, Any],
    chunking_strategy: str = "semantic_v1",
) -> Dict[str, Any]:
    """Projects and sanitizes metadata for a single chunk."""
    file_name = doc_metadata.get("file_name", "")
    source_path = doc_metadata.get("source_path", "")
    
    checksum = sha1_hex(chunk_text)
    ns = generate_namespace_for_file(file_name)
    blob_dir = CONTENT_STORE / ns
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_path = blob_dir / f"{chunk_index:06d}_{checksum}.txt"
    if not blob_path.exists():
        blob_path.write_text(chunk_text, encoding="utf-8")

    meta = {
        "doc_id": sha1_hex(source_path or file_name),
        "file_name": file_name,
        "source_path": source_path,
        "jurisdiction": doc_metadata.get("jurisdiction"),
        "document_type": doc_metadata.get("document_type"),
        "document_title": doc_metadata.get("document_title"),
        "publication_date": doc_metadata.get("publication_date"),
        "page": doc_metadata.get("page"),  # This might come from unstructured elements
        "element_type": doc_metadata.get("element_type"),
        "chunk_index": chunk_index,
        "chunking_strategy": chunking_strategy,
        "content_uri": str(blob_path),
        "checksum": checksum,
    }
    return sanitize_metadata(meta)


def chunk_document(doc: Document) -> List[Dict[str, Any]]:
    """Partitions and chunks a document semantically."""
    source_path = doc.metadata.get("source_path")
    if not source_path:
        return []
        
    print(f"  Partitioning and chunking: {source_path}")
    try:
        # Use unstructured to partition the document
        elements = partition(filename=source_path, strategy="auto")
        
        # Use chunk_by_title to group elements into semantic chunks
        chunks = chunk_by_title(
            elements,
            max_characters=1200, # A reasonable starting point
            new_after_n_chars=1000,
            combine_text_under_n_chars=500,
        )
        
        chunk_data = []
        for i, chunk in enumerate(chunks):
            text = chunk.text
            if not text.strip():
                continue
            
            # Preserve metadata from the original document and add chunk-specific info
            metadata = doc.metadata.copy()
            
            # Try to get page number from the elements within the chunk
            page_numbers = [el.metadata.page_number for el in chunk.metadata.orig_elements if el.metadata.page_number]
            if page_numbers:
                metadata["page"] = min(page_numbers) # Use the first page number of the chunk
            
            # Create a dictionary for each chunk with its text and metadata
            chunk_data.append({
                "text": text,
                "metadata": metadata,
                "chunk_index": i,
            })
        return chunk_data

    except Exception as e:
        print(f"  ERROR chunking {source_path}: {e}")
        return []


def batch(iterable: Iterable[Any], size: int) -> Iterable[List[Any]]:
    arr: List[Any] = []
    for item in iterable:
        arr.append(item)
        if len(arr) >= size:
            yield arr
            arr = []
    if arr:
        yield arr


def ensure_index(pc: Pinecone, name: str, dim: int):
    if name not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{name}' with dim={dim}...")
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # wait for index to be ready
        for _ in range(30):
            time.sleep(2)
            if name in pc.list_indexes().names():
                break
        print("Index created.")
    else:
        print(f"Index '{name}' exists. ✔")


def embed_texts(embedder: OpenAIEmbedding, texts: List[str]) -> List[List[float]]:
    # LlamaIndex OpenAIEmbedding handles batching internally but we keep an upper bound
    # and add retries for robustness.
    vectors: List[List[float]] = []
    for group in batch(texts, EMBED_BATCH):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                out = embedder.get_text_embedding_batch(group)
                # Safety: coerce to desired dimension
                if out and len(out[0]) != EMBED_DIM:
                    # Some models can return larger dims if "dimensions" not honored; trim if needed
                    out = [vec[:EMBED_DIM] for vec in out]
                vectors.extend(out)
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                sleep_s = BASE_SLEEP * (2 ** (attempt - 1))
                print(f"Embedding retry {attempt}/{MAX_RETRIES} after error: {e}. Sleeping {sleep_s:.2f}s")
                time.sleep(sleep_s)
    return vectors


def ingest():
    print("Loading document references from manifest...")
    documents = load_documents_from_manifest()
    if not documents:
        print("No documents found in manifest. Nothing to ingest.")
        return

    # Init embedder
    embedder = OpenAIEmbedding(
        model=EMBED_MODEL_NAME,
        api_key=OPENAI_API_KEY,
        dimensions=EMBED_DIM,
    )

    # Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc, INDEX_NAME, EMBED_DIM)
    index = pc.Index(INDEX_NAME)

    total_chunks = 0
    upserted = 0

    # Process each document from the manifest
    for doc in documents:
        file_name = doc.metadata.get("file_name", "unknown.pdf")
        namespace = generate_namespace_for_file(file_name)

        # Perform semantic chunking
        chunks = chunk_document(doc)
        if not chunks:
            print(f"Skip empty doc: {file_name}")
            continue

        # Prepare embeddings
        texts_to_embed = [c["text"] for c in chunks]
        vectors = embed_texts(embedder, texts_to_embed)

        # Prepare Pinecone upserts in batches
        items_to_upsert: List[Dict[str, Any]] = []
        for i, (chunk_data, vec) in enumerate(zip(chunks, vectors)):
            chunk_text = chunk_data["text"]
            doc_id = doc.id
            
            node_id = sha1_hex(f"{doc_id}|{i}|{chunk_text}")
            
            meta = project_metadata_for_chunk(
                chunk_text=chunk_text,
                chunk_index=i,
                doc_metadata=chunk_data["metadata"],
            )
            items_to_upsert.append({"id": node_id, "values": vec, "metadata": meta})

        total_chunks += len(items_to_upsert)

        # Upsert with retries per batch into per-file namespace
        for group in batch(items_to_upsert, UPSERT_BATCH):
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    index.upsert(vectors=group, namespace=namespace)
                    upserted += len(group)
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES:
                        raise
                    sleep_s = BASE_SLEEP * (2 ** (attempt - 1))
                    print(f"Upsert retry {attempt}/{MAX_RETRIES} after error: {e}. Sleeping {sleep_s:.2f}s")
                    time.sleep(sleep_s)

        print(f"✔ Ingested {len(items_to_upsert)} chunks for '{file_name}' into namespace '{namespace}'")

    print("—" * 60)
    print(f"Done. Documents: {len(documents)}, Chunks: {total_chunks}, Upserted: {upserted}")


if __name__ == "__main__":
    ingest()
