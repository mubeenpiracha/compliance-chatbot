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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional, Tuple

# --- Third-party deps expected in your project ---
# pip install unstructured[all-docs] llama-index pinecone-client tiktoken pydantic
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import Document
from llama_index.readers.file import UnstructuredReader
from llama_index.embeddings.openai import OpenAIEmbedding

try:
    import tiktoken
except Exception:  # fallback if not installed
    tiktoken = None

# -----------------
# CONFIG / CONSTANTS
# -----------------
DATA_DIRS = ["./backend/data/difc", "./backend/data/adgm"]  # hardcoded as requested
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


@dataclass
class Chunk:
    text: str
    page: Optional[int]
    section_path: Optional[str]
    element_type: Optional[str]
    start_char: int
    end_char: int


def chunk_text_token_aware(
    text: str,
    target_tokens: int = 900,
    overlap_tokens: int = 120,
    page: Optional[int] = None,
    section_path: Optional[str] = None,
    element_type: Optional[str] = None,
) -> List[Chunk]:
    """Split text into overlapping windows using tokens (fallback to chars)."""
    enc = get_encoder()
    if enc is None:
        # char-based fallback (~4 chars per token)
        approx = target_tokens * 4
        olap = overlap_tokens * 4
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + approx)
            chunk_text = text[start:end]
            chunks.append(
                Chunk(chunk_text, page, section_path, element_type, start, end)
            )
            if end == n:
                break
            start = max(0, end - olap)
        return chunks

    ids = enc.encode(text)
    chunks: List[Chunk] = []

    def decode(slice_ids: List[int]) -> str:
        return enc.decode(slice_ids)

    start_tok = 0
    n = len(ids)
    while start_tok < n:
        end_tok = min(n, start_tok + target_tokens)
        slice_ids = ids[start_tok:end_tok]
        chunk_text = decode(slice_ids)
        # Estimate char offsets by decoding left & right
        left = decode(ids[:start_tok])
        start_char = len(left)
        end_char = start_char + len(chunk_text)
        chunks.append(
            Chunk(chunk_text, page, section_path, element_type, start_char, end_char)
        )
        if end_tok == n:
            break
        start_tok = max(0, end_tok - overlap_tokens)
    return chunks


# -----------------
# INGESTION
# -----------------

def build_documents() -> List[Document]:
    reader = UnstructuredReader()
    documents: List[Document] = []
    
    for data_dir in DATA_DIRS:
        dpath = Path(data_dir)
        if not dpath.exists():
            print(f"WARN: directory missing: {data_dir}")
            continue
            
        # Find all PDF files in this directory
        pdf_files = list(dpath.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files in {data_dir}")
        
        for pdf_file in pdf_files:
            try:
                print(f"  Processing: {pdf_file.name}")
                docs = reader.load_data(file=str(pdf_file))
                
                # Process each document from this file
                for doc in docs:
                    # normalize minimal metadata
                    file_name = pdf_file.name
                    source_path = str(pdf_file)
                    
                    # best-effort jurisdiction from path
                    jurisdiction = "DIFC" if "difc" in str(dpath).lower() else (
                        "ADGM" if "adgm" in str(dpath).lower() else None
                    )
                    
                    # Update document metadata
                    doc.metadata.update({
                        "file_name": file_name,
                        "source_path": source_path,
                        "jurisdiction": jurisdiction,
                    })
                    documents.append(doc)
                    
            except Exception as e:
                print(f"  ERROR processing {pdf_file.name}: {e}")
                continue
                
    return documents


def generate_namespace_for_file(file_name: str) -> str:
    stem = Path(file_name).stem
    return slugify(stem)[:64] or "default"


def deterministic_node_id(
    doc_id: str, page: Optional[int], start_char: int, text: str
) -> str:
    base = f"{doc_id}|{page if page is not None else ''}|{start_char}|{sha1_hex(text)}"
    return sha1_hex(base)


# Lightweight schema projection for Pinecone

def project_metadata_for_chunk(
    file_name: str,
    source_path: str,
    jurisdiction: Optional[str],
    page: Optional[int],
    section_path: Optional[str],
    element_type: Optional[str],
    chunk_index: int,
    chunking_strategy: str,
    text: str,
) -> Dict[str, Any]:
    checksum = sha1_hex(text)
    # persist heavy text to content store and store a pointer
    ns = generate_namespace_for_file(file_name)
    blob_dir = CONTENT_STORE / ns
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_path = blob_dir / f"{chunk_index:06d}_{checksum}.txt"
    if not blob_path.exists():
        blob_path.write_text(text, encoding="utf-8")

    meta = {
        "doc_id": sha1_hex(source_path or file_name),
        "file_name": file_name,
        "source_path": source_path,
        "jurisdiction": jurisdiction,
        "page": page,
        "element_type": element_type,
        "section_path": section_path,
        "chunk_index": chunk_index,
        "chunking_strategy": chunking_strategy,
        "content_uri": str(blob_path),
        "checksum": checksum,
    }
    return sanitize_metadata(meta)


def chunk_document(doc: Document) -> List[Chunk]:
    """Two-pass strategy: per-page consolidation, then token-aware windows.
    We preserve tables by trusting Unstructured page segmentation (no table splitting hints exposed here),
    but the windows avoid mid-paragraph breaks most of the time.
    """
    # Unstructured content is in doc.text. Page numbers may be present in metadata; if not, we treat as single blob.
    # If the loader provided elements with page info, they'd be in metadata; llama-index's Document keeps raw text.
    text = doc.get_text() if hasattr(doc, "get_text") else (doc.text or "")
    if not text.strip():
        return []

    # Try to recover a "section path" from headings if Unstructured added it into metadata (best-effort)
    section_path = doc.metadata.get("section_path")
    element_type = doc.metadata.get("element_type")
    page = doc.metadata.get("page")

    # Single-pass token-aware chunking; if future element/page info available, we could pre-split by page.
    return chunk_text_token_aware(
        text,
        target_tokens=900,
        overlap_tokens=120,
        page=page,
        section_path=section_path,
        element_type=element_type,
    )


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
    print("Loading documents from:", DATA_DIRS)
    documents = build_documents()
    if not documents:
        print("No documents found. Nothing to ingest.")
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

    # Process per file to use per-file namespaces
    for doc in documents:
        file_name = doc.metadata.get("file_name", "unknown.pdf")
        source_path = doc.metadata.get("source_path", file_name)
        jurisdiction = doc.metadata.get("jurisdiction")
        namespace = generate_namespace_for_file(file_name)

        chunks = chunk_document(doc)
        if not chunks:
            print(f"Skip empty doc: {file_name}")
            continue

        # Prepare embeddings
        texts = [c.text for c in chunks]
        vectors = embed_texts(embedder, texts)

        # Prepare Pinecone upserts in batches
        items: List[Dict[str, Any]] = []
        for i, (c, vec) in enumerate(zip(chunks, vectors)):
            node_id = deterministic_node_id(
                doc_id=sha1_hex(source_path or file_name),
                page=c.page,
                start_char=c.start_char,
                text=c.text,
            )
            meta = project_metadata_for_chunk(
                file_name=file_name,
                source_path=source_path,
                jurisdiction=jurisdiction,
                page=c.page,
                section_path=c.section_path,
                element_type=c.element_type,
                chunk_index=i,
                chunking_strategy="token_aware_v1",
                text=c.text,
            )
            items.append({"id": node_id, "values": vec, "metadata": meta})

        total_chunks += len(items)

        # Upsert with retries per batch into per-file namespace
        for group in batch(items, UPSERT_BATCH):
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

        print(f"✔ Ingested {len(items)} chunks for '{file_name}' into namespace '{namespace}'")

    print("—" * 60)
    print(f"Done. Files: {len(documents)}, Chunks: {total_chunks}, Upserted: {upserted}")


if __name__ == "__main__":
    ingest()
