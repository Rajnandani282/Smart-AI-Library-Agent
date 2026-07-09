"""
rag_pipeline.py
───────────────
Builds and manages the RAG (Retrieval-Augmented Generation) pipeline:
  • Loads text documents from the knowledge_base/ directory
  • Splits them into chunks
  • Embeds with sentence-transformers
  • Stores in ChromaDB (default) or FAISS
  • Provides a retrieve() function used by the Flask app
"""

import os
import csv
import json
import logging
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Config from .env ─────────────────────────────────────────────────────────
VECTOR_STORE     = os.getenv("VECTOR_STORE", "chroma").lower()
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR       = os.getenv("CHROMA_PERSIST_DIR", "./vector_store/chroma_db")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./vector_store/faiss_index")
RAG_TOP_K        = int(os.getenv("RAG_TOP_K", "6"))
KB_DIR           = Path(__file__).parent / "knowledge_base"

# ─── Embedding model (lazy-loaded) ───────────────────────────────────────────
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


# ══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_text_files() -> List[Tuple[str, str]]:
    """
    Loads all .txt files from knowledge_base/syllabi,
    knowledge_base/department_guides, and knowledge_base/reading_lists.
    Returns list of (doc_id, text) tuples.
    """
    docs = []
    for folder in ["syllabi", "department_guides", "reading_lists"]:
        folder_path = KB_DIR / folder
        if not folder_path.exists():
            continue
        for txt_file in folder_path.glob("*.txt"):
            text = txt_file.read_text(encoding="utf-8")
            docs.append((f"{folder}/{txt_file.stem}", text))
            logger.debug(f"Loaded text: {txt_file.name}")
    return docs


def _load_catalog_csv() -> List[Tuple[str, str]]:
    """
    Converts the book catalog CSV into paragraph-style text chunks,
    one chunk per book, for embedding.
    """
    csv_path = KB_DIR / "book_catalog" / "catalog.csv"
    if not csv_path.exists():
        logger.warning("catalog.csv not found — skipping")
        return []

    docs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            availability = (
                f"{row['copies_available']} of {row['copies_total']} copies available"
            )
            text = (
                f"Book Title: {row['title']}\n"
                f"Author: {row['author']}\n"
                f"ISBN: {row['isbn']}\n"
                f"Publisher: {row['publisher']}, {row['year']}, {row['edition']} edition\n"
                f"Department: {row['department']}\n"
                f"Subject: {row['subject']}\n"
                f"Availability: {availability}\n"
                f"Shelf Location: {row['shelf_location']}\n"
                f"Language: {row['language']}\n"
                f"Description: {row['description']}\n"
                f"Topics/Tags: {row['tags'].replace(';', ', ')}\n"
            )
            docs.append((f"catalog/{row['book_id']}", text))
    logger.info(f"Loaded {len(docs)} books from catalog.csv")
    return docs


def _split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Simple character-level text splitter with overlap.
    Falls back to LangChain's RecursiveCharacterTextSplitter when available.
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " "],
        )
        return splitter.split_text(text)
    except ImportError:
        # Fallback: manual split
        chunks, start = [], 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks


def _prepare_documents() -> Tuple[List[str], List[str], List[dict]]:
    """
    Loads all knowledge-base sources, splits into chunks.
    Returns (ids, texts, metadatas).
    """
    all_texts_raw = _load_catalog_csv() + _load_text_files()

    ids, texts, metadatas = [], [], []
    idx = 0
    for doc_id, full_text in all_texts_raw:
        chunks = _split_text(full_text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc_id}_chunk{i}")
            texts.append(chunk)
            metadatas.append({"source": doc_id, "chunk_index": i})
            idx += 1

    logger.info(f"Total chunks prepared for indexing: {len(texts)}")
    return ids, texts, metadatas


# ══════════════════════════════════════════════════════════════════════════════
#  VECTOR STORE — ChromaDB
# ══════════════════════════════════════════════════════════════════════════════

_chroma_collection = None

def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    import chromadb
    from chromadb.config import Settings

    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection_name = "library_kb"
    existing = [c.name for c in client.list_collections()]

    if collection_name in existing:
        logger.info("ChromaDB: loading existing collection")
        _chroma_collection = client.get_collection(collection_name)
    else:
        logger.info("ChromaDB: building new collection — this may take a minute …")
        _chroma_collection = client.create_collection(collection_name)
        _index_into_chroma(_chroma_collection)

    return _chroma_collection


def _index_into_chroma(collection):
    ids, texts, metadatas = _prepare_documents()
    embedder = get_embedder()

    BATCH = 128
    for i in range(0, len(texts), BATCH):
        batch_ids   = ids[i : i + BATCH]
        batch_texts = texts[i : i + BATCH]
        batch_meta  = metadatas[i : i + BATCH]
        batch_embs  = embedder.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embs,
            metadatas=batch_meta,
        )
    logger.info(f"ChromaDB: indexed {len(texts)} chunks")


def _retrieve_chroma(query: str, top_k: int = RAG_TOP_K) -> List[str]:
    collection = _get_chroma_collection()
    embedder   = get_embedder()
    q_emb      = embedder.encode([query]).tolist()
    results    = collection.query(query_embeddings=q_emb, n_results=top_k)
    return results["documents"][0] if results["documents"] else []


# ══════════════════════════════════════════════════════════════════════════════
#  VECTOR STORE — FAISS
# ══════════════════════════════════════════════════════════════════════════════

_faiss_index  = None
_faiss_texts  = None

def _get_faiss_index():
    global _faiss_index, _faiss_texts
    if _faiss_index is not None:
        return _faiss_index, _faiss_texts

    import faiss
    import numpy as np
    import pickle

    idx_file  = Path(FAISS_INDEX_PATH + ".index")
    text_file = Path(FAISS_INDEX_PATH + ".pkl")

    if idx_file.exists() and text_file.exists():
        logger.info("FAISS: loading existing index")
        _faiss_index = faiss.read_index(str(idx_file))
        with open(text_file, "rb") as f:
            _faiss_texts = pickle.load(f)
    else:
        logger.info("FAISS: building index — this may take a minute …")
        _, texts, _ = _prepare_documents()
        embedder    = get_embedder()
        embeddings  = embedder.encode(texts, show_progress_bar=True)
        dim         = embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype("float32"))

        idx_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(idx_file))
        with open(text_file, "wb") as f:
            pickle.dump(texts, f)

        _faiss_index = index
        _faiss_texts = texts
        logger.info(f"FAISS: indexed {len(texts)} chunks")

    return _faiss_index, _faiss_texts


def _retrieve_faiss(query: str, top_k: int = RAG_TOP_K) -> List[str]:
    import faiss
    import numpy as np

    index, texts = _get_faiss_index()
    embedder     = get_embedder()
    q_emb        = embedder.encode([query]).astype("float32")
    _, I         = index.search(q_emb, top_k)
    return [texts[i] for i in I[0] if i < len(texts)]


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def initialize_vector_store():
    """
    Called once at application startup to warm up the vector store.
    Subsequent calls are no-ops (cached).
    """
    logger.info(f"Initializing vector store backend: {VECTOR_STORE}")
    if VECTOR_STORE == "faiss":
        _get_faiss_index()
    else:
        _get_chroma_collection()
    logger.info("Vector store ready.")


def retrieve(query: str, top_k: int = RAG_TOP_K) -> str:
    """
    Retrieves the top-k most relevant chunks for `query`.
    Returns a single concatenated string for injection into the system prompt.
    """
    if VECTOR_STORE == "faiss":
        chunks = _retrieve_faiss(query, top_k)
    else:
        chunks = _retrieve_chroma(query, top_k)

    if not chunks:
        return "No relevant information found in the knowledge base."

    return "\n\n---\n\n".join(chunks)


def rebuild_index():
    """
    Force-rebuilds the vector store from scratch.
    Call this after adding new documents to knowledge_base/.
    """
    global _chroma_collection, _faiss_index, _faiss_texts
    _chroma_collection = None
    _faiss_index = None
    _faiss_texts = None

    if VECTOR_STORE == "faiss":
        # Remove old index files
        for ext in [".index", ".pkl"]:
            p = Path(FAISS_INDEX_PATH + ext)
            if p.exists():
                p.unlink()
    else:
        # Drop and recreate ChromaDB collection
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection("library_kb")
        except Exception:
            pass

    initialize_vector_store()
    logger.info("Index rebuilt successfully.")
