"""Shared settings: paths, model names and the local embedding model."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
FAQ_CSV = DATA_DIR / "faq.csv"
TICKETS_DB = DATA_DIR / "tickets.db"
GUIDE_PDF = DATA_DIR / "telecom_guide.pdf"

CHROMA_DIR = ROOT / "chroma_store"
LOG_DIR = ROOT / "logs"
INTERACTION_LOG = LOG_DIR / "interactions.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # FR-09, NFR-02
# FR-13 names qwen/qwen3.6-27b; Groq no longer serves it, so default to its successor.
LLM_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
MAX_ANSWER_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "450"))  # keeps free-tier OTPM limits happy
TOP_K = 3                                                      # FR-07
CHUNK_SIZE = 600                                                # FR-16
CHUNK_OVERLAP = 100


class MiniLMEmbeddings:
    """LangChain-compatible wrapper around ChromaDB's bundled ONNX build of
    all-MiniLM-L6-v2. Same model as the sentence-transformers version, but runs on
    onnxruntime (no torch), so installs are ~1 GB smaller and start-up is seconds."""

    def __init__(self):
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

        self._fn = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts):
        return [[float(x) for x in v] for v in self._fn(list(texts))]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embeddings():
    """Local MiniLM embeddings (FR-09, NFR-02); ~80 MB model downloaded once, then cached."""
    return MiniLMEmbeddings()


@lru_cache(maxsize=1)
def get_client():
    """One persistent Chroma client per process (parallel retrievers share it)."""
    import chromadb

    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _open_store(collection: str):
    from langchain_chroma import Chroma

    return Chroma(
        client=get_client(),
        collection_name=collection,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


_stores: dict = {}


def get_store(collection: str):
    """Open (or create) a persisted Chroma collection; cached per process."""
    if collection not in _stores:
        _stores[collection] = _open_store(collection)
    return _stores[collection]


def warm_stores(collections) -> None:
    """Open every collection up front, before any parallel retrieval."""
    get_embeddings()
    for name in collections:
        get_store(name)


def ensure_ingested() -> list[str]:
    """Build any empty collection from data/ (e.g. on a fresh cloud deploy, where
    chroma_store/ isn't in git). Existing collections are left alone (NFR-05)."""
    import ingest_faq
    import ingest_guides
    import ingest_tickets

    built = []
    for module in (ingest_faq, ingest_tickets, ingest_guides):
        if get_store(module.COLLECTION)._collection.count() == 0:
            module.main()
            built.append(module.COLLECTION)
    return built


def rebuild_collection(collection: str, docs, ids):
    """Drop and re-create a collection so re-running an ingest is idempotent (FR-17)."""
    _open_store(collection).delete_collection()
    _stores.pop(collection, None)
    store = get_store(collection)
    store.add_documents(docs, ids=ids)
    return store
