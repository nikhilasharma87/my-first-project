"""Merged retriever over all knowledge collections (FR-06 to FR-08).

To add a knowledge source (NFR-06): write an ingest_<name>.py that fills a
Chroma collection, then register it in COLLECTIONS below.
"""
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableParallel

from config import TOP_K, ensure_ingested, get_store, warm_stores

# collection name -> label shown to the LLM and in the Sources panel
COLLECTIONS = {
    "faq": "FAQ",
    "tickets": "TICKETS",
    "guides": "GUIDES",
}


def _search(collection: str, label: str):
    def run(question: str) -> list[Document]:
        hits = get_store(collection).similarity_search_with_score(question, k=TOP_K)
        docs = []
        for doc, distance in hits:  # cosine distance -> similarity in [0, 1]
            score = min(1.0, max(0.0, 1.0 - float(distance)))
            doc.metadata = {**doc.metadata, "collection": label, "score": round(score, 3)}
            docs.append(doc)
        return docs

    return RunnableLambda(run, name=f"retrieve_{collection}")


# Queries all collections in parallel: TOP_K docs each, 9 in total.
merged_retriever = RunnableParallel(
    {name: _search(name, label) for name, label in COLLECTIONS.items()}
) | RunnableLambda(lambda by_source: [d for docs in by_source.values() for d in docs], name="merge")


_warmed = False


def retrieve(question: str) -> list[Document]:
    global _warmed
    if not _warmed:
        ensure_ingested()          # fresh deploy: build chroma_store/ from data/
        warm_stores(COLLECTIONS)   # open stores serially before the parallel fan-out
        _warmed = True
    return merged_retriever.invoke(question)


def format_context(docs: list[Document]) -> str:
    """Source-labelled context block for the prompt (FR-08)."""
    blocks = []
    for doc in docs:
        meta = doc.metadata
        blocks.append(f"[{meta['collection']} | {meta['ref']}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)
