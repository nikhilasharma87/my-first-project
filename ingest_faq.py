"""Load data/faq.csv into the `faq` collection: one row = one document (FR-14)."""
import pandas as pd
from langchain_core.documents import Document

from config import FAQ_CSV, rebuild_collection

COLLECTION = "faq"


def load_docs():
    df = pd.read_csv(FAQ_CSV).fillna("")
    docs, ids = [], []
    for row in df.itertuples(index=False):
        docs.append(
            Document(
                page_content=f"Q: {row.question}\nA: {row.answer}",
                metadata={
                    "source": "FAQ",
                    "ref": f"FAQ #{row.id}",
                    "title": str(row.question),
                    "category": str(row.category),
                },
            )
        )
        ids.append(f"faq-{row.id}")
    return docs, ids


def main():
    docs, ids = load_docs()
    rebuild_collection(COLLECTION, docs, ids)
    print(f"[faq] ingested {len(docs)} FAQ entries")


if __name__ == "__main__":
    main()
