"""Load resolved tickets from data/tickets.db into the `tickets` collection (FR-15).

Only tickets with status = 'resolved' are used; escalated/open tickets have no
verified resolution to ground an answer in.
"""
import sqlite3

from langchain_core.documents import Document

from config import TICKETS_DB, rebuild_collection

COLLECTION = "tickets"


def load_docs():
    con = sqlite3.connect(TICKETS_DB)
    try:
        rows = con.execute(
            "SELECT ticket_id, category, issue_type, description, resolution "
            "FROM tickets WHERE lower(status) = 'resolved' ORDER BY id"
        ).fetchall()
    finally:
        con.close()

    docs, ids = [], []
    for ticket_id, category, issue_type, description, resolution in rows:
        docs.append(
            Document(
                page_content=(
                    f"Issue: {issue_type}\n"
                    f"Customer report: {description}\n"
                    f"Resolution: {resolution}"
                ),
                metadata={
                    "source": "TICKET",
                    "ref": ticket_id,
                    "title": issue_type,
                    "category": category,
                },
            )
        )
        ids.append(f"ticket-{ticket_id}")
    return docs, ids


def main():
    docs, ids = load_docs()
    rebuild_collection(COLLECTION, docs, ids)
    print(f"[tickets] ingested {len(docs)} resolved tickets")


if __name__ == "__main__":
    main()
