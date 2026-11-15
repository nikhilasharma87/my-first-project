"""Interactive CLI for the telecom support assistant (FR-18, FR-19)."""
import sys

from chain import MissingApiKey, log_interaction, stream_answer
from config import ensure_ingested
from retriever import retrieve


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_ingested()
    print("NovaCell support assistant (Nova). Ask a question, or type 'quit' to exit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() == "quit":
            break

        docs = retrieve(question)
        print("Bot: ", end="", flush=True)
        parts = []
        try:
            for token in stream_answer(question, docs):
                parts.append(token)
                print(token, end="", flush=True)
        except MissingApiKey as exc:
            print(f"\n[setup] {exc}")
            break
        print("\n\nSources:")
        for d in docs:
            m = d.metadata
            print(f"  - {m['collection']:<7} {m['ref']:<24} (relevance {m['score']})")
        print()
        log_interaction(question, "".join(parts), docs, channel="cli")


if __name__ == "__main__":
    main()
