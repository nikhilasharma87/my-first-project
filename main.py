"""Interactive CLI for the NovaCell support assistant (FR-18, FR-19).

Same routing as the web app: support questions go to RAG, refund requests to the
Refund Agent (type "Yes, proceed" or "Cancel" when asked to confirm).
"""
import logging
import sys

from assistant import REFUND_REQUEST, RefundAgent, RefundSession, route
from chain import MissingApiKey, log_interaction, stream_answer, to_history
from config import ensure_ingested
from retriever import retrieve


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ensure_ingested()
    print("NovaCell support assistant (Nova). Ask a question, or type 'quit' to exit.")
    print("Type 'clear' to start a new conversation.\n")

    messages: list[dict] = []
    session = RefundSession()
    agent = RefundAgent()

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
        if question.lower() == "clear":
            messages.clear()
            session.clear()
            print("(conversation cleared)\n")
            continue

        history = to_history(messages)
        decision = route(question, session, history)

        if decision.intent == REFUND_REQUEST:
            reply = agent.handle(question, session, history)
            events = decision.events + reply.events
            print("Bot:", reply.text.replace("**", ""))
            if reply.needs_confirmation:
                print("     Reply 'Yes, proceed' to confirm or 'Cancel'.")
            print("\nSteps:\n  " + "\n  ".join(events) + "\n")
            answer_text, docs = reply.text, []
        else:
            docs = retrieve(question)
            print("Bot: ", end="", flush=True)
            parts = []
            try:
                for token in stream_answer(question, docs, history):
                    parts.append(token)
                    print(token, end="", flush=True)
            except MissingApiKey as exc:
                print(f"\n[setup] {exc}")
                break
            answer_text = "".join(parts)
            events = decision.events + [f"[RAG] retrieved {len(docs)} documents"]
            print("\n\nSources:")
            for d in docs:
                m = d.metadata
                print(f"  - {m['collection']:<7} {m['ref']:<24} (relevance {m['score']})")
            print()

        messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer_text}]
        log_interaction(question, answer_text, docs, channel="cli", route=decision.intent, events=events)


if __name__ == "__main__":
    main()
