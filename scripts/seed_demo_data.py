"""Create clearly labelled DEMO conversation data for the Insights dashboard.

Writes logs/demo_interactions.jsonl (separate from the real log, every record has
channel="demo"). Questions go through the real retriever and refund requests through
the real Refund Agent (offline mode, no API calls), so sources, relevance scores and
refund outcomes are genuine; only the answer text is a placeholder.

Run:    .venv\\Scripts\\python scripts\\seed_demo_data.py
Remove: delete logs/demo_interactions.jsonl (or untick "Include demo data").
"""
import json
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ensure_ingested  # noqa: E402
from insights import DEMO_LOG  # noqa: E402
from refund_agent import RefundAgent, RefundSession  # noqa: E402
from retriever import retrieve  # noqa: E402

# (question, weight, answered?) — unanswered ones are real gaps in the sample knowledge base
QUESTIONS = [
    ("Why is my mobile internet so slow?", 14, True),
    ("My 4G is really slow in the evenings", 6, True),
    ("Data not working after recharge", 5, True),
    ("Why is my data balance not updating?", 4, True),
    ("How do I check my current data balance?", 9, True),
    ("I have no signal at home", 7, True),
    ("Is there a network outage in my area?", 6, True),
    ("My calls keep dropping. What should I do?", 8, True),
    ("There is an echo on every call", 3, True),
    ("How do I enable Wi-Fi calling?", 4, True),
    ("I was charged twice for my monthly plan. What should I do?", 7, True),
    ("Why is my bill higher than usual?", 6, True),
    ("I can't download my itemised bill", 3, True),
    ("My eSIM won't activate. How do I fix it?", 7, True),
    ("SIM card not detected after restart", 4, True),
    ("How do I set up roaming before I travel?", 9, True),
    ("No service while abroad", 3, True),
    ("I can't log in to the NovaCell app.", 8, True),
    ("My phone doesn't support VoLTE", 2, True),
    ("Someone changed my plan without permission", 2, True),
    # gaps: not covered by the sample FAQ / tickets / guide
    ("Do you have family plans?", 5, False),
    ("Is 5G available in Pune?", 4, False),
    ("How do I change my plan to a cheaper one?", 4, False),
    ("What are the call rates to the UK?", 3, False),
    ("Can I get a duplicate bill by post?", 2, False),
    ("How do I transfer my number to another person?", 2, False),
]

# refund conversations: list of customer turns (run through the real agent)
REFUND_FLOWS = [
    (["I want a refund of ₹499.", "Yes, proceed"], 6),
    (["I want a refund of ₹999."], 5),
    (["Give me a refund of ₹2,000."], 2),
    (["I want a refund.", "499", "Yes, proceed"], 3),
    (["Refund my ₹499 recharge", "Cancel"], 2),
    (["I want a refund of ₹499.", "maybe"], 1),
]

DAYS = 14
MESSAGES_PER_DAY = (8, 16)


def record(question, answer, docs, ts, session_id, route="requirement_inquiry", events=None, fallback=False):
    return {
        "type": "interaction", "id": uuid.uuid4().hex[:12], "session_id": session_id,
        "ts": ts.isoformat(timespec="seconds"), "channel": "demo", "route": route,
        "events": events or [], "question": question, "answer": answer, "fallback": fallback,
        "sources": [{"collection": d.metadata["collection"], "ref": d.metadata["ref"],
                     "score": d.metadata["score"]} for d in docs],
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_ingested()
    rng = random.Random(42)
    agent = RefundAgent(llm_factory=False)
    cache = {q: retrieve(q) for q, _, _ in QUESTIONS}
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    out = []
    for day in range(DAYS, 0, -1):
        base = now - timedelta(days=day)
        budget = rng.randint(*MESSAGES_PER_DAY)
        while budget > 0:
            sid = uuid.uuid4().hex[:12]
            ts = base + timedelta(hours=rng.randint(7, 22), minutes=rng.randint(0, 59))
            if rng.random() < 0.15:  # a refund conversation
                turns = rng.choices([f for f, _ in REFUND_FLOWS], [w for _, w in REFUND_FLOWS])[0]
                session = RefundSession()
                for text in turns:
                    reply = agent.handle(text, session, [])
                    out.append(record(text, reply.text, [], ts, sid, route="refund_request",
                                      events=["[Intent] refund_request"] + reply.events))
                    ts += timedelta(seconds=rng.randint(20, 90))
                budget -= len(turns)
                continue
            for _ in range(rng.choice([1, 1, 1, 2, 2, 3])):  # 1-3 questions per conversation
                q, _, answered = rng.choices(QUESTIONS, [w for _, w, _ in QUESTIONS])[0]
                docs = cache[q]
                ev = ["[Intent] requirement_inquiry", f"[RAG] retrieved {len(docs)} documents"]
                answer = "(demo data: placeholder answer)" if answered else (
                    "I don't have verified information on that. Please call 611 or use the NovaCell app.")
                rec = record(q, answer, docs, ts, sid, events=ev, fallback=not answered)
                out.append(rec)
                # ~40% of answers get rated; weak or missing answers are rated down more often
                if rng.random() < 0.4:
                    best = max((d.metadata["score"] for d in docs), default=0)
                    p_up = 0.15 if not answered else (0.9 if best >= 0.55 else 0.6)
                    out.append({"type": "feedback", "id": rec["id"], "ts": rec["ts"],
                                "rating": "up" if rng.random() < p_up else "down"})
                ts += timedelta(seconds=rng.randint(30, 120))
                budget -= 1

    DEMO_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DEMO_LOG.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = sum(r["type"] == "interaction" for r in out)
    print(f"Wrote {n} demo messages ({len(out) - n} ratings) over {DAYS} days to {DEMO_LOG}")


if __name__ == "__main__":
    main()
