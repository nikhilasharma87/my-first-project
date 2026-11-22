"""Conversation analytics for the Insights dashboard.

Reads the interaction log Nova already writes (logs/interactions.jsonl, plus the
optional, clearly labelled logs/demo_interactions.jsonl) and turns it into:
- volume and outcome metrics (conversations, messages, answered rate, feedback)
- what customers ask about (topic per message)
- knowledge-base refinement actions, each backed by evidence from the log

Pure functions over pandas DataFrames so they can be tested without Streamlit.
"""
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

from config import FAQ_CSV, INTERACTION_LOG, LOG_DIR, TICKETS_DB

DEMO_LOG = LOG_DIR / "demo_interactions.jsonl"

# Thresholds on retrieval relevance (cosine similarity 0..1, see retriever.py)
TOPIC_MIN_SCORE = 0.35      # below this the best match says little about the topic
WEAK_MATCH_SCORE = 0.45     # answered, but the best source is only loosely related
FAQ_COVERED_SCORE = 0.45    # an FAQ at or above this covers the question

TOPIC_NAMES = {
    "data": "Mobile data",
    "connectivity": "Network & signal",
    "voice": "Calls",
    "sim": "SIM & eSIM",
    "billing": "Billing",
    "roaming": "Roaming",
    "device": "Devices",
    "account": "Account & app",
}
REFUNDS = "Refunds"
OTHER = "Other / unclear"

_PLACEHOLDER_PATTERNS = {
    "Old brand name 'MyTelecom'": re.compile(r"MyTelecom", re.I),
    "Placeholder web address (example.com)": re.compile(r"example\.com", re.I),
}


# ---------------------------------------------------------------- knowledge base catalogue

def load_catalogue() -> pd.DataFrame:
    """Every FAQ and ticket document with its category and text (for topic lookup and
    content checks)."""
    faq = pd.read_csv(FAQ_CSV).fillna("")
    rows = [
        {"ref": f"FAQ #{r.id}", "collection": "FAQ", "category": r.category,
         "title": r.question, "text": f"{r.question} {r.answer}"}
        for r in faq.itertuples(index=False)
    ]
    con = sqlite3.connect(TICKETS_DB)
    try:
        for tid, cat, issue, desc, res, status in con.execute(
            "SELECT ticket_id, category, issue_type, description, resolution, status FROM tickets"
        ):
            if str(status).lower() == "resolved":
                rows.append({"ref": tid, "collection": "TICKETS", "category": cat,
                             "title": issue, "text": f"{issue} {desc} {res}"})
    finally:
        con.close()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- loading the log

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a half-written line should not break the dashboard
    return out


def load_log(include_demo: bool = True, paths: list[Path] | None = None) -> pd.DataFrame:
    """One row per assistant answer, with feedback, topic and retrieval signals attached."""
    if paths is None:
        paths = [INTERACTION_LOG] + ([DEMO_LOG] if include_demo else [])
    records = [r for p in paths for r in _read_jsonl(p)]
    interactions = [r for r in records if r.get("type") == "interaction"]
    if not interactions:
        return pd.DataFrame()

    feedback = {}
    for r in records:
        if r.get("type") == "feedback":
            feedback[r["id"]] = r.get("rating")  # latest click wins

    catalogue = load_catalogue().set_index("ref")["category"].to_dict()
    rows = []
    for r in interactions:
        sources = r.get("sources") or []
        best = _best(sources, ("FAQ", "TICKETS"))
        best_faq = _best(sources, ("FAQ",))
        best_ticket = _best(sources, ("TICKETS",))
        route = r.get("route") or "requirement_inquiry"
        rows.append({
            "id": r["id"],
            "session_id": r.get("session_id") or r["id"],
            "ts": pd.to_datetime(r.get("ts"), utc=True, errors="coerce"),
            "channel": r.get("channel", "web"),
            "route": route,
            "question": r.get("question", ""),
            "answer": r.get("answer", ""),
            "fallback": bool(r.get("fallback")) and route != "refund_request",
            "events": r.get("events") or [],
            "best_ref": best["ref"] if best else None,
            "best_score": best["score"] if best else None,
            "best_faq_ref": best_faq["ref"] if best_faq else None,
            "best_faq_score": best_faq["score"] if best_faq else None,
            "best_ticket_ref": best_ticket["ref"] if best_ticket else None,
            "best_ticket_score": best_ticket["score"] if best_ticket else None,
            "retrieved_refs": [s["ref"] for s in sources],
            "retrieved": [(s.get("collection"), s["ref"]) for s in sources],
            "input_tokens": (r.get("usage") or {}).get("input_tokens"),
            "output_tokens": (r.get("usage") or {}).get("output_tokens"),
            "rating": feedback.get(r["id"]),
            "topic": _topic(route, best, catalogue),
            "refund_outcome": refund_outcome(r.get("events") or []) if route == "refund_request" else None,
        })
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    df["date"] = df["ts"].dt.tz_convert(None).dt.date
    df["question_key"] = df["question"].map(normalise_question)
    return df


def _best(sources: list[dict], collections: tuple) -> dict | None:
    pool = [s for s in sources if s.get("collection") in collections and s.get("score") is not None]
    return max(pool, key=lambda s: s["score"]) if pool else None


def _topic(route: str, best: dict | None, catalogue: dict) -> str:
    if route == "refund_request":
        return REFUNDS
    if not best or best["score"] < TOPIC_MIN_SCORE:
        return OTHER
    return TOPIC_NAMES.get(catalogue.get(best["ref"], ""), OTHER)


def typical(values: pd.Series) -> str:
    """Most frequent value; ties go to the one seen first (stable, readable labels)."""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def normalise_question(q: str) -> str:
    """Group trivially different phrasings: case, punctuation, brand name, extra spaces."""
    q = q.lower().replace("mytelecom", "novacell")
    q = re.sub(r"[^\w₹ ]+", " ", q)
    return " ".join(q.split())


def refund_outcome(events: list[str]) -> str:
    text = "\n".join(events)
    if "[Refund] success" in text:
        return "Refunded"
    if "[Support] already submitted" in text:
        return "Already with Support"
    if "[Support] submitted" in text:
        return "Sent to Support Team"
    if "[User] cancelled" in text:
        return "Cancelled by customer"
    if "[Validation] eligible=False" in text or "no eligible recharge" in text:
        return "Not eligible"
    if "awaiting_user_confirmation" in text or "ambiguous confirmation" in text:
        return "Awaiting confirmation"
    if "awaiting_amount" in text:
        return "Asked for amount"
    return "Other"


def refund_conversations(df: pd.DataFrame) -> pd.Series:
    """Final outcome of each refund conversation (the last refund message decides it)."""
    refunds = df[df["route"] == "refund_request"].sort_values("ts")
    if refunds.empty:
        return pd.Series(dtype="int64")
    return refunds.groupby("session_id")["refund_outcome"].last().value_counts()


def rejection_reasons(df: pd.DataFrame) -> pd.Series:
    reasons = []
    for events in df.loc[df["route"] == "refund_request", "events"]:
        for e in events:
            m = re.match(r"\[Validation\] eligible=False \((.+)\)", e)
            if m:
                reasons.append(m.group(1))
    return pd.Series(reasons, dtype="object").value_counts()


# ---------------------------------------------------------------- metrics

def filter_log(df: pd.DataFrame, days: int | None, channels: list[str] | None) -> pd.DataFrame:
    if df.empty:
        return df
    out = df
    if days:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        out = out[out["ts"] >= cutoff]
    if channels:
        out = out[out["channel"].isin(channels)]
    return out


def kpis(df: pd.DataFrame) -> dict:
    rag = df[df["route"] != "refund_request"]
    rated = df["rating"].dropna()
    return {
        "conversations": df["session_id"].nunique(),
        "messages": len(df),
        "answered_rate": (1 - rag["fallback"].mean()) if len(rag) else None,
        "unanswered": int(rag["fallback"].sum()),
        "refund_requests": int((df["route"] == "refund_request").sum()),
        "helpful_rate": (rated.eq("up").mean()) if len(rated) else None,
        "ratings": len(rated),
    }


def by_topic(df: pd.DataFrame) -> pd.DataFrame:
    """Messages per topic split into answered / not answered, plus feedback."""
    g = df.groupby("topic")
    out = pd.DataFrame({
        "messages": g.size(),
        "conversations": g["session_id"].nunique(),
        "not_answered": g["fallback"].sum().astype(int),
        "thumbs_down": g["rating"].apply(lambda s: int((s == "down").sum())),
        "avg_match": g["best_score"].mean().round(2),
    })
    out["answered"] = out["messages"] - out["not_answered"]
    out["answered_pct"] = (out["answered"] / out["messages"]).round(3)
    return out.sort_values("messages", ascending=False).reset_index()


def daily_volume(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "messages"])
    s = df.groupby("date").size()
    full = pd.date_range(min(s.index), max(s.index), freq="D").date
    return s.reindex(full, fill_value=0).rename_axis("date").reset_index(name="messages")


def top_questions(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    g = df.groupby("question_key")
    out = pd.DataFrame({
        "question": g["question"].agg(typical),  # most common wording
        "times_asked": g.size(),
        "topic": g["topic"].agg(typical),
        "answered_pct": (1 - g["fallback"].mean()).round(3),
        "thumbs_down": g["rating"].apply(lambda s: int((s == "down").sum())),
    })
    return out.sort_values(["times_asked", "thumbs_down"], ascending=False).head(n).reset_index(drop=True)


# ---------------------------------------------------------------- knowledge-base refinement

def recommendations(df: pd.DataFrame, catalogue: pd.DataFrame | None = None) -> pd.DataFrame:
    """Concrete knowledge-base actions, each with the evidence and how many customer
    messages it affects. Sorted by impact."""
    catalogue = load_catalogue() if catalogue is None else catalogue
    titles = catalogue.set_index("ref")["title"].to_dict()
    recs = []

    rag = df[df["route"] != "refund_request"]

    # 1. Content gaps: Nova had no verified answer.
    for key, grp in rag[rag["fallback"]].groupby("question_key"):
        recs.append({
            "priority": 1, "action": "Write a new FAQ",
            "item": typical(grp["question"]),
            "evidence": "Nova had no verified answer and sent the customer to 611",
            "messages": len(grp), "topic": typical(grp["topic"]),
        })

    answered = rag[~rag["fallback"]]

    # 2. Answered only from a one-off support ticket: promote it to a proper FAQ.
    ticket_only = answered[
        (answered["best_faq_score"].fillna(0) < FAQ_COVERED_SCORE)
        & (answered["best_ticket_score"].fillna(0) >= FAQ_COVERED_SCORE)
    ]
    for ref, grp in ticket_only.groupby("best_ticket_ref"):
        recs.append({
            "priority": 2, "action": "Turn a support ticket into an FAQ",
            "item": f"{ref}: {titles.get(ref, '')}",
            "evidence": "Customers ask this, but only a single resolved ticket covers it",
            "messages": len(grp), "topic": typical(grp["topic"]),
        })

    # 3. Weak matches: answered, but the best source is only loosely related.
    weak = answered[answered["best_score"].fillna(0) < WEAK_MATCH_SCORE]
    weak = weak[~weak.index.isin(ticket_only.index)]
    for key, grp in weak.groupby("question_key"):
        recs.append({
            "priority": 3, "action": "Strengthen existing content",
            "item": typical(grp["question"]),
            "evidence": f"Best match only {grp['best_score'].mean():.2f} relevance "
                        f"({typical(grp['best_ref'].dropna())})",
            "messages": len(grp), "topic": typical(grp["topic"]),
        })

    # 4. Answers customers rated unhelpful (gaps are already listed under action 1).
    for key, grp in df[(df["rating"] == "down") & ~df["fallback"]].groupby("question_key"):
        recs.append({
            "priority": 2, "action": "Review an unhelpful answer",
            "item": typical(grp["question"]),
            "evidence": f"{len(grp)} thumbs-down; sources used: {typical(grp['best_ref'].dropna()) if grp['best_ref'].notna().any() else 'none'}",
            "messages": len(grp), "topic": typical(grp["topic"]),
        })

    # 5. Content hygiene: outdated brand names and placeholder addresses in the sources.
    for label, pattern in _PLACEHOLDER_PATTERNS.items():
        hits = catalogue[catalogue["text"].str.contains(pattern)]
        if len(hits):
            used = df["retrieved_refs"].explode()
            affected = int(used.isin(hits["ref"]).groupby(level=0).any().sum()) if len(used) else 0
            recs.append({
                "priority": 3, "action": "Fix outdated wording",
                "item": f"{label} in {len(hits)} documents",
                "evidence": ", ".join(hits["ref"].head(6)) + (" …" if len(hits) > 6 else ""),
                "messages": affected, "topic": "All",
            })

    # 6. Content nobody needed in this period.
    retrieved = set(df["retrieved_refs"].explode().dropna())
    unused = catalogue[~catalogue["ref"].isin(retrieved)]
    if len(df) >= 20 and len(unused):
        recs.append({
            "priority": 4, "action": "Review unused content",
            "item": f"{len(unused)} of {len(catalogue)} FAQs/tickets were never retrieved",
            "evidence": ", ".join(unused["ref"].head(8)) + (" …" if len(unused) > 8 else ""),
            "messages": 0, "topic": "All",
        })

    out = pd.DataFrame(recs, columns=["priority", "action", "item", "evidence", "messages", "topic"])
    return out.sort_values(["priority", "messages"], ascending=[True, False]).reset_index(drop=True)
