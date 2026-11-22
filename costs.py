"""Token cost model for the Insights dashboard.

Mirrors the "RAG Cost Estimation" workbook (Inference, Embedding and Cost Savings
sheets), but fills each assumption from Nova itself where it can be measured:

| Workbook assumption              | Measured from                                         |
|----------------------------------|-------------------------------------------------------|
| User query tokens (20)           | questions in the conversation log                     |
| System prompt tokens (250)       | the real grounding prompt in chain.py                 |
| PDFs / FAQs / tickets per query  | the documents actually retrieved for each answer      |
| Output tokens per query (300)    | Nova's real answers (or Groq's own usage counts)      |
| KB size for embedding            | every document stored in ChromaDB                     |
| Share of queries the bot handles | answered rate in the log                              |

Nova-specific extras the workbook leaves out: the last 4 exchanges of history sent
with each answer, and the intent-classifier call made for every message.

Tokens are estimated as words x tokens-per-word (the workbook uses 1.3). When Groq
reports real usage for an answer it is logged and shown next to the estimate.
Pure functions, no Streamlit, so everything here is unit tested.
"""
from dataclasses import dataclass

import pandas as pd

TOKENS_PER_WORD = 1.3          # workbook, Embedding!C2
CLASSIFIER_OUTPUT_TOKENS = 12  # {"intent": "requirement_inquiry"}

# Per-query input/output tokens from the workbook's Inference sheet.
WORKBOOK_PROFILE = {
    "User query": 20,
    "System prompt": 250,
    "Conversation history": 0,
    "FAQs": 180,
    "Tickets": 240,
    "Guides (PDF)": 1800,
    "Output (answer)": 300,
}
INPUT_PARTS = ["User query", "System prompt", "Conversation history", "FAQs", "Tickets", "Guides (PDF)"]
COLLECTION_PART = {"FAQ": "FAQs", "TICKETS": "Tickets", "GUIDES": "Guides (PDF)"}

# Workbook model price list, $ per 1M tokens (input, output). Editable on the page.
WORKBOOK_MODELS = pd.DataFrame([
    {"model": "GPT-6.2", "input_per_1m": 8.0, "output_per_1m": 40.0},
    {"model": "GPT-6.2 Sol", "input_per_1m": 2.0, "output_per_1m": 10.0},
    {"model": "GPT-6.1 Terra", "input_per_1m": 1.0, "output_per_1m": 5.0},
    {"model": "GPT-6 Luna", "input_per_1m": 0.1, "output_per_1m": 0.5},
])
WORKBOOK_EMBED_PRICE = 0.13     # $ per 1M tokens (workbook's paid embedding API)
WORKBOOK_QUERIES_PER_DAY = 5000
WORKBOOK_DAYS = 30
WORKBOOK_HUMAN = {"queries_per_agent_per_day": 600, "salary": 400.0, "admin": 100.0}


def words(text: str | None) -> int:
    return len((text or "").split())


# ---------------------------------------------------------------- knowledge base
def kb_documents() -> pd.DataFrame:
    """Every stored chunk with its word count: collection, ref, words."""
    from config import ensure_ingested, get_client
    from retriever import COLLECTIONS

    ensure_ingested()
    rows = []
    for name, label in COLLECTIONS.items():
        got = get_client().get_collection(name).get(include=["documents", "metadatas"])
        for text, meta in zip(got["documents"], got["metadatas"]):
            rows.append({"collection": label, "ref": (meta or {}).get("ref"), "words": words(text)})
    return pd.DataFrame(rows, columns=["collection", "ref", "words"])


def prompt_words() -> dict:
    """Fixed prompt text Nova sends: grounding prompt (without context) and classifier prompt."""
    from chain import FALLBACK, SYSTEM_PROMPT
    from intent import CLASSIFIER_PROMPT

    return {"system": words(SYSTEM_PROMPT.format(fallback=FALLBACK, context="")),
            "classifier": words(CLASSIFIER_PROMPT)}


# ---------------------------------------------------------------- measured profile
def _history_words(df: pd.DataFrame, exchanges: int = 4) -> pd.Series:
    """Words of the previous `exchanges` question/answer pairs in the same conversation."""
    out = pd.Series(0, index=df.index, dtype=float)
    for _, g in df.sort_values("ts").groupby("session_id"):
        qa = (g["question"].map(words) + g["answer"].map(words)).tolist()
        for i, idx in enumerate(g.index):
            out[idx] = sum(qa[max(0, i - exchanges):i])
    return out


@dataclass
class Measured:
    words: dict            # component -> average words per answered query
    source: dict           # component -> where the number came from
    rag_queries: int
    classifier_input_words: float
    real_input: float | None    # Groq-reported averages, when logged
    real_output: float | None
    real_count: int


def measure(df: pd.DataFrame, kb: pd.DataFrame, prompts: dict, top_k: int = 3) -> Measured:
    """Average words per RAG answer for each workbook component, from the log and the KB."""
    rag = df[df["route"] == "requirement_inquiry"] if len(df) else df
    w, src = {}, {}
    w["System prompt"], src["System prompt"] = prompts["system"], "chain.py grounding prompt"

    if len(rag):
        w["User query"], src["User query"] = rag["question"].map(words).mean(), f"{len(rag)} logged questions"
        hist = _history_words(df).loc[rag.index]
        w["Conversation history"], src["Conversation history"] = hist.mean(), "previous 4 exchanges per conversation"
    else:
        w["User query"], src["User query"] = WORKBOOK_PROFILE["User query"] / TOKENS_PER_WORD, "workbook (no questions logged)"
        w["Conversation history"], src["Conversation history"] = 0.0, "no conversations logged"

    ref_words = kb.set_index(["collection", "ref"])["words"].to_dict() if len(kb) else {}
    avg_doc = kb.groupby("collection")["words"].mean().to_dict() if len(kb) else {}
    for coll, part in COLLECTION_PART.items():
        per_query = []
        for retrieved in (rag["retrieved"] if len(rag) else []):
            hits = [ref_words.get((c, r), avg_doc.get(c, 0)) for c, r in retrieved if c == coll]
            if retrieved:
                per_query.append(sum(hits))
        if per_query:
            w[part], src[part] = sum(per_query) / len(per_query), "documents retrieved for logged answers"
        else:
            w[part], src[part] = top_k * avg_doc.get(coll, 0), f"top-{top_k} x average {coll.lower()} size"

    real_answers = rag[(rag["channel"] != "demo") & ~rag["fallback"]] if len(rag) else rag
    if len(real_answers):
        w["Output (answer)"] = real_answers["answer"].map(words).mean()
        src["Output (answer)"] = f"{len(real_answers)} real answers"
    else:
        w["Output (answer)"] = WORKBOOK_PROFILE["Output (answer)"] / TOKENS_PER_WORD
        src["Output (answer)"] = "workbook (no real answers logged yet)"

    classifier_in = prompts["classifier"] + w["User query"] + w["Conversation history"]
    usage = rag.dropna(subset=["input_tokens"]) if len(rag) and "input_tokens" in rag else rag.iloc[0:0]
    return Measured(
        words=w, source=src, rag_queries=len(rag), classifier_input_words=classifier_in,
        real_input=float(usage["input_tokens"].mean()) if len(usage) else None,
        real_output=float(usage["output_tokens"].mean()) if len(usage) else None,
        real_count=len(usage),
    )


def profile_table(m: Measured, tokens_per_word: float = TOKENS_PER_WORD) -> pd.DataFrame:
    """Side-by-side tokens per query: workbook assumption vs measured from Nova."""
    rows = [{"component": c, "workbook": WORKBOOK_PROFILE[c],
             "measured": round(m.words[c] * tokens_per_word), "source": m.source[c]}
            for c in WORKBOOK_PROFILE]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the three workbook sheets
def inference_costs(input_tokens: float, output_tokens: float, models: pd.DataFrame,
                    queries_per_month: float) -> pd.DataFrame:
    """Inference sheet: cost per query and per month for each model."""
    t = models.dropna(subset=["input_per_1m", "output_per_1m"]).copy()
    t["input_cost_per_query"] = input_tokens * t["input_per_1m"] / 1e6
    t["output_cost_per_query"] = output_tokens * t["output_per_1m"] / 1e6
    t["cost_per_query"] = t["input_cost_per_query"] + t["output_cost_per_query"]
    t["cost_per_month"] = t["cost_per_query"] * queries_per_month
    return t.reset_index(drop=True)


def embedding_costs(kb_tokens: float, query_tokens: float, queries_per_month: float,
                    price_per_1m: float, reindexes_per_month: float = 1) -> dict:
    """Embedding sheet: knowledge-base (re)indexing plus embedding every user query."""
    kb_cost = kb_tokens * reindexes_per_month * price_per_1m / 1e6
    query_total = query_tokens * queries_per_month
    query_cost = query_total * price_per_1m / 1e6
    return {"kb_tokens": kb_tokens, "kb_cost": kb_cost, "query_tokens": query_total,
            "query_cost": query_cost, "total": kb_cost + query_cost}


def savings(queries_per_day: float, days: float, per_agent_per_day: float, salary: float, admin: float,
            chatbot_cost: float, bot_share: float = 1.0) -> dict:
    """Cost Savings sheet. `bot_share` < 1 sends the unanswered share back to human agents."""
    monthly = queries_per_day * days
    per_agent_month = per_agent_per_day * days
    agents_all = monthly / per_agent_month if per_agent_month else 0.0
    per_agent_cost = salary + admin
    human_only = agents_all * per_agent_cost
    agents_left = agents_all * (1 - bot_share)
    with_bot = chatbot_cost + agents_left * per_agent_cost
    saved = human_only - with_bot
    return {"monthly_queries": monthly, "agents_without_bot": agents_all, "agents_with_bot": agents_left,
            "cost_per_agent": per_agent_cost, "human_only": human_only, "with_bot": with_bot,
            "savings": saved, "savings_pct": saved / human_only if human_only else 0.0}
