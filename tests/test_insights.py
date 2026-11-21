"""Insights dashboard: analytics over a small hand-written log, plus a smoke run of the page."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import insights as I  # noqa: E402

NOW = datetime.now(timezone.utc)


def rec(q, *, sid, sources, fallback=False, route="requirement_inquiry", events=None, days_ago=1, rid=None):
    return {"type": "interaction", "id": rid or f"id_{abs(hash((q, sid, days_ago))) % 10**8}", "session_id": sid,
            "ts": (NOW - timedelta(days=days_ago)).isoformat(), "channel": "web", "route": route,
            "events": events or [], "question": q, "answer": "a", "fallback": fallback,
            "sources": [{"collection": c, "ref": r, "score": s} for c, r, s in sources]}


@pytest.fixture
def log_path(tmp_path):
    rows = [
        rec("Why is my internet slow?", sid="s1", sources=[("FAQ", "FAQ #2", 0.82), ("TICKETS", "TK-008", 0.64)], rid="a1"),
        rec("why is my internet slow", sid="s2", sources=[("FAQ", "FAQ #2", 0.80), ("TICKETS", "TK-008", 0.60)], rid="a2"),
        rec("I can't log in to the NovaCell app.", sid="s3",
            sources=[("FAQ", "FAQ #25", 0.25), ("TICKETS", "TK-013", 0.45)], rid="a3"),
        rec("Do you have family plans?", sid="s4", sources=[("FAQ", "FAQ #15", 0.30)], fallback=True, rid="a4"),
        rec("Do you have family plans", sid="s5", sources=[("FAQ", "FAQ #15", 0.31)], fallback=True, rid="a5"),
        rec("I want a refund of ₹499.", sid="s6", sources=[], route="refund_request",
            events=["[Agent] awaiting_user_confirmation"], rid="a6"),
        rec("Yes, proceed", sid="s6", sources=[], route="refund_request",
            events=["[User] confirmed", "[Refund] success"], rid="a7"),
        rec("Give me a refund of ₹2,000.", sid="s7", sources=[], route="refund_request",
            events=["[Validation] eligible=False (The requested amount exceeds the recharge amount.)"], rid="a8"),
        rec("Old question", sid="s8", sources=[("FAQ", "FAQ #2", 0.8)], days_ago=40, rid="a9"),
        {"type": "feedback", "id": "a3", "rating": "down"},
        {"type": "feedback", "id": "a1", "rating": "up"},
        {"type": "feedback", "id": "a1", "rating": "down"},   # latest click wins
    ]
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n{broken line\n", encoding="utf-8")
    return p


def test_load_and_kpis(log_path):
    df = I.filter_log(I.load_log(paths=[log_path]), days=14, channels=None)
    k = I.kpis(df)
    assert k["messages"] == 8 and k["conversations"] == 7          # 40-day-old row filtered out
    assert k["refund_requests"] == 3 and k["unanswered"] == 2
    assert k["answered_rate"] == pytest.approx(3 / 5)
    assert k["helpful_rate"] == 0.0 and k["ratings"] == 2            # a1 changed to down


def test_topics(log_path):
    df = I.load_log(paths=[log_path])
    topics = I.by_topic(df).set_index("topic")
    assert topics.loc["Refunds", "messages"] == 3
    assert topics.loc["Mobile data", "messages"] >= 2               # FAQ #2 is a 'data' FAQ
    assert topics.loc["Account & app", "messages"] == 1             # via TK-013
    assert topics.loc["Other / unclear", "not_answered"] == 2       # family plans: weak match


def test_top_questions_groups_phrasings(log_path):
    tq = I.top_questions(I.filter_log(I.load_log(paths=[log_path]), 14, None))
    first = tq.iloc[0]
    assert first["times_asked"] == 2 and "slow" in first["question"]


def test_refund_outcomes_per_conversation(log_path):
    df = I.load_log(paths=[log_path])
    out = I.refund_conversations(df)
    assert out.to_dict() == {"Refunded": 1, "Not eligible": 1}
    assert I.rejection_reasons(df).index[0] == "The requested amount exceeds the recharge amount."


def test_recommendations(log_path):
    df = I.filter_log(I.load_log(paths=[log_path]), 14, None)
    recs = I.recommendations(df)
    by_action = recs.groupby("action")["item"].apply(list).to_dict()
    assert by_action["Write a new FAQ"] == ["Do you have family plans?"]
    assert any(i.startswith("TK-013") for i in by_action["Turn a support ticket into an FAQ"])
    assert "I can't log in to the NovaCell app." in by_action["Review an unhelpful answer"]
    assert any("MyTelecom" in i for i in by_action["Fix outdated wording"])
    assert recs.iloc[0]["action"] == "Write a new FAQ"               # gaps first
    gap = recs[recs["action"] == "Write a new FAQ"].iloc[0]
    assert gap["messages"] == 2


def test_empty_log(tmp_path):
    assert I.load_log(paths=[tmp_path / "missing.jsonl"]).empty


def test_dashboard_page_renders(monkeypatch, log_path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(I, "INTERACTION_LOG", log_path, raising=False)
    import config
    monkeypatch.setattr(config, "INTERACTION_LOG", log_path)
    monkeypatch.setattr(I, "DEMO_LOG", log_path.parent / "no_demo.jsonl")

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()   # dashboard is a page of the app
    at.switch_page("pages/dashboard.py").run()
    assert not at.exception, at.exception
    text = " ".join(h.value for h in at.get("html"))
    assert "Conversations" in text and "Write a new FAQ" in text
