"""Token cost model: reproduces the RAG Cost Estimation workbook, then checks the measured profile."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import costs as C  # noqa: E402
import insights as I  # noqa: E402


def test_inference_sheet_matches_workbook():
    inp = sum(C.WORKBOOK_PROFILE[c] for c in C.INPUT_PARTS)
    assert inp == 2490 and C.WORKBOOK_PROFILE["Output (answer)"] == 300
    t = C.inference_costs(inp, 300, C.WORKBOOK_MODELS, 5000 * 30).set_index("model")
    assert t.loc["GPT-6.2", "cost_per_query"] == pytest.approx(0.03192)
    assert t.loc["GPT-6.2", "cost_per_month"] == pytest.approx(4788)
    assert t.loc["GPT-6.2 Sol", "cost_per_month"] == pytest.approx(1197)
    assert t.loc["GPT-6.1 Terra", "cost_per_month"] == pytest.approx(598.5)
    assert t.loc["GPT-6 Luna", "cost_per_month"] == pytest.approx(59.85)


def test_embedding_sheet_matches_workbook():
    e = C.embedding_costs(5_551_000, 20, 150_000, 0.13)
    assert e["kb_cost"] == pytest.approx(0.72163)
    assert e["query_cost"] == pytest.approx(0.39)
    assert round(e["total"], 2) == 1.11


def test_savings_sheet_matches_workbook():
    s = C.savings(5000, 30, 600, 400, 100, chatbot_cost=1197 + 1.11)
    assert s["agents_without_bot"] == pytest.approx(8.3333, rel=1e-4)
    assert s["human_only"] == pytest.approx(4166.6667, rel=1e-6)
    assert s["savings"] == pytest.approx(2968.56, abs=0.01)
    assert round(s["savings_pct"], 2) == 0.71


def test_unanswered_share_goes_back_to_agents():
    s = C.savings(5000, 30, 600, 400, 100, chatbot_cost=1000, bot_share=0.8)
    assert s["agents_with_bot"] == pytest.approx(8.3333 * 0.2, rel=1e-4)
    assert s["with_bot"] == pytest.approx(1000 + 4166.6667 * 0.2, rel=1e-6)


def _log(tmp_path):
    import json
    rows = [
        {"type": "interaction", "id": "a", "session_id": "s", "ts": "2026-11-20T10:00:00+00:00", "channel": "web",
         "route": "requirement_inquiry", "question": "why is my internet slow", "answer": "one two three four",
         "fallback": False, "sources": [{"collection": "FAQ", "ref": "FAQ #2", "score": .8},
                                        {"collection": "TICKETS", "ref": "TK-008", "score": .6}],
         "usage": {"input_tokens": 1500, "output_tokens": 90}},
        {"type": "interaction", "id": "b", "session_id": "s", "ts": "2026-11-20T10:01:00+00:00", "channel": "web",
         "route": "requirement_inquiry", "question": "and abroad?", "answer": "five six",
         "fallback": False, "sources": [{"collection": "FAQ", "ref": "FAQ #2", "score": .7}]},
    ]
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return I.load_log(paths=[p])


def test_measured_profile(tmp_path):
    df = _log(tmp_path)
    kb = pd.DataFrame([{"collection": "FAQ", "ref": "FAQ #2", "words": 50},
                       {"collection": "TICKETS", "ref": "TK-008", "words": 70},
                       {"collection": "GUIDES", "ref": "g1", "words": 90}])
    m = C.measure(df, kb, {"system": 300, "classifier": 100})
    assert m.words["User query"] == pytest.approx((5 + 2) / 2)
    assert m.words["FAQs"] == 50 and m.words["Tickets"] == 35       # second answer retrieved no ticket
    assert m.words["Guides (PDF)"] == 0                               # none retrieved in the log
    assert m.words["Conversation history"] == pytest.approx((0 + 9) / 2)  # 5 + 4 words of the first exchange
    assert m.words["Output (answer)"] == pytest.approx(3)
    assert m.real_count == 1 and m.real_input == 1500
    table = C.profile_table(m, 2.0).set_index("component")
    assert table.loc["System prompt", "measured"] == 600 and table.loc["FAQs", "workbook"] == 180


def test_measure_without_log():
    kb = pd.DataFrame([{"collection": "FAQ", "ref": "FAQ #1", "words": 40}])
    m = C.measure(pd.DataFrame(), kb, {"system": 300, "classifier": 100})
    assert m.words["FAQs"] == 120 and m.rag_queries == 0 and m.real_count == 0


def test_real_knowledge_base_and_prompts():
    kb = C.kb_documents()
    assert set(kb["collection"]) == {"FAQ", "TICKETS", "GUIDES"}
    assert (kb["collection"] == "FAQ").sum() == 25
    p = C.prompt_words()
    assert 200 < p["system"] < 600 and p["classifier"] > 50
