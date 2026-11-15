"""Offline checks: ingestion counts, retrieval, chain wiring and the Streamlit UI.

The Groq model is replaced by a fake chat model so no API key or network is needed.
Run:  .venv\\Scripts\\python -m pytest tests -q   (or just: .venv\\Scripts\\python tests\\test_offline.py)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

import chain  # noqa: E402
from config import INTERACTION_LOG  # noqa: E402
from retriever import COLLECTIONS, retrieve  # noqa: E402

FAKE_ANSWER = "1. Toggle airplane mode off and on [FAQ #2]\n2. Reset your APN to default [TK-001]"


def fake_llm():
    return GenericFakeChatModel(messages=iter([AIMessage(content=FAKE_ANSWER)] * 50))


def test_collections_populated():
    from config import ensure_ingested, get_store

    ensure_ingested()

    counts = {name: get_store(name)._collection.count() for name in COLLECTIONS}
    assert counts == {"faq": 25, "tickets": 19, "guides": counts["guides"]}
    assert counts["guides"] > 0


def test_retrieval_returns_nine_labelled_docs():
    docs = retrieve("Why is my mobile internet so slow?")
    assert len(docs) == 9
    assert [d.metadata["collection"] for d in docs] == ["FAQ"] * 3 + ["TICKETS"] * 3 + ["GUIDES"] * 3
    assert docs[0].metadata["ref"] == "FAQ #2"


def test_escalated_ticket_not_ingested():
    refs = {d.metadata["ref"] for d in retrieve("Unauthorised plan change on account")}
    assert "TK-020" not in refs


def test_chain_streams_and_prompt_has_context(monkeypatch):
    seen = {}

    class Spy(GenericFakeChatModel):
        def _stream(self, messages, *a, **kw):
            seen["system"] = messages[0].content
            yield from super()._stream(messages, *a, **kw)

    monkeypatch.setattr(chain, "get_llm", lambda: Spy(messages=iter([AIMessage(content=FAKE_ANSWER)])))
    docs = retrieve("Why is my mobile internet so slow?")
    tokens = list(chain.stream_answer("Why is my mobile internet so slow?", docs))
    assert len(tokens) > 1 and "".join(tokens) == FAKE_ANSWER
    assert "[FAQ | FAQ #2]" in seen["system"] and "call **611**" in seen["system"]


def test_streamlit_app_flow(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(chain, "get_llm", fake_llm)
    before = INTERACTION_LOG.read_text(encoding="utf-8").count("\n") if INTERACTION_LOG.exists() else 0

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception
    at.chat_input[0].set_value("Why is my mobile internet so slow?").run()
    assert not at.exception
    assert any("airplane mode" in m.markdown[0].value for m in at.chat_message if m.markdown)
    assert at.expander[0].label == "Sources (9)"

    # sample-question button
    at.sidebar.button[1].click().run()
    assert len([m for m in at.chat_message if m.name == "assistant"]) == 2

    lines = INTERACTION_LOG.read_text(encoding="utf-8").splitlines()[before:]
    records = [json.loads(line) for line in lines]
    assert [r["channel"] for r in records if r["type"] == "interaction"] == ["web", "web"]
    assert len(records[0]["sources"]) == 9

    # clear conversation
    at.sidebar.button[-1].click().run()
    assert len(at.chat_message) == 0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
