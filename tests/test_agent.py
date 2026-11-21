"""Refund agent tests: the seven agent_PRD demo scenarios plus guardrails.

No API key or network: intent detection uses a fake classifier and the agent runs
either deterministically or with a scripted tool-calling fake model.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage  # noqa: E402

import refund_tools  # noqa: E402
from assistant import REFUND_REQUEST, REQUIREMENT_INQUIRY, route  # noqa: E402
from intent import heuristic_intent  # noqa: E402
from refund_agent import RefundAgent, RefundSession, RefundState, parse_amounts, parse_confirmation  # noqa: E402


def heuristic_classifier(message, history):
    return heuristic_intent(message), "fallback"


@pytest.fixture
def session():
    return RefundSession()


@pytest.fixture
def agent():
    return RefundAgent(llm_factory=False)  # deterministic mode


def turn(agent, session, text):
    decision = route(text, session, [], classifier=heuristic_classifier)
    assert decision.intent == REFUND_REQUEST, f"{text!r} routed to {decision.intent}"
    return agent.handle(text, session, [])


def recharge(session, rid):
    return next(r for r in session.store.users[session.user_id]["recharges"] if r["recharge_id"] == rid)


@pytest.fixture
def spy_process(monkeypatch):
    calls = []
    real = refund_tools.process_refund

    def spy(*a, **kw):
        calls.append(a)
        return real(*a, **kw)

    import refund_agent
    monkeypatch.setattr(refund_agent, "process_refund", spy)
    return calls


# ---------------------------------------------------------------- scenarios

def test_scenario1_successful_refund_after_confirmation(agent, session, spy_process):
    reply = turn(agent, session, "I want a refund of ₹499.")
    assert reply.needs_confirmation and session.state == RefundState.AWAITING_CONFIRMATION
    assert spy_process == []                                 # original request is not confirmation
    assert recharge(session, "rch_457")["refund_status"] == "not_refunded"

    reply = turn(agent, session, "Yes, proceed")
    assert session.state == RefundState.REFUND_PROCESSED
    assert len(spy_process) == 1
    assert "ref_" in reply.text and "₹499" in reply.text
    assert recharge(session, "rch_457")["refund_status"] == "refunded"
    assert recharge(session, "rch_456")["refund_status"] == "not_refunded"   # other recharge untouched
    assert reply.events == ["[User] confirmed", "[Agent] process_refund(amount=499)", "[Refund] success"]


def test_scenario2_amount_exceeds_recharge(agent, session, spy_process):
    reply = turn(agent, session, "Give me a refund of ₹2,000.")
    assert session.state == RefundState.REFUND_REJECTED
    assert "exceeds the recharge amount" in reply.text
    assert spy_process == [] and not session.store.support_requests


def test_scenario3_recharge_too_old(agent, session, spy_process):
    recharge(session, "rch_457")["recharge_age_days"] = 15
    reply = turn(agent, session, "I want a refund of ₹499")
    assert "7-day refund window" in reply.text
    assert spy_process == []


def test_scenario4_already_refunded(agent, session, spy_process):
    recharge(session, "rch_457")["refund_status"] = "refunded"
    reply = turn(agent, session, "I want a refund of ₹499")
    assert "already been refunded" in reply.text
    assert spy_process == []


def test_scenario5_rag_query_routes_to_rag(session):
    decision = route("Why is my mobile internet slow?", session, [], classifier=heuristic_classifier)
    assert decision.intent == REQUIREMENT_INQUIRY


def test_scenario6_above_limit_goes_to_support(agent, session, spy_process):
    reply = turn(agent, session, "I want a refund of ₹999.")
    assert session.state == RefundState.REFUND_SUBMITTED_FOR_REVIEW
    assert not reply.needs_confirmation
    assert "submitted to our Support Team" in reply.text
    assert "process it or take the necessary action" in reply.text
    for word in ("approved", "rejected", "processed", "refunded"):
        assert word not in reply.text.lower().replace("process it", "")
    assert spy_process == []
    assert recharge(session, "rch_456")["refund_status"] == "not_refunded"
    assert "[Agent] submit_to_support(amount=999)" in reply.events

    again = turn(agent, session, "I want a refund of ₹999.")
    assert "already submitted" in again.text and len(session.store.support_requests) == 1


def test_scenario7_follow_up_gets_history(monkeypatch):
    import chain
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    seen = {}

    class Spy(GenericFakeChatModel):
        def _stream(self, messages, *a, **kw):
            seen["messages"] = messages
            yield from super()._stream(messages, *a, **kw)

    monkeypatch.setattr(chain, "get_llm", lambda **kw: Spy(messages=iter([AIMessage(content="ok")])))
    history = chain.to_history([
        {"role": "user", "content": "How do I activate international roaming?"},
        {"role": "assistant", "content": "Go to Plan & Services > International Roaming."},
    ])
    "".join(chain.stream_answer("Should I do that before I travel?", [], history))
    contents = [m.content for m in seen["messages"]]
    assert contents[1] == "How do I activate international roaming?"
    assert contents[-1] == "Should I do that before I travel?"


# ---------------------------------------------------------------- confirmation guardrails

@pytest.mark.parametrize("reply,expected", [
    ("Yes, proceed", "confirmed"), ("yes", "confirmed"), ("Confirm refund", "confirmed"),
    ("Cancel", "rejected"), ("no", "rejected"), ("No, don't", "rejected"),
    ("maybe", "ambiguous"), ("ok", "ambiguous"), ("yes but wait", "ambiguous"),
    ("Yes, cancel", "ambiguous"), ("hmm not sure", "ambiguous"), ("no thanks", "rejected"),
])
def test_parse_confirmation(reply, expected):
    assert parse_confirmation(reply) == expected


def test_ambiguous_then_cancel_never_refunds(agent, session, spy_process):
    turn(agent, session, "refund ₹499 please")
    reply = agent.handle("maybe", session, [])
    assert reply.needs_confirmation and session.awaiting_confirmation
    turn(agent, session, "Cancel")
    assert session.state == RefundState.REFUND_CANCELLED and spy_process == []
    assert recharge(session, "rch_457")["refund_status"] == "not_refunded"


def test_confirmation_in_original_request_is_not_enough(agent, session, spy_process):
    turn(agent, session, "Refund ₹499, yes proceed")
    assert session.awaiting_confirmation and spy_process == []


def test_clear_conversation_drops_pending_refund(agent, session, spy_process):
    turn(agent, session, "I want a refund of ₹499")
    session.clear()
    decision = route("Yes, proceed", session, [], classifier=heuristic_classifier)
    assert decision.intent == REQUIREMENT_INQUIRY      # nothing pending any more
    assert spy_process == []


def test_process_refund_refuses_over_limit_or_unvalidated(session):
    out = refund_tools.process_refund(session.store, session.user_id, 999, "rch_456")
    assert not out["success"] and recharge(session, "rch_456")["refund_status"] == "not_refunded"
    out = refund_tools.process_refund(session.store, session.user_id, 2000, "rch_457")
    assert not out["success"]


# ---------------------------------------------------------------- missing amount

def test_missing_amount_with_two_eligible_asks_which(agent, session):
    reply = turn(agent, session, "I want a refund.")
    assert session.state == RefundState.AWAITING_AMOUNT
    assert "₹999" in reply.text and "₹499" in reply.text and "Which amount" in reply.text
    reply = turn(agent, session, "499")
    assert session.awaiting_confirmation and session.pending.recharge_id == "rch_457"


def test_missing_amount_single_small_recharge_offers_confirmation(agent, session):
    recharge(session, "rch_456")["refund_status"] = "refunded"
    reply = turn(agent, session, "I want a refund.")
    assert reply.needs_confirmation and reply.pending_amount == 499


def test_missing_amount_single_large_recharge_offers_then_submits(agent, session, spy_process):
    recharge(session, "rch_457")["refund_status"] = "refunded"
    reply = turn(agent, session, "I want a refund.")
    assert session.state == RefundState.AWAITING_AMOUNT and "₹999" in reply.text
    reply = turn(agent, session, "yes")
    assert session.state == RefundState.REFUND_SUBMITTED_FOR_REVIEW and spy_process == []


def test_refunding_one_leaves_other_refundable(agent, session):
    turn(agent, session, "I want a refund of ₹499")
    turn(agent, session, "Yes, proceed")
    reply = turn(agent, session, "I want a refund of ₹499")
    assert "already been refunded" in reply.text
    assert refund_tools.validate_refund(session.store, session.user_id, 999)["eligible"]


def test_amount_parsing():
    assert parse_amounts("Give me a refund of ₹2,000.") == [2000]
    assert parse_amounts("refund rs 499 from 2 days ago") == [499]
    assert parse_amounts("my recharge rch_456 within 7 days") == []
    assert parse_amounts("refund -50") == [-50]


def test_zero_or_negative_amount_rejected(agent, session):
    reply = turn(agent, session, "refund ₹0")
    assert "greater than zero" in reply.text


# ---------------------------------------------------------------- intent rules (FR-05a)

@pytest.mark.parametrize("text,intent", [
    ("I want a refund of ₹999.", REFUND_REQUEST),
    ("Cancel my last recharge", REFUND_REQUEST),
    ("Please reverse my ₹499 recharge", REFUND_REQUEST),
    ("How do I cancel a recharge?", REQUIREMENT_INQUIRY),
    ("What is the refund policy?", REQUIREMENT_INQUIRY),
    ("Why is my internet slow?", REQUIREMENT_INQUIRY),
])
def test_heuristic_intent(text, intent):
    assert heuristic_intent(text) == intent


# ---------------------------------------------------------------- LLM tool-calling path

class ScriptedToolModel:
    """Minimal stand-in for a tool-calling chat model: replays scripted AIMessages."""

    def __init__(self, script):
        self.script = list(script)
        self.tools = None

    def bind_tools(self, tools):
        self.tools = [t.name for t in tools]
        return self

    def invoke(self, messages):
        return self.script.pop(0)


def call(name, args, i):
    return {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}


def test_llm_tool_loop_only_sees_read_tools_and_validates(session):
    model = ScriptedToolModel([
        AIMessage(content="", tool_calls=[call("get_user_account", {}, 1)]),
        AIMessage(content="", tool_calls=[call("validate_refund", {"amount": 499, "user_id": "hacker"}, 2)]),
        AIMessage(content="done"),
    ])
    reply = RefundAgent(llm_factory=lambda: model).handle("Can I get my 499 back?", session, [])
    assert model.tools == ["get_user_account", "validate_refund"]          # FR-23a
    assert reply.needs_confirmation and session.pending.amount == 499
    assert reply.events[:3] == ["[Agent] get_user_account", "[Agent] validate_refund(amount=499)",
                                "[Validation] eligible=True"]


def test_llm_invented_amount_is_ignored(session):
    model = ScriptedToolModel([
        AIMessage(content="", tool_calls=[call("get_user_account", {}, 1)]),
        AIMessage(content="", tool_calls=[call("validate_refund", {"amount": 999}, 2)]),
        AIMessage(content="done"),
    ])
    reply = RefundAgent(llm_factory=lambda: model).handle("I want a refund.", session, [])
    assert any(e.startswith("[Guard] ignored amount 999") for e in reply.events)   # FR-44
    assert session.state == RefundState.AWAITING_AMOUNT
    assert not session.store.support_requests


def test_llm_failure_falls_back_to_deterministic(session):
    def boom():
        raise RuntimeError("rate limited")

    reply = RefundAgent(llm_factory=boom).handle("I want a refund of ₹499", session, [])
    assert reply.needs_confirmation
    assert any("deterministic mode" in e for e in reply.events)
