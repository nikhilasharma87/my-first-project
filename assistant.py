"""Front door for every chat turn: intent detection, then RAG or the Refund Agent.

    User ─▶ route() ─┬─ requirement_inquiry ─▶ existing RAG (retrieve + grounded answer)
                     └─ refund_request ──────▶ RefundAgent

The RAG pipeline itself is unchanged; it now also receives recent history for
follow-ups (agent_PRD FR-45 to FR-50). Mid-refund replies ("Yes, proceed", "499")
go straight to the agent without an LLM call.
"""
import logging
from dataclasses import dataclass, field

from intent import REFUND_REQUEST, REQUIREMENT_INQUIRY, classify_intent
from refund_agent import AgentReply, RefundAgent, RefundSession, RefundState, is_affirmative, parse_amounts, parse_confirmation

log = logging.getLogger("novacell.agent")


@dataclass
class Route:
    intent: str
    events: list = field(default_factory=list)


def route(message: str, session: RefundSession, history: list, classifier=classify_intent) -> Route:
    events: list[str] = []

    def ev(line: str) -> None:
        events.append(line)
        log.info(line)

    if session.awaiting_confirmation and parse_confirmation(message) != "ambiguous":
        ev(f"[Intent] {REFUND_REQUEST} (confirmation reply)")
        return Route(REFUND_REQUEST, events)
    if session.awaiting_amount and (parse_amounts(message) or is_affirmative(message)):
        ev(f"[Intent] {REFUND_REQUEST} (amount reply)")
        return Route(REFUND_REQUEST, events)

    intent, method = classifier(message, history)
    ev(f"[Intent] {intent}" + (" (keyword fallback)" if method == "fallback" else ""))

    if intent == REQUIREMENT_INQUIRY and session.awaiting_amount:
        session.clear()  # customer moved on; don't let a later "yes" pick up a stale offer
    # An ambiguous reply while a refund awaits confirmation: a support question goes to RAG
    # and the refund stays pending; anything refund-related reaches the agent, which asks again.
    return Route(intent, events)


__all__ = ["Route", "route", "AgentReply", "RefundSession", "RefundAgent", "RefundState",
           "REFUND_REQUEST", "REQUIREMENT_INQUIRY"]
