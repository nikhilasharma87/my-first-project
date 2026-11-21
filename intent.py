"""Intent detection: requirement_inquiry vs refund_request (agent_PRD 6.1).

Uses the shared Groq configuration with structured output (FR-04, FR-05). If the
LLM is unavailable (no key, rate limit, malformed output), a deterministic keyword
rule keeps the app working and is logged as the fallback.
"""
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

log = logging.getLogger("novacell.agent")

REQUIREMENT_INQUIRY = "requirement_inquiry"
REFUND_REQUEST = "refund_request"
INTENTS = (REQUIREMENT_INQUIRY, REFUND_REQUEST)


class IntentResult(BaseModel):
    """Classify the customer's latest message."""

    intent: Literal["requirement_inquiry", "refund_request"] = Field(
        description="refund_request only if the customer is asking NovaCell to refund, cancel or "
        "reverse one of their recharges/payments (or is answering the agent's refund questions). "
        "Everything else, including questions ABOUT how refunds or cancellations work, is "
        "requirement_inquiry."
    )


CLASSIFIER_PROMPT = """You route messages for NovaCell's support chat. Classify the customer's \
LATEST message as one of:
- refund_request: they want NovaCell to refund, cancel or reverse a recharge or payment of theirs, \
or they are replying to the refund agent (e.g. giving the amount it asked for).
- requirement_inquiry: any support or information question, including how refunds or \
cancellations work.

Examples:
"Why is my internet slow?" -> requirement_inquiry
"I want a refund of ₹999." -> refund_request
"Please reverse my last recharge" -> refund_request
"Cancel my ₹499 recharge" -> refund_request
"How do I cancel a recharge?" -> requirement_inquiry
"What is your refund policy?" -> requirement_inquiry
"Should I do that before I travel?" (after a roaming answer) -> requirement_inquiry

Earlier messages are given only to resolve follow-ups."""

_REFUND_WORDS = re.compile(
    r"\b(refund|money back|reverse|reversal|charge ?back|cancel (my|the|this|that|last)\b.*\b(recharge|payment|top.?up))",
    re.I,
)
_QUESTION_START = re.compile(r"^\s*(how|what|what's|where|when|why|is there|does|do you|which)\b", re.I)


def heuristic_intent(message: str) -> str:
    """Deterministic fallback (FR-05a: 'how do I cancel…' is an inquiry, 'cancel my…' is a request)."""
    if _QUESTION_START.search(message):
        return REQUIREMENT_INQUIRY
    return REFUND_REQUEST if _REFUND_WORDS.search(message) else REQUIREMENT_INQUIRY


def classify_intent(message: str, history: list | None = None, llm=None) -> tuple[str, str]:
    """Return (intent, method) where method is 'llm' or 'fallback'."""
    try:
        if llm is None:
            from chain import get_llm

            llm = get_llm(max_tokens=60, streaming=False, max_retries=2)
        from langchain_core.messages import HumanMessage, SystemMessage

        structured = llm.with_structured_output(IntentResult)
        result = structured.invoke(
            [SystemMessage(CLASSIFIER_PROMPT), *(history or [])[-4:], HumanMessage(message)]
        )
        if isinstance(result, IntentResult) and result.intent in INTENTS:
            return result.intent, "llm"
        raise ValueError(f"unexpected classifier output: {result!r}")
    except Exception as exc:  # no key, rate limit, parse error
        log.warning("[Intent] LLM classifier unavailable (%s); using keyword fallback", type(exc).__name__)
        return heuristic_intent(message), "fallback"
