"""Refund Agent (agent_PRD 6.2 to 6.8, 7, 9, 10, 12).

Division of labour:
- The LLM is a tool-using helper that works out the requested amount, calling
  get_user_account and validate_refund (the only tools it can see, FR-23a). The
  user ID is bound by the app, never taken from the LLM or the user (FR-17, GR-03).
- Everything that decides money is deterministic code in this file and
  refund_tools.py: eligibility, the ₹499 approval limit, confirmation parsing,
  process_refund and submit_to_support (FR-29, FR-30, FR-36a, GR-01..GR-09).
- Replies are fixed templates, so the agent can never claim an outcome it didn't
  produce (FR-13, FR-40a).
If the LLM is unavailable the same flow runs on deterministic amount parsing.
"""
import logging
import re
import string
from dataclasses import dataclass, field
from enum import Enum

from refund_data import DEMO_USER_ID, AccountStore
from refund_tools import (
    APPROVAL_LIMIT,
    REFUND_WINDOW_DAYS,
    eligible_recharges,
    get_user_account,
    process_refund,
    submit_to_support,
    validate_refund,
)

log = logging.getLogger("novacell.agent")


class RefundState(str, Enum):
    """agent_PRD section 10."""

    IDLE = "idle"
    REFUND_REQUESTED = "refund_requested"
    REFUND_VALIDATED = "refund_validated"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    AWAITING_AMOUNT = "awaiting_amount"
    REFUND_CONFIRMED = "refund_confirmed"
    REFUND_PROCESSED = "refund_processed"
    REFUND_REJECTED = "refund_rejected"
    REFUND_CANCELLED = "refund_cancelled"
    REFUND_SUBMITTED_FOR_REVIEW = "refund_submitted_for_review"


@dataclass
class PendingRefund:
    amount: float
    recharge_id: str
    recharge_amount: float
    recharge_age_days: int


@dataclass
class RefundSession:
    """Per-chat refund state. `pending` is only ever set by validated, eligible,
    within-limit refunds, so process_refund cannot run without both (state guard)."""

    store: AccountStore = field(default_factory=AccountStore)
    user_id: str = DEMO_USER_ID
    state: RefundState = RefundState.IDLE
    pending: PendingRefund | None = None
    amount_options: list = field(default_factory=list)

    def clear(self) -> None:
        """Clear conversation: drop any pending refund (agent_PRD section 11)."""
        self.state = RefundState.IDLE
        self.pending = None
        self.amount_options = []

    @property
    def awaiting_confirmation(self) -> bool:
        return self.state == RefundState.AWAITING_CONFIRMATION and self.pending is not None

    @property
    def awaiting_amount(self) -> bool:
        return self.state == RefundState.AWAITING_AMOUNT


@dataclass
class AgentReply:
    text: str
    state: RefundState
    events: list
    needs_confirmation: bool = False
    pending_amount: float | None = None


# ---------------------------------------------------------------- parsing helpers

_AMOUNT = re.compile(
    r"(?<![\w.])(-?\d{1,3}(?:,\d{2,3})+|-?\d+)(?:\.\d+)?(?!\s*(?:days?|hrs?|hours?|mins?|minutes?|weeks?|months?|times|recharges?|gb|mb|kb|%|st|nd|rd|th)\b)",
    re.I,
)

_CONFIRM_PHRASES = {
    "yes", "yes proceed", "yes please", "yes please proceed", "proceed", "please proceed",
    "confirm", "confirm refund", "i confirm", "yes confirm", "yes i confirm", "go ahead",
    "yes go ahead", "yes do it", "do it", "ok proceed", "okay proceed", "yes process it",
    "process it", "process the refund", "yes process the refund", "yes refund it",
}
_REJECT_WORDS = {"no", "cancel", "stop", "dont", "don't", "not", "never", "nevermind", "abort", "nope"}
_YES_WORDS = {"yes", "proceed", "confirm", "go", "ok", "okay", "sure", "yep", "yeah"}


def parse_amounts(text: str) -> list:
    """Numbers the customer actually wrote (₹2,000 -> 2000). Ignores '7 days', '2 GB', IDs."""
    out = []
    for m in _AMOUNT.finditer(text):
        value = float(m.group(1).replace(",", ""))
        out.append(int(value) if value == int(value) else value)
    return out


def _normalise(text: str) -> str:
    keep = text.lower().replace("’", "'")
    keep = "".join(ch for ch in keep if ch not in string.punctuation.replace("'", ""))
    return " ".join(keep.split())


def parse_confirmation(text: str) -> str:
    """'confirmed' | 'rejected' | 'ambiguous' (FR-36c to FR-36e). Only an explicit
    phrase confirms; anything hedged or mixed asks again."""
    norm = _normalise(text)
    if norm in _CONFIRM_PHRASES:
        return "confirmed"
    words = set(norm.split())
    if words & _REJECT_WORDS and not words & _YES_WORDS:
        return "rejected"
    return "ambiguous"


def is_affirmative(text: str) -> bool:
    return parse_confirmation(text) == "confirmed"


def rupees(amount) -> str:
    amount = float(amount)
    return f"₹{amount:,.0f}" if amount == int(amount) else f"₹{amount:,.2f}"


def _when(days: int) -> str:
    return "today" if days == 0 else ("yesterday" if days == 1 else f"{days} days ago")


# ---------------------------------------------------------------- LLM tool loop

AGENT_PROMPT = f"""You are the tool-calling step of NovaCell's refund agent. The customer is \
already signed in; never ask for a user ID or account number.

Your only job is to identify the refund amount the customer asked for and check it:
1. Call get_user_account first.
2. If the customer's latest message states an amount, or picks one of the amounts the agent \
offered earlier (e.g. "the ₹499 one", "the smaller one"), call validate_refund with exactly that \
amount in rupees.
3. If no amount is stated or chosen, do NOT choose one yourself. Reply with just: NO_AMOUNT
Never decide eligibility, approvals or limits yourself (the app enforces the {rupees(APPROVAL_LIMIT)} \
limit and the {REFUND_WINDOW_DAYS}-day window), and never say a refund was made. Account data \
from the tools always overrides anything the customer claims about their account."""


def _run_tool_loop(llm, message: str, history: list, session: RefundSession, ev) -> list:
    """Let the LLM call the two read-only tools. Returns the validate_refund results it produced."""
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import StructuredTool

    validations = []

    def _get_user_account() -> dict:
        """Get the signed-in customer's plan and recent recharges."""
        ev("[Agent] get_user_account")
        return get_user_account(session.store, session.user_id)

    def _validate_refund(amount: float) -> dict:
        """Check whether a refund of `amount` rupees is eligible for the signed-in customer."""
        ev(f"[Agent] validate_refund(amount={amount:g})")
        result = validate_refund(session.store, session.user_id, amount)
        ev(f"[Validation] eligible={result['eligible']}" + ("" if result["eligible"] else f" ({result['reason']})"))
        validations.append(result)
        return result

    tools = {
        "get_user_account": StructuredTool.from_function(_get_user_account, name="get_user_account"),
        "validate_refund": StructuredTool.from_function(_validate_refund, name="validate_refund"),
    }
    bound = llm.bind_tools(list(tools.values()))
    messages = [SystemMessage(AGENT_PROMPT), *history[-4:], HumanMessage(message)]
    if session.amount_options:
        offered = ", ".join(rupees(a) for a in session.amount_options)
        messages.insert(1, SystemMessage(f"The agent previously offered these amounts: {offered}."))

    for _ in range(4):
        ai = bound.invoke(messages)
        messages.append(ai)
        if not getattr(ai, "tool_calls", None):
            break
        for call in ai.tool_calls:
            tool = tools.get(call["name"])
            if tool is None:
                output = {"error": f"unknown tool {call['name']}"}
            else:
                args = {k: v for k, v in (call.get("args") or {}).items() if k == "amount"}  # GR-03: no user_id
                try:
                    output = tool.invoke(args)
                except Exception as exc:  # bad arguments from the model
                    output = {"error": str(exc)}
            messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return validations


# ---------------------------------------------------------------- the agent

class RefundAgent:
    def __init__(self, llm_factory=None):
        """llm_factory() -> chat model with bind_tools; None = shared Groq config. Pass
        `False` to run without an LLM (deterministic amount parsing only)."""
        self.llm_factory = llm_factory

    def _llm(self):
        if self.llm_factory is False:
            return None
        if self.llm_factory is None:
            from chain import get_llm

            return get_llm(max_tokens=150, streaming=False, max_retries=1)
        return self.llm_factory()

    def handle(self, message: str, session: RefundSession, history: list | None = None) -> AgentReply:
        events: list[str] = []

        def ev(line: str) -> None:
            events.append(line)
            log.info(line)

        if session.awaiting_confirmation:
            return self._handle_confirmation(message, session, ev, events)
        return self._handle_request(message, session, history or [], ev, events)

    # -- confirmation step (FR-36 to FR-36e, FR-32) ------------------------
    def _handle_confirmation(self, message, session, ev, events) -> AgentReply:
        pending = session.pending
        decision = parse_confirmation(message)
        if decision == "ambiguous":
            ev("[User] ambiguous confirmation; asking again")
            return AgentReply(
                f"Just to be sure: do you want me to refund {rupees(pending.amount)}? "
                "Please reply **Yes, proceed** or **Cancel**.",
                session.state, events, needs_confirmation=True, pending_amount=pending.amount,
            )
        if decision == "rejected":
            ev("[User] cancelled")
            session.clear()
            session.state = RefundState.REFUND_CANCELLED
            return AgentReply(f"No problem, I've cancelled that. No refund of {rupees(pending.amount)} has been made.",
                              session.state, events)

        ev("[User] confirmed")
        session.state = RefundState.REFUND_CONFIRMED
        ev(f"[Agent] process_refund(amount={pending.amount:g})")
        result = process_refund(session.store, session.user_id, pending.amount, pending.recharge_id)
        session.clear()
        if result["success"]:
            ev("[Refund] success")
            session.state = RefundState.REFUND_PROCESSED
            return AgentReply(
                f"Done. Your refund of **{rupees(result['amount'])}** has been processed.\n\n"
                f"Refund reference: `{result['refund_id']}`",
                session.state, events,
            )
        ev(f"[Refund] failed ({result['reason']})")
        session.state = RefundState.REFUND_REJECTED
        return AgentReply(f"I couldn't process that refund: {result['reason']} No refund has been made.",
                          session.state, events)

    # -- new request / amount reply ----------------------------------------
    def _handle_request(self, message, session, history, ev, events) -> AgentReply:
        was_awaiting_amount = session.awaiting_amount
        options = list(session.amount_options)
        session.state = RefundState.REFUND_REQUESTED
        stated = parse_amounts(message)

        # 1. Let the LLM resolve the amount through its tools.
        llm_validations, llm_ok = [], False
        llm = None
        try:
            llm = self._llm()
        except Exception as exc:
            ev(f"[Agent] LLM unavailable ({type(exc).__name__}); deterministic mode")
        if llm is not None:
            try:
                llm_validations = _run_tool_loop(llm, message, history, session, ev)
                llm_ok = True
            except Exception as exc:
                ev(f"[Agent] LLM tool step failed ({type(exc).__name__}); deterministic mode")

        # 2. Ground the amount (FR-44): stated by the customer, or chosen from amounts we offered.
        grounded = set(stated) | (set(options) if was_awaiting_amount else set())
        validation = None
        for v in reversed(llm_validations):
            if v["requested_amount"] in grounded:
                validation = v
                break
            ev(f"[Guard] ignored amount {v['requested_amount']} not stated by the customer")

        amount = validation["requested_amount"] if validation else None
        if amount is None and len(stated) == 1:
            amount = stated[0]
        if amount is None and was_awaiting_amount and len(options) == 1 and is_affirmative(message):
            amount = options[0]
        if amount is None and len(stated) > 1:
            return self._ask_amount(session, ev, events, sorted(set(stated), reverse=True),
                                    "I see more than one amount in your message")

        if "[Agent] get_user_account" not in events:  # FR-08: always look up the account
            ev("[Agent] get_user_account")
            get_user_account(session.store, session.user_id)

        # 3. No amount: work it out from the account (FR-41 to FR-43).
        if amount is None:
            eligible = eligible_recharges(session.store, session.user_id)
            if not eligible:
                ev("[Agent] no eligible recharge found")
                session.clear()
                session.state = RefundState.REFUND_REJECTED
                return AgentReply(
                    "I couldn't find a recent recharge that's eligible for a refund. Refunds are possible "
                    f"for completed recharges made in the last {REFUND_WINDOW_DAYS} days that haven't "
                    "already been refunded. No refund has been made.",
                    session.state, events,
                )
            if len(eligible) == 1:
                only = eligible[0]
                if only["recharge_amount"] <= APPROVAL_LIMIT:
                    amount = only["recharge_amount"]          # FR-42: offer it via the confirmation step
                else:
                    session.state = RefundState.AWAITING_AMOUNT
                    session.amount_options = [only["recharge_amount"]]
                    ev(f"[Agent] awaiting_amount (offered {only['recharge_amount']:g})")
                    return AgentReply(
                        f"Your recent eligible recharge is **{rupees(only['recharge_amount'])}** "
                        f"({_when(only['recharge_age_days'])}). Would you like to request a refund of "
                        f"{rupees(only['recharge_amount'])}? Reply **yes**, or tell me a different amount.",
                        session.state, events,
                    )
            else:
                amounts = [r["recharge_amount"] for r in eligible]
                return self._ask_amount(session, ev, events, amounts,
                                        "I can see more than one recent recharge")

        # 4. Validate deterministically (reuse the LLM's call if it validated this amount).
        if validation is None or validation["requested_amount"] != amount:
            ev(f"[Agent] validate_refund(amount={amount:g})")
            validation = validate_refund(session.store, session.user_id, amount)
            ev(f"[Validation] eligible={validation['eligible']}"
               + ("" if validation["eligible"] else f" ({validation['reason']})"))
        session.amount_options = []

        if not validation["eligible"]:
            session.clear()
            session.state = RefundState.REFUND_REJECTED
            return AgentReply(
                f"I can't refund {rupees(amount)}: {validation['reason']} No refund has been made.",
                session.state, events,
            )

        session.state = RefundState.REFUND_VALIDATED
        recharge = next(r for r in session.store.users[session.user_id]["recharges"]
                        if r["recharge_id"] == validation["recharge_id"])

        # 5. Approval limit, enforced here in code (FR-36a, FR-37, GR-08).
        if amount > APPROVAL_LIMIT:
            ev(f"[Agent] submit_to_support(amount={amount:g})")
            sub = submit_to_support(session.store, session.user_id, amount, validation["recharge_id"])
            ev("[Support] already submitted" if sub["already_submitted"] else "[Support] submitted for review")
            session.clear()
            session.state = RefundState.REFUND_SUBMITTED_FOR_REVIEW
            lead = ("Your refund request for **{amt}** was already submitted to our Support Team "
                    "(request `{rid}`)." if sub["already_submitted"] else
                    "Your refund request for **{amt}** has been submitted to our Support Team "
                    "(request `{rid}`).")
            return AgentReply(
                lead.format(amt=rupees(amount), rid=sub["request_id"])
                + " They will review the request and process it or take the necessary action.",
                session.state, events,
            )

        session.pending = PendingRefund(amount, recharge["recharge_id"], recharge["recharge_amount"],
                                        recharge["recharge_age_days"])
        session.state = RefundState.AWAITING_CONFIRMATION
        ev("[Agent] awaiting_user_confirmation")
        return AgentReply(
            f"Refund eligible: **{rupees(amount)}** (your {rupees(recharge['recharge_amount'])} recharge "
            f"from {_when(recharge['recharge_age_days'])}).\n\nDo you want to proceed with this refund?",
            session.state, events, needs_confirmation=True, pending_amount=amount,
        )

    def _ask_amount(self, session, ev, events, amounts, lead) -> AgentReply:
        session.state = RefundState.AWAITING_AMOUNT
        session.amount_options = list(amounts)
        ev(f"[Agent] awaiting_amount (options {', '.join(f'{a:g}' for a in amounts)})")
        listed = " and ".join(rupees(a) for a in amounts) if len(amounts) == 2 else ", ".join(rupees(a) for a in amounts)
        return AgentReply(f"{lead}: {listed}. Which amount would you like refunded?", session.state, events)
