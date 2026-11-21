"""Refund tools with deterministic business rules (agent_PRD 6.4 to 6.7).

Only get_user_account and validate_refund are offered to the LLM.
process_refund and submit_to_support are called by application code alone
(FR-23a), after validation and, for process_refund, explicit confirmation.
"""
import copy
import uuid

from refund_data import AccountStore

APPROVAL_LIMIT = 499        # ₹; refunds above this go to the Support Team (FR-36a, FR-37)
REFUND_WINDOW_DAYS = 7      # FR-25


def get_user_account(store: AccountStore, user_id: str) -> dict:
    """FR-19: the user's plan and recharges, as a structured copy."""
    user = store.users.get(user_id)
    if user is None:
        return {"found": False, "user_id": user_id, "reason": "No account found."}
    return {"found": True, "user_id": user_id, **copy.deepcopy(user)}


def _select_recharge(recharges: list, amount: float) -> dict | None:
    """FR-27a: pick the recharge a refund applies to, deterministically.

    1. A recharge whose amount equals the request exactly (even if already refunded).
       If several match, the most recent not-yet-refunded one, else the most recent.
    2. Otherwise the most recent still-refundable recharge that covers the amount.
    3. Otherwise the most recent recharge (validation will then explain the rejection).
    """
    if not recharges:
        return None
    by_recency = sorted(recharges, key=lambda r: r["recharge_age_days"])
    exact = [r for r in by_recency if r["recharge_amount"] == amount]
    if exact:
        open_exact = [r for r in exact if r["refund_status"] != "refunded"]
        return (open_exact or exact)[0]
    covering = [
        r for r in by_recency
        if r["recharge_status"] == "completed"
        and r["recharge_age_days"] <= REFUND_WINDOW_DAYS
        and r["refund_status"] != "refunded"
        and r["recharge_amount"] >= amount
    ]
    if covering:
        return covering[0]
    refundable = [r for r in by_recency if r["refund_status"] != "refunded"]
    return (refundable or by_recency)[0]


def validate_refund(store: AccountStore, user_id: str, amount) -> dict:
    """FR-20, FR-24 to FR-30: eligibility decided by code, never by the LLM."""
    try:
        requested = float(amount)
    except (TypeError, ValueError):
        return {"eligible": False, "reason": "The refund amount is not a valid number.",
                "requested_amount": amount, "recharge_amount": None, "recharge_id": None}
    if requested == int(requested):
        requested = int(requested)

    user = store.users.get(user_id)
    if user is None or not user["recharges"]:
        return {"eligible": False, "reason": "No recharge was found on the account.",
                "requested_amount": requested, "recharge_amount": None, "recharge_id": None}

    result = {"requested_amount": requested, "recharge_amount": None, "recharge_id": None}
    if requested <= 0:                                                       # FR-26
        return {**result, "eligible": False, "reason": "The refund amount must be greater than zero."}

    recharge = _select_recharge(user["recharges"], requested)
    result.update(recharge_amount=recharge["recharge_amount"], recharge_id=recharge["recharge_id"])

    if recharge["refund_status"] == "refunded":                              # FR-28, GR-06
        reason = "This recharge has already been refunded."
    elif recharge["recharge_status"] != "completed":                        # FR-24
        reason = "Only completed recharges can be refunded."
    elif recharge["recharge_age_days"] > REFUND_WINDOW_DAYS:                 # FR-25
        reason = f"The recharge is outside the {REFUND_WINDOW_DAYS}-day refund window."
    elif requested > recharge["recharge_amount"]:                           # FR-27, GR-04
        reason = "The requested amount exceeds the recharge amount."
    else:
        return {**result, "eligible": True, "reason": "Refund is eligible."}
    return {**result, "eligible": False, "reason": reason}


def eligible_recharges(store: AccountStore, user_id: str) -> list[dict]:
    """Recharges that would pass validation for their full amount (used for FR-41 to FR-42a)."""
    user = store.users.get(user_id) or {"recharges": []}
    out = []
    for r in sorted(user["recharges"], key=lambda r: r["recharge_age_days"]):
        if validate_refund(store, user_id, r["recharge_amount"])["eligible"]:
            out.append(r)
    return out


def process_refund(store: AccountStore, user_id: str, amount, recharge_id: str) -> dict:
    """FR-21, FR-31 to FR-35: simulated refund only; no payment API is called.

    Re-validates and checks the approval limit itself, so it can never refund an
    ineligible or over-limit amount even if called by mistake (GR-01, GR-08)."""
    check = validate_refund(store, user_id, amount)
    if not check["eligible"] or check["recharge_id"] != recharge_id:
        return {"success": False, "reason": check["reason"], "amount": amount, "status": "not_refunded"}
    if check["requested_amount"] > APPROVAL_LIMIT:
        return {"success": False, "reason": "Amount is above the approval limit.",
                "amount": amount, "status": "not_refunded"}

    for r in store.users[user_id]["recharges"]:
        if r["recharge_id"] == recharge_id:
            r["refund_status"] = "refunded"                                  # FR-28a: whole recharge consumed
    refund = {
        "success": True,
        "refund_id": f"ref_{uuid.uuid4().hex[:6]}",
        "amount": check["requested_amount"],
        "recharge_id": recharge_id,
        "status": "refunded",
    }
    store.refunds.append(refund)
    return refund


def submit_to_support(store: AccountStore, user_id: str, amount, recharge_id: str) -> dict:
    """FR-21a, FR-38: hand an over-limit refund to the Support Team. Does not change
    the refund status and does not simulate the team's review (FR-40a, GR-09)."""
    existing = store.support_requests.get(recharge_id)
    if existing:
        return {**existing, "already_submitted": True}
    record = {
        "submitted": True,
        "request_id": f"sup_{uuid.uuid4().hex[:6]}",
        "user_id": user_id,
        "amount": amount,
        "recharge_id": recharge_id,
        "status": "submitted_for_review",
        "already_submitted": False,
    }
    store.support_requests[recharge_id] = record
    return record
