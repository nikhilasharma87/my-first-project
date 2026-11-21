"""Simulated account data for the refund demo (agent_PRD FR-14 to FR-18a).

No real CRM or billing system: a hard-coded user whose ID stands in for the
authenticated session (FR-15, FR-16). Each chat session gets its own copy, so
refunds update the simulated state (FR-33) without touching other sessions.
"""
import copy
from dataclasses import dataclass, field

DEMO_USER_ID = "user_123"  # treated as coming from the authenticated session; never asked for (FR-17)

USERS = {
    "user_123": {
        "name": "Rahul",
        "plan": "Pro",
        "plan_amount": 999,
        "recharges": [
            {
                "recharge_id": "rch_456",
                "recharge_amount": 999,
                "recharge_status": "completed",
                "recharge_age_days": 2,
                "refund_status": "not_refunded",
            },
            {
                "recharge_id": "rch_457",
                "recharge_amount": 499,
                "recharge_status": "completed",
                "recharge_age_days": 1,
                "refund_status": "not_refunded",
            },
        ],
    }
}


@dataclass
class AccountStore:
    """Per-session copy of the simulated account data plus refund/support records."""

    users: dict = field(default_factory=lambda: copy.deepcopy(USERS))
    refunds: list = field(default_factory=list)              # processed simulated refunds
    support_requests: dict = field(default_factory=dict)     # recharge_id -> submission record

    def reset(self) -> None:
        self.users = copy.deepcopy(USERS)
        self.refunds.clear()
        self.support_requests.clear()
