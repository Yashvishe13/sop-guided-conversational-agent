"""Small persisted records shared by the store, the email dispatcher, and the service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

EmailOpStatus = Literal["pending", "dispatching", "sent", "queued", "failed", "delivery_unknown"]


@dataclass
class EmailOpRecord:
    """Ledger row for one consented email send (one logical write).

    ``op_key`` is the idempotency key, derived from session, summary hash, and consent turn.
    ``recipient_ref`` is a reference (the party ID), never the raw address.
    """

    op_key: str
    session_id: str
    recipient_ref: str
    draft_hash: str
    status: EmailOpStatus
    created_at: datetime
    updated_at: datetime
    receipt_id: str | None = None
    detail: str | None = None


@dataclass
class TurnRecord:
    """Idempotency record for one accepted client turn (duplicate submits replay ``response_json``)."""

    session_id: str
    client_turn_id: str
    turn_index: int
    request_hash: str
    response_json: str
    created_at: datetime
