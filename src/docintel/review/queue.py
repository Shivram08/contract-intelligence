"""The human-in-the-loop review queue, persisted in Postgres.

``extract.needs_review`` already decides *which* results a person should see:
an ungrounded quote, an error-severity rule violation, low confidence on a
Tier-3 clause, a run stopped by a budget. Until this module that decision was
a boolean on a response -- the caller had to notice it and keep the result
somewhere. A flag nobody stores is a flag nobody acts on.

The queue is a table in the same database as the index (CLAUDE.md section 3:
"one container, what banks actually run"), with a deliberately small lifecycle:

    open --resolve--> accepted | corrected | rejected

**The resolution is the label.** A reviewer who marks an item ``corrected``
attaches the corrected extraction, which is exactly the human-labelled data the
judge calibration in CLAUDE.md section 6 needs. Storing it beside the model's
output and the reasons it was routed makes disagreement analysable later
without reconstructing anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Final

from docintel.schemas import ExtractionResult, ReviewItem

__all__ = ["QueueEntry", "Resolution", "ReviewQueue"]

SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS review_items (
    id           BIGSERIAL PRIMARY KEY,
    document_id  TEXT        NOT NULL,
    reasons      JSONB       NOT NULL,
    result       JSONB       NOT NULL,
    status       TEXT        NOT NULL DEFAULT 'open'
                 CHECK (status IN ('open', 'accepted', 'corrected', 'rejected')),
    reviewer     TEXT,
    note         TEXT,
    correction   JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at  TIMESTAMPTZ,
    -- A corrected item must carry its correction; nothing else may.
    CHECK ((status = 'corrected') = (correction IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS review_items_open
    ON review_items (created_at) WHERE status = 'open';
"""


class Resolution(StrEnum):
    """What a reviewer decided."""

    ACCEPTED = "accepted"
    """The model's extraction was right; the routing was a false alarm."""

    CORRECTED = "corrected"
    """The extraction was wrong and the reviewer supplied the right one."""

    REJECTED = "rejected"
    """Unusable, and not worth correcting -- e.g. the wrong document."""


@dataclass(frozen=True, slots=True)
class QueueEntry:
    id: int
    document_id: str
    reasons: list[str]
    result: ExtractionResult
    status: str
    reviewer: str | None
    note: str | None
    correction: ExtractionResult | None
    created_at: datetime
    resolved_at: datetime | None


_COLUMNS: Final = (
    "id, document_id, reasons, result, status, reviewer, note, correction, created_at, resolved_at"
)


def _entry(row: tuple[Any, ...]) -> QueueEntry:
    def as_json(value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value

    correction = as_json(row[7])
    return QueueEntry(
        id=int(row[0]),
        document_id=str(row[1]),
        reasons=list(as_json(row[2])),
        result=ExtractionResult.model_validate(as_json(row[3])),
        status=str(row[4]),
        reviewer=row[5],
        note=row[6],
        correction=None if correction is None else ExtractionResult.model_validate(correction),
        created_at=row[8],
        resolved_at=row[9],
    )


class ReviewQueue:
    """Enqueue, list and resolve review items on one psycopg connection."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def ensure_schema(self) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(SCHEMA)
        self._commit()

    def enqueue(self, item: ReviewItem) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO review_items (document_id, reasons, result) "
                "VALUES (%s, %s::jsonb, %s::jsonb) RETURNING id",
                (
                    item.document_id,
                    json.dumps(item.reasons),
                    item.result.model_dump_json(),
                ),
            )
            row = cursor.fetchone()
        self._commit()
        assert row is not None
        return int(row[0])

    def get(self, item_id: int) -> QueueEntry | None:
        with self._connection.cursor() as cursor:
            cursor.execute(f"SELECT {_COLUMNS} FROM review_items WHERE id = %s", (item_id,))
            row = cursor.fetchone()
        return None if row is None else _entry(row)

    def items(self, status: str = "open", limit: int = 50) -> list[QueueEntry]:
        """Oldest first: a queue that shows the newest first starves the old ones."""
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"SELECT {_COLUMNS} FROM review_items WHERE status = %s "
                "ORDER BY created_at, id LIMIT %s",
                (status, limit),
            )
            rows = cursor.fetchall()
        return [_entry(row) for row in rows]

    def resolve(
        self,
        item_id: int,
        resolution: Resolution,
        reviewer: str,
        note: str | None = None,
        correction: ExtractionResult | None = None,
    ) -> QueueEntry | None:
        """Close an open item. Returns None if it does not exist or is not open.

        Resolving twice is refused rather than overwritten: the first decision
        is the label, and silently replacing it would corrupt the calibration
        set it feeds.
        """
        if (resolution is Resolution.CORRECTED) != (correction is not None):
            raise ValueError("a correction is required for 'corrected', and only for it")
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE review_items SET status = %s, reviewer = %s, note = %s, "
                f"correction = %s::jsonb, resolved_at = now() "
                f"WHERE id = %s AND status = 'open' RETURNING {_COLUMNS}",
                (
                    resolution.value,
                    reviewer,
                    note,
                    None if correction is None else correction.model_dump_json(),
                    item_id,
                ),
            )
            row = cursor.fetchone()
        self._commit()
        return None if row is None else _entry(row)

    def _commit(self) -> None:
        # Autocommit connections (the API's and the tests') have nothing to
        # commit; a transactional one must not leave the write pending.
        if not getattr(self._connection, "autocommit", True):
            self._connection.commit()
