"""A cumulative spend ledger with a hard cap across every live eval run.

``--budget`` caps one baseline in one run. Nothing capped the project: ten
runs at a $5 budget each are $50, and the API credit behind this project is
$30. The ledger is committed, so total spend is part of the record rather
than an estimate reconstructed afterwards.

**The check is worst-case, before any money moves.** A run is refused unless
``spent + budget x live baselines`` fits under the cap. Every baseline already
stops at its own budget, so that sum bounds what the run can possibly cost.

**What gets recorded is live spend only** -- the cost of cache misses, taken
from the provider's reported usage. A replay costs nothing and records nothing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

__all__ = ["DEFAULT_CAP_USD", "LEDGER_PATH", "Ledger", "SpendCapError"]

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
LEDGER_PATH: Final = REPO_ROOT / "evals" / "spend_ledger.json"

#: $25 against a $30 credit: the last $5 is never touched by a script.
DEFAULT_CAP_USD: Final = 25.0


class SpendCapError(RuntimeError):
    """A planned run could exceed the project-wide cap."""


@dataclass(slots=True)
class Ledger:
    path: Path = LEDGER_PATH
    cap_usd: float = field(
        default_factory=lambda: float(os.environ.get("DOCINTEL_SPEND_CAP", DEFAULT_CAP_USD))
    )
    entries: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = LEDGER_PATH, cap_usd: float | None = None) -> Ledger:
        entries: list[dict[str, Any]] = []
        if path.exists():
            entries = list(json.loads(path.read_text(encoding="utf-8")).get("entries", []))
        ledger = cls(path=path, entries=entries)
        if cap_usd is not None:
            ledger.cap_usd = cap_usd
        return ledger

    @property
    def spent_usd(self) -> float:
        return sum(float(e.get("cost_usd", 0.0)) for e in self.entries)

    @property
    def remaining_usd(self) -> float:
        return self.cap_usd - self.spent_usd

    def check(self, planned_usd: float) -> None:
        """Refuse a run whose worst case does not fit under the cap."""
        if planned_usd > self.remaining_usd + 1e-9:
            raise SpendCapError(
                f"worst case ${planned_usd:.2f} exceeds the remaining "
                f"${self.remaining_usd:.2f} (spent ${self.spent_usd:.2f} of the "
                f"${self.cap_usd:.2f} cap). Lower --budget or --limit."
            )

    def record(self, cost_usd: float, **details: Any) -> None:
        """Append one run and write the ledger. Zero-cost runs are not recorded."""
        if cost_usd <= 0:
            return
        self.entries.append(
            {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "cost_usd": round(cost_usd, 6),
                **details,
            }
        )
        payload = {
            "cap_usd": self.cap_usd,
            "spent_usd": round(self.spent_usd, 6),
            "entries": self.entries,
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
