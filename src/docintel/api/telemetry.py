"""Re-export of :mod:`docintel.telemetry`.

``CLAUDE.md`` section 4 places ``telemetry.py`` under ``api/``, but the spans it
defines wrap stages inside the *core* pipeline -- the model call in
``agent/loop.py`` and validation in ``extract.py``. Importing from ``api`` there
would make the extraction core depend on the HTTP layer, so the implementation
lives at ``docintel.telemetry`` and this module preserves the documented import
path.
"""

from docintel.telemetry import (
    COST_BUCKETS,
    EXTRACTION_COST,
    EXTRACTION_LATENCY,
    GROUNDING,
    LATENCY_BUCKETS,
    REGISTRY,
    REVIEW,
    RUNS,
    STAGE_LATENCY,
    STAGES,
    TOKENS,
    VIOLATIONS,
    extraction_span,
    record_extraction,
    set_tracer,
    setup_tracing,
    stage_span,
    tracer,
)

__all__ = [
    "COST_BUCKETS",
    "EXTRACTION_COST",
    "EXTRACTION_LATENCY",
    "GROUNDING",
    "LATENCY_BUCKETS",
    "REGISTRY",
    "REVIEW",
    "RUNS",
    "STAGES",
    "STAGE_LATENCY",
    "TOKENS",
    "VIOLATIONS",
    "extraction_span",
    "record_extraction",
    "set_tracer",
    "setup_tracing",
    "stage_span",
    "tracer",
]
