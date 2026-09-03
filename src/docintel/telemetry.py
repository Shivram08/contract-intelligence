"""OpenTelemetry spans and Prometheus metrics.

**The stage boundaries here are the same ones the eval harness measures.** That
is the point of this module rather than an accident of it: `evals/runners.py`
records `retrieval_ms`, `model_ms` and `validation_ms` per document, and the
spans below wrap the identical boundaries. If a trace in Jaeger and a row in
`docs/RESULTS.md` disagreed about what "retrieval" means, one of them would be
lying and there would be no way to tell which.

That concern is not hypothetical. This project has already produced four numbers
that measured something other than what they claimed -- including cache-read
time reported as model latency. So the metric names are shared constants, and
the histogram buckets are chosen from measured values rather than from defaults.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

from prometheus_client import CollectorRegistry, Counter, Histogram

__all__ = [
    "STAGES",
    "extraction_span",
    "record_extraction",
    "setup_tracing",
    "stage_span",
]

#: The pipeline stages, in order. Shared with the eval harness so a trace and a
#: results table describe the same thing.
STAGES: Final[tuple[str, ...]] = ("retrieval", "rerank", "model", "validation")

SERVICE_NAME: Final = "docintel"

#: Latency buckets in seconds, chosen from measured behaviour rather than
#: defaults. Validation runs in ~4ms, retrieval in ~1.5s, and the model in
#: 25-225s, so the range has to span five orders of magnitude -- Prometheus's
#: default buckets top out at 10s and would put every model call in +Inf.
LATENCY_BUCKETS: Final = (
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
)

#: Cost buckets in USD. Measured per-document cost ran $0.05-$0.67.
COST_BUCKETS: Final = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)

REGISTRY: Final = CollectorRegistry()

STAGE_LATENCY: Final = Histogram(
    "docintel_stage_duration_seconds",
    "Wall clock per pipeline stage.",
    labelnames=("stage",),
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

EXTRACTION_LATENCY: Final = Histogram(
    "docintel_extraction_duration_seconds",
    "End-to-end extraction wall clock.",
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

EXTRACTION_COST: Final = Histogram(
    "docintel_extraction_cost_usd",
    "Cost per extraction in USD.",
    buckets=COST_BUCKETS,
    registry=REGISTRY,
)

TOKENS: Final = Counter(
    "docintel_tokens_total",
    "Tokens consumed, by kind. cache_read is billed at 0.1x and cache_write at "
    "1.25x input, so they are counted separately rather than summed.",
    labelnames=("kind",),
    registry=REGISTRY,
)

RUNS: Final = Counter(
    "docintel_runs_total",
    "Extractions by terminal state. An incomplete run is a distinct outcome, "
    "not a zero-scoring success -- conflating them is how presence F1 once "
    "became partly a measurement of max_turns.",
    labelnames=("status",),
    registry=REGISTRY,
)

GROUNDING: Final = Counter(
    "docintel_grounding_spans_total",
    "Evidence spans by grounding outcome. `relocated` is counted because 100% "
    "of spans required it -- models essentially never report correct offsets, "
    "so a zero violation rate is conditional on relocation succeeding.",
    labelnames=("status",),
    registry=REGISTRY,
)

VIOLATIONS: Final = Counter(
    "docintel_rule_violations_total",
    "Deterministic rule violations by rule id and severity.",
    labelnames=("rule_id", "severity"),
    registry=REGISTRY,
)

REVIEW: Final = Counter(
    "docintel_review_routed_total",
    "Extractions routed to human review.",
    registry=REGISTRY,
)


def setup_tracing(app: Any = None, endpoint: str | None = None) -> Any:
    """Configure OTLP tracing, returning the tracer.

    Degrades to a no-op tracer when the OTel SDK or a collector is absent, so
    the service runs without Jaeger rather than refusing to start. Observability
    that takes the application down with it is worse than none.
    """
    endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
        if endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
            )
        trace.set_tracer_provider(provider)

        if app is not None:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            # /metrics and /healthz are excluded: a scrape every 15s would
            # otherwise dominate the trace volume and tell nobody anything.
            FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics,healthz")

        # Installed, not just returned. Returning it and forgetting to install
        # it would leave every stage_span a silent no-op -- traces would be
        # empty and nothing would say why.
        resolved = trace.get_tracer(SERVICE_NAME)
        set_tracer(resolved)
        return resolved
    except Exception:  # pragma: no cover - depends on the environment
        fallback = _NoopTracer()
        set_tracer(fallback)
        return fallback


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None: ...
    def add_event(self, name: str, attributes: Any = None) -> None: ...
    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *args: Any) -> None: ...


class _NoopTracer:
    """Used when the OTel SDK or collector is unavailable."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoopSpan:
        return _NoopSpan()


_TRACER: Any = None


def tracer() -> Any:
    global _TRACER
    if _TRACER is None:
        _TRACER = _NoopTracer()
    return _TRACER


def set_tracer(value: Any) -> None:
    global _TRACER
    _TRACER = value


@contextmanager
def stage_span(stage: str, **attributes: Any) -> Iterator[Any]:
    """Time one pipeline stage into both a span and a histogram.

    One context manager for both so the two can never disagree about where a
    stage started and stopped.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")

    import time

    started = time.perf_counter()
    with tracer().start_as_current_span(f"docintel.{stage}") as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        try:
            yield span
        finally:
            STAGE_LATENCY.labels(stage=stage).observe(time.perf_counter() - started)


@contextmanager
def extraction_span(document_id: str) -> Iterator[Any]:
    """Wrap one whole extraction."""
    import time

    started = time.perf_counter()
    with tracer().start_as_current_span("docintel.extract") as span:
        span.set_attribute("docintel.document_id", document_id)
        try:
            yield span
        finally:
            EXTRACTION_LATENCY.observe(time.perf_counter() - started)


def record_extraction(result: Any, grounding: Any = None, status: str = "completed") -> None:
    """Publish one extraction's metrics.

    Takes the ``ExtractionResult`` rather than loose numbers, so the metrics and
    the API response are derived from the same object and cannot drift.
    """
    RUNS.labels(status=status).inc()
    EXTRACTION_COST.observe(result.usage.cost_usd)

    usage = result.usage
    for kind, value in (
        ("input", usage.input_tokens),
        ("output", usage.output_tokens),
        ("cache_read", usage.cache_read_input_tokens),
        ("cache_write", usage.cache_creation_input_tokens),
    ):
        if value:
            TOKENS.labels(kind=kind).inc(value)

    # Per-stage latency from the result, so /metrics agrees with the response
    # body rather than being measured separately.
    for stage, millis in (result.latency_ms or {}).items():
        if stage in STAGES:
            STAGE_LATENCY.labels(stage=stage).observe(millis / 1000)

    for violation in result.violations:
        VIOLATIONS.labels(rule_id=violation.rule_id, severity=str(violation.severity)).inc()

    if result.needs_review:
        REVIEW.inc()

    if grounding is not None:
        for check in grounding.checks:
            GROUNDING.labels(status=str(check.status)).inc()
