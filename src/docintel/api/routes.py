"""HTTP routes.

The handlers are thin on purpose. Extraction logic lives in
``docintel.extract``, which the eval harness also calls -- so the API and the
measured numbers exercise the same code path. A route that reimplemented any of
it would mean `docs/RESULTS.md` describes a system the service does not run.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Final

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

from docintel.api.telemetry import extraction_span, record_extraction
from docintel.review.queue import QueueEntry, Resolution, ReviewQueue
from docintel.schemas import Document, ExtractionResult

__all__ = ["JobState", "router"]

router: Final = APIRouter()

#: Guard against a request that would cost more than a document is worth. The
#: measured range was $0.05-$0.67 per document; 200k characters is above every
#: contract in CUAD (max ~338k chars, but that is one outlier).
MAX_DOCUMENT_CHARS: Final = 400_000


class ExtractRequest(BaseModel):
    """A document to extract from.

    Text is supplied by the caller rather than fetched by id, because offsets in
    the response index into exactly the bytes that were sent. Resolving an id to
    text server-side would introduce a second copy that could differ.
    """

    document_id: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)
    metadata: dict[str, str] = Field(default_factory=dict)

    def to_document(self) -> Document:
        return Document(document_id=self.document_id, text=self.text, metadata=self.metadata)


class BatchRequest(BaseModel):
    documents: list[ExtractRequest] = Field(min_length=1, max_length=100)


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatus
    document_count: int


class JobView(BaseModel):
    job_id: str
    status: JobStatus
    submitted_at: datetime
    completed: int
    total: int
    results: list[ExtractionResult] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)


class HealthView(BaseModel):
    """Health, reflecting dependencies rather than just liveness.

    ``status`` is ``ok`` only when the database answers. A health check that
    returns 200 because the process is alive tells a load balancer nothing about
    whether requests will succeed.
    """

    status: str
    database: str
    extractor: str
    version: str


@dataclass(slots=True)
class JobState:
    """In-process batch job store.

    Deliberately in-memory and deliberately called out: jobs do not survive a
    restart and do not work behind more than one replica. A real deployment
    needs Postgres or a queue here. It is a job-id API with honest limits rather
    than a durable one pretending otherwise -- see LIMITATIONS.md.
    """

    jobs: dict[str, JobView] = field(default_factory=dict)

    def create(self, count: int) -> JobView:
        job = JobView(
            job_id=str(uuid.uuid4()),
            status=JobStatus.QUEUED,
            submitted_at=datetime.now(UTC),
            completed=0,
            total=count,
        )
        self.jobs[job.job_id] = job
        return job


def get_extractor(request: Request) -> Any:
    extractor = getattr(request.app.state, "extractor", None)
    if extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="extractor unavailable: the service started without a model client",
        )
    return extractor


def get_review_queue(request: Request) -> ReviewQueue | None:
    """The queue on the app's connection, created (and its table) on first use.

    None when the service has no database: extraction still works and the
    review flag is still on the response, but nothing is persisted.
    """
    queue: ReviewQueue | None = getattr(request.app.state, "review_queue", None)
    if queue is not None:
        return queue
    connection = getattr(request.app.state, "connection", None)
    if connection is None:
        return None
    queue = ReviewQueue(connection)
    queue.ensure_schema()
    request.app.state.review_queue = queue
    return queue


def require_review_queue(request: Request) -> ReviewQueue:
    queue = get_review_queue(request)
    if queue is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="review queue unavailable: the service has no database",
        )
    return queue


def _enqueue(queue: ReviewQueue | None, outcome: Any) -> int | None:
    review = getattr(outcome, "review", None)
    if queue is None or review is None:
        return None
    return queue.enqueue(review)


def get_jobs(request: Request) -> JobState:
    jobs: JobState = request.app.state.jobs
    return jobs


@router.get("/healthz", response_model=HealthView, tags=["ops"])
def healthz(request: Request) -> HealthView:
    """Readiness, including whether the database actually answers."""
    database = "unknown"
    connection = getattr(request.app.state, "connection", None)
    if connection is None:
        database = "absent"
    else:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            database = "ok"
        except Exception as exc:
            database = f"error: {type(exc).__name__}"

    service = getattr(request.app.state, "extractor", None)
    if service is None:
        extractor = "absent"
    elif (
        type(getattr(service, "extractor", None)).__name__ == "Extractor"
        and type(getattr(getattr(service, "extractor", None), "client", None)).__name__
        == "_StubClient"
    ):
        # Named distinctly so a demo run can never be mistaken for a measurement.
        extractor = "stub"
    else:
        extractor = "ok"
    healthy = database == "ok" and extractor in {"ok", "stub"}
    return HealthView(
        status="ok" if healthy else "degraded",
        database=database,
        extractor=extractor,
        version=getattr(request.app.state, "version", "0.1.0"),
    )


@router.post(
    "/extract",
    response_model=ExtractionResult,
    tags=["extraction"],
    summary="Extract all 12 clause types from one contract",
)
def extract(
    payload: Annotated[ExtractRequest, Body()],
    extractor: Annotated[Any, Depends(get_extractor)],
    response: Response,
    queue: Annotated[ReviewQueue | None, Depends(get_review_queue)],
) -> ExtractionResult:
    """Synchronous single-document extraction.

    Returns the full ``ExtractionResult``, including rule violations and the
    review flag. **Violations are part of a 200 response, not an error.** An
    extraction that tripped a rule is a successful extraction with findings
    attached; turning it into a 4xx would hide exactly the information the
    caller needs, and the review queue exists because some results are meant to
    be looked at rather than rejected.
    """
    document = payload.to_document()
    with extraction_span(document.document_id) as span:
        outcome = extractor.run(document)
        result: ExtractionResult = outcome.result
        span.set_attribute("docintel.clauses_present", sum(1 for c in result.clauses if c.present))
        span.set_attribute("docintel.violations", len(result.violations))
        span.set_attribute("docintel.needs_review", result.needs_review)
        span.set_attribute("docintel.cost_usd", result.usage.cost_usd)
        record_extraction(
            result,
            grounding=getattr(outcome, "grounding", None),
            status="completed" if not result.stopped_on_budget else "incomplete",
        )
    # A flagged result is persisted, not just flagged. The id travels in a
    # header so the response body stays exactly ExtractionResult.
    review_id = _enqueue(queue, outcome)
    if review_id is not None:
        response.headers["X-Review-Id"] = str(review_id)
    return result


@router.post(
    "/extract/batch",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["extraction"],
)
def extract_batch(
    payload: Annotated[BatchRequest, Body()],
    request: Request,
    extractor: Annotated[Any, Depends(get_extractor)],
    jobs: Annotated[JobState, Depends(get_jobs)],
) -> JobAccepted:
    """Accept a batch and return a job id immediately.

    202 rather than 200: extraction took 25-225 seconds per document when
    measured, so a synchronous batch of 100 would exceed any sane client
    timeout.
    """
    job = jobs.create(len(payload.documents))
    documents = [item.to_document() for item in payload.documents]
    queue = get_review_queue(request)

    background = getattr(request.state, "background_tasks", None)
    if background is not None:  # pragma: no cover - exercised via the app
        background.add_task(_run_job, extractor, jobs, job.job_id, documents, queue)
    else:
        _run_job(extractor, jobs, job.job_id, documents, queue)

    return JobAccepted(
        job_id=job.job_id, status=jobs.jobs[job.job_id].status, document_count=job.total
    )


@router.get("/extract/batch/{job_id}", response_model=JobView, tags=["extraction"])
def get_job(job_id: str, jobs: Annotated[JobState, Depends(get_jobs)]) -> JobView:
    job = jobs.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
    return job


def _run_job(
    extractor: Any,
    jobs: JobState,
    job_id: str,
    documents: list[Document],
    queue: ReviewQueue | None = None,
) -> None:
    """Process a batch. One document failing does not fail the job.

    A batch where document 40 of 100 raises should return 99 results and one
    error, not lose the 39 already paid for.
    """
    job = jobs.jobs[job_id]
    job.status = JobStatus.RUNNING
    for document in documents:
        try:
            with extraction_span(document.document_id):
                outcome = extractor.run(document)
            job.results.append(outcome.result)
            record_extraction(outcome.result, grounding=getattr(outcome, "grounding", None))
            _enqueue(queue, outcome)
        except Exception as exc:
            job.errors[document.document_id] = f"{type(exc).__name__}: {exc}"
        finally:
            job.completed += 1
    job.status = JobStatus.SUCCEEDED if not job.errors else JobStatus.FAILED


# --------------------------------------------------------------------------
# Review queue
# --------------------------------------------------------------------------


class ReviewView(BaseModel):
    id: int
    document_id: str
    reasons: list[str]
    status: str
    reviewer: str | None
    note: str | None
    created_at: datetime
    resolved_at: datetime | None
    result: ExtractionResult
    correction: ExtractionResult | None

    @classmethod
    def of(cls, entry: QueueEntry) -> ReviewView:
        return cls(
            id=entry.id,
            document_id=entry.document_id,
            reasons=entry.reasons,
            status=entry.status,
            reviewer=entry.reviewer,
            note=entry.note,
            created_at=entry.created_at,
            resolved_at=entry.resolved_at,
            result=entry.result,
            correction=entry.correction,
        )


class ResolveRequest(BaseModel):
    resolution: Resolution
    reviewer: str = Field(min_length=1, max_length=200)
    note: str | None = Field(default=None, max_length=4_000)
    #: Required when the resolution is "corrected", refused otherwise.
    correction: ExtractionResult | None = None


@router.get("/review", response_model=list[ReviewView], tags=["review"])
def list_review(
    queue: Annotated[ReviewQueue, Depends(require_review_queue)],
    status_filter: Annotated[str, Query(alias="status")] = "open",
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[ReviewView]:
    """Items awaiting a person, oldest first."""
    if status_filter not in {"open", *(r.value for r in Resolution)}:
        raise HTTPException(status_code=422, detail=f"unknown status {status_filter!r}")
    return [ReviewView.of(e) for e in queue.items(status_filter, limit)]


@router.get("/review/{item_id}", response_model=ReviewView, tags=["review"])
def get_review(
    item_id: int, queue: Annotated[ReviewQueue, Depends(require_review_queue)]
) -> ReviewView:
    entry = queue.get(item_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"no review item {item_id}")
    return ReviewView.of(entry)


@router.post("/review/{item_id}/resolve", response_model=ReviewView, tags=["review"])
def resolve_review(
    item_id: int,
    payload: Annotated[ResolveRequest, Body()],
    queue: Annotated[ReviewQueue, Depends(require_review_queue)],
) -> ReviewView:
    """Close an item. 409 if it was already resolved: the first decision stands."""
    if (payload.resolution is Resolution.CORRECTED) != (payload.correction is not None):
        raise HTTPException(
            status_code=422,
            detail="a correction is required for 'corrected', and only for it",
        )
    entry = queue.resolve(
        item_id, payload.resolution, payload.reviewer, payload.note, payload.correction
    )
    if entry is None:
        existing = queue.get(item_id)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"no review item {item_id}")
        raise HTTPException(
            status_code=409, detail=f"review item {item_id} is already {existing.status}"
        )
    return ReviewView.of(entry)
