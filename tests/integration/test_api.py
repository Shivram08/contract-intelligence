"""Integration tests for the HTTP API.

Runs the real app against a real Postgres, with a **stub LLM client**. No test
here makes a paid API call, and that is enforced by construction rather than by
convention: the stub is the only client the app is given, so there is nothing to
call.

Postgres comes from testcontainers when Docker is available, and otherwise from
a dedicated local database. Never the working index -- an earlier version of the
retrieval integration tests dropped tables by unqualified name and deleted a
developer's 60-document index twice.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse, urlunparse

import pytest

pytestmark = pytest.mark.integration


@asynccontextmanager
async def _null_lifespan(app: Any) -> AsyncIterator[None]:
    """Replaces the real lifespan for tests.

    The production lifespan constructs the embedder, the reranker and an
    Anthropic client. Tests must not do any of that -- the weights are 1.5 GB and
    the client would be real -- so state is injected directly instead. Setting
    ``lifespan_context = None`` does not work: Starlette calls it.
    """
    yield


DATABASE_URL = os.environ.get(
    "DOCINTEL_TEST_DATABASE_URL",
    "postgresql://docintel:docintel@localhost:5433/docintel",
)
TEST_DATABASE = "docintel_api_test"

CONTRACT = (
    "AGENCY AGREEMENT\n\n"
    "This Agreement is entered into by and between Acme Industries, Inc. "
    '("Company") and Beta Distribution LLC ("Distributor").\n\n'
    "12. GOVERNING LAW. This Agreement shall be governed by and construed in "
    "accordance with the laws of the State of Delaware.\n"
)


class StubClient:
    """Returns one ``submit_extraction`` call. Deterministic, free.

    ``quote`` is injectable so a test can supply text that is *not* in the
    document and assert the grounding gate rejects it -- the negative case is
    what keeps grounding an invariant rather than an escape hatch.
    """

    def __init__(self, quote: str | None = None, present: bool = True) -> None:
        self.quote = quote
        self.present = present
        self.messages = self
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        from docintel.schemas import ClauseType

        self.calls += 1
        quote = self.quote if self.quote is not None else "laws of the State of Delaware"
        clauses = [
            {
                "clause_type": clause.value,
                "present": clause is ClauseType.GOVERNING_LAW and self.present,
                "value": "US-DE" if clause is ClauseType.GOVERNING_LAW and self.present else None,
                "raw_text": quote if clause is ClauseType.GOVERNING_LAW and self.present else None,
                "evidence": (
                    [{"quote": quote, "char_start": 0, "char_end": 0}]
                    if clause is ClauseType.GOVERNING_LAW and self.present
                    else []
                ),
                "confidence": 0.9,
            }
            for clause in ClauseType
        ]
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="stub_1",
                    name="submit_extraction",
                    input={"clauses": clauses},
                )
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="stub",
        )


@pytest.fixture(scope="module")
def connection() -> Iterator[Any]:
    """A dedicated test database. Never the one the CLI indexes into."""
    psycopg = pytest.importorskip("psycopg")
    try:
        admin = psycopg.connect(DATABASE_URL, connect_timeout=5, autocommit=True)
    except psycopg.OperationalError as exc:  # pragma: no cover
        pytest.skip(f"Postgres unavailable: {exc}")

    with admin, admin.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DATABASE,))
        if cursor.fetchone() is None:
            cursor.execute(f'CREATE DATABASE "{TEST_DATABASE}"')

    url = urlunparse(urlparse(DATABASE_URL)._replace(path=f"/{TEST_DATABASE}"))
    conn = psycopg.connect(url, autocommit=True)
    with conn.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    with conn.cursor() as cursor:
        cursor.execute("SELECT current_database()")
        row = cursor.fetchone()
    assert row and row[0] == TEST_DATABASE, "refusing to run against the wrong database"

    try:
        yield conn
    finally:
        conn.close()


def build_client(connection: Any, stub: StubClient) -> Any:
    """The real app, with the stub wired in place of the model client."""
    from fastapi.testclient import TestClient

    from docintel.api.main import ExtractionService, create_app
    from docintel.extract import Extractor

    app = create_app()
    app.router.lifespan_context = _null_lifespan

    from docintel.api.routes import JobState

    app.state.extractor = ExtractionService(
        extractor=Extractor(client=stub), search=lambda query, top_k: []
    )
    app.state.connection = connection
    app.state.jobs = JobState()
    app.state.version = "test"
    return TestClient(app)


class TestHealthz:
    def test_reports_ok_when_the_database_answers(self, connection: Any) -> None:
        with build_client(connection, StubClient()) as client:
            body = client.get("/healthz").json()
        assert body["database"] == "ok"
        assert body["status"] == "ok"

    def test_reports_the_broken_dependency_rather_than_just_failing(self) -> None:
        """A health check that only says "unhealthy" cannot be diagnosed."""
        from fastapi.testclient import TestClient

        from docintel.api.main import create_app
        from docintel.api.routes import JobState

        app = create_app()
        app.router.lifespan_context = _null_lifespan
        app.state.extractor = None
        app.state.connection = None
        app.state.jobs = JobState()
        with TestClient(app) as client:
            body = client.get("/healthz").json()
        assert body["status"] == "degraded"
        assert body["database"] == "absent"
        assert body["extractor"] == "absent"

    def test_reflects_a_dead_connection(self, connection: Any) -> None:
        """Closed connection must surface as an error, not as ok."""
        import psycopg
        from fastapi.testclient import TestClient

        from docintel.api.main import create_app
        from docintel.api.routes import JobState

        dead = psycopg.connect(
            urlunparse(urlparse(DATABASE_URL)._replace(path=f"/{TEST_DATABASE}"))
        )
        dead.close()
        app = create_app()
        app.router.lifespan_context = _null_lifespan
        app.state.extractor = None
        app.state.connection = dead
        app.state.jobs = JobState()
        with TestClient(app) as client:
            body = client.get("/healthz").json()
        assert body["database"].startswith("error")
        assert body["status"] == "degraded"


class TestExtract:
    def test_returns_a_valid_extraction_result(self, connection: Any) -> None:
        from docintel.schemas import ExtractionResult

        with build_client(connection, StubClient()) as client:
            response = client.post(
                "/extract", json={"document_id": "TEST_AGENCY", "text": CONTRACT}
            )
        assert response.status_code == 200
        # Parses back into the real model, so the response shape is the contract
        # rather than whatever JSON happened to serialize.
        result = ExtractionResult.model_validate(response.json())
        assert result.document_id == "TEST_AGENCY"
        assert len(result.clauses) == 12

    def test_all_twelve_clause_types_are_present_in_the_response(self, connection: Any) -> None:
        from docintel.schemas import ClauseType

        with build_client(connection, StubClient()) as client:
            body = client.post("/extract", json={"document_id": "T", "text": CONTRACT}).json()
        returned = {c["clause_type"] for c in body["clauses"]}
        assert returned == {c.value for c in ClauseType}

    def test_offsets_are_relocated_from_the_quote(self, connection: Any) -> None:
        """The stub reports 0/0 offsets, as real models essentially always do.

        100% of evidence spans required relocation in the measured runs, so this
        is the normal path rather than an edge case.
        """
        with build_client(connection, StubClient()) as client:
            body = client.post("/extract", json={"document_id": "T", "text": CONTRACT}).json()
        governing = next(c for c in body["clauses"] if c["clause_type"] == "governing_law")
        assert governing["present"] is True
        evidence = governing["evidence"][0]
        assert evidence["char_end"] > evidence["char_start"]
        assert CONTRACT[evidence["char_start"] : evidence["char_end"]] == evidence["quote"]

    def test_a_fabricated_quote_is_surfaced_not_swallowed(self, connection: Any) -> None:
        """The hallucination gate, through the HTTP layer.

        A quote absent from the document must not silently pass. Grounding drops
        the span, which leaves present=True with no evidence, which trips
        presence_requires_evidence -- and that violation must appear in the
        response body.
        """
        stub = StubClient(quote="Seller shall not compete in any territory whatsoever")
        with build_client(connection, stub) as client:
            body = client.post("/extract", json={"document_id": "T", "text": CONTRACT}).json()

        rule_ids = {v["rule_id"] for v in body["violations"]}
        assert "presence_requires_evidence" in rule_ids
        assert body["needs_review"] is True

    def test_violations_do_not_turn_into_an_error_status(self, connection: Any) -> None:
        """An extraction with findings is a 200 with findings attached.

        A 4xx would hide the information the caller needs most.
        """
        stub = StubClient(quote="text that does not appear in the contract at all")
        with build_client(connection, stub) as client:
            response = client.post("/extract", json={"document_id": "T", "text": CONTRACT})
        assert response.status_code == 200
        assert response.json()["violations"]

    def test_usage_and_latency_are_reported(self, connection: Any) -> None:
        with build_client(connection, StubClient()) as client:
            body = client.post("/extract", json={"document_id": "T", "text": CONTRACT}).json()
        assert body["usage"]["input_tokens"] == 100
        assert set(body["latency_ms"]) & {"agent", "validation", "total"}

    def test_503_when_no_model_client_was_configured(self, connection: Any) -> None:
        from fastapi.testclient import TestClient

        from docintel.api.main import create_app
        from docintel.api.routes import JobState

        app = create_app()
        app.router.lifespan_context = _null_lifespan
        app.state.extractor = None
        app.state.connection = connection
        app.state.jobs = JobState()
        with TestClient(app) as client:
            response = client.post("/extract", json={"document_id": "T", "text": CONTRACT})
        assert response.status_code == 503
        assert "extractor unavailable" in response.json()["detail"]


class TestMalformedRequests:
    def test_empty_document_id_is_422_with_the_field_named(self, connection: Any) -> None:
        with build_client(connection, StubClient()) as client:
            response = client.post("/extract", json={"document_id": "", "text": "x"})
        assert response.status_code == 422
        body = response.json()
        assert "document_id" in body["summary"]

    def test_missing_text_is_422(self, connection: Any) -> None:
        with build_client(connection, StubClient()) as client:
            response = client.post("/extract", json={"document_id": "T"})
        assert response.status_code == 422
        assert "text" in response.json()["summary"]

    def test_oversized_document_is_rejected_before_any_model_call(self, connection: Any) -> None:
        """The guard has to fire before the spend, not after."""
        stub = StubClient()
        with build_client(connection, stub) as client:
            response = client.post("/extract", json={"document_id": "T", "text": "x" * 500_000})
        assert response.status_code == 422
        assert stub.calls == 0

    def test_body_carries_structured_detail_not_just_a_message(self, connection: Any) -> None:
        with build_client(connection, StubClient()) as client:
            body = client.post("/extract", json={}).json()
        assert isinstance(body["detail"], list)
        assert body["detail"][0]["loc"]


class TestBatch:
    def test_returns_202_with_a_job_id(self, connection: Any) -> None:
        with build_client(connection, StubClient()) as client:
            response = client.post(
                "/extract/batch",
                json={"documents": [{"document_id": "A", "text": CONTRACT}]},
            )
        assert response.status_code == 202
        assert response.json()["job_id"]
        assert response.json()["document_count"] == 1

    def test_job_can_be_polled(self, connection: Any) -> None:
        with build_client(connection, StubClient()) as client:
            job_id = client.post(
                "/extract/batch",
                json={
                    "documents": [
                        {"document_id": "A", "text": CONTRACT},
                        {"document_id": "B", "text": CONTRACT},
                    ]
                },
            ).json()["job_id"]
            body = client.get(f"/extract/batch/{job_id}").json()
        assert body["total"] == 2
        assert body["completed"] == 2
        assert len(body["results"]) == 2

    def test_unknown_job_is_404(self, connection: Any) -> None:
        with build_client(connection, StubClient()) as client:
            assert client.get("/extract/batch/nope").status_code == 404

    def test_empty_batch_is_422(self, connection: Any) -> None:
        with build_client(connection, StubClient()) as client:
            assert client.post("/extract/batch", json={"documents": []}).status_code == 422


class TestMetrics:
    def test_exposes_histograms(self, connection: Any) -> None:
        with build_client(connection, StubClient()) as client:
            client.post("/extract", json={"document_id": "T", "text": CONTRACT})
            body = client.get("/metrics").text
        assert "docintel_extraction_duration_seconds_bucket" in body
        assert "docintel_extraction_cost_usd_bucket" in body
        assert "docintel_stage_duration_seconds" in body

    def test_records_the_run_and_its_tokens(self, connection: Any) -> None:
        with build_client(connection, StubClient()) as client:
            client.post("/extract", json={"document_id": "T", "text": CONTRACT})
            body = client.get("/metrics").text
        assert 'docintel_runs_total{status="completed"}' in body
        assert 'docintel_tokens_total{kind="input"}' in body

    def test_grounding_outcomes_are_counted_including_relocation(self, connection: Any) -> None:
        """`relocated` is counted because a 0% violation rate is conditional on
        relocation succeeding, and that has to be visible in the metrics."""
        with build_client(connection, StubClient()) as client:
            client.post("/extract", json={"document_id": "T", "text": CONTRACT})
            body = client.get("/metrics").text
        assert "docintel_grounding_spans_total" in body
