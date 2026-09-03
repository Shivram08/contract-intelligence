"""FastAPI application factory.

Dependencies are built once at startup and hung on ``app.state``: the embedder
and reranker load model weights, and doing that per request would add tens of
seconds to every call. M2 measured ~47s of reranker latency when the model was
constructed per invocation rather than once.

The app starts **degraded rather than refusing** when the database or the model
client is unavailable. ``/healthz`` reports which dependency is missing and
``/extract`` returns 503 with a reason. A service that will not boot without
every dependency cannot be diagnosed through its own health endpoint, which is
the moment you most want it.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Final

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from docintel.api.routes import JobState, router
from docintel.api.telemetry import REGISTRY, setup_tracing
from docintel.schemas import Document

__all__ = ["ExtractionService", "create_app"]


@dataclass(slots=True)
class ExtractionService:
    """Binds an ``Extractor`` to its search function.

    ``Extractor.extract`` takes the search callable per call so an ablation can
    swap retrieval arms without rebuilding the extractor. The API only ever uses
    one arm, so it is bound once here and the routes get a single ``run``. This
    adapter exists so neither side has to bend: the eval harness keeps its
    injectable search, and the routes stay thin.
    """

    extractor: Any
    search: Any

    def run(self, document: Document) -> Any:
        return self.extractor.extract(document, search=self.search)


VERSION: Final = "0.1.0"


def _build_extractor() -> tuple[Any, Any]:
    """Construct the extractor and its database connection.

    Returns ``(extractor, connection)``, either of which may be None. Failures
    are swallowed here on purpose -- the caller records them for ``/healthz``
    rather than crashing the process, so an operator can ask the service what is
    wrong instead of reading container logs.
    """
    connection: Any = None
    try:
        import psycopg
        from pgvector.psycopg import register_vector

        from docintel.config import get_settings, resolve_anthropic_api_key
        from docintel.extract import Extractor
        from docintel.retrieval.embed import BgeEmbedder
        from docintel.retrieval.hybrid import HybridRetriever
        from docintel.retrieval.rerank import BgeReranker
        from docintel.retrieval.rerank import rerank as apply_rerank

        settings = get_settings()
        connection = psycopg.connect(str(settings.database_url), connect_timeout=5)
        register_vector(connection)

        retriever = HybridRetriever(
            connection=connection,
            embedder=BgeEmbedder(),
            candidates_per_retriever=settings.retrieval.candidates_per_retriever,
            rrf_k=settings.retrieval.rrf_k,
        )
        reranker = BgeReranker()

        def search(query: str, top_k: int) -> Any:
            from docintel.api.telemetry import stage_span

            with stage_span("retrieval", **{"docintel.query": query[:120]}) as span:
                hits = list(retriever.search(query, top_k=top_k))
                span.set_attribute("docintel.hits", len(hits))
            if not hits:
                return hits
            with stage_span("rerank", **{"docintel.candidates": len(hits)}):
                return apply_rerank(reranker, query, hits, top_k=top_k)

        # Demo mode. Makes the full stack -- traces, metrics, the whole request
        # path -- demonstrable without spending money, which matters because a
        # live extraction costs $0.05-$0.67. The stub is a fixed canned
        # response, and /healthz reports `extractor: stub` so nobody can mistake
        # a demo run for a measurement.
        if os.environ.get("DOCINTEL_LLM_STUB") == "1":
            return (
                ExtractionService(extractor=Extractor(client=_StubClient()), search=search),
                connection,
            )

        api_key = resolve_anthropic_api_key()
        if not api_key and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return None, connection

        import anthropic

        return (
            ExtractionService(
                extractor=Extractor(client=anthropic.Anthropic(api_key=api_key)),
                search=search,
            ),
            connection,
        )
    except Exception:
        return None, connection


class _StubClient:
    """A deterministic stand-in for the Anthropic client.

    Returns one ``submit_extraction`` call quoting text taken from the document
    itself, so the grounding check does real work and the response is a real
    ``ExtractionResult`` rather than a fixture. Costs nothing and never varies.
    """

    def __init__(self) -> None:
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        from docintel.schemas import ClauseType

        text = ""
        for message in kwargs.get("messages", []):
            content = message.get("content")
            if isinstance(content, str):
                text = content
                break

        # Quote a real substring so grounding resolves rather than rejecting.
        quote = ""
        marker = "<contract>"
        if marker in text:
            body = text.split(marker, 1)[1].split("</contract>", 1)[0].strip()
            quote = " ".join(body.split()[:12])

        clauses = [
            {
                "clause_type": clause.value,
                "present": clause is ClauseType.PARTIES and bool(quote),
                "value": None,
                "raw_text": quote if clause is ClauseType.PARTIES and quote else None,
                "evidence": (
                    [{"quote": quote, "char_start": 0, "char_end": 0}]
                    if clause is ClauseType.PARTIES and quote
                    else []
                ),
                "confidence": 0.5,
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
                input_tokens=0,
                output_tokens=0,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="stub",
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_tracing(app)
    extractor, connection = _build_extractor()
    app.state.extractor = extractor
    app.state.connection = connection
    app.state.jobs = JobState()
    app.state.version = VERSION
    try:
        yield
    finally:
        if connection is not None:
            connection.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Contract Intelligence Service",
        version=VERSION,
        summary=(
            "Clause extraction from commercial contracts with verbatim "
            "grounding and deterministic validation."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """422 with the field paths, not just "unprocessable entity".

        The default body is already structured; this keeps it and adds a
        human-readable summary, because a caller sending a 400k-character
        document should be told which limit it broke.
        """
        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.errors(),
                "summary": "; ".join(
                    f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                    for err in exc.errors()[:5]
                ),
            },
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Prometheus exposition.

        Served from the module's own registry rather than the global default, so
        an eval run importing the same metrics in-process cannot pollute what
        the service reports.
        """
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
