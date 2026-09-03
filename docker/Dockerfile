# syntax=docker/dockerfile:1.7
#
# Multi-stage. The build stage carries uv and compilers; the runtime stage
# carries neither -- only the resolved virtualenv and the source.
#
# Pinned by digest, not just tag: `3.11-slim-bookworm` is a moving target, and a
# service whose base image changes underneath it is not reproducible. Update the
# digest deliberately.

FROM python:3.11-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84 AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_FROZEN=1

# uv is copied from its own published image rather than curl|sh, so the version
# is pinned and the install is not a network script.
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, in their own layer keyed on the lockfile. Source changes
# then do not re-resolve or re-download anything.
# --extra models pulls torch + sentence-transformers. It roughly quadruples the
# image, and it is not optional: without it BgeEmbedder raises ImportError,
# _build_extractor swallows it, and the service starts permanently degraded with
# /healthz reporting "extractor: absent" and no hint why. Shipping an image that
# cannot do dense retrieval is shipping a broken service.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra models

COPY src/ ./src/
COPY data/reference/ ./data/reference/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra models


FROM python:3.11-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    # Model weights land on a mounted volume, so a container restart does not
    # re-download ~1.5 GB of bge-base and bge-reranker.
    HF_HOME=/models \
    TRANSFORMERS_OFFLINE=0

# Non-root, and the uid is fixed so a mounted volume's ownership is predictable
# across hosts.
RUN groupadd --gid 10001 docintel \
 && useradd --uid 10001 --gid docintel --create-home --shell /usr/sbin/nologin docintel \
 && mkdir -p /models && chown -R docintel:docintel /models

WORKDIR /app

COPY --from=build --chown=docintel:docintel /app/.venv /app/.venv
COPY --from=build --chown=docintel:docintel /app/src /app/src
COPY --from=build --chown=docintel:docintel /app/data/reference /app/data/reference

USER docintel

EXPOSE 8000

# Uses the app's own readiness endpoint rather than a TCP probe, so "healthy"
# means the database answers -- not merely that the port is open.
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request,sys,json; \
b=urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=4).read(); \
sys.exit(0 if json.loads(b)['database']=='ok' else 1)"

CMD ["uvicorn", "docintel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
