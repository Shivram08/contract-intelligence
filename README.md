# Contract Intelligence Service

[![CI](https://github.com/Shivram08/contract-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Shivram08/contract-intelligence/actions/workflows/ci.yml)

Clause extraction from commercial contracts, with enforced output structure and
measured evaluation against expert annotations.

**A rigorously evaluated prototype, not a production system.** See
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) once written.

> **Status: in progress.** This README is written last, as a memo, at M5. See
> [`CLAUDE.md`](CLAUDE.md) for the full spec and
> [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md) for the pre-flight data audit.

## Running it

```bash
uv sync --all-groups                  # add --extra models for real embeddings
docker compose -f docker/docker-compose.yml up -d
uv run python -m docintel.cli index --split dev
uv run python -m docintel.cli search "which state's law governs this agreement"
```

Postgres is published on **5433**, not 5432, so a natively installed Postgres
cannot shadow the container. See [`.env.example`](.env.example).

## Data and attribution

Built on the **Contract Understanding Atticus Dataset (CUAD) v1** — 510
commercial contracts with 13,101 expert-labeled clauses across 41 categories.

> Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). *CUAD: An Expert-Annotated
> NLP Dataset for Legal Contract Review.* Proceedings of the Neural Information
> Processing Systems Track on Datasets and Benchmarks.

CUAD is distributed by [The Atticus Project](https://www.atticusprojectai.org/cuad)
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset is not
vendored in this repository; see [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md) for the
expected layout under `data/raw/`.
