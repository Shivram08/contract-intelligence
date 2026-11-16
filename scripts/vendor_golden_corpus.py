"""Copy the golden contracts' text into the repo, with a SHA-256 manifest.

CI has no copy of CUAD (~165 MB, gitignored), so before this the eval gate ran
over zero contracts and failed on F1 0.000 -- a red build that said nothing
about the model. Only the contracts the golden case set touches are vendored
(81 files, ~3.7 MB). CUAD is CC BY 4.0, so redistribution is permitted with the
attribution written alongside the files.

The manifest pins each file's bytes. ``tests/unit/test_golden_corpus.py``
re-hashes them in CI, so a git line-ending rewrite fails loudly instead of
shifting every gold offset.

Usage::

    uv run python scripts/vendor_golden_corpus.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from evals.cases import cases_by_document, load_cases  # noqa: E402

from docintel.text import contract_key  # noqa: E402

DEFAULT_CUAD_DIR: Final = REPO_ROOT / "data" / "raw" / "CUAD_v1"
DEFAULT_OUT: Final = REPO_ROOT / "evals" / "golden" / "corpus"

ATTRIBUTION: Final = """\
# Golden-set corpus

Plain-text contracts from the **Contract Understanding Atticus Dataset (CUAD)
v1**, copied byte-for-byte from `full_contract_txt/` for the contracts the
golden case set (`../cases.json`) uses. Vendored so the CI eval gate can run
without the full dataset. Regenerate with `scripts/vendor_golden_corpus.py`.

> Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). *CUAD: An
> Expert-Annotated NLP Dataset for Legal Contract Review.* NeurIPS Datasets and
> Benchmarks Track.

Distributed by The Atticus Project (https://www.atticusprojectai.org/cuad)
under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Unmodified.
The Atticus Project makes no representations about the licence status of the
underlying contracts, which are public SEC EDGAR filings.
"""


def vendor(cuad_dir: Path, out: Path) -> int:
    wanted = set(cases_by_document(load_cases()))
    target = out / "full_contract_txt"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    manifest: dict[str, str] = {}
    for path in sorted((cuad_dir / "full_contract_txt").glob("*.txt")):
        if contract_key(path.stem) not in wanted:
            continue
        data = path.read_bytes()
        (target / path.name).write_bytes(data)
        manifest[path.name] = hashlib.sha256(data).hexdigest()

    found = {contract_key(Path(name).stem) for name in manifest}
    if found != wanted:
        print(f"missing from CUAD: {sorted(wanted - found)}", file=sys.stderr)
        return 1

    (out / "manifest.json").write_text(
        json.dumps({"source": "CUAD_v1/full_contract_txt", "files": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(ATTRIBUTION, encoding="utf-8")
    print(f"vendored {len(manifest)} contracts to {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuad-dir", type=Path, default=DEFAULT_CUAD_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    return vendor(args.cuad_dir, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
