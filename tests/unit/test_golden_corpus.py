"""Guards on the vendored golden-set corpus.

CI has no copy of CUAD, so the text of the golden contracts is committed under
``evals/golden/corpus/``. Two things can silently break that:

1. Git rewriting the bytes (line-ending normalization). Every gold span and every
   cached prompt is keyed on exact characters, so a rewritten file turns the eval
   gate into a measurement of git's settings. The manifest pins each file's
   SHA-256 at vendoring time.
2. The corpus drifting from the case set. A case whose contract is missing used
   to be scored as an all-absent run -- the gate then reported F1 0.000 and
   failed for a reason that had nothing to do with the model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evals.cases import cases_by_document, load_cases
from evals.run_eval import missing_documents

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "evals" / "golden" / "corpus"
MANIFEST = CORPUS / "manifest.json"


def _manifest() -> dict[str, str]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files: dict[str, str] = payload["files"]
    return files


class TestVendoredCorpus:
    def test_every_file_matches_its_recorded_hash(self) -> None:
        for name, digest in _manifest().items():
            data = (CORPUS / "full_contract_txt" / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest, name

    def test_corpus_holds_no_unlisted_files(self) -> None:
        on_disk = {p.name for p in (CORPUS / "full_contract_txt").glob("*.txt")}
        assert on_disk == set(_manifest())

    def test_corpus_covers_every_golden_contract(self) -> None:
        wanted = set(cases_by_document(load_cases()))
        assert missing_documents(CORPUS, wanted) == set()


class TestMissingDocuments:
    def test_reports_ids_with_no_text_file(self, tmp_path: Path) -> None:
        txt = tmp_path / "full_contract_txt"
        txt.mkdir()
        (txt / "A.txt").write_text("alpha", encoding="utf-8")
        assert missing_documents(tmp_path, {"A", "B"}) == {"B"}

    def test_an_absent_corpus_misses_everything(self, tmp_path: Path) -> None:
        assert missing_documents(tmp_path / "nope", {"A"}) == {"A"}


def test_gate_refuses_to_score_an_incomplete_corpus(tmp_path: Path) -> None:
    from evals.run_eval import main

    (tmp_path / "full_contract_txt").mkdir()
    # Exit 2 is "refused to measure", distinct from 1 ("measured a regression").
    code = main(["--gate", "--quiet", "--baselines", "1_regex", "--cuad-dir", str(tmp_path)])
    assert code == 2
