"""Guards the src-layout packaging.

`docintel` lives under `src/`, so it is importable only if hatchling's
`packages` setting and the editable install actually line up. That is a real
failure mode -- a typo in `[tool.hatch.build.targets.wheel]` produces a repo
where every other test passes locally (cwd on sys.path) and nothing imports
inside the container.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_docintel_is_importable_from_the_installed_package() -> None:
    module = importlib.import_module("docintel")
    assert module.__file__ is not None
    installed_path = Path(module.__file__).resolve()
    assert installed_path.parent == REPO_ROOT / "src" / "docintel"


def test_declared_python_floor_matches_ruff_and_mypy_targets() -> None:
    """A mismatch here silently disables pyupgrade rules or type narrowing."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["requires-python"] == ">=3.11"
    assert config["tool"]["ruff"]["target-version"] == "py311"
    assert config["tool"]["mypy"]["python_version"] == "3.11"
