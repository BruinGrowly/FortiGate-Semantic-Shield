"""
Tests for ingesting the Semantic Substrate Primer document.
"""

import json
import sys
import unittest
from pathlib import Path
from typing import Tuple

try:
    from semantic_substrate_database.meaning_based_programming import MeaningBasedExecutor  # type: ignore
except ImportError:  # pragma: no cover - fallback for direct execution
    SRC_PATH = Path(__file__).resolve().parent.parent / "src"
    if str(SRC_PATH) not in sys.path:
        sys.path.append(str(SRC_PATH))
    from meaning_based_programming import MeaningBasedExecutor  # type: ignore


def _version_tuple(path: Path) -> Tuple[int, ...]:
    """Convert a primer filename into a comparable version tuple."""
    stem = path.stem.split("_")[-1]
    parts = []
    for segment in stem.split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            parts.append(0)
    return tuple(parts) if parts else (0,)


def _latest_primer_path() -> Path:
    """Locate the most recent Semantic Substrate Primer file."""
    start = Path(__file__).resolve().parent
    repo_root = None
    for candidate in [start] + list(start.parents):
        matches = sorted(
            (p for p in candidate.glob("Semantic_Substrate_Primer_*.json") if p.is_file()),
            key=_version_tuple,
        )
        if matches:
            repo_root = candidate
            break
    if repo_root is None:
        raise FileNotFoundError("Semantic Substrate Primer file not found in repository.")
    candidates = sorted(
        (p for p in repo_root.glob("Semantic_Substrate_Primer_*.json") if p.is_file()),
        key=_version_tuple,
    )
    if not candidates:
        raise FileNotFoundError("Semantic Substrate Primer file not found in repository.")
    return candidates[-1]


class TestPrimerIngestion(unittest.TestCase):
    """Validate that the primer ingestion process surfaces expected metadata."""

    @classmethod
    def setUpClass(cls):
        cls.primer_path = _latest_primer_path()
        with open(cls.primer_path, "r", encoding="utf-8") as primer_file:
            cls.primer_data = json.load(primer_file)

    def test_executor_loads_primer_metadata(self):
        executor = MeaningBasedExecutor()
        metadata = getattr(executor, "primer_metadata", {})

        expected_version = self.primer_data.get("_version")
        self.assertEqual(metadata.get("version"), expected_version)
        if metadata.get("source_path"):
            self.assertEqual(
                Path(metadata["source_path"]).resolve(),
                self.primer_path.resolve(),
            )

        principles = executor.universal_principles
        primer_principles = self.primer_data.get("universal_principles", {})
        self.assertEqual(len(principles), len(primer_principles))

        anchor_expected = primer_principles.get("principle_1", {})
        anchor = principles.get("universal_anchor")
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.get("name"), anchor_expected.get("name"))
        self.assertEqual(anchor.get("description"), anchor_expected.get("statement"))
        self.assertEqual(anchor.get("statement"), anchor_expected.get("statement"))
        self.assertEqual(anchor.get("substrate_role"), anchor_expected.get("substrate_role"))
        self.assertEqual(anchor.get("primer_version"), expected_version)
        self.assertIsInstance(anchor.get("coordinates"), tuple)
