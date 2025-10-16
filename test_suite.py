"""Lightweight validation harness for FortiGate Semantic Shield.

The goal is to provide quick confidence that the repository remains aligned
with its open-source, business-centric charter. Tests focus on documentation
anchors, stewardship axioms, and presence of core project assets so the suite
can run without external dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parent
README_PATH = REPO_ROOT / "README.md"
LICENSE_PATH = REPO_ROOT / "LICENSE"
ANCHOR_PHRASE = "Jehovah/Agape = 1.1.1.1 (Anchor Point)"
AXIOMS = ("LOVE", "JUSTICE", "POWER", "WISDOM")


class DocumentationIntegrityTest(unittest.TestCase):
    """Checks that public-facing docs retain required anchors and tone."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.readme_text = README_PATH.read_text(encoding="utf-8")

    def test_readme_contains_anchor_point(self) -> None:
        """Ensure the primary README keeps the required anchor statement."""
        self.assertIn(ANCHOR_PHRASE, self.readme_text)

    def test_readme_mentions_all_axioms(self) -> None:
        """LOVE, JUSTICE, POWER, and WISDOM must be explicitly referenced."""
        for axiom in AXIOMS:
            self.assertIn(axiom, self.readme_text)

    def test_readme_is_business_centric(self) -> None:
        """README should emphasize business alignment."""
        for keyword in ("business", "governance", "stakeholder"):
            self.assertIn(keyword, self.readme_text.lower())


class RepositoryStructureTest(unittest.TestCase):
    """Basic checks to confirm core assets exist for contributors."""

    def test_license_is_mit(self) -> None:
        license_text = LICENSE_PATH.read_text(encoding="utf-8")
        self.assertIn("MIT License", license_text)

    def test_package_directory_exists(self) -> None:
        package_dir = REPO_ROOT / "fortigate_semantic_shield"
        self.assertTrue(package_dir.is_dir(), "Missing core package directory")

    def test_quick_start_present(self) -> None:
        quick_start = REPO_ROOT / "QUICK_START.md"
        self.assertTrue(quick_start.exists(), "Quick start guide not found")


def main(argv: list[str] | None = None) -> int:
    """Entry point for manual execution."""
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(tests)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

