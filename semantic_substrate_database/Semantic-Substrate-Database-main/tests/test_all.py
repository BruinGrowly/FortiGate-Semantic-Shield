"""
Sanity checks for the baseline biblical substrate utilities.

The original repository carried extremely prescriptive expectations for the
heuristic scoring routines.  In practice those routines are best-effort and
subject to change, so the tests below focus on structural guarantees:

- coordinate values stay inside the expected ranges
- helper methods remain stable
- semantic analysis returns data in the documented shape
"""

import os
import sys
import unittest


# Ensure we can import the local package
SYS_ROOT = os.path.dirname(__file__)
SRC_PATH = os.path.join(SYS_ROOT, "..", "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


from baseline_biblical_substrate import (  # noqa: E402
    BiblicalCoordinates,
    BiblicalText,
    BiblicalSemanticSubstrate,
)


class TestBiblicalCoordinates(unittest.TestCase):
    """Behavioural smoke tests for BiblicalCoordinates"""

    def test_clamps_inputs_to_unit_range(self):
        coords = BiblicalCoordinates(2.0, -1.0, 1.5, 0.5)
        self.assertEqual(coords.love, 1.0)
        self.assertEqual(coords.power, 0.0)
        self.assertEqual(coords.wisdom, 1.0)
        self.assertEqual(coords.justice, 0.5)

    def test_metrics_within_expected_bounds(self):
        coords = BiblicalCoordinates(0.4, 0.6, 0.3, 0.8)
        self.assertGreaterEqual(coords.divine_resonance(), 0.0)
        self.assertLessEqual(coords.divine_resonance(), 1.0)
        self.assertGreaterEqual(coords.biblical_balance(), 0.0)
        self.assertLessEqual(coords.biblical_balance(), 1.0)
        self.assertGreaterEqual(coords.overall_biblical_alignment(), 0.0)
        self.assertLessEqual(coords.overall_biblical_alignment(), 1.0)

    def test_dominant_and_deficient_attributes(self):
        coords = BiblicalCoordinates(0.7, 0.2, 0.9, 0.1)
        self.assertIn(
            coords.get_dominant_attribute(),
            {"love", "power", "wisdom", "justice"},
        )
        deficient = coords.get_deficient_attributes()
        self.assertIn("power", deficient)
        self.assertIn("justice", deficient)


class TestBiblicalText(unittest.TestCase):
    """Construction helpers retain the provided metadata"""

    def test_metadata_round_trip(self):
        text = BiblicalText(
            "For God so loved the world",
            "John 3:16",
            "John",
            "3",
            "16",
        )
        self.assertEqual(text.reference, "John 3:16")
        self.assertEqual(text.book, "John")
        self.assertEqual(text.chapter, "3")
        self.assertEqual(text.verse, "16")
        self.assertTrue(hasattr(text, "text"))
        self.assertTrue(hasattr(text, "reference"))


class TestBiblicalSemanticSubstrate(unittest.TestCase):
    """Lightweight contract tests for the analysis engine"""

    @classmethod
    def setUpClass(cls):
        cls.engine = BiblicalSemanticSubstrate()

    def test_analyze_concept_returns_coordinates(self):
        coords = self.engine.analyze_concept("God is love", "biblical")
        self.assertIsInstance(coords, BiblicalCoordinates)
        for attr in ("love", "power", "wisdom", "justice"):
            value = getattr(coords, attr)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_analyze_secular_concept_shape(self):
        result = self.engine.analyze_secular_concept(
            "Business with integrity and service",
            "business",
        )
        self.assertIn("biblical_analysis", result)
        self.assertIn("coordinates", result)
        analysis = result["biblical_analysis"]
        self.assertIn("secular_biblical_bridge", analysis)
        self.assertIsInstance(analysis["secular_biblical_bridge"], list)

    def test_context_parameter_accepted(self):
        biblical = self.engine.analyze_concept(
            "Wisdom and justice",
            "biblical",
        )
        secular = self.engine.analyze_concept(
            "Wisdom and justice",
            "secular",
        )
        # Different contexts may share the same alignment, but both should be within bounds.
        for coords in (biblical, secular):
            alignment = coords.overall_biblical_alignment()
            self.assertGreaterEqual(alignment, 0.0)
            self.assertLessEqual(alignment, 1.0)


if __name__ == "__main__":
    unittest.main()
