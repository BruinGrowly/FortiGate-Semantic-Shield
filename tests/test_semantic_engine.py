"""
Unit tests for the Semantic Engine.

This module tests the coordinate generation logic in the semantic engine
to ensure it is deterministic, robust, and produces valid output.
"""
import pytest
from guardian_ai.semantic_engine import analyze

def test_determinism():
    """Tests that the same input text always produces the same coordinates."""
    text = "This is a test of determinism."
    coords1 = analyze(text)
    coords2 = analyze(text)
    assert coords1 == coords2, "The analyze function is not deterministic."

def test_output_range():
    """Tests that all coordinate values are within the [0.0, 1.0] range."""
    text = "This is a test of the output range."
    coords = analyze(text)
    for value in coords.to_tuple():
        assert 0.0 <= value <= 1.0, f"Coordinate value {value} is out of range."

def test_empty_string():
    """Tests that an empty string produces valid coordinates."""
    coords = analyze("")
    assert coords is not None
    for value in coords.to_tuple():
        assert 0.0 <= value <= 1.0, "Empty string produced out-of-range coordinates."

def test_non_ascii_characters():
    """Tests that the engine can handle non-ASCII characters."""
    text = "这是一个测试"  # "This is a test" in Chinese
    coords = analyze(text)
    assert coords is not None
    for value in coords.to_tuple():
        assert 0.0 <= value <= 1.0, "Non-ASCII text produced out-of-range coordinates."
