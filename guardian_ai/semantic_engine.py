"""
Semantic Engine for the Guardian AI System.

This module is responsible for analyzing text and generating its
corresponding 4D semantic coordinates (Love, Justice, Power, Wisdom).
"""
import hashlib
from .models import SemanticCoordinates

def analyze(text: str) -> SemanticCoordinates:
    """
    Analyzes a string of text and returns its 4D semantic coordinates.

    This implementation uses a SHA-256 hash to generate a deterministic
    and content-dependent set of coordinates. The first 16 bytes of the
    hash are used to generate the four 32-bit floating-point values.
    """
    # Use SHA-256 to create a deterministic hash of the input text.
    hasher = hashlib.sha256()
    hasher.update(text.encode('utf-8'))
    digest = hasher.digest()

    # The max value for a 4-byte integer is 2^32 - 1.
    max_val = 2**32 - 1

    # Take the first 16 bytes of the hash and split them into four 4-byte chunks.
    # Convert each chunk to an integer and normalize it to a float between 0 and 1.
    love = int.from_bytes(digest[0:4], 'big') / max_val
    justice = int.from_bytes(digest[4:8], 'big') / max_val
    power = int.from_bytes(digest[8:12], 'big') / max_val
    wisdom = int.from_bytes(digest[12:16], 'big') / max_val

    return SemanticCoordinates(love=love, justice=justice, power=power, wisdom=wisdom)
