"""
SS Intuition utilities for the FortiGate Semantic Shield.

This module encodes heuristics inspired by the Semantic Substrate: it uses the
golden ratio to distribute weight across the LOVE, JUSTICE, POWER, WISDOM axes
so automation can stay balanced while still pursuing higher throughput.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
CARDINAL_AXES = ("LOVE", "JUSTICE", "POWER", "WISDOM")
COMPASS_PRESETS: Dict[str, Tuple[str, str, str, str]] = {
    "theological": CARDINAL_AXES,
    "secular": ("TRUST", "FAIRNESS", "EFFICACY", "INSIGHT"),
}
AXIS_ALIAS_MAP: Dict[str, Tuple[str, ...]] = {
    "LOVE": ("LOVE", "TRUST"),
    "JUSTICE": ("JUSTICE", "FAIRNESS"),
    "POWER": ("POWER", "EFFICACY"),
    "WISDOM": ("WISDOM", "INSIGHT"),
}


@dataclass(frozen=True)
class CompassProfile:
    """Holds normalized weights for the four cardinal axes."""

    love: float
    justice: float
    power: float
    wisdom: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "LOVE": self.love,
            "JUSTICE": self.justice,
            "POWER": self.power,
            "WISDOM": self.wisdom,
        }

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.love, self.justice, self.power, self.wisdom)


def golden_ratio_profile(love_bias: float = 1.0) -> CompassProfile:
    """
    Create a compass profile anchored in the golden ratio.

    The LOVE axis is treated as the anchor anchor; the remaining axes cascade
    according to successive powers of phi^-1 to keep the profile in harmony.
    """

    inverse_phi = 1.0 / GOLDEN_RATIO
    weights = [
        love_bias,
        love_bias * inverse_phi,
        love_bias * inverse_phi**2,
        love_bias * inverse_phi**3,
    ]
    total = sum(weights)
    normalized = [value / total for value in weights]
    return CompassProfile(*normalized)


def blend_profiles(profiles: Iterable[CompassProfile]) -> CompassProfile:
    """
    Combine multiple profiles by averaging their contributions.

    Useful for blending governance input (e.g., finance + operations) with
    sector overlays while preserving compass equilibrium.
    """

    accum = [0.0, 0.0, 0.0, 0.0]
    count = 0
    for profile in profiles:
        vector = profile.as_tuple()
        accum = [a + b for a, b in zip(accum, vector)]
        count += 1

    if count == 0:
        raise ValueError("At least one profile must be provided")

    averaged = [value / count for value in accum]
    return CompassProfile(*averaged)


def fibonacci_window(sequence: Iterable[float]) -> List[float]:
    """
    Apply a Fibonacci-inspired smoothing window to a sequence of metrics.

    Later samples receive a phi-weighted emphasis to reflect the intuition that
    recent context should influence execution while still honoring history.
    """

    fib_weights = [1, 1, 2, 3, 5, 8]
    window_length = min(len(fib_weights), len(list(sequence)))
    if window_length == 0:
        return []

    weights = fib_weights[:window_length]
    total_weight = sum(weights)

    smoothed = []
    recent_values = list(sequence)[-window_length:]
    for idx in range(window_length):
        weighted_sum = sum(
            value * weight
            for value, weight in zip(recent_values[: idx + 1], weights[: idx + 1])
        )
        smoothed.append(weighted_sum / sum(weights[: idx + 1]))

    # Normalize smoothed output so it can be safely used as scaling factors.
    max_value = max(smoothed) if smoothed else 1.0
    return [value / max_value for value in smoothed]


def golden_batch_size(base: int, boost: float = 1.0) -> int:
    """
    Compute an event batch size using powers of phi to explore near-optimal loads.

    The result intentionally hovers around the base value to maintain balance
    under varying latency conditions.
    """

    candidate = base * (GOLDEN_RATIO ** (boost - 1.0))
    return max(1, int(round(candidate)))


def compass_profile_with_labels(
    profile: CompassProfile, preset: Optional[str] = None
) -> Dict[str, float]:
    """
    Represent a compass profile with axis labels appropriate for the requested preset.

    The theological preset (default) uses LOVE/JUSTICE/POWER/WISDOM, while the
    secular preset surfaces TRUST/FAIRNESS/EFFICACY/INSIGHT. Values always map back
    to the canonical axis ordering to preserve consistency.
    """

    axes = COMPASS_PRESETS.get((preset or "theological").lower(), CARDINAL_AXES)
    return {axis: value for axis, value in zip(axes, profile.as_tuple())}


def resolve_bias_overrides(overrides: Dict[str, float]) -> Dict[str, float]:
    """
    Map bias overrides (which may use canonical or alias labels) onto canonical axes.

    Returns a dictionary keyed by the canonical axis names (LOVE, JUSTICE, POWER, WISDOM).
    """

    resolved: Dict[str, float] = {}
    alias_lookup: Dict[str, str] = {}
    for canonical_axis, aliases in AXIS_ALIAS_MAP.items():
        for alias in aliases:
            alias_lookup[alias.upper()] = canonical_axis

    for label, value in overrides.items():
        canonical = alias_lookup.get(label.upper())
        if canonical is not None:
            resolved[canonical] = value

    return resolved

