"""
FortiGate Semantic Shield package.

Exports the primary interfaces for integrating the semantic substrate
intelligence layer with FortiGate appliances.
"""

__all__ = [
    "FortiGateSemanticShield",
    "FortiGatePolicyApplier",
    "FortiGateTelemetryCollector",
    "LearningPersistenceManager",
]

__version__ = "1.0.0"

from .intelligence import FortiGateSemanticShield  # noqa: E402
from .device_interface import (  # noqa: E402
    FortiGatePolicyApplier,
    FortiGateTelemetryCollector,
    LearningPersistenceManager,
)
