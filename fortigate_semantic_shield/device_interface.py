"""
FortiGate device integration layer.

These classes provide the bridge between the semantic intelligence engine and
FortiGate appliances. The default implementations log intended actions; replace
the placeholders with concrete FortiOS REST/CLI operations in production.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class FortiGateAPIConfig:
    """Connection details for FortiGate REST integration."""

    host: str
    token: str
    verify_ssl: bool = True
    vdom: Optional[str] = None
    timeout: float = 5.0  # seconds
    extra_headers: Dict[str, str] = field(default_factory=dict)


class FortiGatePolicyApplier:
    """
    Applies high-level defense actions to a FortiGate device.

    Plug in REST, pyFG, or SSH/CLI logic inside the `apply_*` methods to
    implement real changes on the firewall.
    """

    def __init__(self, config: Optional[FortiGateAPIConfig] = None) -> None:
        self.config = config
        self.logger = LOGGER.getChild(self.__class__.__name__)

    def apply_response(self, response: Any) -> None:
        """Dispatch a semantic defense response to FortiGate subsystems."""
        self.logger.debug("Applying response %s", getattr(response, "response_id", "unknown"))

        blocking_rules = getattr(response, "blocking_rules", [])
        routing_mods = getattr(response, "routing_modifications", {})
        quarantine_actions = getattr(response, "quarantine_actions", [])
        healing_protocols = getattr(response, "healing_protocols", [])

        if blocking_rules:
            self.apply_blocking_rules(blocking_rules)

        if routing_mods:
            self.apply_routing_modifications(routing_mods)

        if quarantine_actions:
            self.apply_quarantine_actions(quarantine_actions)

        if healing_protocols:
            self.apply_healing_protocols(healing_protocols)

    # -- Individual action hooks -------------------------------------------------

    def apply_blocking_rules(self, rules: Iterable[Dict[str, Any]]) -> None:
        """Push blocking rules to the FortiGate firewall."""
        for rule in rules:
            self.logger.info("[FortiGate] Would apply blocking rule: %s", rule)

    def apply_quarantine_actions(self, actions: Iterable[str]) -> None:
        """Trigger quarantine workflows."""
        for action in actions:
            self.logger.info("[FortiGate] Would queue quarantine action: %s", action)

    def apply_routing_modifications(self, modifications: Dict[str, Any]) -> None:
        """Adjust routing or policy forwarding on the device."""
        self.logger.info("[FortiGate] Would apply routing modifications: %s", modifications)

    def apply_healing_protocols(self, protocols: Iterable[str]) -> None:
        """Launch healing / remediation protocols."""
        for protocol in protocols:
            self.logger.info("[FortiGate] Would trigger healing protocol: %s", protocol)


class FortiGateTelemetryCollector:
    """
    Maintains recent FortiGate telemetry (CPU, memory, sessions, etc.) and
    provides guardrail checks for the semantic engine.
    """

    def __init__(
        self,
        cpu_threshold: float = 85.0,
        memory_threshold: float = 80.0,
        queue_threshold: int = 1000,
    ) -> None:
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.queue_threshold = queue_threshold
        self.latest_metrics: Dict[str, Any] = {}
        self.logger = LOGGER.getChild(self.__class__.__name__)

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Ingest fresh telemetry from the FortiGate appliance."""
        self.latest_metrics.update(metrics)
        self.logger.debug("Telemetry updated: %s", metrics)

    def should_pause_learning(self) -> bool:
        """Return True if resource guardrails are exceeded."""
        cpu = float(self.latest_metrics.get("cpu_usage", 0.0))
        memory = float(self.latest_metrics.get("memory_usage", 0.0))
        queue = int(self.latest_metrics.get("session_queue", 0))

        pause = (
            cpu >= self.cpu_threshold
            or memory >= self.memory_threshold
            or queue >= self.queue_threshold
        )

        if pause:
            self.logger.warning(
                "Guardrails triggered (cpu=%.1f%%, memory=%.1f%%, queue=%d)",
                cpu,
                memory,
                queue,
            )
        return pause


class LearningPersistenceManager:
    """
    Handles periodic snapshotting and rotation of the learning database.

    Use this to archive the SQLite file to persistent storage on the FortiGate
    or a remote management server.
    """

    def __init__(
        self,
        database_path: str,
        export_directory: str,
        max_snapshots: int = 5,
    ) -> None:
        self.database_path = database_path
        self.export_directory = export_directory
        self.max_snapshots = max_snapshots
        self.logger = LOGGER.getChild(self.__class__.__name__)

    def export_snapshot(self) -> Optional[str]:
        """Create a timestamped copy of the learning database."""
        import shutil
        from datetime import datetime

        if not os.path.exists(self.database_path):
            self.logger.error("Database not found: %s", self.database_path)
            return None

        os.makedirs(self.export_directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(
            self.export_directory, f"learning_snapshot_{timestamp}.db"
        )

        shutil.copy2(self.database_path, export_path)
        self.logger.info("Exported learning snapshot to %s", export_path)
        self._enforce_rotation()
        return export_path

    def _enforce_rotation(self) -> None:
        """Keep only the latest N snapshots."""
        snapshots = sorted(
            (
                os.path.join(self.export_directory, name)
                for name in os.listdir(self.export_directory)
                if name.endswith(".db")
            ),
            key=os.path.getmtime,
            reverse=True,
        )

        for old_snapshot in snapshots[self.max_snapshots :]:
            try:
                os.remove(old_snapshot)
                self.logger.info("Removed old snapshot %s", old_snapshot)
            except OSError as exc:
                self.logger.warning("Failed to remove %s: %s", old_snapshot, exc)
