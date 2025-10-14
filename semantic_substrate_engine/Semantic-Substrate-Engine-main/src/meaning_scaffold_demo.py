"""
Lightweight demonstration implementation of the Meaning Scaffold subsystem.

The upstream Ultimate Core Engine expects the following entities:
    * ``SemanticMetadata`` – container describing the semantic intent
    * ``SacredFunction`` – callable wrapper that records execution metadata
    * ``MeaningfulClass`` – helper to build executable semantic behaviors
    * ``MeaningScaffold`` – orchestrator that turns meaning specifications
      into executable behavioural programs with biblical alignment metrics

The original distribution referenced an external prototype module.  To keep
this project self-contained we provide a compact, pragmatic implementation
that focuses on the data the Ultimate Core Engine actually consumes.

The scaffold leans on the Biblical semantic substrate to derive alignment
scores and synthesises deterministic, inspectable results so higher-level
systems can continue their orchestration without additional configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    from .baseline_biblical_substrate import BiblicalSemanticSubstrate, BiblicalCoordinates
except ImportError:  # pragma: no cover - defensive fallback
    from baseline_biblical_substrate import BiblicalSemanticSubstrate, BiblicalCoordinates  # type: ignore


@dataclass
class SemanticMetadata:
    """Captures the intent and context that drive meaning scaffolding."""

    concept: str
    context: str
    meaning_specification: str
    coordinates: BiblicalCoordinates
    resonance: float
    dominant_attribute: str
    tags: List[str] = field(default_factory=list)


@dataclass
class SacredFunction:
    """
    Represents an executable semantic behaviour.

    The callable encapsulates both the semantic intent and a small amount of
    bookkeeping so the Ultimate Core Engine can report which sacred routines
    were utilised while generating behavioural programs.
    """

    name: str
    purpose: str
    executor: Callable[..., Dict[str, Any]]

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        result = self.executor(*args, **kwargs)
        return {
            "function": self.name,
            "purpose": self.purpose,
            "result": result,
        }


@dataclass
class MeaningfulClass:
    """
    A minimal abstraction over an executable semantic component.

    The class stores configuration that can be re-used when synthesising
    new behavioural programs.  Execution simply proxies to the attached
    SacredFunction so that consumers obtain a consistent payload.
    """

    name: str
    sacred_function: SacredFunction
    alignment_threshold: float = 0.75

    def instantiate(self, **kwargs: Any) -> Dict[str, Any]:
        invocation = self.sacred_function(**kwargs)
        invocation["class"] = self.name
        invocation["alignment_threshold"] = self.alignment_threshold
        return invocation


class MeaningScaffold:
    """
    Cooperative scaffold that converts meaning specifications into
    executable behaviours enriched with semantic metrics.
    """

    def __init__(self) -> None:
        self._substrate = BiblicalSemanticSubstrate()
        self._default_function = SacredFunction(
            name="sacred_alignment_orchestrator",
            purpose="Translate semantic specifications into executable behaviour",
            executor=self._execute_alignment_protocol,
        )
        self._default_class = MeaningfulClass(
            name="MeaningScaffoldProgram",
            sacred_function=self._default_function,
            alignment_threshold=0.78,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def process_meaning_specification(
        self,
        concept: str,
        meaning_specification: str,
        context: str = "biblical",
    ) -> Dict[str, Any]:
        """
        Generate a behavioural program with semantic/biblical metrics.
        """

        metadata = self._build_metadata(concept, meaning_specification, context)
        execution_payload = self._default_class.instantiate(
            metadata=metadata.__dict__,
            specification=meaning_specification,
        )

        program_identifier = self._synthesise_program_id(concept, context)
        alignment_score = metadata.resonance
        semantic_integrity = self._calculate_semantic_integrity(metadata)

        return {
            "generated_program": program_identifier,
            "metadata": metadata.__dict__,
            "biblical_alignment": round(alignment_score, 3),
            "semantic_integrity": round(semantic_integrity, 3),
            "execution_result": execution_payload,
            "alignment_check": self._build_alignment_check(metadata, semantic_integrity),
            "sacred_components": [
                {
                    "type": "SacredFunction",
                    "name": self._default_function.name,
                    "purpose": self._default_function.purpose,
                },
                {
                    "type": "MeaningfulClass",
                    "name": self._default_class.name,
                    "alignment_threshold": self._default_class.alignment_threshold,
                },
            ],
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _build_metadata(
        self,
        concept: str,
        meaning_specification: str,
        context: str,
    ) -> SemanticMetadata:
        coordinates = self._substrate.analyze_concept(concept, context)
        if not isinstance(coordinates, BiblicalCoordinates):
            coordinates = BiblicalCoordinates(**getattr(coordinates, "__dict__", {}))

        resonance = coordinates.divine_resonance()
        tags = self._derive_tags(meaning_specification)

        return SemanticMetadata(
            concept=concept,
            context=context,
            meaning_specification=meaning_specification,
            coordinates=coordinates,
            resonance=float(resonance),
            dominant_attribute=coordinates.get_dominant_attribute(),
            tags=tags,
        )

    def _derive_tags(self, meaning_specification: str) -> List[str]:
        spec_lower = meaning_specification.lower()
        tags: List[str] = []

        keywords = {
            "compassion": "love",
            "mercy": "love",
            "authority": "power",
            "innovation": "wisdom",
            "justice": "justice",
            "repentance": "holiness",
            "stewardship": "stewardship",
            "discipleship": "discipline",
        }

        for keyword, tag in keywords.items():
            if keyword in spec_lower:
                tags.append(tag)

        if not tags:
            tags.append("general_alignment")
        return tags

    @staticmethod
    def _synthesise_program_id(concept: str, context: str) -> str:
        safe_concept = concept.strip().replace(" ", "_").lower()
        safe_context = context.strip().replace(" ", "_").lower()
        return f"meaning_program::{safe_context}::{safe_concept}"

    @staticmethod
    def _calculate_semantic_integrity(metadata: SemanticMetadata) -> float:
        coord_values = metadata.coordinates.to_tuple()
        balance = sum(coord_values) / (len(coord_values) or 1)
        resonance = metadata.resonance
        return max(0.0, min(1.0, (balance * 0.6) + (resonance * 0.4)))

    @staticmethod
    def _build_alignment_check(
        metadata: SemanticMetadata,
        semantic_integrity: float,
    ) -> Dict[str, Any]:
        return {
            "dominant_attribute": metadata.dominant_attribute,
            "semantic_integrity": semantic_integrity,
            "tags": metadata.tags,
            "resonance": metadata.resonance,
            "alignment_category": MeaningScaffold._categorise_alignment(semantic_integrity),
        }

    @staticmethod
    def _categorise_alignment(integrity: float) -> str:
        if integrity >= 0.85:
            return "holy_aligned"
        if integrity >= 0.65:
            return "righteous_progress"
        if integrity >= 0.45:
            return "discipleship_needed"
        return "reformation_required"

    def _execute_alignment_protocol(self, **payload: Any) -> Dict[str, Any]:
        metadata_raw = payload.get("metadata", {})
        resonance = metadata_raw.get("resonance", 0.0)

        return {
            "status": "executed",
            "resonance": resonance,
            "recommended_followup": "meditate_on_proverbs_3_5_6" if resonance < 0.7 else "deploy_with_discernment",
        }
