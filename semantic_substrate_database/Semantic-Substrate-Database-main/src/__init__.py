"""
Semantic Substrate Database v2.0 - ICE-Centric Database System

A revolutionary database system that stores and queries semantic meaning using
the ICE (Intent-Context-Execution) framework with 4D divine coordinate system.

Requires Semantic Substrate Engine v3.0+ for full functionality:
pip install semantic-substrate-engine
"""

from .semantic_substrate_database import SemanticSubstrateDatabase
from .enhanced_semantic_database import EnhancedSemanticSubstrateDatabase
from .meaning_based_database import MeaningBasedDatabase

# Database version
__version__ = "2.0.0"
__author__ = "BruinGrowly"
__license__ = "MIT"

__all__ = [
    "SemanticSubstrateDatabase",
    "EnhancedSemanticSubstrateDatabase",
    "MeaningBasedDatabase"
]

# Backwards compatibility alias
EnhancedSemanticDatabase = EnhancedSemanticSubstrateDatabase

# Note: ICE Framework components are imported from semantic_substrate_engine package
# For full ICE functionality, install: pip install semantic-substrate-engine
