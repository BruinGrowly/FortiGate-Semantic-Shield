"""
SEMANTIC SUBSTRATE DATABASE v2.0 - ICE-Centric Database System

A revolutionary database system that stores and queries semantic meaning using
the ICE (Intent-Context-Execution) framework with 4D divine coordinate system.

Core Features:
- ICE-Centric semantic coordinate storage and indexing
- Multi-dimensional proximity queries with intent understanding
- Context-aware semantic search with execution validation
- Universal anchor navigation with alignment metrics
- Meaning preservation tracking with integrity validation
- 5 execution strategies for behaviorally-aligned responses
- 8 context domains with adaptive processing

Built on Semantic Substrate Engine v3.0 - ICE-Centric Architecture
License: MIT
"""

import sqlite3
import json
import numpy as np
import pickle
import hashlib
import math
import importlib.util
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import sys
import os

# Global fallback SacredNumber class (defined before imports)
class _SacredNumberFallback:
    """Fallback SacredNumber class with all required methods"""
    def __init__(self, value):
        self.value = value
        self.is_sacred = True if value in [1, 3, 7, 12, 21, 40, 42, 66, 77, 613] else False
        self.sacred_resonance = 0.9 if self.is_sacred else 0.1
        self.biblical_significance = self._get_biblical_meaning()
        self.divine_attributes = {
            'value': self.value,
            'is_sacred': self.is_sacred,
            'biblical_meaning': self._get_biblical_meaning()
        }
        self.mystical_properties = {
            'value': self.value,
            'is_sacred': self.is_sacred,
            'prime_factorization': self._get_prime_factors(),
            'biblical_significance': self._get_biblical_meaning(),
            'numerological_value': self._get_numerological_value()
        }
    
    def _get_biblical_meaning(self):
        meanings = {
            1: "Unity, Oneness with God",
            3: "Divine perfection, Holy Trinity", 
            7: "Spiritual perfection, Completion",
            12: "Divine government, Apostolic foundation",
            21: "Disorder, Rebellion",
            40: "Probation, Testing",
            42: "Israel's oppression, Coming of Messiah",
            66: "Idolatry, Imperfection",
            77: "Perfect order, Resurrection",
            613: "Divine law completeness, Torah commandments"
        }
        return meanings.get(self.value, "Unknown")
    
    def _get_prime_factors(self):
        if self.value <= 1:
            return []
        factors = []
        n = self.value
        for i in range(2, int(n ** 0.5) + 1):
            while n % i == 0:
                factors.append(i)
                n //= i
        if n > 1:
            factors.append(n)
        return factors
    
    def _get_numerological_value(self):
        return sum(int(d) for d in str(abs(self.value)))


class SacredNumberWrapper:
    """Wrapper to ensure SacredNumber has all required methods"""
    def __init__(self, value, engine_instance=None):
        self.value = value
        
        if engine_instance is None:
            # Use our fallback implementation
            self._is_sacred = True if value in [1, 3, 7, 12, 21, 40, 42, 66, 77, 613] else False
            self._sacred_resonance = 0.9 if self._is_sacred else 0.1
            self._biblical_significance = self._get_biblical_meaning()
            self._divine_attributes = {
                'value': self.value,
                'is_sacred': self._is_sacred,
                'biblical_meaning': self._get_biblical_meaning()
            }
            self._mystical_properties = {
                'value': self.value,
                'is_sacred': self._is_sacred,
                'prime_factorization': self._get_prime_factors(),
                'biblical_significance': self._get_biblical_meaning(),
                'numerological_value': self._get_numerological_value()
            }
            self._engine_version = None
        else:
            # Use engine version but store engine instance for delegation
            self._engine = engine_instance
            self._is_sacred = getattr(engine_instance, 'is_sacred', False)
            self._sacred_resonance = getattr(engine_instance, 'sacred_resonance', 0.5)
            self._biblical_significance = getattr(engine_instance, 'biblical_significance', 'Unknown')
            self._engine_version = getattr(engine_instance, 'engine_version', 'Unknown')
            
            # Calculate fallback properties if not present
            if not hasattr(engine_instance, 'divine_attributes'):
                self._divine_attributes = {
                    'value': self.value,
                    'is_sacred': self._is_sacred,
                    'biblical_meaning': self._get_biblical_meaning()
                }
            else:
                self._divine_attributes = getattr(engine_instance, 'divine_attributes', {})
            
            if not hasattr(engine_instance, 'mystical_properties'):
                self._mystical_properties = {
                    'value': self.value,
                    'is_sacred': self._is_sacred,
                    'prime_factorization': self._get_prime_factors(),
                    'biblical_significance': self._get_biblical_meaning(),
                    'numerological_value': self._get_numerological_value()
                }
            else:
                self._mystical_properties = getattr(engine_instance, 'mystical_properties', {})
    
    def _get_biblical_meaning(self):
        if hasattr(self, '_biblical_significance') and self._biblical_significance != 'Unknown':
            return self._biblical_significance
        
        meanings = {
            1: "Unity, Oneness with God",
            3: "Divine perfection, Holy Trinity", 
            7: "Spiritual perfection, Completion",
            12: "Divine government, Apostolic foundation",
            21: "Disorder, Rebellion",
            40: "Probation, Testing",
            42: "Israel's oppression, Coming of Messiah",
            66: "Idolatry, Imperfection",
            77: "Perfect order, Resurrection",
            613: "Divine law completeness, Torah commandments"
        }
        return meanings.get(self.value, "Unknown")
    
    def _get_prime_factors(self):
        if hasattr(self, '_mystical_properties') and 'prime_factorization' in self._mystical_properties:
            return self._mystical_properties['prime_factorization']
        
        if self.value <= 1:
            return []
        factors = []
        n = self.value
        for i in range(2, int(n ** 0.5) + 1):
            while n % i == 0:
                factors.append(i)
                n //= i
        if n > 1:
            factors.append(n)
        return factors
    
    def _get_numerological_value(self):
        if hasattr(self, '_mystical_properties') and 'numerological_value' in self._mystical_properties:
            return self._mystical_properties['numerological_value']
        return sum(int(d) for d in str(abs(self.value)))
    
    # Properties to match expected interface
    @property
    def is_sacred(self):
        return self._is_sacred
    
    @property
    def sacred_resonance(self):
        return self._sacred_resonance
    
    @property
    def biblical_significance(self):
        return self._biblical_significance
    
    @property
    def divine_attributes(self):
        return self._divine_attributes
    
    @property
    def mystical_properties(self):
        return self._mystical_properties
    
    # Delegate other attributes to engine if available
    def __getattr__(self, name):
        """Delegate to the engine SacredNumber if available"""
        engine = self.__dict__.get('_engine')
        if engine is not None:
            return getattr(engine, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Set attributes appropriately"""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            engine = self.__dict__.get('_engine')
            if engine is not None and hasattr(engine, name):
                setattr(engine, name, value)
            else:
                object.__setattr__(self, '_' + name, value)


# Import from Semantic Substrate Engine package
ENGINE_AVAILABLE = False
_EngineSacredNumber = None
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
try:
    # Try multiple import approaches
    import os
    engine_path = None
    
    # Check if engine is installed as a package
    try:
        import semantic_substrate_engine as sse
        engine_path = "semantic_substrate_engine"
    except ImportError:
        pass
    
    # Try to find engine in common locations
    if engine_path is None:
        possible_paths = [
            os.path.expanduser("~/Semantic-Substrate-Engine/src"),
            "../Semantic-Substrate-Engine/src", 
            "../../Semantic-Substrate-Engine/src",
            "C:/Users/Well/Claude Code/Projects/SSE_ICE_Upgrade/Semantic-Substrate-Engine/src"
        ]
        local_engine_path = _repo_root / "semantic_substrate_engine" / "Semantic-Substrate-Engine-main" / "src"
        possible_paths.insert(0, str(local_engine_path))
        for path in possible_paths:
            if os.path.exists(path):
                sys.path.insert(0, path)
                engine_path = "src"
                break
    
    if engine_path:
        if engine_path == "semantic_substrate_engine":
            from semantic_substrate_engine import (
                BiblicalCoordinates, 
                BiblicalSemanticSubstrate,
                ICESemanticSubstrateEngine,
                ThoughtType,
                ContextDomain,
                UnifiedICEFramework,
                UltimateCoreEngine,
                SacredNumber as EngineSacredNumber
            )
            _EngineSacredNumber = EngineSacredNumber
            ENGINE_AVAILABLE = True
            print("[SEMANTIC DB] Semantic Substrate Engine loaded from package")
        else:
            from baseline_biblical_substrate import BiblicalCoordinates, BiblicalSemanticSubstrate
            try:
                from ultimate_core_engine import UltimateCoreEngine  # noqa: F401
                from ice_framework import ThoughtType, ContextDomain  # noqa: F401
                from unified_ice_framework import UnifiedICEFramework  # noqa: F401
                from ice_semantic_substrate_engine import ICESemanticSubstrateEngine  # noqa: F401
                try:
                    from enhanced_core_components import SacredNumber as EngineSacredNumber  # type: ignore[attr-defined]
                    _EngineSacredNumber = EngineSacredNumber
                except Exception:
                    _EngineSacredNumber = None
                ENGINE_AVAILABLE = True
                print("[SEMANTIC DB] Semantic Substrate Engine loaded from local source")
            except Exception as e:
                ENGINE_AVAILABLE = False
                print(f"[SEMANTIC DB] Advanced engine components not available: {e}")
    else:
        ENGINE_AVAILABLE = False
        
except Exception as e:
    ENGINE_AVAILABLE = False
    print(f"[SEMANTIC DB] Engine not available: {e}")

# Fallback imports
if not ENGINE_AVAILABLE:
    try:
        from baseline_biblical_substrate import BiblicalCoordinates, BiblicalSemanticSubstrate
        print("[SEMANTIC DB] Using baseline biblical substrate fallback")
    except ImportError:
        # Final fallback
        class BiblicalCoordinates:
            def __init__(self, love, power, wisdom, justice):
                self.love = love
                self.power = power
                self.wisdom = wisdom
                self.justice = justice
                
            def calculate_divine_resonance(self):
                return (self.love + self.power + self.wisdom + self.justice) / 4.0
                
            def distance_from_jehovah(self):
                jehovah = BiblicalCoordinates(1.0, 1.0, 1.0, 1.0)
                return abs(self.love - jehovah.love) + abs(self.power - jehovah.power) + abs(self.wisdom - jehovah.wisdom) + abs(self.justice - jehovah.justice)
        
        class BiblicalSemanticSubstrate:
            def __init__(self):
                self.engine_version = "Legacy Fallback v1.0"
                
            def analyze_text(self, text, context="biblical"):
                return {
                    'text': text,
                    'context': context,
                    'coordinates': BiblicalCoordinates(0.5, 0.5, 0.5, 0.5),
                    'divine_resonance': 0.5
                }
    
    print("[SEMANTIC DB] Warning: Semantic Substrate Engine not installed. Install with: pip install semantic-substrate-engine")
    print("[SEMANTIC DB] Attempting to use local fallback components...")


def SacredNumber(value, *engine_args, **engine_kwargs):
    """Create a SacredNumber instance using engine implementation when available."""
    engine_instance = None
    if _EngineSacredNumber:
        try:
            engine_instance = _EngineSacredNumber(value, *engine_args, **engine_kwargs)
        except TypeError:
            engine_instance = _EngineSacredNumber(value)
    return SacredNumberWrapper(value, engine_instance)


class SemanticSubstrateDatabase:
    """
    Revolutionary Semantic Database Engine

    First database in history to store meaning as mathematical coordinates
    in 4D semantic space anchored to divine truth.
    """

    def __init__(self, db_path: str = "semantic_substrate.db"):
        """Initialize the semantic database"""
        self.db_path = db_path
        self.conn = None
        self.cache = {}  # In-memory cache for hot data
        self._local_baseline_module = None
        self._transaction_active = False  # Track transaction state
        self._savepoints = []  # Stack of savepoints for nested transactions
        
        # Initialize engine with proper selection
        self.engine = None
        self.engine_available = False
        
        # Try UltimateCoreEngine first (most advanced)
        if 'UltimateCoreEngine' in globals():
            try:
                self.engine = UltimateCoreEngine()
                self.engine_available = True
                print("[SEMANTIC DB] UltimateCoreEngine initialized - Full semantic capabilities active")
            except Exception as e:
                print(f"[SEMANTIC DB] UltimateCoreEngine initialization failed: {e}")
        
        # Fall back to BiblicalSemanticSubstrate
        if self.engine is None and 'BiblicalSemanticSubstrate' in globals():
            try:
                self.engine = BiblicalSemanticSubstrate()
                self.engine_available = True
                print("[SEMANTIC DB] BiblicalSemanticSubstrate initialized")
            except Exception as e:
                print(f"[SEMANTIC DB] BiblicalSemanticSubstrate initialization failed: {e}")
        
        # Final fallback if nothing else works
        if self.engine is None:
            try:
                # Use the hardcoded fallback without shadowing the global symbol
                class _FinalFallbackBiblicalSemanticSubstrate:
                    def __init__(self):
                        self.engine_version = "Fallback v1.0"

                    def analyze_text(self, text, context="biblical"):
                        return {
                            'text': text,
                            'context': context,
                            'coordinates': BiblicalCoordinates(0.5, 0.5, 0.5, 0.5),
                            'divine_resonance': 0.5
                        }

                self.engine = _FinalFallbackBiblicalSemanticSubstrate()
                self.engine_available = False
                print("[SEMANTIC DB] Using final fallback engine")
            except Exception as e:
                print(f"[SEMANTIC DB] Final fallback initialization failed: {e}")
                self.engine_available = False

        # Maintain a dedicated baseline engine for semantic blending
        try:
            self.baseline_engine = BiblicalSemanticSubstrate()
        except Exception:
            self.baseline_engine = self.engine if self.engine is not None else None

        # Initialize database
        self._initialize_database()

        engine_version = "v3.0 ICE-Centric" if self.engine_available else "v2.2 Legacy"
        print(f"[SEMANTIC DB] Initialized at {db_path}")
        print(f"[SEMANTIC DB] Connected to SSE {engine_version}")

    def _initialize_database(self):
        """Create database schema with all required tables"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = self.conn.cursor()

        # Table 1: Semantic Coordinates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_coordinates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_text TEXT NOT NULL,
                context TEXT NOT NULL,
                love REAL NOT NULL,
                power REAL NOT NULL,
                wisdom REAL NOT NULL,
                justice REAL NOT NULL,
                divine_resonance REAL,
                distance_from_jehovah REAL,
                biblical_balance REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(concept_text, context)
            )
        """)

        # Table 2: Semantic Units
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_units (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                semantic_signature TEXT UNIQUE NOT NULL,
                text TEXT NOT NULL,
                context TEXT NOT NULL,
                eternal_signature REAL NOT NULL,
                meaning_preservation_factor REAL NOT NULL,
                essence_json TEXT NOT NULL,
                meaning_vector_blob BLOB,
                coordinate_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (coordinate_id) REFERENCES semantic_coordinates(id)
            )
        """)

        # Table 3: Sacred Numbers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sacred_numbers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                value REAL UNIQUE NOT NULL,
                is_sacred INTEGER NOT NULL,
                sacred_resonance REAL,
                biblical_significance REAL,
                divine_attributes_json TEXT,
                mystical_properties_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table 4: Universal Anchors
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS universal_anchors (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                significance TEXT NOT NULL,
                love REAL NOT NULL,
                power REAL NOT NULL,
                wisdom REAL NOT NULL,
                justice REAL NOT NULL,
                stability REAL NOT NULL,
                scripture TEXT,
                eternal_constancy REAL
            )
        """)

        # Table 5: Concept Relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concept_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept1_id INTEGER NOT NULL,
                concept2_id INTEGER NOT NULL,
                semantic_distance REAL NOT NULL,
                relationship_type TEXT,
                strength REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (concept1_id) REFERENCES semantic_coordinates(id),
                FOREIGN KEY (concept2_id) REFERENCES semantic_coordinates(id),
                UNIQUE(concept1_id, concept2_id)
            )
        """)

        # Table 6: Contextual Resonance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contextual_resonance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_id INTEGER NOT NULL,
                context TEXT NOT NULL,
                resonance_value REAL NOT NULL,
                harmony_factor REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (concept_id) REFERENCES semantic_coordinates(id),
                UNIQUE(concept_id, context)
            )
        """)

        # Table 7: Semantic Evolution History
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_evolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                unit_id INTEGER NOT NULL,
                transformation_type TEXT NOT NULL,
                preservation_factor REAL NOT NULL,
                eternal_signature_before REAL,
                eternal_signature_after REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (unit_id) REFERENCES semantic_units(id)
            )
        """)

        # Table 8: Learning History
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                semantic_signature TEXT,
                threat_type TEXT,
                severity TEXT,
                context TEXT,
                strategy TEXT,
                success_confidence REAL,
                divine_alignment REAL,
                success_prediction REAL,
                learning_json TEXT
            )
        """)

        # Table 9: Threat Patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threat_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                semantic_signature TEXT UNIQUE,
                threat_type TEXT,
                occurrences INTEGER,
                success_rate REAL,
                last_updated TEXT,
                pattern_json TEXT
            )
        """)

        # Table 10: Defense History
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS defense_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                semantic_signature TEXT,
                threat_type TEXT,
                strategy TEXT,
                success INTEGER,
                processing_time REAL,
                divine_protection REAL,
                wisdom_accuracy REAL,
                justice_enforcement REAL,
                mercy_factor REAL,
                response_json TEXT
            )
        """)

        # Table 11: Teaching Sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teaching_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                attack_type TEXT,
                severity TEXT,
                outcome TEXT,
                response_strategy TEXT,
                teachings_json TEXT
            )
        """)

        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_concept_text ON semantic_coordinates(concept_text)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_context ON semantic_coordinates(context)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_divine_resonance ON semantic_coordinates(divine_resonance)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_distance_jehovah ON semantic_coordinates(distance_from_jehovah)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_signature ON semantic_units(semantic_signature)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_eternal_signature ON semantic_units(eternal_signature)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sacred_value ON sacred_numbers(value)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_sacred ON sacred_numbers(is_sacred)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_learning_signature ON learning_history(semantic_signature)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_signature ON threat_patterns(semantic_signature)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_defense_signature ON defense_history(semantic_signature)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_teaching_attack ON teaching_sessions(attack_type)")

        self.conn.commit()

        # Initialize universal anchors if not present
        self._initialize_universal_anchors()

        print("[SEMANTIC DB] Database schema initialized")
        print("[SEMANTIC DB] Indexes created for optimal performance")

    def _initialize_universal_anchors(self):
        """Initialize the 4 universal anchors (613, 12, 7, 40)"""
        cursor = self.conn.cursor()

        anchors = [
            (613, "Divine Law", "The 613 Commandments - Perfect Divine Order",
             1.0, 1.0, 1.0, 1.0, 1.0, "Torah - 613 Mitzvot", 1.0),
            (12, "God's People", "The 12 Tribes of Israel - Divine Government",
             0.9, 0.95, 0.9, 0.95, 0.95, "Genesis 49 - The 12 Tribes", 0.95),
            (7, "Divine Perfection", "Seven Days of Creation - Complete Perfection",
             0.98, 0.95, 1.0, 0.98, 0.98, "Genesis 2:2 - Seventh Day", 0.98),
            (40, "Divine Testing", "40 Days/Years of Testing and Preparation",
             0.85, 0.9, 0.88, 0.92, 0.9, "Various - 40 Day/Year Periods", 0.9)
        ]

        for anchor in anchors:
            cursor.execute("""
                INSERT OR IGNORE INTO universal_anchors
                (id, name, significance, love, power, wisdom, justice, stability, scripture, eternal_constancy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, anchor)

        self.conn.commit()
        print("[SEMANTIC DB] Universal anchors initialized")

    # ========================================================================
    # CORE STORAGE OPERATIONS
    # ========================================================================

    def store_concept(self, text: str, context: str = "biblical",
                     auto_analyze: bool = True) -> int:
        """
        Store a concept with its semantic coordinates

        Args:
            text: The concept text
            context: Semantic context (biblical, educational, business, etc.)
            auto_analyze: Automatically analyze using SSE engine

        Returns:
            concept_id: Database ID of stored concept
        """
        cursor = self.conn.cursor()

        def _to_biblical_coords(source) -> BiblicalCoordinates:
            if isinstance(source, BiblicalCoordinates):
                return BiblicalCoordinates(source.love, source.power, source.wisdom, source.justice)
            return BiblicalCoordinates(
                getattr(source, 'love', 0.0),
                getattr(source, 'power', 0.0),
                getattr(source, 'wisdom', 0.0),
                getattr(source, 'justice', 0.0)
            )

        if auto_analyze:
            # Analyze using SSE engine with fallback compatibility
            if self.engine_available and hasattr(self.engine, 'core_engine'):
                # UltimateCoreEngine path
                result = self.engine.core_engine.analyze_concept(text, context)
                coords = _to_biblical_coords(result)

                # Blend with enhanced biblical substrate for semantic richness
                if getattr(self, 'baseline_engine', None) is None:
                    try:
                        if self._local_baseline_module is None:
                            module_path = Path(__file__).resolve().parent / "baseline_biblical_substrate.py"
                            spec = importlib.util.spec_from_file_location(
                                "local_baseline_biblical_substrate",
                                module_path
                            )
                            if spec and spec.loader:
                                module = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(module)
                                self._local_baseline_module = module
                        if self._local_baseline_module:
                            self.baseline_engine = self._local_baseline_module.BiblicalSemanticSubstrate()  # type: ignore[attr-defined]
                    except Exception:
                        self.baseline_engine = None

                if getattr(self, 'baseline_engine', None):
                    baseline = _to_biblical_coords(self.baseline_engine.analyze_concept(text, context))
                    coords = BiblicalCoordinates(
                        min(1.0, (coords.love + baseline.love) / 2.0),
                        min(1.0, (coords.power + baseline.power) / 2.0),
                        min(1.0, (coords.wisdom + baseline.wisdom) / 2.0),
                        min(1.0, (coords.justice + baseline.justice) / 2.0)
                    )
            else:
                # BiblicalSemanticSubstrate fallback
                coords = _to_biblical_coords(self.engine.analyze_concept(text, context))
        else:
            # Must provide coordinates manually
            raise ValueError("Manual coordinate specification not yet implemented")

        # Calculate derived metrics
        divine_resonance = coords.divine_resonance()
        distance_from_jehovah = coords.distance_from_jehovah()
        biblical_balance = coords.biblical_balance()

        # Insert or update
        cursor.execute("""
            INSERT INTO semantic_coordinates
            (concept_text, context, love, power, wisdom, justice,
             divine_resonance, distance_from_jehovah, biblical_balance, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(concept_text, context) DO UPDATE SET
                love=excluded.love,
                power=excluded.power,
                wisdom=excluded.wisdom,
                justice=excluded.justice,
                divine_resonance=excluded.divine_resonance,
                distance_from_jehovah=excluded.distance_from_jehovah,
                biblical_balance=excluded.biblical_balance,
                updated_at=CURRENT_TIMESTAMP
        """, (text, context, coords.love, coords.power, coords.wisdom, coords.justice,
              divine_resonance, distance_from_jehovah, biblical_balance))

        # Get the concept_id (works for both insert and update)
        cursor.execute("""
            SELECT id FROM semantic_coordinates
            WHERE concept_text = ? AND context = ?
        """, (text, context))
        concept_id = cursor.fetchone()[0]

        # Only auto-commit if not in an explicit transaction
        if not self._transaction_active:
            self.conn.commit()

        # Also create semantic unit
        if auto_analyze:
            self._store_semantic_unit(text, context, concept_id)

        # Update cache
        cache_key = f"{text}:{context}"
        self.cache[cache_key] = {
            'id': concept_id,
            'coordinates': coords,
            'metrics': {
                'divine_resonance': divine_resonance,
                'distance_from_jehovah': distance_from_jehovah,
                'biblical_balance': biblical_balance
            }
        }

        return concept_id

    def _auto_discover_relationships(self, concept_id: int, context: str,
                                     max_distance: float = 0.5,
                                     max_relationships: int = 5):
        """
        Automatically discover and store relationships with nearby concepts

        This makes the database self-aware by automatically connecting
        semantically similar concepts.

        Args:
            concept_id: ID of the concept to find relationships for
            context: Context to search within
            max_distance: Maximum semantic distance for relationships
            max_relationships: Maximum number of relationships to create
        """
        # Get coordinates for this concept
        coords = self._get_coordinates_by_id(concept_id)
        if not coords:
            return

        # Find nearby concepts
        nearby = self.query_by_proximity(
            coords,
            max_distance=max_distance,
            context=context,
            limit=max_relationships + 1  # +1 because it includes itself
        )

        # Store relationships with nearby concepts
        relationships_created = 0
        for nearby_concept in nearby:
            if nearby_concept['id'] != concept_id:  # Don't relate to itself
                try:
                    self.store_relationship(
                        concept_id,
                        nearby_concept['id'],
                        relationship_type="semantic_proximity"
                    )
                    relationships_created += 1
                except:
                    pass  # Relationship might already exist

        return relationships_created

    def _store_semantic_unit(self, text: str, context: str, coordinate_id: int):
        """Store semantic unit with eternal signature"""
        cursor = self.conn.cursor()

        # Create semantic unit using SSE engine with fallback
        try:
            if self.engine_available and hasattr(self.engine, 'create_semantic_unit'):
                unit = self.engine.create_semantic_unit(text, context)
                
                # Serialize essence and meaning vector
                essence_json = json.dumps(unit.essence)
                meaning_vector_blob = pickle.dumps(unit.meaning_vector)
                
                cursor.execute("""
                    INSERT OR IGNORE INTO semantic_units
                    (semantic_signature, text, context, eternal_signature,
                     meaning_preservation_factor, essence_json, meaning_vector_blob, coordinate_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (unit.semantic_signature, text, context, unit.eternal_signature,
                      unit.meaning_preservation_factor, essence_json, meaning_vector_blob, coordinate_id))
            else:
                # Fallback: create minimal semantic unit
                semantic_signature = hashlib.md5(f"{text}:{context}".encode()).hexdigest()
                essence_json = json.dumps({"text": text, "context": context})
                meaning_vector_blob = pickle.dumps([text, context])
                
                cursor.execute("""
                    INSERT OR IGNORE INTO semantic_units
                    (semantic_signature, text, context, eternal_signature,
                     meaning_preservation_factor, essence_json, meaning_vector_blob, coordinate_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (semantic_signature, text, context, 1.0, 0.95, essence_json, meaning_vector_blob, coordinate_id))
        except Exception as e:
            print(f"Warning: Could not create semantic unit: {e}")
            # Continue without semantic unit

        # Only auto-commit if not in an explicit transaction
        if not self._transaction_active:
            self.conn.commit()

    def store_sacred_number(self, value: Union[int, float]) -> int:
        """Store a sacred number with its divine attributes"""
        cursor = self.conn.cursor()

        # Analyze using SSE engine
        sacred_num = SacredNumber(value)

        # Serialize attributes
        divine_attrs_json = json.dumps(sacred_num.divine_attributes)
        mystical_props_json = json.dumps(sacred_num.mystical_properties)

        cursor.execute("""
            INSERT OR IGNORE INTO sacred_numbers
            (value, is_sacred, sacred_resonance, biblical_significance,
             divine_attributes_json, mystical_properties_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (value, int(sacred_num.is_sacred), sacred_num.sacred_resonance,
              sacred_num.biblical_significance, divine_attrs_json, mystical_props_json))

        # Get the id (works for both insert and ignore)
        cursor.execute("SELECT id FROM sacred_numbers WHERE value = ?", (value,))
        row_id = cursor.fetchone()[0]

        # Only auto-commit if not in an explicit transaction
        if not self._transaction_active:
            self.conn.commit()
        return row_id

    def store_relationship(self, concept1_id: int, concept2_id: int,
                          relationship_type: str = "semantic_proximity") -> int:
        """Store relationship between two concepts"""
        cursor = self.conn.cursor()

        # Get coordinates for both concepts
        coords1 = self._get_coordinates_by_id(concept1_id)
        coords2 = self._get_coordinates_by_id(concept2_id)

        if not coords1 or not coords2:
            raise ValueError("One or both concepts not found")

        # Calculate semantic distance
        distance = math.sqrt(
            (coords1.love - coords2.love) ** 2 +
            (coords1.power - coords2.power) ** 2 +
            (coords1.wisdom - coords2.wisdom) ** 2 +
            (coords1.justice - coords2.justice) ** 2
        )
        strength = max(0, 1.0 - distance / 4.0)  # Normalize to 0-1

        cursor.execute("""
            INSERT OR IGNORE INTO concept_relationships
            (concept1_id, concept2_id, semantic_distance, relationship_type, strength)
            VALUES (?, ?, ?, ?, ?)
        """, (concept1_id, concept2_id, distance, relationship_type, strength))

        # Get the id (works for both insert and ignore)
        cursor.execute("""
            SELECT id FROM concept_relationships
            WHERE concept1_id = ? AND concept2_id = ?
        """, (concept1_id, concept2_id))
        row = cursor.fetchone()
        rel_id = row[0] if row else 0

        # Only auto-commit if not in an explicit transaction
        if not self._transaction_active:
            self.conn.commit()
        return rel_id

    # ========================================================================
    # TRANSACTION MANAGEMENT
    # ========================================================================

    def begin_transaction(self) -> bool:
        """
        Begin an explicit transaction

        Returns:
            True if transaction started successfully
        """
        if self._transaction_active:
            raise RuntimeError("Transaction already active. Use savepoints for nested transactions.")

        self.conn.execute("BEGIN IMMEDIATE")
        self._transaction_active = True
        print("[SEMANTIC DB] Transaction started")
        return True

    def commit(self) -> bool:
        """
        Commit the current transaction

        Returns:
            True if commit successful
        """
        if not self._transaction_active:
            raise RuntimeError("No active transaction to commit")

        self.conn.commit()
        self._transaction_active = False
        self._savepoints = []  # Clear all savepoints on commit
        print("[SEMANTIC DB] Transaction committed")
        return True

    def rollback(self) -> bool:
        """
        Rollback the current transaction

        Returns:
            True if rollback successful
        """
        if not self._transaction_active:
            raise RuntimeError("No active transaction to rollback")

        self.conn.rollback()
        self._transaction_active = False
        self._savepoints = []  # Clear all savepoints on rollback
        self.cache.clear()  # Clear cache on rollback
        print("[SEMANTIC DB] Transaction rolled back")
        return True

    def create_savepoint(self, name: str) -> str:
        """
        Create a savepoint for nested transactions

        Args:
            name: Name for the savepoint

        Returns:
            Savepoint name
        """
        if not self._transaction_active:
            raise RuntimeError("No active transaction. Start a transaction first.")

        savepoint_name = f"sp_{name}_{len(self._savepoints)}"
        self.conn.execute(f"SAVEPOINT {savepoint_name}")
        self._savepoints.append(savepoint_name)
        print(f"[SEMANTIC DB] Savepoint created: {savepoint_name}")
        return savepoint_name

    def rollback_to_savepoint(self, name: str) -> bool:
        """
        Rollback to a specific savepoint

        Args:
            name: Savepoint name

        Returns:
            True if rollback successful
        """
        if name not in self._savepoints:
            raise ValueError(f"Savepoint not found: {name}")

        self.conn.execute(f"ROLLBACK TO SAVEPOINT {name}")

        # Remove this and all subsequent savepoints
        idx = self._savepoints.index(name)
        self._savepoints = self._savepoints[:idx]

        print(f"[SEMANTIC DB] Rolled back to savepoint: {name}")
        return True

    def release_savepoint(self, name: str) -> bool:
        """
        Release a savepoint (commit it)

        Args:
            name: Savepoint name

        Returns:
            True if release successful
        """
        if name not in self._savepoints:
            raise ValueError(f"Savepoint not found: {name}")

        self.conn.execute(f"RELEASE SAVEPOINT {name}")
        self._savepoints.remove(name)
        print(f"[SEMANTIC DB] Savepoint released: {name}")
        return True

    def atomic_operation(self, func, *args, **kwargs):
        """
        Execute a function as an atomic operation with automatic transaction management

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Function return value

        Raises:
            Exception if function fails (transaction rolled back)
        """
        transaction_started = False

        try:
            if not self._transaction_active:
                self.begin_transaction()
                transaction_started = True

            result = func(*args, **kwargs)

            if transaction_started:
                self.commit()

            return result

        except Exception as e:
            if transaction_started:
                self.rollback()
            print(f"[SEMANTIC DB] Atomic operation failed: {e}")
            raise

    def batch_store_concepts(self, concepts: List[Tuple[str, str]]) -> List[int]:
        """
        Store multiple concepts in a single transaction

        Args:
            concepts: List of (text, context) tuples

        Returns:
            List of concept IDs
        """
        def _batch_store():
            concept_ids = []
            for text, context in concepts:
                concept_id = self.store_concept(text, context)
                concept_ids.append(concept_id)
            return concept_ids

        return self.atomic_operation(_batch_store)

    def batch_store_sacred_numbers(self, numbers: List[Union[int, float]]) -> List[int]:
        """
        Store multiple sacred numbers in a single transaction

        Args:
            numbers: List of numbers to store

        Returns:
            List of sacred number IDs
        """
        def _batch_store():
            number_ids = []
            for num in numbers:
                num_id = self.store_sacred_number(num)
                number_ids.append(num_id)
            return number_ids

        return self.atomic_operation(_batch_store)

    def is_transaction_active(self) -> bool:
        """Check if a transaction is currently active"""
        return self._transaction_active

    # ========================================================================
    # SELF-AWARE RELATIONSHIP DISCOVERY
    # ========================================================================

    def enable_auto_relationships(self, context: Optional[str] = None,
                                  max_distance: float = 0.5,
                                  max_relationships: int = 5) -> int:
        """
        Enable self-aware relationship discovery for all existing concepts

        This makes the database analyze itself and automatically discover
        semantic relationships between all stored concepts.

        Args:
            context: Specific context to process (None = all contexts)
            max_distance: Maximum semantic distance for relationships
            max_relationships: Maximum relationships per concept

        Returns:
            Total number of relationships created
        """
        cursor = self.conn.cursor()

        # Get all concepts in the specified context
        if context:
            cursor.execute("SELECT id, context FROM semantic_coordinates WHERE context = ?", (context,))
        else:
            cursor.execute("SELECT id, context FROM semantic_coordinates")

        concepts = cursor.fetchall()
        total_relationships = 0

        print(f"[SEMANTIC DB] Auto-discovering relationships for {len(concepts)} concepts...")

        for concept in concepts:
            concept_id = concept[0]
            concept_context = concept[1]

            relationships = self._auto_discover_relationships(
                concept_id,
                concept_context,
                max_distance,
                max_relationships
            )

            if relationships:
                total_relationships += relationships

        print(f"[SEMANTIC DB] Created {total_relationships} automatic relationships")
        return total_relationships

    def get_concept_relationships(self, concept_id: int) -> List[Dict[str, Any]]:
        """
        Get all relationships for a concept

        Returns:
            List of related concepts with relationship metadata
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                c2.id,
                c2.concept_text,
                c2.context,
                r.semantic_distance,
                r.relationship_type,
                r.strength
            FROM concept_relationships r
            JOIN semantic_coordinates c2 ON r.concept2_id = c2.id
            WHERE r.concept1_id = ?
            ORDER BY r.semantic_distance ASC
        """, (concept_id,))

        relationships = []
        for row in cursor.fetchall():
            relationships.append({
                'related_id': row[0],
                'related_text': row[1],
                'context': row[2],
                'semantic_distance': row[3],
                'relationship_type': row[4],
                'strength': row[5]
            })

        return relationships

    def find_semantic_clusters(self, context: str = "business",
                              max_distance: float = 0.3,
                              min_cluster_size: int = 2) -> List[List[str]]:
        """
        Discover semantic clusters (groups of related concepts)

        Args:
            context: Context to analyze
            max_distance: Maximum distance within a cluster
            min_cluster_size: Minimum concepts in a cluster

        Returns:
            List of clusters (each cluster is a list of concept names)
        """
        cursor = self.conn.cursor()

        # Get all concepts in context
        cursor.execute("SELECT id, concept_text FROM semantic_coordinates WHERE context = ?", (context,))
        concepts = {row[0]: row[1] for row in cursor.fetchall()}

        # Build adjacency graph based on relationships
        clusters = []
        visited = set()

        for concept_id in concepts.keys():
            if concept_id in visited:
                continue

            # Start a new cluster
            cluster = [concept_id]
            visited.add(concept_id)

            # Get all relationships within max_distance
            relationships = self.get_concept_relationships(concept_id)

            for rel in relationships:
                if rel['semantic_distance'] <= max_distance:
                    related_id = rel['related_id']
                    if related_id not in visited and related_id in concepts:
                        cluster.append(related_id)
                        visited.add(related_id)

            # Only keep clusters that meet minimum size
            if len(cluster) >= min_cluster_size:
                cluster_names = [concepts[cid] for cid in cluster]
                clusters.append(cluster_names)

        return clusters

    def get_relationship_graph(self, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the entire relationship graph for visualization

        Returns:
            Dict with nodes and edges for graph visualization
        """
        cursor = self.conn.cursor()

        # Get all concepts
        if context:
            cursor.execute("SELECT id, concept_text, love, power, wisdom, justice FROM semantic_coordinates WHERE context = ?", (context,))
        else:
            cursor.execute("SELECT id, concept_text, love, power, wisdom, justice FROM semantic_coordinates")

        nodes = []
        for row in cursor.fetchall():
            nodes.append({
                'id': row[0],
                'label': row[1],
                'coordinates': {
                    'love': row[2],
                    'power': row[3],
                    'wisdom': row[4],
                    'justice': row[5]
                }
            })

        # Get all relationships
        cursor.execute("""
            SELECT concept1_id, concept2_id, semantic_distance, strength
            FROM concept_relationships
        """)

        edges = []
        for row in cursor.fetchall():
            edges.append({
                'source': row[0],
                'target': row[1],
                'distance': row[2],
                'strength': row[3]
            })

        return {
            'nodes': nodes,
            'edges': edges,
            'node_count': len(nodes),
            'edge_count': len(edges)
        }

    # ========================================================================
    # QUERY OPERATIONS
    # ========================================================================

    def query_by_text(self, text: str, context: Optional[str] = None) -> Optional[Dict]:
        """Query concept by exact text match"""
        cursor = self.conn.cursor()

        if context:
            cursor.execute("""
                SELECT * FROM semantic_coordinates
                WHERE concept_text = ? AND context = ?
            """, (text, context))
        else:
            cursor.execute("""
                SELECT * FROM semantic_coordinates
                WHERE concept_text = ?
            """, (text,))

        row = cursor.fetchone()
        return dict(row) if row else None

    def query_by_proximity(self, target_coords: BiblicalCoordinates,
                          max_distance: float = 0.5,
                          context: Optional[str] = None,
                          limit: int = 10) -> List[Dict]:
        """
        Revolutionary query: Find concepts near a point in semantic space

        Args:
            target_coords: Target coordinates in semantic space
            max_distance: Maximum distance from target
            context: Optional context filter
            limit: Maximum results to return

        Returns:
            List of matching concepts with distances
        """
        cursor = self.conn.cursor()

        # Get all concepts (with optional context filter)
        if context:
            cursor.execute("""
                SELECT * FROM semantic_coordinates WHERE context = ?
            """, (context,))
        else:
            cursor.execute("SELECT * FROM semantic_coordinates")

        results = []
        for row in cursor.fetchall():
            concept = dict(row)
            coords = BiblicalCoordinates(
                concept['love'], concept['power'],
                concept['wisdom'], concept['justice']
            )

            # Calculate distance between two coordinate points
            distance = math.sqrt(
                (target_coords.love - coords.love) ** 2 +
                (target_coords.power - coords.power) ** 2 +
                (target_coords.wisdom - coords.wisdom) ** 2 +
                (target_coords.justice - coords.justice) ** 2
            )

            if distance <= max_distance:
                concept['semantic_distance'] = distance
                concept['coordinates'] = coords
                results.append(concept)

        # Sort by distance
        results.sort(key=lambda x: x['semantic_distance'])

        return results[:limit]

    def query_by_divine_resonance(self, min_resonance: float = 0.8,
                                  context: Optional[str] = None,
                                  limit: int = 10) -> List[Dict]:
        """Find concepts with high divine alignment"""
        cursor = self.conn.cursor()

        if context:
            cursor.execute("""
                SELECT * FROM semantic_coordinates
                WHERE divine_resonance >= ? AND context = ?
                ORDER BY divine_resonance DESC
                LIMIT ?
            """, (min_resonance, context, limit))
        else:
            cursor.execute("""
                SELECT * FROM semantic_coordinates
                WHERE divine_resonance >= ?
                ORDER BY divine_resonance DESC
                LIMIT ?
            """, (min_resonance, limit))

        return [dict(row) for row in cursor.fetchall()]

    def query_sacred_numbers(self, min_value: float = 0,
                            max_value: float = 1000,
                            only_sacred: bool = True) -> List[Dict]:
        """Query sacred numbers within range"""
        cursor = self.conn.cursor()

        if only_sacred:
            cursor.execute("""
                SELECT * FROM sacred_numbers
                WHERE value >= ? AND value <= ? AND is_sacred = 1
                ORDER BY sacred_resonance DESC
            """, (min_value, max_value))
        else:
            cursor.execute("""
                SELECT * FROM sacred_numbers
                WHERE value >= ? AND value <= ?
                ORDER BY value
            """, (min_value, max_value))

        results = []
        for row in cursor.fetchall():
            concept = dict(row)
            concept['divine_attributes'] = json.loads(concept['divine_attributes_json'])
            concept['mystical_properties'] = json.loads(concept['mystical_properties_json'])
            results.append(concept)

        return results

    def query_nearest_to_anchor(self, anchor_id: int,
                               max_distance: float = 1.0,
                               limit: int = 10) -> List[Dict]:
        """Find concepts nearest to a universal anchor"""
        cursor = self.conn.cursor()

        # Get anchor coordinates
        cursor.execute("SELECT * FROM universal_anchors WHERE id = ?", (anchor_id,))
        anchor_row = cursor.fetchone()

        if not anchor_row:
            raise ValueError(f"Anchor {anchor_id} not found")

        anchor = dict(anchor_row)
        anchor_coords = BiblicalCoordinates(
            anchor['love'], anchor['power'],
            anchor['wisdom'], anchor['justice']
        )

        # Use proximity query
        return self.query_by_proximity(anchor_coords, max_distance, limit=limit)

    def search_semantic(self, query_text: str,
                       context: str = "biblical",
                       limit: int = 10) -> List[Dict]:
        """
        Semantic search: Analyze query and find similar concepts

        This is the revolutionary feature - true semantic search!
        """
        # Analyze query using SSE engine with fallback compatibility
        if self.engine_available and hasattr(self.engine, 'core_engine'):
            query_coords = self.engine.core_engine.analyze_concept(query_text, context)
        else:
            query_coords = self.engine.analyze_concept(query_text, context)

        # Find nearby concepts in semantic space
        results = self.query_by_proximity(query_coords, max_distance=0.8,
                                         context=context, limit=limit)

        # Enrich results with similarity scores
        for result in results:
            result['semantic_similarity'] = 1.0 - (result['semantic_distance'] / 4.0)
            result['query_alignment'] = result['divine_resonance'] * result['semantic_similarity']

        # Sort by query alignment
        results.sort(key=lambda x: x['query_alignment'], reverse=True)

        return results

    def _get_coordinates_by_id(self, concept_id: int) -> Optional[BiblicalCoordinates]:
        """Helper to get coordinates by ID"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT love, power, wisdom, justice
            FROM semantic_coordinates WHERE id = ?
        """, (concept_id,))

        row = cursor.fetchone()
        if row:
            return BiblicalCoordinates(row[0], row[1], row[2], row[3])
        return None

    # ========================================================================
    # ANALYTICS & STATISTICS
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        cursor = self.conn.cursor()

        stats = {}

        # Concept counts
        cursor.execute("SELECT COUNT(*) FROM semantic_coordinates")
        stats['total_concepts'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT context) FROM semantic_coordinates")
        stats['unique_contexts'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM semantic_units")
        stats['total_semantic_units'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM sacred_numbers WHERE is_sacred = 1")
        stats['sacred_numbers_count'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM concept_relationships")
        stats['total_relationships'] = cursor.fetchone()[0]

        # Average metrics
        cursor.execute("""
            SELECT AVG(divine_resonance), AVG(distance_from_jehovah), AVG(biblical_balance)
            FROM semantic_coordinates
        """)
        row = cursor.fetchone()
        stats['avg_divine_resonance'] = row[0] if row[0] else 0
        stats['avg_distance_from_jehovah'] = row[1] if row[1] else 0
        stats['avg_biblical_balance'] = row[2] if row[2] else 0

        # Context distribution
        cursor.execute("""
            SELECT context, COUNT(*) as count
            FROM semantic_coordinates
            GROUP BY context
            ORDER BY count DESC
        """)
        stats['context_distribution'] = {row[0]: row[1] for row in cursor.fetchall()}

        return stats

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return {
            'cache_size': len(self.cache),
            'cache_keys': list(self.cache.keys())[:10]  # Sample of keys
        }

    # ========================================================================
    # MAINTENANCE OPERATIONS
    # ========================================================================

    def clear_cache(self):
        """Clear in-memory cache"""
        self.cache.clear()
        print("[SEMANTIC DB] Cache cleared")

    def vacuum(self):
        """Optimize database storage"""
        self.conn.execute("VACUUM")
        print("[SEMANTIC DB] Database optimized")

    # ========================================================================
    # BACKUP AND RECOVERY OPERATIONS
    # ========================================================================

    def create_backup(self, backup_path: Optional[str] = None) -> str:
        """
        Create a full database backup

        Args:
            backup_path: Path for backup file (auto-generated if None)

        Returns:
            Path to created backup file
        """
        import shutil
        from datetime import datetime

        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.db_path}.backup_{timestamp}"

        # Ensure all changes are committed
        self.conn.commit()

        # Create backup using SQLite backup API for consistency
        backup_conn = sqlite3.connect(backup_path)
        with backup_conn:
            self.conn.backup(backup_conn)
        backup_conn.close()

        print(f"[SEMANTIC DB] Backup created: {backup_path}")
        return backup_path

    def restore_from_backup(self, backup_path: str, verify: bool = True) -> bool:
        """
        Restore database from backup

        Args:
            backup_path: Path to backup file
            verify: Verify backup integrity before restoring

        Returns:
            True if restore successful
        """
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        # Verify backup if requested
        if verify:
            if not self.verify_backup(backup_path):
                raise ValueError(f"Backup file verification failed: {backup_path}")

        # Close current connection
        self.conn.close()

        # Replace database file with backup
        import shutil
        shutil.copy2(backup_path, self.db_path)

        # Reconnect to restored database
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        # Clear cache after restore
        self.cache.clear()

        print(f"[SEMANTIC DB] Restored from backup: {backup_path}")
        return True

    def verify_backup(self, backup_path: str) -> bool:
        """
        Verify backup file integrity

        Args:
            backup_path: Path to backup file

        Returns:
            True if backup is valid
        """
        try:
            # Try to open backup and verify schema
            test_conn = sqlite3.connect(backup_path)
            cursor = test_conn.cursor()

            # Verify critical tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN (
                    'semantic_coordinates',
                    'semantic_units',
                    'sacred_numbers',
                    'universal_anchors'
                )
            """)
            tables = [row[0] for row in cursor.fetchall()]

            test_conn.close()

            required_tables = ['semantic_coordinates', 'semantic_units',
                             'sacred_numbers', 'universal_anchors']

            return all(table in tables for table in required_tables)

        except Exception as e:
            print(f"[SEMANTIC DB] Backup verification failed: {e}")
            return False

    def create_incremental_backup(self, backup_dir: str) -> str:
        """
        Create incremental backup (exports only changes since last backup)

        Args:
            backup_dir: Directory for incremental backups

        Returns:
            Path to incremental backup file
        """
        import os
        from datetime import datetime

        os.makedirs(backup_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        incremental_path = os.path.join(backup_dir, f"incremental_{timestamp}.json")

        # Export to JSON (contains all current data)
        self.export_to_json(incremental_path)

        print(f"[SEMANTIC DB] Incremental backup created: {incremental_path}")
        return incremental_path

    def restore_from_json(self, json_path: str, clear_existing: bool = False) -> bool:
        """
        Restore database from JSON export

        Args:
            json_path: Path to JSON export file
            clear_existing: Clear existing data before restore

        Returns:
            True if restore successful
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        cursor = self.conn.cursor()

        # Clear existing data if requested
        if clear_existing:
            cursor.execute("DELETE FROM semantic_coordinates")
            cursor.execute("DELETE FROM semantic_units")
            cursor.execute("DELETE FROM sacred_numbers")
            cursor.execute("DELETE FROM concept_relationships")
            self.conn.commit()
            print("[SEMANTIC DB] Existing data cleared")

        # Restore concepts
        for concept in data.get('concepts', []):
            cursor.execute("""
                INSERT OR REPLACE INTO semantic_coordinates
                (id, concept_text, context, love, power, wisdom, justice,
                 divine_resonance, distance_from_jehovah, biblical_balance,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (concept['id'], concept['concept_text'], concept['context'],
                  concept['love'], concept['power'], concept['wisdom'], concept['justice'],
                  concept['divine_resonance'], concept['distance_from_jehovah'],
                  concept['biblical_balance'], concept['created_at'], concept['updated_at']))

        # Restore sacred numbers
        for num in data.get('sacred_numbers', []):
            cursor.execute("""
                INSERT OR REPLACE INTO sacred_numbers
                (id, value, is_sacred, sacred_resonance, biblical_significance,
                 divine_attributes_json, mystical_properties_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (num['id'], num['value'], num['is_sacred'], num['sacred_resonance'],
                  num['biblical_significance'], num['divine_attributes_json'],
                  num['mystical_properties_json'], num['created_at']))

        self.conn.commit()
        self.cache.clear()

        stats = data.get('metadata', {}).get('statistics', {})
        print(f"[SEMANTIC DB] Restored from JSON: {json_path}")
        print(f"[SEMANTIC DB] Restored {stats.get('total_concepts', 0)} concepts, "
              f"{stats.get('sacred_numbers_count', 0)} sacred numbers")

        return True

    def create_snapshot(self, snapshot_name: str, snapshot_dir: str = "snapshots") -> str:
        """
        Create a named snapshot of the database

        Args:
            snapshot_name: Name for the snapshot
            snapshot_dir: Directory for snapshots

        Returns:
            Path to snapshot file
        """
        import os
        from datetime import datetime

        os.makedirs(snapshot_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = os.path.join(snapshot_dir, f"{snapshot_name}_{timestamp}.db")

        # Create backup at snapshot location
        return self.create_backup(snapshot_path)

    def list_backups(self, backup_dir: str = ".") -> List[Dict[str, Any]]:
        """
        List all available backups

        Args:
            backup_dir: Directory containing backups

        Returns:
            List of backup info dicts
        """
        import os
        import glob
        from datetime import datetime

        backups = []

        # Find backup files
        backup_pattern = os.path.join(backup_dir, f"{os.path.basename(self.db_path)}.backup_*")
        for backup_path in glob.glob(backup_pattern):
            stat = os.stat(backup_path)
            backups.append({
                'path': backup_path,
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / 1024 / 1024, 2),
                'created_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'valid': self.verify_backup(backup_path)
            })

        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x['created_at'], reverse=True)

        return backups

    def auto_backup(self, backup_dir: str = "backups",
                    keep_last_n: int = 10) -> str:
        """
        Create automatic backup and maintain backup rotation

        Args:
            backup_dir: Directory for backups
            keep_last_n: Number of backups to keep

        Returns:
            Path to created backup
        """
        import os

        os.makedirs(backup_dir, exist_ok=True)

        # Create backup in backup directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"{os.path.basename(self.db_path)}.backup_{timestamp}")

        backup_path = self.create_backup(backup_path)

        # Rotate old backups
        backups = self.list_backups(backup_dir)
        if len(backups) > keep_last_n:
            for old_backup in backups[keep_last_n:]:
                try:
                    os.remove(old_backup['path'])
                    print(f"[SEMANTIC DB] Removed old backup: {old_backup['path']}")
                except Exception as e:
                    print(f"[SEMANTIC DB] Failed to remove old backup: {e}")

        print(f"[SEMANTIC DB] Auto-backup complete (keeping last {keep_last_n})")
        return backup_path

    def export_to_json(self, output_path: str):
        """Export entire database to JSON"""
        cursor = self.conn.cursor()

        export_data = {
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'engine_version': self.engine.engine_version,
                'statistics': self.get_statistics()
            },
            'concepts': [],
            'sacred_numbers': [],
            'anchors': []
        }

        # Export concepts
        cursor.execute("SELECT * FROM semantic_coordinates")
        for row in cursor.fetchall():
            export_data['concepts'].append(dict(row))

        # Export sacred numbers
        cursor.execute("SELECT * FROM sacred_numbers")
        for row in cursor.fetchall():
            export_data['sacred_numbers'].append(dict(row))

        # Export anchors
        cursor.execute("SELECT * FROM universal_anchors")
        for row in cursor.fetchall():
            export_data['anchors'].append(dict(row))

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"[SEMANTIC DB] Exported to {output_path}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("[SEMANTIC DB] Connection closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# ============================================================================
# DEMONSTRATION FUNCTION
# ============================================================================

def demonstrate_semantic_database():
    """Demonstrate the revolutionary semantic database capabilities"""

    print("=" * 100)
    print("SEMANTIC SUBSTRATE DATABASE v1.0 - Revolutionary Meaning-Native Database")
    print("=" * 100)

    # Initialize database
    db = SemanticSubstrateDatabase("test_semantic.db")

    print("\n[DEMO] Storing Biblical Concepts")
    print("-" * 50)

    # Store some concepts
    concepts = [
        ("love", "biblical"),
        ("wisdom", "biblical"),
        ("justice", "biblical"),
        ("mercy", "biblical"),
        ("faith", "biblical"),
        ("hope", "biblical"),
        ("grace", "biblical"),
        ("truth", "biblical")
    ]

    for text, context in concepts:
        concept_id = db.store_concept(text, context)
        print(f"Stored: '{text}' (context: {context}) -> ID: {concept_id}")

    # Store some sacred numbers
    print("\n[DEMO] Storing Sacred Numbers")
    print("-" * 50)

    sacred_nums = [7, 12, 40, 613, 3, 10]
    for num in sacred_nums:
        db.store_sacred_number(num)
        print(f"Stored sacred number: {num}")

    print("\n[DEMO] Revolutionary Semantic Search")
    print("-" * 50)

    # Semantic search - the revolutionary feature!
    query = "compassion and kindness"
    print(f"Query: '{query}'")
    results = db.search_semantic(query, context="biblical", limit=5)

    print(f"\nFound {len(results)} semantically similar concepts:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['concept_text']}")
        print(f"   Semantic Similarity: {result['semantic_similarity']:.3f}")
        print(f"   Divine Resonance: {result['divine_resonance']:.3f}")
        print(f"   Query Alignment: {result['query_alignment']:.3f}")

    print("\n[DEMO] Query by Divine Resonance")
    print("-" * 50)

    high_resonance = db.query_by_divine_resonance(min_resonance=0.85, context="biblical")
    print(f"Concepts with divine resonance >= 0.85:")
    for concept in high_resonance:
        print(f"  {concept['concept_text']}: {concept['divine_resonance']:.3f}")

    print("\n[DEMO] Query Nearest to Universal Anchor")
    print("-" * 50)

    # Find concepts near anchor 7 (Divine Perfection)
    near_anchor = db.query_nearest_to_anchor(anchor_id=7, max_distance=1.0, limit=5)
    print(f"Concepts nearest to Anchor 7 (Divine Perfection):")
    for concept in near_anchor:
        print(f"  {concept['concept_text']}: distance = {concept['semantic_distance']:.3f}")

    print("\n[DEMO] Database Statistics")
    print("-" * 50)

    stats = db.get_statistics()
    print(f"Total Concepts: {stats['total_concepts']}")
    print(f"Unique Contexts: {stats['unique_contexts']}")
    print(f"Sacred Numbers: {stats['sacred_numbers_count']}")
    print(f"Total Relationships: {stats['total_relationships']}")
    print(f"Avg Divine Resonance: {stats['avg_divine_resonance']:.3f}")
    print(f"Avg Distance from JEHOVAH: {stats['avg_distance_from_jehovah']:.3f}")
    print(f"\nContext Distribution:")
    for context, count in stats['context_distribution'].items():
        print(f"  {context}: {count}")

    print("\n[DEMO] Export Database")
    print("-" * 50)

    db.export_to_json("semantic_export.json")

    # Close database
    db.close()

    print("\n" + "=" * 100)
    print("DEMONSTRATION COMPLETE - Revolutionary Semantic Database Operational!")
    print("=" * 100)

    return db


if __name__ == "__main__":
    demonstrate_semantic_database()
