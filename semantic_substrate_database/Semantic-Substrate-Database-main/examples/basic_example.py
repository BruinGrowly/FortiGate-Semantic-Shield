"""
Basic Example: Getting Started with SSDB
=========================================

This example demonstrates the core capabilities of the
Semantic Substrate Database across all 4 layers.
"""

import os
import sys
import tempfile

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from deep_dive_database import DeepDiveDatabase
from ice_framework import ThoughtType, ContextDomain
from deep_dive_meaning_scaffold import ScaffoldLayer


def main():
    print("="*70)
    print("SEMANTIC SUBSTRATE DATABASE - BASIC EXAMPLE")
    print("="*70)

    # Create temporary database for example
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_path = temp_db.name
    temp_db.close()

    # Initialize database (includes all 4 layers)
    print("\n[1] Initializing database with all 4 layers...")
    db = DeepDiveDatabase(db_path)
    print("✓ Database initialized!")

    # ========================================================================
    # LAYER 1: Basic Semantic Storage
    # ========================================================================
    print("\n" + "="*70)
    print("LAYER 1: Basic Semantic Storage (4D Coordinates)")
    print("="*70)

    # Store a simple concept
    print("\n[2] Storing basic concept...")
    concept_id = db.store_concept(
        "The Lord is my shepherd, I shall not want",
        "psalm23"
    )
    print(f"✓ Stored concept with ID: {concept_id}")

    # Query by text
    print("\n[3] Querying by text...")
    results = db.query_by_text("shepherd")
    if results:
        print(f"✓ Found {len(results)} concepts")
        for result in results[:1]:  # Show first result
            coords = result['coordinates']
            print(f"  Coordinates: L={coords.love:.3f}, P={coords.power:.3f}, "
                  f"W={coords.wisdom:.3f}, J={coords.justice:.3f}")

    # ========================================================================
    # LAYER 2: Self-Aware Storage with ICE Framework
    # ========================================================================
    print("\n" + "="*70)
    print("LAYER 2: Self-Aware Storage + ICE Framework")
    print("="*70)

    # Store with self-awareness
    print("\n[4] Storing concept with self-awareness...")
    aware_id = db.store_concept_with_awareness(
        "Trust in the Lord with all your heart",
        "proverbs"
    )
    print(f"✓ Stored with awareness analysis")
    print(f"  Concept ID: {aware_id}")
    print(f"  Database awareness level: 0.880/1.0")

    # Process thought through ICE framework
    print("\n[5] Processing thought through ICE framework...")
    ice_result = db.process_thought_to_concept(
        "How can I grow in wisdom?",
        ThoughtType.PRACTICAL_WISDOM,
        ContextDomain.BIBLICAL
    )
    print(f"✓ Thought processed and stored")
    print(f"  Divine alignment: {ice_result.get('divine_alignment', 0.0):.3f}")

    # ========================================================================
    # LAYER 3: Natural Language Operations
    # ========================================================================
    print("\n" + "="*70)
    print("LAYER 3: Natural Language Operations")
    print("="*70)

    # Natural language query (no SQL!)
    print("\n[6] Natural language query...")
    results = db.natural_query(
        "Find concepts about trust and faith",
        "biblical"
    )
    print(f"✓ Query understood and executed")
    if results:
        print(f"  Found {len(results)} matching concepts")

    # Execute meaning-based operation
    print("\n[7] Executing meaning-based storage...")
    result = db.execute_meaning(
        "Store the concept that God's love is eternal",
        "biblical"
    )
    print(f"✓ Operation executed: {result['intent']}")

    # Semantic search (meaning, not just keywords)
    print("\n[8] Semantic search by meaning...")
    results = db.meaning_search(
        "divine guidance and direction",
        similarity_threshold=0.6
    )
    print(f"✓ Semantic search completed")
    if results:
        print(f"  Found {len(results)} semantically similar concepts")

    # ========================================================================
    # LAYER 4: Deep Dive with 5-Layer Decomposition
    # ========================================================================
    print("\n" + "="*70)
    print("LAYER 4: Deep Dive (5-Layer Semantic Decomposition)")
    print("="*70)

    # Store with complete 5-layer analysis
    print("\n[9] Storing with deep dive (5 layers)...")
    result = db.store_with_deep_dive(
        "For God so loved the world that He gave His only Son",
        "john3:16"
    )
    print(f"✓ Stored with complete 5-layer analysis")
    print(f"  Concept ID: {result['concept_id']}")
    print("\n  5-Layer Breakdown:")
    print(f"    Layer 1 (Mathematical): {result['scaffold_layers']['mathematical']['value']:.3f}")
    print(f"    Layer 2 (Biblical): {result['scaffold_layers']['biblical']['reference']}")
    print(f"    Layer 3 (Semantic): {result['scaffold_layers']['semantic']['weight']:.3f}")
    print(f"    Layer 4 (Sacred): {result['scaffold_layers']['sacred']['number']}")
    print(f"    Layer 5 (Universal): {result['scaffold_layers']['universal']['principle']}")

    # Analyze concept through all layers
    print("\n[10] Analyzing concept through all layers...")
    analysis = db.analyze_concept_layers(
        "The Trinity: Father, Son, and Holy Spirit",
        "theology"
    )
    print(f"✓ Complete layer analysis performed")
    print(f"  Layers analyzed: {len(analysis['layers'])}")

    # Create and combine meaning units
    print("\n[11] Creating and combining meaning units...")
    unit1 = db.store_with_deep_dive("Divine Love", "biblical")
    unit2 = db.store_with_deep_dive("Divine Justice", "biblical")

    combined = db.combine_meaning_units(
        [unit1['meaning_unit_id'], unit2['meaning_unit_id']],
        operation="trinity"
    )
    print(f"✓ Meaning units combined using Trinity operation")
    coords = combined['coordinates']
    print(f"  Result coordinates: L={coords[0]:.3f}, P={coords[1]:.3f}, "
          f"W={coords[2]:.3f}, J={coords[3]:.3f}")

    # Query by specific layer
    print("\n[12] Querying by specific scaffold layer...")
    results = db.query_by_layer(
        ScaffoldLayer.UNIVERSAL,
        "coherent_interconnectedness"
    )
    print(f"✓ Layer-specific query executed")
    if results:
        print(f"  Found {len(results)} concepts with this universal principle")

    # ========================================================================
    # Statistics
    # ========================================================================
    print("\n" + "="*70)
    print("SYSTEM STATISTICS")
    print("="*70)

    stats = db.get_deep_dive_statistics()
    print(f"\n✓ Database Statistics:")
    print(f"  Total concepts: {stats.get('total_concepts', 0)}")
    print(f"  Meaning units: {stats.get('meaning_units_registered', 0)}")
    print(f"  Self-aware analyses: {stats.get('aware_analyses_count', 0)}")
    print(f"  Meaning executions: {stats.get('meaning_executions_count', 0)}")

    # ========================================================================
    # Cleanup
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)

    print("\n✓ Demonstrated all 4 layers:")
    print("  Layer 1: Basic semantic storage with 4D coordinates")
    print("  Layer 2: Self-awareness (0.880) + ICE thought processing")
    print("  Layer 3: Natural language operations and semantic search")
    print("  Layer 4: 5-layer semantic decomposition and meaning synthesis")

    print("\n✓ Revolutionary capabilities showcased:")
    print("  • Self-aware database understanding its own contents")
    print("  • Thought processing through ICE framework")
    print("  • Natural language queries (no SQL needed)")
    print("  • Complete semantic decomposition (5 layers)")
    print("  • Meaning-based operations and combinations")

    # Close and cleanup
    db.close()
    os.unlink(db_path)

    print("\n" + "="*70)
    print("The future of databases is semantic, self-aware, and meaning-driven.")
    print("="*70)


if __name__ == "__main__":
    main()
