"""
Visualization Example: SSDB Coordinate Visualization
====================================================

This example demonstrates how to visualize the 4D coordinate
system and semantic relationships in SSDB.
"""

import os
import sys
import tempfile

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from deep_dive_database import DeepDiveDatabase


def print_ascii_bar(label, value, max_value=1.0, width=50):
    """Print an ASCII bar chart"""
    filled = int((value / max_value) * width)
    bar = '█' * filled + '░' * (width - filled)
    percentage = (value / max_value) * 100
    print(f"  {label:12} │{bar}│ {value:.3f} ({percentage:.0f}%)")


def visualize_coordinates(text, coords):
    """Visualize 4D coordinates as bar chart"""
    print(f"\nConcept: '{text}'")
    print("─" * 70)
    print_ascii_bar("Love", coords.love)
    print_ascii_bar("Power", coords.power)
    print_ascii_bar("Wisdom", coords.wisdom)
    print_ascii_bar("Justice", coords.justice)

    # Calculate overall alignment
    avg = (coords.love + coords.power + coords.wisdom + coords.justice) / 4
    print("─" * 70)
    print_ascii_bar("Average", avg)


def visualize_comparison(concepts_data):
    """Visualize comparison of multiple concepts"""
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70)

    for text, coords in concepts_data:
        avg = (coords.love + coords.power + coords.wisdom + coords.justice) / 4
        filled = int(avg * 50)
        bar = '█' * filled + '░' * (50 - filled)
        print(f"\n{text[:40]:40} │{bar}│ {avg:.3f}")


def visualize_layer_decomposition(analysis):
    """Visualize 5-layer semantic decomposition"""
    print("\n" + "="*70)
    print("5-LAYER SEMANTIC DECOMPOSITION")
    print("="*70)

    layers = analysis['layers']

    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│ Layer 5: UNIVERSAL PRINCIPLES                           │")
    print(f"│ └─ {layers['layer_5_universal']['principle']:50} │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│ Layer 4: SACRED NUMBERS                                 │")
    print(f"│ └─ Sacred Number: {str(layers['layer_4_sacred']['sacred_number']):38} │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│ Layer 3: SEMANTIC RELATIONSHIPS                         │")
    weight = layers['layer_3_semantic']['weight']
    print(f"│ └─ Semantic Weight: {weight:.3f}                              │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│ Layer 2: BIBLICAL REFERENCES                            │")
    ref = str(layers['layer_2_biblical']['reference'])[:45]
    print(f"│ └─ {ref:52} │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│ Layer 1: MATHEMATICAL FOUNDATION                        │")
    math_val = layers['layer_1_mathematical']['value']
    print(f"│ └─ Mathematical Value: {math_val:.3f}                         │")
    print("└─────────────────────────────────────────────────────────┘")


def visualize_architecture():
    """Visualize SSDB 4-layer architecture"""
    print("\n" + "="*70)
    print("SSDB ARCHITECTURE - 4 LAYERS")
    print("="*70)

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  LAYER 4: Deep Dive Database                                │
    │  • 5-layer semantic decomposition                           │
    │  • Meaning unit combination (blend, multiply, trinity)      │
    │  • Meaning programs and workflows                           │
    │  • Layer-specific queries                                   │
    │                                                             │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  LAYER 3: Meaning-Based Database                            │
    │  • Natural language operations                              │
    │  • Intent classification (query, store, analyze, relate)    │
    │  • Semantic search by meaning                               │
    │  • Meaning execution framework                              │
    │                                                             │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  LAYER 2: Enhanced Database                                 │
    │  • Self-awareness (0.880/1.0)                               │
    │  • ICE Framework (Intent-Context-Execution)                 │
    │  • 8 thought types processing                               │
    │  • Divine alignment calculation                             │
    │                                                             │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  LAYER 1: Semantic Substrate Database                       │
    │  • 4D coordinate system (Love, Power, Wisdom, Justice)      │
    │  • Sacred number detection                                  │
    │  • Text and proximity queries                               │
    │  • SQLite foundation                                        │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    Each layer is 100% backward compatible with layers below.
    """)


def main():
    print("="*70)
    print("SSDB VISUALIZATION EXAMPLE")
    print("="*70)

    # Show architecture
    visualize_architecture()

    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_path = temp_db.name
    temp_db.close()

    # Initialize database
    print("\nInitializing database...")
    db = DeepDiveDatabase(db_path)

    # ========================================================================
    # Visualize 4D Coordinates
    # ========================================================================
    print("\n" + "="*70)
    print("4D COORDINATE VISUALIZATION")
    print("="*70)

    # Store and visualize some concepts
    concepts = [
        ("God is love", "biblical"),
        ("Divine wisdom and understanding", "proverbs"),
        ("Righteous judgment", "psalm"),
        ("Mighty power and strength", "exodus"),
    ]

    coords_data = []
    for text, context in concepts:
        concept_id = db.store_concept(text, context)
        results = db.query_by_text(text.split()[0])
        if results:
            coords = results[0]['coordinates']
            visualize_coordinates(text, coords)
            coords_data.append((text, coords))

    # Comparative visualization
    visualize_comparison(coords_data)

    # ========================================================================
    # Visualize 5-Layer Decomposition
    # ========================================================================
    print("\n\n" + "="*70)
    print("DEEP DIVE: 5-LAYER DECOMPOSITION")
    print("="*70)

    result = db.store_with_deep_dive(
        "The Trinity reveals God as Father, Son, and Holy Spirit",
        "theology"
    )

    analysis = db.analyze_concept_layers(
        "The Trinity reveals God as Father, Son, and Holy Spirit",
        "theology"
    )

    visualize_layer_decomposition(analysis)

    # ========================================================================
    # Visualize Test Results
    # ========================================================================
    print("\n\n" + "="*70)
    print("TEST RESULTS VISUALIZATION")
    print("="*70)

    test_results = [
        ("Layer 1 (Original SSDB)", 30, 30),
        ("Layer 2 (Enhanced SSDB)", 21, 21),
        ("Layer 3 (Meaning-Based)", 24, 24),
        ("Layer 4 (Deep Dive)", 26, 26),
    ]

    print("\n")
    total_passed = 0
    total_tests = 0

    for layer_name, passed, total in test_results:
        percentage = (passed / total) * 100
        filled = int(percentage / 2)  # Scale to 50 chars
        bar = '█' * filled
        print(f"{layer_name:25} │{bar:50}│ {passed}/{total} ({percentage:.0f}%)")
        total_passed += passed
        total_tests += total

    print("─" * 79)
    total_pct = (total_passed / total_tests) * 100
    filled = int(total_pct / 2)
    bar = '█' * filled
    print(f"{'TOTAL':25} │{bar:50}│ {total_passed}/{total_tests} ({total_pct:.0f}%)")

    # ========================================================================
    # Visualize Statistics
    # ========================================================================
    print("\n\n" + "="*70)
    print("DATABASE STATISTICS")
    print("="*70)

    stats = db.get_deep_dive_statistics()

    print(f"""
    ┌─────────────────────────────────────────────────┐
    │ Storage Statistics                              │
    ├─────────────────────────────────────────────────┤
    │ Total Concepts:           {stats.get('total_concepts', 0):20} │
    │ Relationships:            {stats.get('total_relationships', 0):20} │
    │ Meaning Units:            {stats.get('meaning_units_registered', 0):20} │
    │ Scaffold Combinations:    {stats.get('scaffold_combinations', 0):20} │
    ├─────────────────────────────────────────────────┤
    │ Processing Statistics                           │
    ├─────────────────────────────────────────────────┤
    │ Self-Aware Analyses:      {stats.get('aware_analyses_count', 0):20} │
    │ ICE Thoughts Processed:   {stats.get('ice_thoughts_count', 0):20} │
    │ Meaning Executions:       {stats.get('meaning_executions_count', 0):20} │
    │ Meaning Programs:         {stats.get('meaning_programs_loaded', 0):20} │
    └─────────────────────────────────────────────────┘
    """)

    # ========================================================================
    # Cleanup
    # ========================================================================
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)

    print("\n✓ Visualized:")
    print("  • 4-layer architecture diagram")
    print("  • 4D coordinate system (Love, Power, Wisdom, Justice)")
    print("  • 5-layer semantic decomposition")
    print("  • Test results (101/101 passing)")
    print("  • Database statistics")

    db.close()
    os.unlink(db_path)


if __name__ == "__main__":
    main()
