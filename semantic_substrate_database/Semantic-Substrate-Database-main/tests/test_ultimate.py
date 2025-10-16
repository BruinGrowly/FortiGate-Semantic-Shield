# Quick test of Ultimate Core Engine
print("Testing Ultimate Core Engine v2.2")

try:
    from semantic_substrate_engine.ultimate_core_engine import UltimateCoreEngine  # type: ignore
    from semantic_substrate_engine.enhanced_core_components import (  # type: ignore
        SacredNumber,
        SemanticUnit,
        UniversalAnchor,
        SevenUniversalPrinciples,
    )
except ImportError:
    import sys
    from pathlib import Path

    ENGINE_SRC = (
        Path(__file__).resolve().parents[3]
        / "semantic_substrate_engine"
        / "Semantic-Substrate-Engine-main"
        / "src"
    )
    if str(ENGINE_SRC) not in sys.path:
        sys.path.insert(0, str(ENGINE_SRC))

    from ultimate_core_engine import UltimateCoreEngine  # type: ignore
    from enhanced_core_components import (  # type: ignore
        SacredNumber,
        SemanticUnit,
        UniversalAnchor,
        SevenUniversalPrinciples,
    )

try:
    print("[OK] Import successful")
    
    engine = UltimateCoreEngine()
    print("[OK] Engine initialized")
    
    # Test a simple sacred number
    sacred_40 = SacredNumber(40)
    print(f"[OK] Sacred Number 40: {sacred_40.is_sacred}, Resonance: {sacred_40.sacred_resonance:.3f}")
    
    # Test semantic unit
    unit = SemanticUnit("wisdom", "biblical")
    print(f"[OK] Semantic Unit: {unit.text}, Essence Love: {unit.essence['love']:.2f}")
    
    # Test universal anchor
    anchor = UniversalAnchor()
    print(f"[OK] Universal Anchor: {len(anchor.anchor_points)} anchors loaded")
    
    # Test seven principles
    principles = SevenUniversalPrinciples()
    print(f"[OK] Seven Principles: {len(principles.principles)} principles loaded")
    
    # Test advanced mathematics
    try:
        from mathematics.semantic_mathematics_engine import SemanticMathematicsEngine
        math_engine = SemanticMathematicsEngine()
        print("[OK] Advanced Mathematics Engine available")
    except ImportError:
        print("[INFO] Advanced Mathematics Engine not available (standalone mode)")
    
    print("\n[SUCCESS] ALL ENHANCED COMPONENTS WORKING!")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
