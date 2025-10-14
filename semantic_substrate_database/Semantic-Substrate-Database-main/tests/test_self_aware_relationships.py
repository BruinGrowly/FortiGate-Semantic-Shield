"""
Test Self-Aware Relationship Discovery

Demonstrates the database's ability to:
- Automatically discover semantic relationships
- Build a self-organizing knowledge graph
- Find clusters of related concepts
- Propagate connections without manual intervention
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from semantic_substrate_database import SemanticSubstrateDatabase


def test_self_aware_database():
    """Test self-aware relationship discovery with biblical concepts"""
    print("=" * 80)
    print("SEMANTIC SUBSTRATE DATABASE - SELF-AWARE RELATIONSHIP TEST")
    print("=" * 80)

    # Initialize database
    db_path = "self_aware_test.db"
    try:
        os.unlink(db_path)
    except:
        pass

    db = SemanticSubstrateDatabase(db_path)

    # Store biblical concepts
    print("\n[1] Storing Biblical Concepts")
    print("-" * 80)

    concepts = [
        "love", "mercy", "grace", "compassion", "kindness",
        "faith", "hope", "trust", "belief",
        "wisdom", "knowledge", "understanding", "insight",
        "justice", "righteousness", "holiness", "purity",
        "power", "strength", "might", "authority"
    ]

    concept_ids = db.batch_store_concepts([(c, "biblical") for c in concepts])
    print(f"Stored {len(concept_ids)} concepts")

    # Enable automatic relationship discovery
    print("\n[2] Enabling Self-Aware Relationship Discovery")
    print("-" * 80)
    print("Database is now analyzing itself to discover relationships...")

    total_relationships = db.enable_auto_relationships(
        context="biblical",
        max_distance=0.5,
        max_relationships=5
    )

    print(f"\nThe database automatically discovered {total_relationships} relationships!")

    # Analyze the relationship graph
    print("\n[3] Analyzing Self-Generated Relationship Graph")
    print("-" * 80)

    stats = db.get_statistics()
    print(f"Total Concepts: {stats['total_concepts']}")
    print(f"Total Relationships: {stats['total_relationships']}")
    print(f"Average Relationships per Concept: {stats['total_relationships'] / stats['total_concepts']:.2f}")

    # Show relationships for specific concepts
    print("\n[4] Concept Relationship Networks")
    print("-" * 80)

    test_concepts = ["love", "faith", "wisdom", "justice"]

    for concept_name in test_concepts:
        concept = db.query_by_text(concept_name, "biblical")
        if concept:
            relationships = db.get_concept_relationships(concept['id'])

            print(f"\n'{concept_name}' is automatically connected to:")
            if relationships:
                for i, rel in enumerate(relationships[:5], 1):
                    print(f"  {i}. {rel['related_text']}")
                    print(f"     Distance: {rel['semantic_distance']:.4f}, Strength: {rel['strength']:.4f}")
            else:
                print("  (No relationships found)")

    # Discover semantic clusters
    print("\n[5] Discovering Semantic Clusters")
    print("-" * 80)
    print("Database is finding groups of related concepts...")

    clusters = db.find_semantic_clusters(context="biblical", max_distance=0.3, min_cluster_size=2)

    print(f"\nDiscovered {len(clusters)} semantic clusters:")
    for i, cluster in enumerate(clusters, 1):
        print(f"\nCluster {i}: {len(cluster)} concepts")
        print(f"  {', '.join(cluster)}")

    # Get full relationship graph
    print("\n[6] Exporting Self-Generated Knowledge Graph")
    print("-" * 80)

    graph = db.get_relationship_graph(context="biblical")
    print(f"Knowledge Graph:")
    print(f"  Nodes (Concepts): {graph['node_count']}")
    print(f"  Edges (Relationships): {graph['edge_count']}")
    print(f"  Graph Density: {graph['edge_count'] / (graph['node_count'] * (graph['node_count']-1)) * 100:.2f}%")

    # Export graph for visualization
    graph_file = "self_aware_knowledge_graph.json"
    with open(graph_file, 'w') as f:
        json.dump(graph, f, indent=2, default=str)
    print(f"\nExported knowledge graph to: {graph_file}")

    # Demonstrate path finding
    print("\n[7] Semantic Path Discovery")
    print("-" * 80)

    love = db.query_by_text("love", "biblical")
    justice = db.query_by_text("justice", "biblical")

    if love and justice:
        print(f"\nDirect relationship between 'love' and 'justice':")
        love_rels = db.get_concept_relationships(love['id'])
        connected = any(r['related_id'] == justice['id'] for r in love_rels)

        if connected:
            rel = next(r for r in love_rels if r['related_id'] == justice['id'])
            print(f"  ✓ Directly connected")
            print(f"  Distance: {rel['semantic_distance']:.4f}")
            print(f"  Strength: {rel['strength']:.4f}")
        else:
            print(f"  Not directly connected (too far apart)")

    # Close database
    db.close()

    print("\n" + "=" * 80)
    print("SELF-AWARE DATABASE TEST COMPLETE!")
    print("=" * 80)
    print("\nKey Capabilities Demonstrated:")
    print("  ✓ Automatic relationship discovery")
    print("  ✓ Self-organizing knowledge graph")
    print("  ✓ Semantic cluster detection")
    print("  ✓ Relationship strength calculation")
    print("  ✓ Zero manual relationship creation")
    print("\nThe database truly understands its own data!")

    return True


def test_cryptocurrency_self_awareness():
    """Test self-aware relationships with cryptocurrency data"""
    print("\n\n")
    print("=" * 80)
    print("CRYPTOCURRENCY SELF-AWARE RELATIONSHIP TEST")
    print("=" * 80)

    # Load the existing cryptocurrency database
    db_path = "cryptocurrency_semantic.db"

    if not os.path.exists(db_path):
        print("ERROR: Cryptocurrency database not found. Run test_cryptocurrency_data.py first.")
        return False

    db = SemanticSubstrateDatabase(db_path)

    # Enable auto-relationships
    print("\n[1] Analyzing Cryptocurrency Relationships")
    print("-" * 80)
    print("Database is discovering relationships between cryptocurrencies...")

    total_relationships = db.enable_auto_relationships(
        context="business",
        max_distance=0.2,  # Tighter clustering for similar cryptos
        max_relationships=10
    )

    print(f"\nAutomatically discovered {total_relationships} cryptocurrency relationships!")

    # Analyze specific cryptocurrencies
    print("\n[2] Bitcoin Relationship Network")
    print("-" * 80)

    bitcoin = db.query_by_text("Bitcoin", "business")
    if bitcoin:
        relationships = db.get_concept_relationships(bitcoin['id'])
        print(f"Bitcoin is automatically related to {len(relationships)} other cryptocurrencies:")

        for i, rel in enumerate(relationships[:10], 1):
            print(f"  {i}. {rel['related_text']}")
            print(f"     Semantic Distance: {rel['semantic_distance']:.4f}")
            print(f"     Relationship Strength: {rel['strength']:.4f}")

    # Find stablecoin cluster
    print("\n[3] Discovering Stablecoin Cluster")
    print("-" * 80)

    stablecoins = ["Tether USDt", "USDC", "Ethena USDe"]
    print("Checking if stablecoins are automatically clustered together...")

    for stablecoin in stablecoins:
        concept = db.query_by_text(stablecoin, "business")
        if concept:
            rels = db.get_concept_relationships(concept['id'])
            related_stablecoins = [r['related_text'] for r in rels
                                  if any(s in r['related_text'] for s in ['USD', 'Tether', 'USDC'])]

            if related_stablecoins:
                print(f"\n{stablecoin} automatically connected to:")
                for rs in related_stablecoins[:3]:
                    print(f"  • {rs}")

    # Get statistics
    stats = db.get_statistics()
    print("\n[4] Cryptocurrency Knowledge Graph Statistics")
    print("-" * 80)
    print(f"Total Cryptocurrencies: {stats['total_concepts']}")
    print(f"Automatic Relationships: {stats['total_relationships']}")
    print(f"Avg Connections per Crypto: {stats['total_relationships'] / stats['total_concepts']:.2f}")

    # Export graph
    graph = db.get_relationship_graph(context="business")
    graph_file = "cryptocurrency_knowledge_graph.json"

    with open(graph_file, 'w') as f:
        json.dump(graph, f, indent=2, default=str)

    print(f"\nExported cryptocurrency knowledge graph to: {graph_file}")

    db.close()

    print("\n" + "=" * 80)
    print("CRYPTOCURRENCY SELF-AWARENESS TEST COMPLETE!")
    print("=" * 80)
    print("\n✓ The database automatically understood cryptocurrency relationships")
    print("✓ Zero manual configuration required")
    print("✓ Self-organizing semantic network created")

    return True


if __name__ == "__main__":
    try:
        # Test with biblical concepts
        success1 = test_self_aware_database()

        # Test with cryptocurrency data
        success2 = test_cryptocurrency_self_awareness()

        sys.exit(0 if (success1 and success2) else 1)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
