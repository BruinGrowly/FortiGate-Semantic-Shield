"""
Test Semantic Substrate Database with Real Cryptocurrency Data

Loads cryptocurrency market data and performs semantic analysis:
- Batch loading of cryptocurrency names
- Semantic similarity analysis between crypto names
- Proximity queries in semantic space
- Divine resonance analysis
- Transaction management for bulk operations
"""

import sys
import os
import csv
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from semantic_substrate_database import SemanticSubstrateDatabase


def load_cryptocurrency_data(csv_path: str):
    """Load cryptocurrency data from CSV"""
    cryptos = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cryptos.append({
                'name': row['name'],
                'symbol': row['symbol'],
                'price_usd': row['price_usd'],
                'market_cap': row['market_cap']
            })

    return cryptos


def test_cryptocurrency_semantic_database():
    """Test SSDB with cryptocurrency data"""
    print("=" * 80)
    print("SEMANTIC SUBSTRATE DATABASE - CRYPTOCURRENCY DATA TEST")
    print("=" * 80)

    # Load cryptocurrency data
    csv_path = os.path.join("data", "cryptocurrency.csv")
    print(f"\n[1] Loading Cryptocurrency Data")
    print("-" * 80)

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        return False

    cryptos = load_cryptocurrency_data(csv_path)
    print(f"Loaded {len(cryptos)} cryptocurrency records")

    # Get unique cryptocurrency names
    unique_cryptos = {}
    for crypto in cryptos:
        if crypto['name'] not in unique_cryptos:
            unique_cryptos[crypto['name']] = crypto

    print(f"Found {len(unique_cryptos)} unique cryptocurrencies")
    print(f"\nSample cryptocurrencies:")
    for i, name in enumerate(list(unique_cryptos.keys())[:10], 1):
        print(f"  {i}. {name} ({unique_cryptos[name]['symbol']})")

    # Initialize database
    print(f"\n[2] Initializing Semantic Substrate Database")
    print("-" * 80)

    db_path = "cryptocurrency_semantic.db"

    # Delete existing database
    try:
        os.unlink(db_path)
    except:
        pass

    db = SemanticSubstrateDatabase(db_path)

    # Batch store cryptocurrency names
    print(f"\n[3] Storing Cryptocurrency Names with Semantic Analysis")
    print("-" * 80)

    # Prepare concepts (limit to reasonable number for testing)
    test_limit = 50
    concepts_to_store = [(name, "business") for name in list(unique_cryptos.keys())[:test_limit]]

    print(f"Storing {len(concepts_to_store)} cryptocurrency names...")
    start_time = time.time()

    # Use batch operation with transaction
    concept_ids = db.batch_store_concepts(concepts_to_store)

    elapsed = time.time() - start_time
    print(f"Stored {len(concept_ids)} concepts in {elapsed:.2f} seconds")
    print(f"Average: {elapsed/len(concept_ids)*1000:.2f} ms per concept")

    # Get database statistics
    print(f"\n[4] Database Statistics")
    print("-" * 80)

    stats = db.get_statistics()
    print(f"Total Concepts: {stats['total_concepts']}")
    print(f"Total Semantic Units: {stats['total_semantic_units']}")
    print(f"Avg Divine Resonance: {stats['avg_divine_resonance']:.4f}")
    print(f"Avg Distance from JEHOVAH: {stats['avg_distance_from_jehovah']:.4f}")
    print(f"Avg Biblical Balance: {stats['avg_biblical_balance']:.4f}")

    # Semantic search for similar cryptocurrencies
    print(f"\n[5] Semantic Search - Find Similar Cryptocurrencies")
    print("-" * 80)

    search_queries = [
        "digital currency",
        "blockchain token",
        "decentralized finance",
        "smart contracts"
    ]

    for query in search_queries:
        print(f"\nQuery: '{query}'")
        results = db.search_semantic(query, context="business", limit=5)

        if results:
            print(f"Found {len(results)} similar cryptocurrencies:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['concept_text']}")
                print(f"     Similarity: {result['semantic_similarity']:.4f}")
                print(f"     Divine Resonance: {result['divine_resonance']:.4f}")
                print(f"     Query Alignment: {result['query_alignment']:.4f}")
        else:
            print("  No results found")

    # Find cryptocurrencies with high divine resonance
    print(f"\n[6] Divine Resonance Analysis")
    print("-" * 80)

    high_resonance = db.query_by_divine_resonance(min_resonance=0.0, context="business", limit=10)
    print(f"Top 10 cryptocurrencies by divine resonance:")
    for i, concept in enumerate(high_resonance, 1):
        print(f"  {i}. {concept['concept_text']}: {concept['divine_resonance']:.4f}")

    # Proximity search - find similar cryptocurrencies
    print(f"\n[7] Proximity Search - Semantic Clustering")
    print("-" * 80)

    # Get coordinates for Bitcoin
    bitcoin = db.query_by_text("Bitcoin", "business")
    if bitcoin:
        from baseline_biblical_substrate import BiblicalCoordinates
        btc_coords = BiblicalCoordinates(
            bitcoin['love'],
            bitcoin['power'],
            bitcoin['wisdom'],
            bitcoin['justice']
        )

        print(f"Bitcoin coordinates:")
        print(f"  Love: {btc_coords.love:.4f}")
        print(f"  Power: {btc_coords.power:.4f}")
        print(f"  Wisdom: {btc_coords.wisdom:.4f}")
        print(f"  Justice: {btc_coords.justice:.4f}")

        # Find nearby cryptocurrencies in semantic space
        nearby = db.query_by_proximity(btc_coords, max_distance=0.5, context="business", limit=5)
        print(f"\nCryptocurrencies semantically similar to Bitcoin:")
        for i, result in enumerate(nearby, 1):
            print(f"  {i}. {result['concept_text']}")
            print(f"     Semantic Distance: {result['semantic_distance']:.4f}")

    # Analyze semantic relationships
    print(f"\n[8] Semantic Relationship Analysis")
    print("-" * 80)

    # Compare different types of cryptocurrencies
    comparison_pairs = [
        ("Bitcoin", "Ethereum"),
        ("Bitcoin", "Dogecoin"),
        ("Ethereum", "Cardano"),
        ("Tether USDt", "USDC")
    ]

    for name1, name2 in comparison_pairs:
        concept1 = db.query_by_text(name1, "business")
        concept2 = db.query_by_text(name2, "business")

        if concept1 and concept2:
            # Calculate semantic distance
            from baseline_biblical_substrate import BiblicalCoordinates
            import math

            coords1 = BiblicalCoordinates(
                concept1['love'], concept1['power'],
                concept1['wisdom'], concept1['justice']
            )
            coords2 = BiblicalCoordinates(
                concept2['love'], concept2['power'],
                concept2['wisdom'], concept2['justice']
            )

            distance = math.sqrt(
                (coords1.love - coords2.love) ** 2 +
                (coords1.power - coords2.power) ** 2 +
                (coords1.wisdom - coords2.wisdom) ** 2 +
                (coords1.justice - coords2.justice) ** 2
            )

            similarity = 1.0 - (distance / 4.0)

            print(f"\n{name1} vs {name2}:")
            print(f"  Semantic Distance: {distance:.4f}")
            print(f"  Semantic Similarity: {similarity:.4f}")
            print(f"  Relationship: ", end="")
            if similarity > 0.9:
                print("Very Similar")
            elif similarity > 0.8:
                print("Similar")
            elif similarity > 0.7:
                print("Moderately Similar")
            else:
                print("Different")

    # Export analysis results
    print(f"\n[9] Exporting Analysis Results")
    print("-" * 80)

    export_path = "cryptocurrency_semantic_analysis.json"
    db.export_to_json(export_path)
    print(f"Exported analysis to: {export_path}")

    # Create backup
    print(f"\n[10] Creating Database Backup")
    print("-" * 80)

    backup_path = db.create_backup()
    print(f"Backup created: {backup_path}")

    # Close database
    db.close()

    print("\n" + "=" * 80)
    print("CRYPTOCURRENCY DATA TEST COMPLETE!")
    print("=" * 80)
    print(f"\nKey Findings:")
    print(f"  - Processed {len(concept_ids)} unique cryptocurrencies")
    print(f"  - Generated semantic coordinates in 4D space")
    print(f"  - Calculated divine resonance for each cryptocurrency")
    print(f"  - Performed semantic similarity analysis")
    print(f"  - Identified semantic clusters and relationships")
    print(f"\nDatabase File: {db_path}")
    print(f"Export File: {export_path}")
    print(f"Backup File: {backup_path}")

    return True


if __name__ == "__main__":
    try:
        success = test_cryptocurrency_semantic_database()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
