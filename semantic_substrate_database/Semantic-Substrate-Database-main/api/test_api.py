"""
Test client for Semantic Substrate Database API

Demonstrates all API endpoints with example usage.
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"


class SemanticAPIClient:
    """Client for interacting with Semantic Substrate Database API"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    # Health & Info
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        response = self.session.get(f"{self.base_url}/statistics")
        response.raise_for_status()
        return response.json()

    # Concepts
    def create_concept(self, text: str, context: str = "biblical") -> Dict[str, Any]:
        """Create a new concept"""
        response = self.session.post(
            f"{self.base_url}/concepts",
            json={"text": text, "context": context}
        )
        response.raise_for_status()
        return response.json()

    def get_concept(self, concept_id: int) -> Dict[str, Any]:
        """Get concept by ID"""
        response = self.session.get(f"{self.base_url}/concepts/{concept_id}")
        response.raise_for_status()
        return response.json()

    def list_concepts(self, context: str = None, limit: int = 100) -> list:
        """List all concepts"""
        params = {"limit": limit}
        if context:
            params["context"] = context

        response = self.session.get(f"{self.base_url}/concepts", params=params)
        response.raise_for_status()
        return response.json()

    # Revolutionary Searches
    def semantic_search(self, query: str, context: str = "biblical", limit: int = 10) -> list:
        """Revolutionary semantic search"""
        response = self.session.post(
            f"{self.base_url}/search/semantic",
            json={
                "query": query,
                "context": context,
                "limit": limit
            }
        )
        response.raise_for_status()
        return response.json()

    def proximity_search(self, coordinates: Dict[str, float], max_distance: float = 0.5,
                        context: str = None, limit: int = 10) -> list:
        """Search by proximity in semantic space"""
        response = self.session.post(
            f"{self.base_url}/search/proximity",
            json={
                "coordinates": coordinates,
                "max_distance": max_distance,
                "context": context,
                "limit": limit
            }
        )
        response.raise_for_status()
        return response.json()

    def divine_resonance_search(self, min_resonance: float = 0.8,
                               context: str = None, limit: int = 10) -> list:
        """Search by divine resonance"""
        params = {"min_resonance": min_resonance, "limit": limit}
        if context:
            params["context"] = context

        response = self.session.get(
            f"{self.base_url}/search/divine-resonance",
            params=params
        )
        response.raise_for_status()
        return response.json()

    def anchor_search(self, anchor_id: int, max_distance: float = 1.0, limit: int = 10) -> list:
        """Search nearest to universal anchor"""
        response = self.session.get(
            f"{self.base_url}/search/nearest-anchor/{anchor_id}",
            params={"max_distance": max_distance, "limit": limit}
        )
        response.raise_for_status()
        return response.json()

    # Sacred Numbers
    def create_sacred_number(self, value: float) -> Dict[str, Any]:
        """Store a sacred number"""
        response = self.session.post(
            f"{self.base_url}/sacred-numbers",
            json={"value": value}
        )
        response.raise_for_status()
        return response.json()

    def list_sacred_numbers(self, min_value: float = 0, max_value: float = 1000,
                           only_sacred: bool = True) -> list:
        """List sacred numbers"""
        response = self.session.get(
            f"{self.base_url}/sacred-numbers",
            params={
                "min_value": min_value,
                "max_value": max_value,
                "only_sacred": only_sacred
            }
        )
        response.raise_for_status()
        return response.json()

    # Utilities
    def export_database(self) -> Dict[str, Any]:
        """Export entire database"""
        response = self.session.post(f"{self.base_url}/export")
        response.raise_for_status()
        return response.json()

    def clear_cache(self):
        """Clear cache"""
        response = self.session.delete(f"{self.base_url}/cache")
        response.raise_for_status()


def demonstrate_api():
    """Demonstrate all API functionality"""
    print("=" * 80)
    print("SEMANTIC SUBSTRATE DATABASE API - DEMONSTRATION")
    print("=" * 80)

    client = SemanticAPIClient()

    # Health Check
    print("\n[1] Health Check")
    print("-" * 40)
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Database Connected: {health['database_connected']}")
    print(f"Engine Version: {health['engine_version']}")

    # Create Concepts
    print("\n[2] Creating Biblical Concepts")
    print("-" * 40)
    concepts = ["love", "wisdom", "mercy", "grace", "faith"]
    for concept in concepts:
        result = client.create_concept(concept, "biblical")
        print(f"Created: {result['text']} (ID: {result['id']})")

    # List Concepts
    print("\n[3] Listing Concepts")
    print("-" * 40)
    all_concepts = client.list_concepts(context="biblical", limit=10)
    print(f"Total concepts: {len(all_concepts)}")
    for c in all_concepts[:5]:
        print(f"  - {c['text']}: divine_resonance={c['divine_resonance']:.3f}")

    # Revolutionary Semantic Search
    print("\n[4] Revolutionary Semantic Search")
    print("-" * 40)
    query = "compassion and kindness"
    print(f"Query: '{query}'")
    results = client.semantic_search(query, "biblical", limit=5)
    print(f"\nFound {len(results)} semantically similar concepts:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['concept_text']}")
        print(f"   Similarity: {r['semantic_similarity']:.3f}")
        print(f"   Alignment: {r['query_alignment']:.3f}")

    # Proximity Search
    print("\n[5] Proximity Search (4D Semantic Space)")
    print("-" * 40)
    target = {"love": 0.9, "power": 0.6, "wisdom": 0.8, "justice": 0.8}
    print(f"Target coordinates: {target}")
    results = client.proximity_search(target, max_distance=0.5, limit=5)
    print(f"\nFound {len(results)} concepts nearby:")
    for r in results[:3]:
        print(f"  - {r['concept_text']}: distance={r['semantic_distance']:.3f}")

    # Divine Resonance Search
    print("\n[6] Divine Resonance Search")
    print("-" * 40)
    results = client.divine_resonance_search(min_resonance=0.0, limit=5)
    print(f"Concepts with highest divine alignment:")
    for r in results:
        print(f"  - {r['concept_text']}: {r['divine_resonance']:.3f}")

    # Anchor Navigation
    print("\n[7] Universal Anchor Navigation")
    print("-" * 40)
    anchor_id = 7  # Divine Perfection
    results = client.anchor_search(anchor_id, max_distance=2.0, limit=3)
    print(f"Concepts nearest to Anchor {anchor_id} (Divine Perfection):")
    for r in results:
        print(f"  - {r['concept_text']}: distance={r['semantic_distance']:.3f}")

    # Sacred Numbers
    print("\n[8] Sacred Numbers")
    print("-" * 40)
    sacred_nums = [7, 12, 40, 613]
    for num in sacred_nums:
        client.create_sacred_number(num)
    print(f"Stored {len(sacred_nums)} sacred numbers")

    sacred_list = client.list_sacred_numbers(only_sacred=True)
    print(f"\nSacred numbers in database:")
    for sn in sacred_list[:5]:
        print(f"  - {int(sn['value'])}: resonance={sn['sacred_resonance']:.3f}")

    # Statistics
    print("\n[9] Database Statistics")
    print("-" * 40)
    stats = client.get_statistics()
    print(f"Total Concepts: {stats['total_concepts']}")
    print(f"Sacred Numbers: {stats['sacred_numbers_count']}")
    print(f"Avg Divine Resonance: {stats['avg_divine_resonance']:.3f}")
    print(f"Avg Distance from JEHOVAH: {stats['avg_distance_from_jehovah']:.3f}")

    print("\n" + "=" * 80)
    print("API DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        demonstrate_api()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API server")
        print("Please start the server first:")
        print("  python api/semantic_api.py")
    except Exception as e:
        print(f"\nERROR: {e}")
