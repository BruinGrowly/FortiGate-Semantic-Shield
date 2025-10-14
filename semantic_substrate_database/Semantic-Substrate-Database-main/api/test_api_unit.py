"""
API contract tests for the FastAPI layer.

These tests spin up the application with a temporary SQLite database and
exercise core endpoints.  The goal is to ensure the HTTP surface remains
stable and serialises the expected fields.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def api_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """
    Provide a TestClient with an isolated database.

    A module-scoped fixture keeps the API warm across tests while ensuring the
    underlying SQLite file is unique to the test run.
    """
    db_root = tmp_path_factory.mktemp("api-db")
    db_path = db_root / "semantic_api.db"
    original_env = os.environ.get("SEMANTIC_DB_PATH")
    os.environ["SEMANTIC_DB_PATH"] = str(db_path)

    # Reload the module so the lifespan hook picks up the temp path.
    module = importlib.import_module("api.semantic_api")
    module = importlib.reload(module)

    try:
        with TestClient(module.app) as client:
            yield client
    finally:
        if original_env is None:
            os.environ.pop("SEMANTIC_DB_PATH", None)
        else:
            os.environ["SEMANTIC_DB_PATH"] = original_env
        if Path(db_path).exists():
            os.remove(db_path)


@pytest.fixture
def stored_concept(api_client: TestClient) -> int:
    response = api_client.post(
        "/concepts",
        json={"text": "love", "context": "biblical"},
    )
    response.raise_for_status()
    return response.json()["id"]


def test_root_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Semantic Substrate Database API"
    assert "version" in body


def test_health_check(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["database_connected"] is True


def test_create_and_get_concept(api_client: TestClient) -> None:
    create_response = api_client.post(
        "/concepts",
        json={"text": "wisdom", "context": "biblical"},
    )
    assert create_response.status_code == 201
    created = create_response.json()
    concept_id = created["id"]

    get_response = api_client.get(f"/concepts/{concept_id}")
    assert get_response.status_code == 200
    retrieved = get_response.json()
    assert retrieved["id"] == concept_id
    assert retrieved["text"] == "wisdom"
    assert "coordinates" in retrieved


def test_list_concepts(api_client: TestClient, stored_concept: int) -> None:
    response = api_client.get("/concepts?limit=10")
    assert response.status_code == 200
    concepts = response.json()
    assert isinstance(concepts, list)
    assert any(item["id"] == stored_concept for item in concepts)


def test_semantic_search(api_client: TestClient) -> None:
    payloads = [
        {"text": "love", "context": "biblical"},
        {"text": "mercy", "context": "biblical"},
        {"text": "grace", "context": "biblical"},
    ]
    for data in payloads:
        api_client.post("/concepts", json=data)

    response = api_client.post(
        "/search/semantic",
        json={"query": "compassion and kindness", "context": "biblical", "limit": 5},
    )
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert results
    top = results[0]
    for key in ("concept_text", "semantic_similarity", "query_alignment"):
        assert key in top


def test_proximity_search(api_client: TestClient) -> None:
    response = api_client.post(
        "/search/proximity",
        json={
            "coordinates": {"love": 0.8, "power": 0.5, "wisdom": 0.7, "justice": 0.6},
            "max_distance": 1.0,
            "context": "biblical",
            "limit": 5,
        },
    )
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)


def test_divine_resonance_search(api_client: TestClient) -> None:
    response = api_client.get(
        "/search/divine-resonance",
        params={"min_resonance": 0.0, "context": "biblical", "limit": 5},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_anchor_search(api_client: TestClient) -> None:
    response = api_client.get("/search/nearest-anchor/7", params={"max_distance": 2.0, "limit": 5})
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_sacred_number_endpoints(api_client: TestClient) -> None:
    create = api_client.post("/sacred-numbers", json={"value": 7})
    assert create.status_code == 201

    listing = api_client.get("/sacred-numbers")
    assert listing.status_code == 200
    numbers = listing.json()
    assert isinstance(numbers, list)
    assert any(item["value"] == 7 for item in numbers)


def test_statistics(api_client: TestClient) -> None:
    response = api_client.get("/statistics")
    assert response.status_code == 200
    body = response.json()
    for key in ("total_concepts", "sacred_numbers_count", "avg_divine_resonance"):
        assert key in body


def test_export(api_client: TestClient) -> None:
    response = api_client.post("/export")
    assert response.status_code == 200
    body = response.json()
    for key in ("metadata", "concepts", "sacred_numbers"):
        assert key in body


def test_clear_cache(api_client: TestClient) -> None:
    response = api_client.delete("/cache")
    assert response.status_code == 204


def test_validation_errors(api_client: TestClient) -> None:
    response = api_client.post(
        "/concepts",
        json={"text": "test", "context": "invalid"},
    )
    assert response.status_code == 422


def test_not_found_handling(api_client: TestClient) -> None:
    response = api_client.get("/concepts/999999")
    assert response.status_code == 404
