# Semantic Substrate Database

Semantic Substrate Database (SSDB) is an experimental semantic storage engine that layers
traditional relational persistence with lightweight models for intent, context, and execution.
It ships with several progressively richer database front-ends, a bundle of integration tests,
and a FastAPI surface for programmatic access.

> This repository is not a drop-in replacement for a production-grade database. It is a research
> playground that explores ideas around value-aligned storage, semantic coordinates, and
> self-describing data flows.

## Feature Highlights
- **SQLite-backed persistence** with schema management, backups, and export tooling.
- **Semantic coordinate system** that maps concepts onto four axes: love, power, wisdom, and justice.
- **Sacred number model** with deterministic fallbacks when the external engine is unavailable.
- **ICE framework integration** for translating high-level “thoughts” into executable database actions.
- **Meaning-based programming layer** that turns declarative specifications into stored operations.
- **FastAPI service** (`api/semantic_api.py`) exposing CRUD, search, and analytics endpoints.
- **Comprehensive tests** covering core storage, semantic queries, backups, and the extended layers.

## Semantic Coordinate Engine
SSDB computes love/power/wisdom/justice coordinates by combining:
- the legacy biblical keyword heuristics,
- a modern-language keyword map (e.g. `analytical`, `clarity`, `ethical`), and
- ICE Framework intent/context analysis.

Two environment variables let you adjust how much weight the new layers carry:

> Optional semantic embeddings: Set `SSDB_USE_EMBEDDINGS=1` and install `sentence-transformers` to blend sentence embeddings with the heuristic coordinates.

| Variable | Default | Description |
|----------|---------|-------------|
| `SSDB_SEMANTIC_WEIGHT` | `0.35` | Contribution from the modern keyword map |
| `SSDB_ICE_WEIGHT` | `0.35` | Contribution from ICE Framework execution coordinates |
| `SSDB_USE_EMBEDDINGS` | `false` | Enable sentence-transformer semantic similarity (requires `sentence-transformers`) |
| `SSDB_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model name when embeddings are enabled |
| `SSDB_EMBEDDING_WEIGHT` | `0.25` | Contribution of embedding similarity to coordinates |

Values are clamped to the range `[0.0, 1.0]`. Set them before starting the API or running ingestion jobs if you need more conservative or aggressive semantic blending.

## Repository Layout
- `src/` – Library source code. `semantic_substrate_database.py` is the primary entry point.
- `api/` – FastAPI service and accompanying tests.
- `examples/` – Executable scripts demonstrating the layered database interfaces.
- `tests/` – PyTest suites for core, integration, and deep-dive features.
- `docs/` – Background material, including the ethical-foundation narrative.
- `docker-compose.yml` / `Dockerfile` – Container tooling for API demos and test runs.

## Quick Start
```bash
git clone https://github.com/BruinGrowly/Semantic-Substrate-Database.git
cd Semantic-Substrate-Database
python -m venv .venv
.venv\Scripts\activate  # PowerShell on Windows (use `source .venv/bin/activate` on Unix)
pip install --upgrade pip
pip install -r requirements.txt
```

### Basic Usage
```python
from semantic_substrate_database import SemanticSubstrateDatabase

db = SemanticSubstrateDatabase("semantic.db")
concept_id = db.store_concept("Show compassion to those who suffer", context="ethics")
related = db.query_by_text("compassion", context="ethics")
db.store_sacred_number(613)
stats = db.get_statistics()
db.close()
```

Run a richer demonstration:
```bash
python src/semantic_substrate_database.py
```

## Running Tests
```bash
python -m pytest
```
Large suites (for example `tests/test_deep_dive_database.py`) touch many subsystems, so expect the
full run to take several minutes.

## API Service
```bash
pip install -r api/requirements.txt  # Optional, installs pinned API versions
uvicorn api.semantic_api:app --reload
```

## Docker Tooling
```bash
# Build the image
docker build -t semantic-substrate-db .

# Run the bundled demonstration
docker run --rm -v %CD%\data:/app/data semantic-substrate-db

# Run the test profile defined in docker-compose
docker compose --profile test run --rm ssdb-test
```
The default image command executes the core demonstration script. Use `docker compose up ssdb`
to start the API stack with code mounted for live development.

## Contributing
Issues and pull requests are welcome. Please open a ticket describing the change before submitting
large patches. When contributing code, run `python -m pytest` and ensure style tools (pylint, black,
mypy) report clean output.

## License
MIT © 2025 BruinGrowly
