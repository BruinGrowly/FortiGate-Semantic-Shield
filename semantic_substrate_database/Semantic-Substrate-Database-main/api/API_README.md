# Semantic Substrate Database - REST API Documentation

## Version 1.0

**The world's first REST API for a meaning-native database.**

---

## Quick Start

### Installation

```bash
cd api
pip install -r requirements.txt
```

### Start Server

```bash
python semantic_api.py
```

Server starts at: `http://localhost:8000`

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## API Overview

### Base URL
```
http://localhost:8000
```

### Authentication
Bearer token authentication (placeholder - implement proper JWT in production)

### Response Format
All responses are JSON

### Error Handling
Standard HTTP status codes with error details

---

## Endpoints

### Health & Info

#### GET /
Root endpoint with API information

**Response:**
```json
{
  "name": "Semantic Substrate Database API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

#### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-10T12:00:00",
  "database_connected": true,
  "engine_version": "2.2 - Ultimate with Sacred Components"
}
```

#### GET /statistics
Get database statistics

**Response:**
```json
{
  "total_concepts": 100,
  "unique_contexts": 3,
  "total_semantic_units": 100,
  "sacred_numbers_count": 10,
  "total_relationships": 25,
  "avg_divine_resonance": 0.75,
  "avg_distance_from_jehovah": 0.45,
  "avg_biblical_balance": 0.80,
  "context_distribution": {
    "biblical": 60,
    "educational": 30,
    "business": 10
  }
}
```

---

### Concept Management

#### POST /concepts
Create a new concept with semantic coordinates

**Request Body:**
```json
{
  "text": "love",
  "context": "biblical"
}
```

**Response:** (201 Created)
```json
{
  "id": 1,
  "text": "love",
  "context": "biblical",
  "coordinates": {
    "love": 0.9,
    "power": 0.6,
    "wisdom": 0.8,
    "justice": 0.8
  },
  "divine_resonance": 0.85,
  "distance_from_jehovah": 0.45,
  "biblical_balance": 0.75,
  "created_at": "2025-10-10T12:00:00",
  "updated_at": "2025-10-10T12:00:00"
}
```

#### GET /concepts/{concept_id}
Get concept by ID

**Parameters:**
- `concept_id` (path): Concept ID

**Response:**
```json
{
  "id": 1,
  "text": "love",
  "context": "biblical",
  "coordinates": {...},
  "divine_resonance": 0.85,
  ...
}
```

#### GET /concepts
List all concepts with pagination

**Query Parameters:**
- `context` (optional): Filter by context
- `limit` (default: 100): Max results
- `offset` (default: 0): Offset for pagination

**Response:**
```json
[
  {
    "id": 1,
    "text": "love",
    "context": "biblical",
    ...
  },
  ...
]
```

---

### Revolutionary Search Endpoints

#### POST /search/semantic
**THE KILLER FEATURE** - Semantic search by meaning similarity

**Request Body:**
```json
{
  "query": "compassion and kindness",
  "context": "biblical",
  "limit": 10
}
```

**Response:**
```json
[
  {
    "concept_text": "love",
    "context": "biblical",
    "semantic_distance": 0.05,
    "semantic_similarity": 0.98,
    "query_alignment": 0.95,
    "coordinates": {
      "love": 0.9,
      "power": 0.6,
      "wisdom": 0.8,
      "justice": 0.8
    },
    "divine_resonance": 0.85
  },
  ...
]
```

**Example:**
```bash
curl -X POST "http://localhost:8000/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"query": "compassion", "context": "biblical", "limit": 5}'
```

#### POST /search/proximity
Find concepts near a point in 4D semantic space

**Request Body:**
```json
{
  "coordinates": {
    "love": 0.9,
    "power": 0.6,
    "wisdom": 0.8,
    "justice": 0.8
  },
  "max_distance": 0.5,
  "context": "biblical",
  "limit": 10
}
```

**Response:**
```json
[
  {
    "concept_text": "mercy",
    "semantic_distance": 0.15,
    "coordinates": {...},
    ...
  },
  ...
]
```

#### GET /search/divine-resonance
Find concepts with high divine alignment

**Query Parameters:**
- `min_resonance` (default: 0.8): Minimum divine resonance
- `context` (optional): Filter by context
- `limit` (default: 10): Max results

**Response:**
```json
[
  {
    "concept_text": "love",
    "divine_resonance": 0.95,
    ...
  },
  ...
]
```

**Example:**
```bash
curl "http://localhost:8000/search/divine-resonance?min_resonance=0.8&context=biblical"
```

#### GET /search/nearest-anchor/{anchor_id}
Find concepts nearest to a universal anchor

**Path Parameters:**
- `anchor_id`: Universal anchor ID (613, 12, 7, or 40)

**Query Parameters:**
- `max_distance` (default: 1.0): Maximum distance
- `limit` (default: 10): Max results

**Anchors:**
- **613**: Divine Law (613 Commandments)
- **12**: God's People (12 Tribes)
- **7**: Divine Perfection (7 Days)
- **40**: Divine Testing (40 Days/Years)

**Example:**
```bash
curl "http://localhost:8000/search/nearest-anchor/7?max_distance=1.0&limit=5"
```

---

### Sacred Numbers

#### POST /sacred-numbers
Store a sacred number

**Request Body:**
```json
{
  "value": 7
}
```

**Response:** (201 Created)
```json
{
  "id": 1,
  "value": 7.0,
  "is_sacred": true,
  "sacred_resonance": 0.95,
  "biblical_significance": 0.98,
  "divine_attributes": {
    "love": 0.85,
    "power": 0.90,
    "wisdom": 0.95,
    "justice": 0.92
  }
}
```

#### GET /sacred-numbers
List sacred numbers

**Query Parameters:**
- `min_value` (default: 0): Minimum value
- `max_value` (default: 1000): Maximum value
- `only_sacred` (default: true): Only sacred numbers

**Response:**
```json
[
  {
    "id": 1,
    "value": 7.0,
    "is_sacred": true,
    "sacred_resonance": 0.95,
    ...
  },
  ...
]
```

---

### Utility Endpoints

#### POST /export
Export entire database to JSON

**Response:**
```json
{
  "metadata": {
    "exported_at": "2025-10-10T12:00:00",
    "engine_version": "2.2",
    "statistics": {...}
  },
  "concepts": [...],
  "sacred_numbers": [...],
  "anchors": [...]
}
```

#### DELETE /cache
Clear in-memory cache

**Response:** (204 No Content)

---

## Python Client

### Installation

```python
from api.test_api import SemanticAPIClient

client = SemanticAPIClient("http://localhost:8000")
```

### Examples

#### Create Concept
```python
concept = client.create_concept("love", "biblical")
print(f"Created: {concept['text']} (ID: {concept['id']})")
```

#### Semantic Search
```python
results = client.semantic_search("compassion and kindness", "biblical", limit=5)
for result in results:
    print(f"{result['concept_text']}: {result['semantic_similarity']:.3f}")
```

#### Proximity Search
```python
coordinates = {"love": 0.9, "power": 0.6, "wisdom": 0.8, "justice": 0.8}
results = client.proximity_search(coordinates, max_distance=0.5)
```

#### Divine Resonance
```python
results = client.divine_resonance_search(min_resonance=0.8)
```

#### Anchor Navigation
```python
results = client.anchor_search(anchor_id=7, max_distance=1.0)
```

---

## Testing

### Start Server
```bash
python api/semantic_api.py
```

### Run Test Client
```bash
python api/test_api.py
```

### Interactive Testing
Visit: http://localhost:8000/docs

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "Validation error",
  "detail": "Invalid coordinates",
  "timestamp": "2025-10-10T12:00:00"
}
```

### 404 Not Found
```json
{
  "error": "Not found",
  "detail": "Concept with ID 999 not found",
  "timestamp": "2025-10-10T12:00:00"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "detail": "Database connection failed",
  "timestamp": "2025-10-10T12:00:00"
}
```

---

## Performance

### Response Times (Prototype)
- Health check: <10ms
- Create concept: <100ms
- Get concept: <20ms
- Semantic search: <500ms (50 concepts)
- Proximity search: <200ms (100 concepts)
- Export database: <2s (1000 concepts)

### Optimization Tips
1. Use pagination for large result sets
2. Clear cache periodically
3. Filter by context when possible
4. Batch create operations

---

## Production Deployment

### Requirements
```bash
pip install -r api/requirements.txt
```

### Environment Variables
```bash
export SEMANTIC_DB_PATH="/path/to/production.db"
```

### Run with Gunicorn
```bash
gunicorn api.semantic_api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r api/requirements.txt
CMD ["python", "api/semantic_api.py"]
```

---

## Security

### Current Implementation
- Bearer token placeholder
- CORS enabled (configure for production)
- No rate limiting

### Production Requirements
1. Implement JWT authentication
2. Add rate limiting middleware
3. Configure CORS properly
4. Use HTTPS
5. Add request validation
6. Implement API keys

---

## Advanced Usage

### Batch Operations
```python
# Create multiple concepts
concepts = ["love", "mercy", "grace", "faith", "hope"]
for concept in concepts:
    client.create_concept(concept, "biblical")
```

### Complex Queries
```python
# Find concepts similar to "compassion" but near Divine Perfection anchor
results = client.semantic_search("compassion", "biblical")
near_anchor = client.anchor_search(7, max_distance=1.0)

# Intersection of both
similar_and_near = [
    r for r in results
    if r['concept_text'] in [a['concept_text'] for a in near_anchor]
]
```

### Analytics
```python
# Get full statistics
stats = client.get_statistics()

# Export for analysis
data = client.export_database()

# Save to file
with open('export.json', 'w') as f:
    json.dump(data, f, indent=2)
```

---

## Limitations

### Current Prototype
- SQLite backend (single connection)
- In-memory cache only
- No authentication
- No rate limiting
- Sequential processing

### For Production
- Migrate to PostgreSQL
- Add Redis cache
- Implement JWT auth
- Add rate limiting
- Enable concurrent processing

---

## Support

### Documentation
- API Docs: http://localhost:8000/docs
- Main README: ../DATABASE_README.md
- Quick Start: ../QUICKSTART.md

### Testing
```bash
# Health check
curl http://localhost:8000/health

# Create concept
curl -X POST http://localhost:8000/concepts \
  -H "Content-Type: application/json" \
  -d '{"text": "love", "context": "biblical"}'
```

---

**Semantic Substrate Database API v1.0**
*Revolutionary meaning-native database with RESTful access*
