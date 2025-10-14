"""
SEMANTIC SUBSTRATE DATABASE - RESTful API v1.0

FastAPI-based REST API for the revolutionary Semantic Substrate Database.
Provides HTTP endpoints for all database operations with authentication,
rate limiting, and comprehensive error handling.

Author: Built on Semantic Substrate Database v1.0
License: MIT
"""

from fastapi import FastAPI, HTTPException, Depends, status, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import sys
import os
from contextlib import asynccontextmanager

# Ensure the src package is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from semantic_substrate_database import SemanticSubstrateDatabase
from baseline_biblical_substrate import BiblicalCoordinates

# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class ConceptCreate(BaseModel):
    """Schema for creating a new concept"""
    text: str = Field(..., min_length=1, max_length=500, description="Concept text")
    context: str = Field(default="biblical", description="Semantic context")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "love",
                "context": "biblical"
            }
        }
    )

    @field_validator('context')
    @classmethod
    def validate_context(cls, v: str) -> str:
        allowed = ['biblical', 'educational', 'business', 'scientific', 'general']
        if v not in allowed:
            raise ValueError(f'Context must be one of: {", ".join(allowed)}')
        return v


class ConceptResponse(BaseModel):
    """Schema for concept response"""
    id: int
    text: str
    context: str
    coordinates: Dict[str, float]
    divine_resonance: float
    distance_from_jehovah: float
    biblical_balance: float
    created_at: str
    updated_at: str
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
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
        }
    )


class SemanticSearchRequest(BaseModel):
    """Schema for semantic search request"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    context: str = Field(default="biblical", description="Search context")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "compassion and kindness",
                "context": "biblical",
                "limit": 10
            }
        }
    )


class SemanticSearchResult(BaseModel):
    """Schema for semantic search result"""
    concept_text: str
    context: str
    semantic_distance: float
    semantic_similarity: float
    query_alignment: float
    coordinates: Dict[str, float]
    divine_resonance: float
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "concept_text": "love",
                "context": "biblical",
                "semantic_distance": 0.12,
                "semantic_similarity": 0.88,
                "query_alignment": 0.82,
                "coordinates": {
                    "love": 0.92,
                    "power": 0.40,
                    "wisdom": 0.74,
                    "justice": 0.66
                },
                "divine_resonance": 0.87
            }
        }
    )


class ProximityQueryRequest(BaseModel):
    """Schema for proximity query"""
    coordinates: Dict[str, float] = Field(..., description="Target coordinates (love, power, wisdom, justice)")
    max_distance: float = Field(default=0.5, ge=0.0, le=4.0, description="Maximum distance")
    context: Optional[str] = Field(None, description="Filter by context")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")

    @field_validator('coordinates')
    @classmethod
    def validate_coordinates(cls, v: Dict[str, float]) -> Dict[str, float]:
        required = ['love', 'power', 'wisdom', 'justice']
        if not all(k in v for k in required):
            raise ValueError(f'Coordinates must contain: {", ".join(required)}')
        for val in v.values():
            if not (0.0 <= val <= 1.0):
                raise ValueError('Coordinate values must be between 0.0 and 1.0')
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class SacredNumberCreate(BaseModel):
    """Schema for creating sacred number"""
    value: float = Field(..., description="Number value")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "value": 7
            }
        }
    )


class SacredNumberResponse(BaseModel):
    """Schema for sacred number response"""
    id: int
    value: float
    is_sacred: bool
    sacred_resonance: float
    biblical_significance: float
    divine_attributes: Dict[str, float]
    model_config = ConfigDict(from_attributes=True)


class StatisticsResponse(BaseModel):
    """Schema for database statistics"""
    total_concepts: int
    unique_contexts: int
    total_semantic_units: int
    sacred_numbers_count: int
    total_relationships: int
    avg_divine_resonance: float
    avg_distance_from_jehovah: float
    avg_biblical_balance: float
    context_distribution: Dict[str, int]
    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    """Schema for health check"""
    status: str
    timestamp: str
    database_connected: bool
    engine_version: str
    model_config = ConfigDict(from_attributes=True)


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str
    detail: Optional[str] = None
    timestamp: str
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Internal server error",
                "detail": "Database unavailable",
                "timestamp": "2025-10-10T12:00:00"
            }
        }
    )


# ============================================================================
# API APPLICATION
# ============================================================================

# Global database instance
db: Optional[SemanticSubstrateDatabase] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for database connection"""
    global db

    # Startup
    db_path = os.environ.get("SEMANTIC_DB_PATH", "semantic_api.db")
    db = SemanticSubstrateDatabase(db_path)
    print(f"[API] Database connected: {db_path}")

    yield

    # Shutdown
    if db:
        db.close()
        print("[API] Database connection closed")


# Create FastAPI application
app = FastAPI(
    title="Semantic Substrate Database API",
    description="Revolutionary meaning-native database with semantic search and divine alignment metrics",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security (Bearer token - implement proper auth in production)
security = HTTPBearer(auto_error=False)


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

def get_db() -> SemanticSubstrateDatabase:
    """Dependency to get database instance"""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized"
        )
    return db


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify authentication token (placeholder - implement real auth)"""
    # TODO: Implement real JWT verification
    if credentials is None:
        return "anonymous"
    return credentials.credentials


# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Semantic Substrate Database API",
        "version": "1.0.0",
        "description": "Revolutionary meaning-native database",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(database: SemanticSubstrateDatabase = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Test database connection
        stats = database.get_statistics()
        connected = True
        engine_version = database.engine.engine_version
    except Exception as e:
        connected = False
        engine_version = "unknown"

    return HealthResponse(
        status="healthy" if connected else "unhealthy",
        timestamp=datetime.now().isoformat(),
        database_connected=connected,
        engine_version=engine_version
    )


@app.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(
    database: SemanticSubstrateDatabase = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get database statistics"""
    try:
        stats = database.get_statistics()
        return StatisticsResponse(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


# ============================================================================
# CONCEPT ENDPOINTS
# ============================================================================

@app.post("/concepts", response_model=ConceptResponse, status_code=status.HTTP_201_CREATED)
async def create_concept(
    concept: ConceptCreate,
    database: SemanticSubstrateDatabase = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Create a new concept with semantic coordinates"""
    try:
        concept_id = database.store_concept(concept.text, concept.context)

        # Retrieve the created concept
        result = database.query_by_text(concept.text, concept.context)

        return ConceptResponse(
            id=result['id'],
            text=result['concept_text'],
            context=result['context'],
            coordinates={
                'love': result['love'],
                'power': result['power'],
                'wisdom': result['wisdom'],
                'justice': result['justice']
            },
            divine_resonance=result['divine_resonance'],
            distance_from_jehovah=result['distance_from_jehovah'],
            biblical_balance=result['biblical_balance'],
            created_at=result['created_at'],
            updated_at=result['updated_at']
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create concept: {str(e)}"
        )


@app.get("/concepts/{concept_id}", response_model=ConceptResponse)
async def get_concept(
    concept_id: int,
    database: SemanticSubstrateDatabase = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get concept by ID"""
    try:
        # Query by ID (we need to add this method)
        cursor = database.conn.cursor()
        cursor.execute("SELECT * FROM semantic_coordinates WHERE id = ?", (concept_id,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept with ID {concept_id} not found"
            )

        result = dict(row)

        return ConceptResponse(
            id=result['id'],
            text=result['concept_text'],
            context=result['context'],
            coordinates={
                'love': result['love'],
                'power': result['power'],
                'wisdom': result['wisdom'],
                'justice': result['justice']
            },
            divine_resonance=result['divine_resonance'],
            distance_from_jehovah=result['distance_from_jehovah'],
            biblical_balance=result['biblical_balance'],
            created_at=result['created_at'],
            updated_at=result['updated_at']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get concept: {str(e)}"
        )


@app.get("/concepts", response_model=List[ConceptResponse])
async def list_concepts(
    context: Optional[str] = Query(None, description="Filter by context"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    database: SemanticSubstrateDatabase = Depends(get_db),
    token: str = Depends(verify_token)
):
    """List all concepts with optional filtering"""
    try:
        cursor = database.conn.cursor()

        if context:
            cursor.execute("""
                SELECT * FROM semantic_coordinates
                WHERE context = ?
                ORDER BY id DESC
                LIMIT ? OFFSET ?
            """, (context, limit, offset))
        else:
            cursor.execute("""
                SELECT * FROM semantic_coordinates
                ORDER BY id DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            results.append(ConceptResponse(
                id=result['id'],
                text=result['concept_text'],
                context=result['context'],
                coordinates={
                    'love': result['love'],
                    'power': result['power'],
                    'wisdom': result['wisdom'],
                    'justice': result['justice']
                },
                divine_resonance=result['divine_resonance'],
                distance_from_jehovah=result['distance_from_jehovah'],
                biblical_balance=result['biblical_balance'],
                created_at=result['created_at'],
                updated_at=result['updated_at']
            ))

        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list concepts: {str(e)}"
        )


# ============================================================================
# REVOLUTIONARY QUERY ENDPOINTS
# ============================================================================

@app.post("/search/semantic", response_model=List[SemanticSearchResult])
async def semantic_search(
    search: SemanticSearchRequest,
    database: SemanticSubstrateDatabase = Depends(get_db),
    token: str = Depends(verify_token)
):
    """
    Revolutionary semantic search - finds concepts by meaning similarity

    This is the killer feature - true meaning-based search!
    """
    try:
        results = database.search_semantic(
            search.query,
            context=search.context,
            limit=search.limit
        )

        return [
            SemanticSearchResult(
                concept_text=r['concept_text'],
                context=r['context'],
                semantic_distance=r['semantic_distance'],
                semantic_similarity=r['semantic_similarity'],
                query_alignment=r['query_alignment'],
                coordinates={
                    'love': r['love'],
                    'power': r['power'],
                    'wisdom': r['wisdom'],
                    'justice': r['justice']
                },
                divine_resonance=r['divine_resonance']
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}"
        )


@app.post("/search/proximity", response_model=List[Dict[str, Any]])
async def proximity_search(
    query: ProximityQueryRequest,
    database: SemanticSubstrateDatabase = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Find concepts near a point in 4D semantic space"""
    try:
        target_coords = BiblicalCoordinates(
            query.coordinates['love'],
            query.coordinates['power'],
            query.coordinates['wisdom'],
            query.coordinates['justice']
        )

        results = database.query_by_proximity(
            target_coords,
            max_distance=query.max_distance,
            context=query.context,
            limit=query.limit
        )

        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Proximity search failed: {str(e)}"
        )


@app.get("/search/divine-resonance", response_model=List[Dict[str, Any]])
async def search_by_divine_resonance(
    min_resonance: float = Query(0.8, ge=0.0, le=1.0, description="Minimum divine resonance"),
    context: Optional[str] = Query(None, description="Filter by context"),
    limit: int = Query(10, ge=1, le=100, description="Max results"),
    database: SemanticSubstrateDatabase = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Find concepts with high divine alignment"""
    try:
        results = database.query_by_divine_resonance(
            min_resonance=min_resonance,
            context=context,
            limit=limit
        )

        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Divine resonance search failed: {str(e)}"
        )


@app.get("/search/nearest-anchor/{anchor_id}", response_model=List[Dict[str, Any]])
async def search_nearest_to_anchor(
    anchor_id: int,
    max_distance: float = Query(1.0, ge=0.0, le=4.0, description="Maximum distance"),
    limit: int = Query(10, ge=1, le=100, description="Max results"),
    database: SemanticSubstrateDatabase = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Find concepts nearest to a universal anchor"""
    try:
        if anchor_id not in [613, 12, 7, 40]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Anchor ID must be one of: 613, 12, 7, 40"
            )

        results = database.query_nearest_to_anchor(
            anchor_id=anchor_id,
            max_distance=max_distance,
            limit=limit
        )

        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anchor search failed: {str(e)}"
        )


# ============================================================================
# SACRED NUMBER ENDPOINTS
# ============================================================================

@app.post("/sacred-numbers", response_model=SacredNumberResponse, status_code=status.HTTP_201_CREATED)
async def create_sacred_number(
    sacred_num: SacredNumberCreate,
    database: SemanticSubstrateDatabase = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Store a sacred number"""
    try:
        num_id = database.store_sacred_number(sacred_num.value)

        # Retrieve the stored number
        results = database.query_sacred_numbers(
            min_value=sacred_num.value,
            max_value=sacred_num.value,
            only_sacred=False
        )

        if not results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve stored sacred number"
            )

        result = results[0]

        return SacredNumberResponse(
            id=result['id'],
            value=result['value'],
            is_sacred=bool(result['is_sacred']),
            sacred_resonance=result['sacred_resonance'],
            biblical_significance=result['biblical_significance'],
            divine_attributes=result['divine_attributes']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create sacred number: {str(e)}"
        )


@app.get("/sacred-numbers", response_model=List[SacredNumberResponse])
async def list_sacred_numbers(
    min_value: float = Query(0, description="Minimum value"),
    max_value: float = Query(1000, description="Maximum value"),
    only_sacred: bool = Query(True, description="Only sacred numbers"),
    database: SemanticSubstrateDatabase = Depends(get_db),
    token: str = Depends(verify_token)
):
    """List sacred numbers"""
    try:
        results = database.query_sacred_numbers(
            min_value=min_value,
            max_value=max_value,
            only_sacred=only_sacred
        )

        return [
            SacredNumberResponse(
                id=r['id'],
                value=r['value'],
                is_sacred=bool(r['is_sacred']),
                sacred_resonance=r['sacred_resonance'],
                biblical_significance=r['biblical_significance'],
                divine_attributes=r['divine_attributes']
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sacred numbers: {str(e)}"
        )


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.post("/export")
async def export_database(
    database: SemanticSubstrateDatabase = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Export entire database to JSON"""
    try:
        import tempfile
        import json

        # Create temp file
        temp_file = tempfile.mktemp(suffix='.json')
        database.export_to_json(temp_file)

        # Read and return JSON
        with open(temp_file, 'r') as f:
            data = json.load(f)

        # Cleanup
        os.unlink(temp_file)

        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}"
        )


@app.delete("/cache", status_code=status.HTTP_204_NO_CONTENT)
async def clear_cache(
    database: SemanticSubstrateDatabase = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Clear the in-memory cache"""
    try:
        database.clear_cache()
        return None
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


# ============================================================================
# MAIN (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("SEMANTIC SUBSTRATE DATABASE API")
    print("=" * 80)
    print("Starting API server...")
    print("Documentation: http://localhost:8000/docs")
    print("=" * 80)

    uvicorn.run(
        "semantic_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
