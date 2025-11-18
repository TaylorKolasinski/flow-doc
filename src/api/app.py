"""
FastAPI application for Flow-Doc REST API.

This module sets up the FastAPI application with CORS, lifecycle management,
error handling, and includes all API routes.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes import router, get_service_container
from src.api.models import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Lifespan Context Manager
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Lifespan context manager for startup and shutdown events.

    Startup:
    - Initialize services (analyzer, memory, generator, agent)
    - Check Ollama and ChromaDB connections
    - Log system status

    Shutdown:
    - Cleanup resources
    - Close connections
    - Log shutdown message
    """
    # Startup
    logger.info("=" * 70)
    logger.info("  Flow-Doc API Server Starting")
    logger.info("=" * 70)

    try:
        # Initialize services
        logger.info("Initializing services...")
        services = get_service_container()
        services.initialize(codebase_path="./sample_codebase")

        logger.info("✓ Services initialized successfully")
        logger.info(f"  - Codebase: {services.codebase_path}")
        logger.info(f"  - Analyzer: Ready")
        logger.info(f"  - Memory: Ready (ChromaDB)")
        logger.info(f"  - Generator: Ready (Ollama)")
        logger.info(f"  - Agent: Ready (LangGraph)")

        # Check Ollama connection
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                logger.info(f"✓ Ollama connected ({len(models)} models available)")
            else:
                logger.warning("⚠ Ollama connection check failed")
        except Exception as e:
            logger.warning(f"⚠ Ollama not reachable: {e}")
            logger.warning("  Make sure Ollama is running: ollama serve")

        logger.info("")
        logger.info("API Server Ready!")
        logger.info("=" * 70)
        logger.info("Documentation: http://localhost:8000/docs")
        logger.info("Health Check:  http://localhost:8000/health")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"✗ Startup failed: {e}", exc_info=True)
        raise

    # Yield control to the application
    yield

    # Shutdown
    logger.info("")
    logger.info("=" * 70)
    logger.info("  Flow-Doc API Server Shutting Down")
    logger.info("=" * 70)

    try:
        services = get_service_container()
        services.cleanup()
        logger.info("✓ Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)

    logger.info("Goodbye!")
    logger.info("=" * 70)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Flow-Doc API",
    description="""
# Flow-Doc: AI-Powered Documentation Assistant

REST API for automated code documentation generation using LangGraph, Ollama, and ChromaDB.

## Features

- **Codebase Analysis**: AST-based parsing of Python files
- **Semantic Memory**: ChromaDB for component storage and retrieval
- **Agentic Workflow**: LangGraph-powered documentation generation
- **RAG Queries**: Natural language questions about your codebase
- **Quality Validation**: Iterative refinement with quality checks
- **Export**: Download all generated documentation

## Workflow

1. **Analyze** - Parse codebase and store components
2. **Document** - Generate docs using LLM workflow
3. **Query** - Ask questions about the codebase
4. **Export** - Download generated documentation

## Authentication

Currently no authentication required (development mode).

## Rate Limiting

No rate limiting (development mode).

## Support

- GitHub: https://github.com/yourusername/flow-doc
- Issues: https://github.com/yourusername/flow-doc/issues
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# ============================================================================
# CORS Middleware
# ============================================================================

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # Vue dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc.detail),
            code=f"HTTP_{exc.status_code}"
        ).model_dump()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc.errors()),
            code="VALIDATION_ERROR"
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc),
            code="INTERNAL_ERROR"
        ).model_dump()
    )


# ============================================================================
# Include Routers
# ============================================================================

app.include_router(router, prefix="/api/v1")


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.

    Returns basic information about the API and links to documentation.
    """
    return {
        "message": "Flow-Doc API Server",
        "version": "1.0.0",
        "description": "AI-Powered Documentation Assistant",
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "analyze": "POST /api/v1/analyze",
            "document": "POST /api/v1/document/{component}",
            "query": "POST /api/v1/query",
            "patterns": "GET /api/v1/patterns",
            "visualize": "GET /api/v1/visualize",
            "export": "GET /api/v1/export",
            "health": "GET /api/v1/health"
        }
    }


# ============================================================================
# Development Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
