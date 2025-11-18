"""
Pydantic models for FastAPI request/response schemas.

This module defines all request and response models for the Flow-Doc API,
ensuring type safety and automatic validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


# ============================================================================
# Request Models
# ============================================================================

class AnalyzeRequest(BaseModel):
    """Request to analyze a codebase directory."""

    path: str = Field(
        ...,
        description="Path to the codebase directory to analyze",
        example="./sample_codebase"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "path": "./sample_codebase"
            }
        }


class DocumentRequest(BaseModel):
    """Request to generate documentation for a specific component."""

    component: str = Field(
        ...,
        description="Name of the component to document",
        example="UserService"
    )
    path: str = Field(
        ...,
        description="Relative path to the component file",
        example="services/user.py"
    )
    max_iterations: int = Field(
        default=3,
        description="Maximum number of refinement iterations",
        ge=1,
        le=10
    )

    class Config:
        json_schema_extra = {
            "example": {
                "component": "UserService",
                "path": "services/user.py",
                "max_iterations": 3
            }
        }


class QueryRequest(BaseModel):
    """Request for natural language query over codebase."""

    question: str = Field(
        ...,
        description="Natural language question about the codebase",
        min_length=5,
        example="What authentication methods are used?"
    )
    k: int = Field(
        default=5,
        description="Number of similar components to retrieve",
        ge=1,
        le=20
    )

    @validator('question')
    def validate_question(cls, v):
        """Ensure question is not empty or just whitespace."""
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What authentication methods are used?",
                "k": 5
            }
        }


class WorkflowRequest(BaseModel):
    """Request to run full workflow on multiple components."""

    component_paths: List[Dict[str, str]] = Field(
        ...,
        description="List of components to document",
        example=[
            {"component": "flask_app", "path": "flask_app.py"},
            {"component": "UserService", "path": "services/user.py"}
        ]
    )
    max_iterations: int = Field(default=3, ge=1, le=10)

    class Config:
        json_schema_extra = {
            "example": {
                "component_paths": [
                    {"component": "flask_app", "path": "flask_app.py"}
                ],
                "max_iterations": 3
            }
        }


# ============================================================================
# Response Models
# ============================================================================

class ComponentInfo(BaseModel):
    """Information about a detected code component."""

    name: str = Field(..., description="Component name")
    type: str = Field(..., description="Component type (function, class, route)")
    file_path: str = Field(..., description="File containing the component")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional component details"
    )


class AnalyzeResponse(BaseModel):
    """Response from codebase analysis."""

    files_found: int = Field(..., description="Number of Python files found")
    components: List[str] = Field(..., description="List of component names")
    components_by_type: Dict[str, int] = Field(
        ...,
        description="Count of components by type"
    )
    total_components: int = Field(..., description="Total components stored")
    detailed_components: Optional[List[ComponentInfo]] = Field(
        default=None,
        description="Detailed component information"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "files_found": 5,
                "components": ["UserService", "login", "get_user"],
                "components_by_type": {
                    "class": 2,
                    "function": 8,
                    "route": 5
                },
                "total_components": 15
            }
        }


class DocumentResponse(BaseModel):
    """Response from documentation generation."""

    success: bool = Field(..., description="Whether generation succeeded")
    component: str = Field(..., description="Component name")
    markdown: str = Field(..., description="Generated markdown documentation")
    path: str = Field(..., description="Path to saved documentation file")
    iterations: int = Field(..., description="Number of iterations performed")
    validation_score: float = Field(
        ...,
        description="Validation score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    validation_passed: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default=[], description="Any errors encountered")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "component": "UserService",
                "markdown": "# UserService\\n\\n## Overview\\n...",
                "path": "docs_output/UserService.md",
                "iterations": 2,
                "validation_score": 0.85,
                "validation_passed": True,
                "errors": []
            }
        }


class QueryResponse(BaseModel):
    """Response from RAG query."""

    answer: str = Field(..., description="Generated answer to the question")
    sources: List[str] = Field(..., description="Source components used")
    question: str = Field(..., description="Original question")
    retrieved_count: int = Field(
        ...,
        description="Number of components retrieved"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The codebase uses JWT-based authentication...",
                "sources": [
                    "auth.py::login",
                    "middleware.py::verify_token"
                ],
                "question": "What authentication methods are used?",
                "retrieved_count": 2
            }
        }


class PatternInfo(BaseModel):
    """Information about a detected code pattern."""

    pattern_type: str = Field(..., description="Type of pattern")
    count: int = Field(..., description="Number of occurrences")
    examples: List[str] = Field(..., description="Example instances")


class PatternsResponse(BaseModel):
    """Response with detected code patterns."""

    patterns: List[PatternInfo] = Field(..., description="Detected patterns")
    total_patterns: int = Field(..., description="Total number of patterns")

    class Config:
        json_schema_extra = {
            "example": {
                "patterns": [
                    {
                        "pattern_type": "route",
                        "count": 8,
                        "examples": ["/users", "/login", "/api/data"]
                    }
                ],
                "total_patterns": 8
            }
        }


class PatternSuggestionsResponse(BaseModel):
    """Response with pattern deviation suggestions for a component."""

    component: str = Field(..., description="Component name")
    file_path: str = Field(..., description="File path of the component")
    suggestions: List[str] = Field(..., description="List of improvement suggestions")
    consistency_score: float = Field(..., description="Overall codebase consistency score (0.0-1.0)")
    suggestion_count: int = Field(..., description="Number of suggestions")

    class Config:
        json_schema_extra = {
            "example": {
                "component": "UserService",
                "file_path": "services/user.py",
                "suggestions": [
                    "Function name doesn't follow snake_case convention (80% of codebase uses snake_case)",
                    "Missing docstring. 75% of components are documented.",
                    "No error handling detected. 60% of similar components use try-except blocks."
                ],
                "consistency_score": 0.75,
                "suggestion_count": 3
            }
        }


class HealthStatus(BaseModel):
    """Health check status."""

    status: str = Field(..., description="Overall status (healthy/unhealthy)")
    ollama_connected: bool = Field(..., description="Ollama connection status")
    chromadb_connected: bool = Field(..., description="ChromaDB connection status")
    details: Dict[str, Any] = Field(
        default={},
        description="Additional health details"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "ollama_connected": True,
                "chromadb_connected": True,
                "details": {
                    "ollama_model": "llama3.2",
                    "chromadb_collections": 1
                }
            }
        }


class DependencyNode(BaseModel):
    """Node in the dependency graph."""

    id: str = Field(..., description="Node identifier")
    label: str = Field(..., description="Display label")
    type: str = Field(..., description="Node type (class, function, module)")
    file: str = Field(..., description="Source file")


class DependencyEdge(BaseModel):
    """Edge in the dependency graph."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    type: str = Field(..., description="Relationship type (imports, calls, extends)")


class VisualizeResponse(BaseModel):
    """Response with dependency graph data."""

    nodes: List[DependencyNode] = Field(..., description="Graph nodes")
    edges: List[DependencyEdge] = Field(..., description="Graph edges")
    node_count: int = Field(..., description="Total number of nodes")
    edge_count: int = Field(..., description="Total number of edges")

    class Config:
        json_schema_extra = {
            "example": {
                "nodes": [
                    {"id": "user_service", "label": "UserService", "type": "class", "file": "user.py"}
                ],
                "edges": [
                    {"source": "main", "target": "user_service", "type": "imports"}
                ],
                "node_count": 15,
                "edge_count": 23
            }
        }


class WorkflowResult(BaseModel):
    """Result from a single component workflow execution."""

    component: str
    success: bool
    iterations: int
    validation_score: float
    output_file: str
    errors: List[str] = []


class BatchDocumentResponse(BaseModel):
    """Response from batch documentation generation."""

    total_components: int = Field(..., description="Total components processed")
    successful: int = Field(..., description="Successfully documented")
    failed: int = Field(..., description="Failed components")
    results: List[WorkflowResult] = Field(..., description="Individual results")

    class Config:
        json_schema_extra = {
            "example": {
                "total_components": 5,
                "successful": 4,
                "failed": 1,
                "results": [
                    {
                        "component": "UserService",
                        "success": True,
                        "iterations": 2,
                        "validation_score": 0.85,
                        "output_file": "docs_output/UserService.md",
                        "errors": []
                    }
                ]
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Component not found",
                "detail": "No file found at path: services/user.py",
                "code": "NOT_FOUND"
            }
        }
