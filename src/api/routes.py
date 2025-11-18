"""
API route definitions for Flow-Doc REST API.

This module implements all REST endpoints for the documentation assistant,
including codebase analysis, documentation generation, RAG queries, and more.
"""

import logging
import os
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Response
from fastapi.responses import FileResponse, StreamingResponse
import requests

from src.core.analyzer import CodebaseAnalyzer
from src.core.memory import MemoryStore
from src.core.generator import DocumentationGenerator
from src.core.agent import DocumentationAgent
from src.core.pattern_detector import PatternDetector
from src.api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    DocumentRequest,
    DocumentResponse,
    QueryRequest,
    QueryResponse,
    PatternsResponse,
    PatternInfo,
    PatternSuggestionsResponse,
    HealthStatus,
    VisualizeResponse,
    DependencyNode,
    DependencyEdge,
    ComponentInfo,
    WorkflowRequest,
    BatchDocumentResponse,
    WorkflowResult,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# ============================================================================
# Dependency Injection
# ============================================================================

class ServiceContainer:
    """Container for shared service instances."""

    def __init__(self):
        self.analyzer: CodebaseAnalyzer = None
        self.memory: MemoryStore = None
        self.generator: DocumentationGenerator = None
        self.agent: DocumentationAgent = None
        self.pattern_detector: PatternDetector = None
        self.codebase_path: str = None

    def initialize(self, codebase_path: str = "./sample_codebase"):
        """Initialize all services."""
        logger.info(f"Initializing services with codebase: {codebase_path}")

        self.codebase_path = codebase_path
        self.analyzer = CodebaseAnalyzer(codebase_path)
        self.memory = MemoryStore(persist_directory="./data")
        self.generator = DocumentationGenerator(self.memory)
        self.pattern_detector = PatternDetector()
        self.agent = DocumentationAgent(self.analyzer, self.memory, self.generator, self.pattern_detector)

        logger.info("✓ Services initialized successfully")

    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up services...")
        # Cleanup logic if needed


# Global service container
services = ServiceContainer()


def get_services() -> ServiceContainer:
    """Dependency injection for services."""
    if services.analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Services not initialized. Call /health to check status."
        )
    return services


# ============================================================================
# Health Check Endpoint
# ============================================================================

@router.get(
    "/health",
    response_model=HealthStatus,
    summary="Health Check",
    description="Check system health and service connectivity",
    tags=["System"]
)
async def health_check() -> HealthStatus:
    """
    Check the health status of all services.

    Returns:
        HealthStatus with connection status for Ollama and ChromaDB
    """
    ollama_connected = False
    chromadb_connected = False
    details = {}

    # Check Ollama connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            ollama_connected = True
            models = response.json().get("models", [])
            details["ollama_models"] = [m.get("name") for m in models]
            details["ollama_model_count"] = len(models)
    except Exception as e:
        logger.warning(f"Ollama connection check failed: {e}")
        details["ollama_error"] = str(e)

    # Check ChromaDB connection
    try:
        if services.memory:
            stats = services.memory.get_collection_stats()
            chromadb_connected = True
            details["chromadb_collections"] = 1
            details["chromadb_component_count"] = stats.get("count", 0)
    except Exception as e:
        logger.warning(f"ChromaDB connection check failed: {e}")
        details["chromadb_error"] = str(e)

    # Determine overall status
    status = "healthy" if (ollama_connected and chromadb_connected) else "degraded"
    if not ollama_connected and not chromadb_connected:
        status = "unhealthy"

    # Add service initialization status
    details["services_initialized"] = services.analyzer is not None

    return HealthStatus(
        status=status,
        ollama_connected=ollama_connected,
        chromadb_connected=chromadb_connected,
        details=details
    )


# ============================================================================
# Analyze Endpoint
# ============================================================================

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze Codebase",
    description="Analyze a codebase directory and store components in memory",
    tags=["Analysis"]
)
async def analyze_codebase(
    request: AnalyzeRequest,
    container: ServiceContainer = Depends(get_services)
) -> AnalyzeResponse:
    """
    Analyze all Python files in a codebase directory.

    - Scans for all .py files
    - Parses with AST
    - Extracts functions, classes, routes
    - Stores components in ChromaDB with embeddings

    Args:
        request: AnalyzeRequest with codebase path

    Returns:
        AnalyzeResponse with analysis summary
    """
    logger.info(f"Analyzing codebase at: {request.path}")

    # Validate path
    codebase_path = Path(request.path)
    if not codebase_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Codebase path not found: {request.path}"
        )

    if not codebase_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Path is not a directory: {request.path}"
        )

    try:
        # Reinitialize analyzer with new path
        container.analyzer = CodebaseAnalyzer(str(codebase_path))
        container.codebase_path = str(codebase_path)

        # Scan directory
        files = container.analyzer.scan_directory()
        logger.info(f"Found {len(files)} Python files")

        # Parse each file and store components
        components = []
        components_by_type = defaultdict(int)
        detailed_components = []

        for file_info in files:
            try:
                parsed = container.analyzer.parse_file(file_info['path'])
                if not parsed:
                    continue

                # Extract classes
                for cls in parsed.get('classes', []):
                    component = {
                        "type": "class",
                        "name": cls['name'],
                        "file_path": file_info['path'],
                        "methods": cls.get('methods', []),
                        "docstring": cls.get('docstring', '')
                    }
                    container.memory.store_component(component)
                    components.append(cls['name'])
                    components_by_type['class'] += 1

                    detailed_components.append(ComponentInfo(
                        name=cls['name'],
                        type="class",
                        file_path=file_info['path'],
                        details={"methods": cls.get('methods', [])}
                    ))

                # Extract functions
                for func in parsed.get('functions', []):
                    component = {
                        "type": "function",
                        "name": func['name'],
                        "file_path": file_info['path'],
                        "params": func.get('params', []),
                        "docstring": func.get('docstring', ''),
                        "decorators": func.get('decorators', [])
                    }
                    container.memory.store_component(component)
                    components.append(func['name'])
                    components_by_type['function'] += 1

                    detailed_components.append(ComponentInfo(
                        name=func['name'],
                        type="function",
                        file_path=file_info['path'],
                        details={"params": func.get('params', [])}
                    ))

                # Extract routes
                for route in parsed.get('routes', []):
                    component = {
                        "type": "route",
                        "name": route['handler'],
                        "file_path": file_info['path'],
                        "path": route['path'],
                        "methods": route.get('methods', [])
                    }
                    container.memory.store_component(component)
                    components.append(f"{route['path']} ({route['handler']})")
                    components_by_type['route'] += 1

                    detailed_components.append(ComponentInfo(
                        name=route['handler'],
                        type="route",
                        file_path=file_info['path'],
                        details={"path": route['path'], "methods": route.get('methods', [])}
                    ))

            except Exception as e:
                logger.error(f"Error parsing {file_info['path']}: {e}")
                continue

        logger.info(f"Stored {len(components)} components in memory")

        return AnalyzeResponse(
            files_found=len(files),
            components=components,
            components_by_type=dict(components_by_type),
            total_components=len(components),
            detailed_components=detailed_components
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


# ============================================================================
# Batch Documentation Endpoint
# ============================================================================

@router.post(
    "/document/batch",
    response_model=BatchDocumentResponse,
    summary="Batch Generate Documentation",
    description="Generate documentation for multiple components",
    tags=["Documentation"]
)
async def document_batch(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    container: ServiceContainer = Depends(get_services)
) -> BatchDocumentResponse:
    """
    Generate documentation for multiple components in batch.

    Args:
        request: WorkflowRequest with list of components

    Returns:
        BatchDocumentResponse with all results
    """
    logger.info(f"Batch documentation for {len(request.component_paths)} components")

    results = []
    successful = 0
    failed = 0

    for item in request.component_paths:
        component = item.get('component')
        path = item.get('path')

        try:
            result = container.agent.run(
                component_name=component,
                component_path=path,
                max_iterations=request.max_iterations
            )

            if result['success']:
                successful += 1
            else:
                failed += 1

            results.append(WorkflowResult(
                component=component,
                success=result['success'],
                iterations=result['iterations'],
                validation_score=result['validation_score'],
                output_file=result['output_file'],
                errors=result['errors']
            ))

        except Exception as e:
            logger.error(f"Failed to document {component}: {e}")
            failed += 1
            results.append(WorkflowResult(
                component=component,
                success=False,
                iterations=0,
                validation_score=0.0,
                output_file="",
                errors=[str(e)]
            ))

    return BatchDocumentResponse(
        total_components=len(request.component_paths),
        successful=successful,
        failed=failed,
        results=results
    )


# ============================================================================
# Document Generation Endpoint
# ============================================================================

@router.post(
    "/document/{component}",
    response_model=DocumentResponse,
    summary="Generate Documentation",
    description="Run agentic workflow to generate documentation for a component",
    tags=["Documentation"]
)
async def document_component(
    component: str,
    request: DocumentRequest,
    container: ServiceContainer = Depends(get_services)
) -> DocumentResponse:
    """
    Generate documentation for a specific component using the agentic workflow.

    Executes the complete LangGraph workflow:
    1. Analyze code with AST
    2. Retrieve similar components from memory
    3. Generate documentation with LLM
    4. Validate quality
    5. Iteratively refine if needed
    6. Store results

    Args:
        component: Component name (path parameter)
        request: DocumentRequest with path and settings

    Returns:
        DocumentResponse with generated documentation
    """
    logger.info(f"Generating documentation for component: {component}")

    # Verify component path exists
    full_path = Path(container.codebase_path) / request.path
    if not full_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Component file not found: {request.path}"
        )

    try:
        # Run workflow
        result = container.agent.run(
            component_name=request.component,
            component_path=request.path,
            max_iterations=request.max_iterations
        )

        # Read generated markdown
        markdown_content = ""
        if result.get('output_file'):
            output_path = Path(result['output_file'])
            if output_path.exists():
                with open(output_path, 'r') as f:
                    markdown_content = f.read()

        logger.info(
            f"Documentation generated: {result['success']}, "
            f"iterations={result['iterations']}, score={result['validation_score']:.2f}"
        )

        return DocumentResponse(
            success=result['success'],
            component=result['component'],
            markdown=markdown_content,
            path=result['output_file'],
            iterations=result['iterations'],
            validation_score=result['validation_score'],
            validation_passed=result['validation_passed'],
            errors=result['errors']
        )

    except Exception as e:
        logger.error(f"Documentation generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Documentation generation failed: {str(e)}"
        )


# ============================================================================
# Query Endpoint (RAG)
# ============================================================================

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query Codebase",
    description="Ask natural language questions about the codebase using RAG",
    tags=["Query"]
)
async def query_codebase(
    request: QueryRequest,
    container: ServiceContainer = Depends(get_services)
) -> QueryResponse:
    """
    Query the codebase using RAG (Retrieval Augmented Generation).

    - Retrieves similar components from ChromaDB
    - Generates answer using Ollama LLM
    - Returns answer with source citations

    Args:
        request: QueryRequest with question

    Returns:
        QueryResponse with answer and sources
    """
    logger.info(f"Query: {request.question}")

    try:
        # Use generator's simple_query method
        answer = container.generator.simple_query(
            question=request.question,
            k=request.k
        )

        # Retrieve similar components for sources
        similar = container.memory.retrieve_similar(request.question, k=request.k)
        sources = [
            f"{comp['metadata'].get('file_path', 'unknown')}::{comp['metadata'].get('name', 'unknown')}"
            for comp in similar
        ]

        logger.info(f"Answer generated with {len(sources)} sources")

        return QueryResponse(
            answer=answer,
            sources=sources,
            question=request.question,
            retrieved_count=len(similar)
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


# ============================================================================
# Patterns Endpoint
# ============================================================================

@router.get(
    "/patterns",
    response_model=PatternsResponse,
    summary="Get Code Patterns",
    description="Retrieve detected code patterns grouped by type",
    tags=["Analysis"]
)
async def get_patterns(
    container: ServiceContainer = Depends(get_services)
) -> PatternsResponse:
    """
    Get detected code patterns from memory.

    Returns patterns grouped by type:
    - Routes (API endpoints)
    - Error handling patterns
    - Authentication patterns
    - Data validation patterns
    - etc.

    Returns:
        PatternsResponse with detected patterns
    """
    logger.info("Retrieving code patterns")

    try:
        # Get all patterns from memory
        all_patterns = container.memory.get_all_patterns()

        # Group by type
        pattern_groups = defaultdict(lambda: {"count": 0, "examples": []})

        for pattern in all_patterns:
            pattern_type = pattern.get('type', 'unknown')
            pattern_groups[pattern_type]['count'] += 1

            # Add example (limit to 5 per type)
            if len(pattern_groups[pattern_type]['examples']) < 5:
                example = pattern.get('name', '')
                if pattern_type == 'route' and 'path' in pattern:
                    example = pattern['path']
                pattern_groups[pattern_type]['examples'].append(example)

        # Build response
        patterns = [
            PatternInfo(
                pattern_type=ptype,
                count=data['count'],
                examples=data['examples']
            )
            for ptype, data in pattern_groups.items()
        ]

        total = sum(p.count for p in patterns)

        logger.info(f"Found {len(patterns)} pattern types, {total} total patterns")

        return PatternsResponse(
            patterns=patterns,
            total_patterns=total
        )

    except Exception as e:
        logger.error(f"Pattern retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pattern retrieval failed: {str(e)}"
        )


# ============================================================================
# Visualize Endpoint
# ============================================================================

@router.get(
    "/visualize",
    response_model=VisualizeResponse,
    summary="Get Dependency Graph",
    description="Retrieve component dependency graph data",
    tags=["Visualization"]
)
async def visualize_dependencies(
    container: ServiceContainer = Depends(get_services)
) -> VisualizeResponse:
    """
    Get component dependency graph data.

    Returns nodes and edges for visualization:
    - Nodes: Components (classes, functions, modules)
    - Edges: Relationships (imports, calls, extends)

    Returns:
        VisualizeResponse with graph data
    """
    logger.info("Building dependency graph")

    try:
        nodes = []
        edges = []
        node_ids = set()

        # Get all components from memory
        all_patterns = container.memory.get_all_patterns()

        # Create nodes for each component
        for pattern in all_patterns:
            node_id = f"{pattern.get('file_path', 'unknown')}::{pattern.get('name', 'unknown')}"

            if node_id not in node_ids:
                node_ids.add(node_id)
                nodes.append(DependencyNode(
                    id=node_id,
                    label=pattern.get('name', 'Unknown'),
                    type=pattern.get('type', 'unknown'),
                    file=pattern.get('file_path', 'unknown')
                ))

        # Parse import relationships from stored components
        # This is a simplified version - more sophisticated analysis would be needed
        # for complete dependency tracking
        for pattern in all_patterns:
            if pattern.get('type') == 'function' and 'imports' in pattern:
                source_id = f"{pattern.get('file_path')}::{pattern.get('name')}"
                for imp in pattern.get('imports', []):
                    target_id = f"unknown::{imp}"
                    if target_id in node_ids or source_id in node_ids:
                        edges.append(DependencyEdge(
                            source=source_id,
                            target=target_id,
                            type="imports"
                        ))

        logger.info(f"Graph built: {len(nodes)} nodes, {len(edges)} edges")

        return VisualizeResponse(
            nodes=nodes,
            edges=edges,
            node_count=len(nodes),
            edge_count=len(edges)
        )

    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Visualization failed: {str(e)}"
        )


# ============================================================================
# Export Endpoint
# ============================================================================

@router.get(
    "/export",
    summary="Export Documentation",
    description="Export all generated documentation as a ZIP file",
    tags=["Export"],
    response_class=FileResponse
)
async def export_documentation() -> FileResponse:
    """
    Export all generated documentation files as a ZIP archive.

    Returns:
        ZIP file containing all docs_output/*.md files
    """
    logger.info("Exporting documentation")

    docs_dir = Path("docs_output")
    if not docs_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="No documentation found. Generate some first!"
        )

    try:
        # Create temporary ZIP file
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.zip', delete=False) as tmp_file:
            zip_path = tmp_file.name

        # Create ZIP archive
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            md_files = list(docs_dir.glob("*.md"))

            if not md_files:
                raise HTTPException(
                    status_code=404,
                    detail="No documentation files found"
                )

            for md_file in md_files:
                zipf.write(md_file, arcname=md_file.name)

        logger.info(f"Exported {len(md_files)} documentation files")

        # Return ZIP file
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename="flow-doc-documentation.zip",
            background=BackgroundTasks()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {str(e)}"
        )


# ============================================================================
# Pattern Detection Endpoints
# ============================================================================

@router.get(
    "/patterns/suggestions/{component}",
    response_model=PatternSuggestionsResponse,
    summary="Get Pattern Suggestions",
    description="Analyze a component and provide pattern deviation suggestions",
    tags=["Patterns"]
)
async def get_pattern_suggestions(
    component: str,
    container: ServiceContainer = Depends(get_services)
) -> PatternSuggestionsResponse:
    """
    Get pattern deviation suggestions for a specific component.

    This endpoint:
    1. Retrieves all components from memory
    2. Analyzes patterns across the codebase
    3. Compares the specified component against those patterns
    4. Returns suggestions for deviations

    Args:
        component: Component identifier in format "file_path::component_name" or just "component_name"

    Returns:
        PatternSuggestionsResponse with suggestions and consistency score

    Raises:
        HTTPException 404: If component not found
        HTTPException 500: If pattern analysis fails
    """
    logger.info(f"Getting pattern suggestions for: {component}")

    try:
        # Parse component identifier
        # Support both "file_path::component_name" and just "component_name"
        if "::" in component:
            file_path, component_name = component.split("::", 1)
        else:
            component_name = component
            file_path = None

        # Get all components from memory for pattern analysis
        all_results = container.memory.retrieve_similar("", k=100)

        # Convert memory results to component format
        all_components = []
        target_component = None

        for result in all_results:
            doc = result.get('document', '')
            metadata = result.get('metadata', {})
            comp = {
                'name': metadata.get('name', ''),
                'type': metadata.get('type', ''),
                'file_path': metadata.get('file_path', ''),
                'docstring': doc[:200] if doc else '',
                'methods': metadata.get('methods', []),
                'imports': metadata.get('imports', []),
                'decorators': metadata.get('decorators', []),
                'params': metadata.get('params', []),
                'returns': metadata.get('returns', ''),
                'raises': metadata.get('raises', []),
            }
            all_components.append(comp)

            # Find the target component
            if comp['name'] == component_name:
                if file_path is None or comp['file_path'] == file_path:
                    target_component = comp

        if target_component is None:
            raise HTTPException(
                status_code=404,
                detail=f"Component '{component}' not found in memory. "
                       "Please analyze the codebase first."
            )

        # Analyze patterns across all components
        logger.debug(f"Analyzing patterns from {len(all_components)} components")
        patterns = container.pattern_detector.analyze_patterns(all_components)

        consistency_score = patterns.get('consistency_score', 0.0)
        logger.debug(f"Codebase consistency score: {consistency_score:.1%}")

        # Detect deviations for the target component
        suggestions = container.pattern_detector.detect_deviations(
            target_component,
            patterns
        )

        logger.info(f"Found {len(suggestions)} pattern deviations for {component_name}")

        return PatternSuggestionsResponse(
            component=component_name,
            file_path=target_component['file_path'],
            suggestions=suggestions,
            consistency_score=consistency_score,
            suggestion_count=len(suggestions)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pattern analysis failed: {str(e)}"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def get_service_container() -> ServiceContainer:
    """Get the global service container."""
    return services
