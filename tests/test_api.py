"""Tests for FastAPI endpoints."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routes import services


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_services(temp_data_dir):
    """Mock all services for testing."""
    with patch('src.api.routes.services') as mock_svc:
        # Mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.scan_directory.return_value = [
            {"path": "test.py", "name": "test.py", "size": 100, "modified": "2024-01-01"}
        ]
        mock_analyzer.parse_file.return_value = {
            "classes": [{"name": "TestClass", "methods": ["test_method"], "docstring": "Test"}],
            "functions": [{"name": "test_func", "params": ["arg1"], "docstring": "Test func"}],
            "routes": [{"handler": "test_route", "path": "/test", "methods": ["GET"]}]
        }

        # Mock memory
        mock_memory = MagicMock()
        mock_memory.store_component.return_value = "test.py::TestClass"
        mock_memory.get_collection_stats.return_value = {"count": 10}
        mock_memory.get_all_patterns.return_value = [
            {"type": "function", "name": "test_func", "file_path": "test.py"},
            {"type": "route", "name": "test_route", "path": "/test", "file_path": "test.py"}
        ]
        mock_memory.retrieve_similar.return_value = [
            {"metadata": {"file_path": "test.py", "name": "test_func"}, "score": 0.95}
        ]

        # Mock generator
        mock_generator = MagicMock()
        mock_generator.simple_query.return_value = "This is a test answer"

        # Mock agent
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "success": True,
            "component": "TestComponent",
            "iterations": 2,
            "validation_score": 0.85,
            "validation_passed": True,
            "errors": [],
            "output_file": "docs_output/TestComponent.md"
        }

        # Assign mocks to services
        mock_svc.analyzer = mock_analyzer
        mock_svc.memory = mock_memory
        mock_svc.generator = mock_generator
        mock_svc.agent = mock_agent
        mock_svc.codebase_path = "./sample_codebase"

        yield mock_svc


@pytest.fixture
def client(mock_services):
    """Create test client with mocked services."""
    # Initialize services before creating client to avoid lifespan issues
    from src.api.routes import services as route_services
    route_services.analyzer = mock_services.analyzer
    route_services.memory = mock_services.memory
    route_services.generator = mock_services.generator
    route_services.agent = mock_services.agent
    route_services.codebase_path = mock_services.codebase_path

    # Create client
    return TestClient(app)


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_check_healthy(self, client, mock_services):
        """Test health check when all services are healthy."""
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": [{"name": "llama3.2"}]}
            mock_get.return_value = mock_response

            response = client.get("/api/v1/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["healthy", "degraded"]
            assert "ollama_connected" in data
            assert "chromadb_connected" in data
            assert "details" in data

    def test_health_check_degraded(self, client, mock_services):
        """Test health check when Ollama is down."""
        with patch('requests.get', side_effect=Exception("Connection refused")):
            response = client.get("/api/v1/health")

            assert response.status_code == 200
            data = response.json()
            assert data["ollama_connected"] == False


class TestAnalyzeEndpoint:
    """Test /analyze endpoint."""

    def test_analyze_codebase_success(self, client, mock_services):
        """Test successful codebase analysis."""
        request_data = {"path": "./sample_codebase"}

        response = client.post("/api/v1/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "files_found" in data
        assert "components" in data
        assert "components_by_type" in data
        assert "total_components" in data
        assert data["total_components"] > 0

    def test_analyze_invalid_path(self, client, mock_services):
        """Test analysis with invalid path."""
        request_data = {"path": "/nonexistent/path"}

        response = client.post("/api/v1/analyze", json=request_data)

        assert response.status_code == 404
        data = response.json()
        assert "error" in data or "detail" in data


class TestDocumentEndpoint:
    """Test /document/{component} endpoint."""

    def test_document_component_success(self, client, mock_services):
        """Test successful documentation generation."""
        # Create test Python file in the codebase
        codebase_dir = Path(mock_services.codebase_path)
        codebase_dir.mkdir(exist_ok=True)
        test_file = codebase_dir / "test.py"
        test_file.write_text("# Test Python file\ndef test_func():\n    pass\n")

        # Create test markdown file
        docs_dir = Path("docs_output")
        docs_dir.mkdir(exist_ok=True)
        test_doc = docs_dir / "TestComponent.md"
        test_doc.write_text("# TestComponent\n\nTest documentation")

        request_data = {
            "component": "TestComponent",
            "path": "test.py",
            "max_iterations": 3
        }

        response = client.post("/api/v1/document/TestComponent", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["component"] == "TestComponent"
        assert "markdown" in data
        assert data["iterations"] == 2
        assert data["validation_score"] == 0.85
        assert data["validation_passed"] == True

        # Cleanup
        if test_file.exists():
            test_file.unlink()
        if test_doc.exists():
            test_doc.unlink()

    def test_document_component_not_found(self, client, mock_services):
        """Test documentation generation for non-existent component."""
        request_data = {
            "component": "NonExistent",
            "path": "nonexistent.py",
            "max_iterations": 3
        }

        response = client.post("/api/v1/document/NonExistent", json=request_data)

        assert response.status_code == 404


class TestQueryEndpoint:
    """Test /query endpoint."""

    def test_query_codebase_success(self, client, mock_services):
        """Test successful RAG query."""
        request_data = {
            "question": "What authentication methods are used?",
            "k": 5
        }

        response = client.post("/api/v1/query", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["question"] == request_data["question"]
        assert "retrieved_count" in data
        assert len(data["answer"]) > 0

    def test_query_validation_error(self, client, mock_services):
        """Test query with invalid input."""
        request_data = {
            "question": "   ",  # Empty question (whitespace only)
            "k": 5
        }

        response = client.post("/api/v1/query", json=request_data)

        assert response.status_code == 422  # Validation error


class TestPatternsEndpoint:
    """Test /patterns endpoint."""

    def test_get_patterns_success(self, client, mock_services):
        """Test retrieving code patterns."""
        response = client.get("/api/v1/patterns")

        assert response.status_code == 200
        data = response.json()
        assert "patterns" in data
        assert "total_patterns" in data
        assert isinstance(data["patterns"], list)

        if len(data["patterns"]) > 0:
            pattern = data["patterns"][0]
            assert "pattern_type" in pattern
            assert "count" in pattern
            assert "examples" in pattern


class TestVisualizeEndpoint:
    """Test /visualize endpoint."""

    def test_visualize_dependencies_success(self, client, mock_services):
        """Test dependency graph generation."""
        response = client.get("/api/v1/visualize")

        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert "node_count" in data
        assert "edge_count" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)


class TestExportEndpoint:
    """Test /export endpoint."""

    def test_export_documentation_success(self, client, mock_services):
        """Test exporting documentation as ZIP."""
        # Create test documentation files
        docs_dir = Path("docs_output")
        docs_dir.mkdir(exist_ok=True)
        test_doc1 = docs_dir / "Component1.md"
        test_doc2 = docs_dir / "Component2.md"
        test_doc1.write_text("# Component1")
        test_doc2.write_text("# Component2")

        response = client.get("/api/v1/export")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

        # Cleanup
        if test_doc1.exists():
            test_doc1.unlink()
        if test_doc2.exists():
            test_doc2.unlink()

    def test_export_no_documentation(self, client, mock_services):
        """Test export when no documentation exists."""
        # Ensure docs_output is empty
        docs_dir = Path("docs_output")
        if docs_dir.exists():
            for f in docs_dir.glob("*.md"):
                f.unlink()

        response = client.get("/api/v1/export")

        # Should return 404 when no docs exist
        assert response.status_code == 404


class TestBatchDocumentEndpoint:
    """Test /document/batch endpoint."""

    def test_batch_document_success(self, client, mock_services):
        """Test batch documentation generation."""
        request_data = {
            "component_paths": [
                {"component": "Component1", "path": "comp1.py"},
                {"component": "Component2", "path": "comp2.py"}
            ],
            "max_iterations": 3
        }

        response = client.post("/api/v1/document/batch", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "total_components" in data
        assert "successful" in data
        assert "failed" in data
        assert "results" in data
        assert data["total_components"] == 2


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["message"] == "Flow-Doc API Server"


class TestErrorHandling:
    """Test error handling."""

    def test_validation_error_handling(self, client, mock_services):
        """Test that validation errors are handled properly."""
        # Missing required field
        request_data = {}

        response = client.post("/api/v1/analyze", json=request_data)

        assert response.status_code == 422
        data = response.json()
        assert "error" in data


def cleanup_test_files():
    """Clean up test output files."""
    docs_dir = Path("docs_output")
    if docs_dir.exists():
        for f in docs_dir.glob("*.md"):
            if f.name.startswith("Test") or f.name.startswith("Component"):
                f.unlink()


if __name__ == "__main__":
    try:
        print("Running API tests...")
        pytest.main([__file__, "-v", "-s"])
    finally:
        cleanup_test_files()
