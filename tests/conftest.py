"""
Shared pytest fixtures for flow-doc tests.

This module provides common fixtures used across all test files, including:
- Sample codebase paths and files
- Mock ChromaDB instances
- Mock Ollama responses
- Temporary directories for output
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Generator
from unittest.mock import Mock, MagicMock, patch

import pytest
from chromadb import Client
from chromadb.config import Settings


# ============================================================================
# Sample Code Fixtures
# ============================================================================

@pytest.fixture
def sample_python_file() -> str:
    """Sample Python file content for testing."""
    return '''"""Sample module for testing."""

import os
from typing import List, Dict

class UserService:
    """Service for managing users."""

    def __init__(self, db_path: str):
        """Initialize the service."""
        self.db_path = db_path

    def get_user(self, user_id: int) -> Dict:
        """
        Retrieve a user by ID.

        Args:
            user_id: The user's unique identifier

        Returns:
            User data dictionary

        Raises:
            ValueError: If user_id is invalid
        """
        if user_id < 0:
            raise ValueError("Invalid user_id")
        return {"id": user_id, "name": "Test User"}

    def list_users(self) -> List[Dict]:
        """List all users."""
        return [{"id": 1, "name": "User 1"}]

def standalone_function(param: str) -> str:
    """A standalone utility function."""
    return param.upper()
'''


@pytest.fixture
def sample_fastapi_file() -> str:
    """Sample FastAPI file for testing route parsing."""
    return '''"""FastAPI routes."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    """User model."""
    id: int
    name: str

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello World"}

@app.get("/api/users")
async def get_users():
    """Get all users."""
    return [{"id": 1, "name": "User 1"}]

@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    """Get user by ID."""
    if user_id < 0:
        raise HTTPException(status_code=400, detail="Invalid ID")
    return {"id": user_id, "name": f"User {user_id}"}

@app.post("/api/users")
async def create_user(user: User):
    """Create a new user."""
    return user
'''


@pytest.fixture
def sample_flask_file() -> str:
    """Sample Flask file for testing route parsing."""
    return '''"""Flask routes."""

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def index():
    """Index page."""
    return jsonify({"message": "Hello Flask"})

@app.route("/api/users", methods=["GET"])
def get_users():
    """Get all users."""
    return jsonify([{"id": 1, "name": "User 1"}])

@app.route("/api/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    """Get user by ID."""
    return jsonify({"id": user_id, "name": f"User {user_id}"})
'''


@pytest.fixture
def malformed_python_file() -> str:
    """Malformed Python file for error handling tests."""
    return '''"""This file has syntax errors."""

class BrokenClass:
    def method_without_body(self):
        # Missing implementation

    def method_with_syntax_error(
        # Missing closing parenthesis and colon
'''


@pytest.fixture
def empty_python_file() -> str:
    """Empty Python file for edge case testing."""
    return ""


@pytest.fixture
def sample_codebase_path(tmp_path: Path) -> Path:
    """Create a temporary sample codebase."""
    # Create directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "models").mkdir()
    (tmp_path / "src" / "api").mkdir()
    (tmp_path / "tests").mkdir()

    # Create sample files
    (tmp_path / "src" / "__init__.py").write_text("")

    (tmp_path / "src" / "models" / "user.py").write_text('''"""User model."""

class User:
    """User data model."""

    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"id": self.id, "name": self.name}
''')

    (tmp_path / "src" / "api" / "routes.py").write_text('''"""API routes."""

from fastapi import FastAPI

app = FastAPI()

@app.get("/users")
async def get_users():
    """Get all users."""
    return []
''')

    return tmp_path


# ============================================================================
# ChromaDB Fixtures
# ============================================================================

@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB client for in-memory testing."""
    client = Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=None,  # In-memory
        anonymized_telemetry=False
    ))
    yield client


@pytest.fixture
def mock_chromadb_collection(mock_chromadb):
    """Mock ChromaDB collection with sample data."""
    collection = mock_chromadb.get_or_create_collection(
        name="test_collection"
    )

    # Add sample documents
    collection.add(
        ids=["test_1", "test_2"],
        documents=[
            "Sample document about user authentication",
            "Sample document about data validation"
        ],
        metadatas=[
            {"type": "function", "name": "authenticate", "file_path": "auth.py"},
            {"type": "function", "name": "validate", "file_path": "utils.py"}
        ]
    )

    yield collection


# ============================================================================
# Mock Ollama Fixtures
# ============================================================================

@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {
        "model": "llama3.2",
        "response": "# Component Documentation\n\nThis is a test documentation response.",
        "done": True
    }


@pytest.fixture
def mock_ollama_embeddings():
    """Mock Ollama embeddings."""
    return Mock(
        embed_query=Mock(return_value=[0.1] * 384),
        embed_documents=Mock(return_value=[[0.1] * 384, [0.2] * 384])
    )


@pytest.fixture
def mock_ollama_llm():
    """Mock Ollama LLM."""
    mock_llm = Mock()
    mock_llm.invoke = Mock(return_value="# Test Documentation\n\nThis is generated documentation.")
    return mock_llm


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    yield output_dir


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Temporary directory for data storage."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    yield data_dir


# ============================================================================
# Component Fixtures
# ============================================================================

@pytest.fixture
def sample_component_data() -> Dict[str, Any]:
    """Sample component data for testing."""
    return {
        "type": "function",
        "name": "get_user",
        "file_path": "api/users.py",
        "docstring": "Retrieve a user by ID",
        "params": ["user_id: int"],
        "returns": "Dict",
        "decorators": ["@app.get"],
        "imports": ["from fastapi import FastAPI"],
        "raises": ["HTTPException"],
        "lineno": 10
    }


@pytest.fixture
def sample_class_component() -> Dict[str, Any]:
    """Sample class component data."""
    return {
        "type": "class",
        "name": "UserService",
        "file_path": "services/user.py",
        "docstring": "Service for managing users",
        "methods": ["get_user", "create_user", "update_user", "delete_user"],
        "bases": [],
        "decorators": [],
        "imports": ["from typing import Dict", "from models import User"],
        "lineno": 5
    }


@pytest.fixture
def sample_route_component() -> Dict[str, Any]:
    """Sample route component data."""
    return {
        "type": "route",
        "name": "get_users",
        "file_path": "api/routes.py",
        "path": "/api/v1/users",
        "methods": ["GET"],
        "handler": "get_users",
        "docstring": "Retrieve all users from the database",
        "decorators": ["@app.get", "@require_auth"],
        "params": [],
        "returns": "List[User]",
        "lineno": 15
    }


@pytest.fixture
def sample_components_list() -> List[Dict[str, Any]]:
    """List of sample components for pattern detection."""
    return [
        {
            "type": "route",
            "name": "get_users",
            "file_path": "api/users.py",
            "path": "/api/v1/users",
            "methods": ["GET"],
            "docstring": "Get all users",
            "decorators": ["@require_auth"],
            "imports": ["from flask import jsonify"]
        },
        {
            "type": "route",
            "name": "create_user",
            "file_path": "api/users.py",
            "path": "/api/v1/users",
            "methods": ["POST"],
            "docstring": "Create a new user",
            "decorators": ["@require_auth", "@validate_input"],
            "imports": ["from flask import request"]
        },
        {
            "type": "function",
            "name": "validate_email",
            "file_path": "utils/validation.py",
            "docstring": "Validate email format",
            "params": ["email: str"],
            "returns": "bool",
            "raises": ["ValueError"],
            "imports": ["import re"]
        },
        {
            "type": "class",
            "name": "UserService",
            "file_path": "services/user.py",
            "docstring": "User service class",
            "methods": ["get_user", "create_user"],
            "imports": ["from typing import Dict"]
        }
    ]


# ============================================================================
# Mock Service Fixtures
# ============================================================================

@pytest.fixture
def mock_analyzer(sample_codebase_path):
    """Mock CodebaseAnalyzer."""
    from src.core.analyzer import CodebaseAnalyzer
    analyzer = CodebaseAnalyzer(str(sample_codebase_path))
    return analyzer


@pytest.fixture
def mock_memory_store(tmp_path, mock_ollama_embeddings):
    """Mock MemoryStore with in-memory ChromaDB."""
    with patch('src.core.memory.OllamaEmbeddings', return_value=mock_ollama_embeddings):
        from src.core.memory import MemoryStore
        memory = MemoryStore(persist_directory=str(tmp_path / "chroma_test"))
        yield memory


@pytest.fixture
def mock_generator(mock_memory_store, mock_ollama_llm):
    """Mock DocumentationGenerator."""
    with patch('src.core.generator.ChatOllama', return_value=mock_ollama_llm):
        from src.core.generator import DocumentationGenerator
        generator = DocumentationGenerator(mock_memory_store)
        yield generator


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test."""
    yield
    # Cleanup any temporary files created during testing
    test_files = [
        "test_output.md",
        "test_graph.html",
        "test_graph.png",
        "pattern_report.md"
    ]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    yield


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require Ollama"
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require Ollama)"
    )
