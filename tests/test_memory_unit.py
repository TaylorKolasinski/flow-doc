"""Unit tests for MemoryStore that don't require Ollama."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.core.memory import MemoryStore


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for ChromaDB data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_embeddings():
    """Mock Ollama embeddings."""
    with patch('src.core.memory.OllamaEmbeddings') as mock:
        mock_instance = MagicMock()
        # Return a fixed embedding vector
        mock_instance.embed_query.return_value = [0.1] * 384
        mock.return_value = mock_instance
        yield mock


class TestMemoryStoreUnit:
    """Unit tests for MemoryStore without Ollama dependency."""

    def test_text_representation_function(self, temp_data_dir, mock_embeddings):
        """Test text representation generation for function."""
        store = MemoryStore(persist_directory=temp_data_dir)

        component = {
            "type": "function",
            "name": "get_user",
            "docstring": "Get user by ID",
            "params": ["user_id", "db"],
            "decorators": ["app.get"],
            "file_path": "api/users.py"
        }

        text = store._generate_text_representation(component)

        assert "Type: function" in text
        assert "Name: get_user" in text
        assert "Description: Get user by ID" in text
        assert "Parameters: user_id, db" in text
        assert "Decorators: app.get" in text
        assert "File: api/users.py" in text

    def test_text_representation_class(self, temp_data_dir, mock_embeddings):
        """Test text representation generation for class."""
        store = MemoryStore(persist_directory=temp_data_dir)

        component = {
            "type": "class",
            "name": "UserService",
            "docstring": "User management service",
            "methods": ["get", "create", "update"],
            "file_path": "services/user.py"
        }

        text = store._generate_text_representation(component)

        assert "Type: class" in text
        assert "Name: UserService" in text
        assert "Description: User management service" in text
        assert "Methods: get, create, update" in text

    def test_text_representation_route(self, temp_data_dir, mock_embeddings):
        """Test text representation generation for route."""
        store = MemoryStore(persist_directory=temp_data_dir)

        component = {
            "type": "route",
            "name": "get_users",
            "path": "/api/users",
            "methods": ["GET"],
            "file_path": "api/users.py"
        }

        text = store._generate_text_representation(component)

        assert "Type: route" in text
        assert "Name: get_users" in text
        assert "Route: /api/users" in text
        assert "HTTP Methods: GET" in text

    def test_store_component_creates_correct_id(self, temp_data_dir, mock_embeddings):
        """Test that store_component creates correct document ID."""
        store = MemoryStore(persist_directory=temp_data_dir)

        component = {
            "type": "function",
            "name": "test_func",
            "file_path": "test.py"
        }

        doc_id = store.store_component(component)

        assert doc_id == "test.py::test_func"

    def test_persistence_directory_created(self, mock_embeddings):
        """Test that persistence directory is created if it doesn't exist."""
        temp_dir = tempfile.mkdtemp()
        try:
            custom_path = Path(temp_dir) / "nested" / "path" / "data"
            store = MemoryStore(persist_directory=str(custom_path))

            assert custom_path.exists()
            assert custom_path.is_dir()
        finally:
            shutil.rmtree(temp_dir)

    def test_collection_initialization(self, temp_data_dir, mock_embeddings):
        """Test that collection is properly initialized."""
        store = MemoryStore(
            persist_directory=temp_data_dir,
            collection_name="test_collection"
        )

        assert store.collection is not None
        assert store.collection.name == "test_collection"

    def test_clear_collection(self, temp_data_dir, mock_embeddings):
        """Test clearing collection."""
        store = MemoryStore(persist_directory=temp_data_dir)

        # Add a component
        component = {
            "type": "function",
            "name": "test",
            "file_path": "test.py"
        }
        store.store_component(component)

        initial_count = store.collection.count()
        assert initial_count > 0

        # Clear
        store.clear_collection()

        assert store.collection.count() == 0

    def test_collection_stats_empty(self, temp_data_dir, mock_embeddings):
        """Test stats for empty collection."""
        store = MemoryStore(persist_directory=temp_data_dir)

        stats = store.get_collection_stats()

        assert stats["total_components"] == 0
        assert stats["types"] == {}

    def test_collection_stats_with_data(self, temp_data_dir, mock_embeddings):
        """Test stats with data."""
        store = MemoryStore(persist_directory=temp_data_dir)

        # Add different types of components
        components = [
            {"type": "function", "name": "f1", "file_path": "a.py"},
            {"type": "function", "name": "f2", "file_path": "b.py"},
            {"type": "class", "name": "C1", "file_path": "c.py"},
        ]

        for comp in components:
            store.store_component(comp)

        stats = store.get_collection_stats()

        assert stats["total_components"] == 3
        assert stats["types"]["function"] == 2
        assert stats["types"]["class"] == 1

    def test_get_all_patterns_empty(self, temp_data_dir, mock_embeddings):
        """Test pattern extraction from empty collection."""
        store = MemoryStore(persist_directory=temp_data_dir)

        patterns = store.get_all_patterns()

        assert isinstance(patterns, list)
        assert len(patterns) == 0

    def test_get_all_patterns_with_routes(self, temp_data_dir, mock_embeddings):
        """Test pattern extraction with routes."""
        store = MemoryStore(persist_directory=temp_data_dir)

        routes = [
            {
                "type": "route",
                "name": "get_users",
                "path": "/api/users",
                "methods": ["GET"],
                "file_path": "api/users.py"
            },
            {
                "type": "route",
                "name": "get_user",
                "path": "/api/users/{id}",
                "methods": ["GET"],
                "file_path": "api/users.py"
            }
        ]

        for route in routes:
            store.store_component(route)

        patterns = store.get_all_patterns()

        assert len(patterns) > 0
        assert any("/api/" in p for p in patterns)
        assert any("path parameters" in p for p in patterns)


class TestMemoryStorePersistence:
    """Test persistence functionality."""

    def test_data_persists_across_instances(self, temp_data_dir, mock_embeddings):
        """Test that data persists when creating new instance."""
        # First instance
        store1 = MemoryStore(persist_directory=temp_data_dir)
        component = {
            "type": "function",
            "name": "persist_test",
            "file_path": "test.py"
        }
        doc_id = store1.store_component(component)
        count1 = store1.collection.count()

        # Delete first instance
        del store1

        # Second instance
        store2 = MemoryStore(persist_directory=temp_data_dir)
        count2 = store2.collection.count()

        assert count2 == count1
        assert count2 == 1

        # Verify the document exists
        result = store2.collection.get(ids=[doc_id])
        assert len(result["ids"]) == 1


if __name__ == "__main__":
    print("Running unit tests for MemoryStore...")
    print("These tests use mocked embeddings and don't require Ollama.\n")

    pytest.main([__file__, "-v"])
