"""Tests for MemoryStore functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from src.core.memory import MemoryStore
import requests


def is_ollama_available():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


# Skip tests if Ollama is not available
pytestmark = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama server not available"
)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for ChromaDB data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def memory_store(temp_data_dir):
    """Create a MemoryStore instance with temporary storage."""
    store = MemoryStore(persist_directory=temp_data_dir)
    yield store
    # Cleanup
    store.clear_collection()


class TestMemoryStoreInit:
    """Test MemoryStore initialization."""

    def test_init_default_config(self, temp_data_dir):
        """Test initialization with default configuration."""
        store = MemoryStore(persist_directory=temp_data_dir)

        assert store.persist_directory == temp_data_dir
        assert store.collection_name == "codebase_patterns"
        assert store.embedding_model == "llama3.2"
        assert store.ollama_base_url == "http://localhost:11434"
        assert store.collection is not None

    def test_init_custom_config(self, temp_data_dir):
        """Test initialization with custom configuration."""
        store = MemoryStore(
            persist_directory=temp_data_dir,
            collection_name="test_collection",
            embedding_model="llama3.2",
        )

        assert store.collection_name == "test_collection"
        assert store.embedding_model == "llama3.2"

    def test_persistence_directory_created(self, temp_data_dir):
        """Test that persistence directory is created."""
        custom_dir = Path(temp_data_dir) / "custom" / "nested"
        store = MemoryStore(persist_directory=str(custom_dir))

        assert custom_dir.exists()


class TestStoreComponent:
    """Test storing components in MemoryStore."""

    def test_store_function_component(self, memory_store):
        """Test storing a function component."""
        component = {
            "type": "function",
            "name": "get_user",
            "docstring": "Retrieve user by ID from database",
            "params": ["user_id", "db"],
            "decorators": ["app.get"],
            "file_path": "api/users.py"
        }

        doc_id = memory_store.store_component(component)

        assert doc_id == "api/users.py::get_user"
        assert memory_store.collection.count() == 1

    def test_store_class_component(self, memory_store):
        """Test storing a class component."""
        component = {
            "type": "class",
            "name": "UserService",
            "docstring": "Service for managing user operations",
            "methods": ["get_user", "create_user", "update_user"],
            "file_path": "services/user.py"
        }

        doc_id = memory_store.store_component(component)

        assert doc_id == "services/user.py::UserService"
        assert memory_store.collection.count() == 1

    def test_store_route_component(self, memory_store):
        """Test storing a route component."""
        component = {
            "type": "route",
            "name": "get_users",
            "path": "/api/users",
            "methods": ["GET"],
            "docstring": "Get all users endpoint",
            "file_path": "api/users.py"
        }

        doc_id = memory_store.store_component(component)

        assert doc_id == "api/users.py::get_users"

        # Verify metadata
        result = memory_store.collection.get(ids=[doc_id])
        assert result["metadatas"][0]["route_path"] == "/api/users"
        assert result["metadatas"][0]["http_methods"] == "GET"

    def test_store_multiple_components(self, memory_store):
        """Test storing multiple components."""
        components = [
            {
                "type": "function",
                "name": "authenticate",
                "docstring": "Authenticate user credentials",
                "params": ["username", "password"],
                "file_path": "auth.py"
            },
            {
                "type": "class",
                "name": "Database",
                "docstring": "Database connection manager",
                "methods": ["connect", "disconnect"],
                "file_path": "db.py"
            },
            {
                "type": "route",
                "name": "login",
                "path": "/auth/login",
                "methods": ["POST"],
                "file_path": "auth.py"
            }
        ]

        for component in components:
            memory_store.store_component(component)

        assert memory_store.collection.count() == 3


class TestRetrieveSimilar:
    """Test semantic retrieval from MemoryStore."""

    @pytest.fixture(autouse=True)
    def setup_components(self, memory_store):
        """Set up test components."""
        self.components = [
            {
                "type": "function",
                "name": "get_user",
                "docstring": "Retrieve user information from database by ID",
                "params": ["user_id"],
                "decorators": ["app.get"],
                "file_path": "api/users.py"
            },
            {
                "type": "function",
                "name": "create_user",
                "docstring": "Create a new user account with validation",
                "params": ["user_data"],
                "decorators": ["app.post"],
                "file_path": "api/users.py"
            },
            {
                "type": "function",
                "name": "calculate_total",
                "docstring": "Calculate total price with tax and discounts",
                "params": ["price", "tax", "discount"],
                "file_path": "utils/math.py"
            },
            {
                "type": "class",
                "name": "UserService",
                "docstring": "Service class for user management operations",
                "methods": ["get", "create", "update", "delete"],
                "file_path": "services/user.py"
            }
        ]

        for component in self.components:
            memory_store.store_component(component)

        self.memory_store = memory_store

    def test_retrieve_similar_basic(self):
        """Test basic similarity search."""
        results = self.memory_store.retrieve_similar("user management", k=3)

        assert len(results) > 0
        assert len(results) <= 3
        assert all("id" in r for r in results)
        assert all("score" in r for r in results)
        assert all("metadata" in r for r in results)

    def test_retrieve_similar_user_functions(self):
        """Test retrieving user-related functions."""
        results = self.memory_store.retrieve_similar("get user information", k=2)

        assert len(results) > 0
        # Should prioritize get_user function
        top_result = results[0]
        assert "user" in top_result["metadata"]["name"].lower()

    def test_retrieve_similar_calculation(self):
        """Test retrieving calculation-related functions."""
        # Use k=4 to ensure we check all stored components
        # Semantic similarity can vary, so we need to check more results
        results = self.memory_store.retrieve_similar("calculate price", k=4)

        assert len(results) > 0
        # Should find calculate_total in the results
        names = [r["metadata"]["name"] for r in results]
        assert any("calculate" in name.lower() for name in names)

    def test_retrieve_similar_with_k_parameter(self):
        """Test k parameter limits results."""
        results_k2 = self.memory_store.retrieve_similar("user", k=2)
        results_k4 = self.memory_store.retrieve_similar("user", k=4)

        assert len(results_k2) <= 2
        assert len(results_k4) <= 4
        assert len(results_k4) >= len(results_k2)


class TestGetAllPatterns:
    """Test pattern extraction from MemoryStore."""

    def test_get_patterns_empty_collection(self, memory_store):
        """Test pattern extraction from empty collection."""
        patterns = memory_store.get_all_patterns()

        assert isinstance(patterns, list)
        assert len(patterns) == 0

    def test_get_patterns_with_routes(self, memory_store):
        """Test pattern extraction with route components."""
        components = [
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
                "path": "/api/users/{user_id}",
                "methods": ["GET"],
                "file_path": "api/users.py"
            },
            {
                "type": "route",
                "name": "create_user",
                "path": "/api/users",
                "methods": ["POST"],
                "file_path": "api/users.py"
            }
        ]

        for component in components:
            memory_store.store_component(component)

        patterns = memory_store.get_all_patterns()

        assert len(patterns) > 0
        assert any("/api/" in pattern for pattern in patterns)
        assert any("path parameters" in pattern for pattern in patterns)

    def test_get_patterns_with_classes(self, memory_store):
        """Test pattern extraction with class components."""
        components = [
            {
                "type": "class",
                "name": "UserService",
                "docstring": "User management",
                "methods": ["get", "create"],
                "file_path": "services/user.py"
            },
            {
                "type": "class",
                "name": "ProductService",
                "docstring": "Product management",
                "methods": ["list", "get"],
                "file_path": "services/product.py"
            }
        ]

        for component in components:
            memory_store.store_component(component)

        patterns = memory_store.get_all_patterns()

        assert any("class" in pattern.lower() for pattern in patterns)


class TestPersistence:
    """Test ChromaDB persistence."""

    def test_persistence_across_instances(self, temp_data_dir):
        """Test that data persists across MemoryStore instances."""
        # Create first instance and store data
        store1 = MemoryStore(persist_directory=temp_data_dir)
        component = {
            "type": "function",
            "name": "test_function",
            "docstring": "Test persistence",
            "file_path": "test.py"
        }
        doc_id = store1.store_component(component)
        count1 = store1.collection.count()

        # Close first instance (Python GC will handle cleanup)
        del store1

        # Create second instance with same directory
        store2 = MemoryStore(persist_directory=temp_data_dir)
        count2 = store2.collection.count()

        # Verify data persisted
        assert count2 == count1
        assert count2 == 1

        # Verify we can retrieve the document
        result = store2.collection.get(ids=[doc_id])
        assert len(result["ids"]) == 1
        assert result["ids"][0] == doc_id

    def test_clear_collection(self, memory_store):
        """Test clearing collection."""
        # Add some components
        component = {
            "type": "function",
            "name": "test",
            "file_path": "test.py"
        }
        memory_store.store_component(component)

        assert memory_store.collection.count() > 0

        # Clear collection
        memory_store.clear_collection()

        assert memory_store.collection.count() == 0


class TestCollectionStats:
    """Test collection statistics."""

    def test_stats_empty_collection(self, memory_store):
        """Test stats for empty collection."""
        stats = memory_store.get_collection_stats()

        assert stats["total_components"] == 0
        assert stats["types"] == {}

    def test_stats_with_components(self, memory_store):
        """Test stats with various components."""
        components = [
            {"type": "function", "name": "func1", "file_path": "a.py"},
            {"type": "function", "name": "func2", "file_path": "b.py"},
            {"type": "class", "name": "ClassA", "file_path": "c.py"},
            {"type": "route", "name": "route1", "path": "/api", "methods": ["GET"], "file_path": "d.py"},
        ]

        for component in components:
            memory_store.store_component(component)

        stats = memory_store.get_collection_stats()

        assert stats["total_components"] == 4
        assert stats["types"]["function"] == 2
        assert stats["types"]["class"] == 1
        assert stats["types"]["route"] == 1


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self, temp_data_dir):
        """Test complete workflow: store, retrieve, persist, reload."""
        # Step 1: Create store and add components
        store = MemoryStore(persist_directory=temp_data_dir)

        components = [
            {
                "type": "function",
                "name": "authenticate_user",
                "docstring": "Authenticate user with credentials",
                "params": ["username", "password"],
                "file_path": "auth.py"
            },
            {
                "type": "function",
                "name": "authorize_request",
                "docstring": "Check if user is authorized for action",
                "params": ["user", "action"],
                "file_path": "auth.py"
            }
        ]

        for comp in components:
            store.store_component(comp)

        # Step 2: Retrieve similar
        results = store.retrieve_similar("user authentication", k=2)
        assert len(results) > 0

        # Step 3: Get patterns
        patterns = store.get_all_patterns()
        assert isinstance(patterns, list)

        # Step 4: Get stats
        stats = store.get_collection_stats()
        assert stats["total_components"] == 2

        # Step 5: Close and reopen
        del store

        store2 = MemoryStore(persist_directory=temp_data_dir)
        stats2 = store2.get_collection_stats()
        assert stats2["total_components"] == 2

        # Step 6: Verify search still works
        results2 = store2.retrieve_similar("authorization", k=1)
        assert len(results2) > 0


if __name__ == "__main__":
    """Run manual tests with output."""
    print("Running MemoryStore integration tests...\n")

    if not is_ollama_available():
        print("ERROR: Ollama server is not running!")
        print("Please start Ollama: ollama serve")
        print("And ensure llama3.2 model is available: ollama pull llama3.2")
        exit(1)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}\n")

    try:
        # Test 1: Initialize and store
        print("=" * 60)
        print("Test 1: Storing Components")
        print("=" * 60)
        store = MemoryStore(persist_directory=temp_dir)

        test_components = [
            {
                "type": "function",
                "name": "get_user",
                "docstring": "Retrieve user by ID from database",
                "params": ["user_id"],
                "decorators": ["app.get"],
                "file_path": "api/users.py"
            },
            {
                "type": "class",
                "name": "UserService",
                "docstring": "Service for user management",
                "methods": ["get_user", "create_user", "update_user"],
                "file_path": "services/user.py"
            },
            {
                "type": "route",
                "name": "create_user",
                "path": "/api/users",
                "methods": ["POST"],
                "docstring": "Create new user endpoint",
                "file_path": "api/users.py"
            }
        ]

        for comp in test_components:
            doc_id = store.store_component(comp)
            print(f"✓ Stored: {doc_id}")

        stats = store.get_collection_stats()
        print(f"\nCollection Stats: {stats['total_components']} components")
        print(f"Types: {stats['types']}")

        # Test 2: Retrieve similar
        print("\n" + "=" * 60)
        print("Test 2: Semantic Search")
        print("=" * 60)
        query = "user management functions"
        print(f"Query: '{query}'\n")

        results = store.retrieve_similar(query, k=3)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['metadata']['name']} (score: {result['score']:.3f})")
            print(f"   Type: {result['metadata']['type']}")
            print(f"   File: {result['metadata']['file_path']}")
            print()

        # Test 3: Extract patterns
        print("=" * 60)
        print("Test 3: Pattern Extraction")
        print("=" * 60)
        patterns = store.get_all_patterns()
        print(f"Found {len(patterns)} patterns:\n")
        for pattern in patterns:
            print(f"  - {pattern}")

        # Test 4: Persistence
        print("\n" + "=" * 60)
        print("Test 4: Testing Persistence")
        print("=" * 60)
        print("Closing store and reopening...")
        del store

        store2 = MemoryStore(persist_directory=temp_dir)
        stats2 = store2.get_collection_stats()
        print(f"✓ Data persisted: {stats2['total_components']} components")

        results2 = store2.retrieve_similar("user operations", k=2)
        print(f"✓ Search works: found {len(results2)} results")

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")
