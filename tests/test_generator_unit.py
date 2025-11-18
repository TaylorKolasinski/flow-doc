"""Unit tests for DocumentationGenerator without Ollama dependency."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil
from src.core.generator import DocumentationGenerator
from src.core.memory import MemoryStore


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for ChromaDB data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_ollama():
    """Mock Ollama LLM."""
    with patch('src.core.generator.Ollama') as mock:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "This is a test response from the LLM."
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_embeddings():
    """Mock Ollama embeddings."""
    with patch('src.core.memory.OllamaEmbeddings') as mock:
        mock_instance = MagicMock()
        mock_instance.embed_query.return_value = [0.1] * 384
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def memory_store_with_data(temp_data_dir, mock_embeddings):
    """Create MemoryStore with test data."""
    store = MemoryStore(persist_directory=temp_data_dir)

    # Add test components
    test_components = [
        {
            "type": "function",
            "name": "get_user",
            "docstring": "Get user by ID",
            "params": ["user_id"],
            "decorators": ["app.get"],
            "file_path": "api/users.py"
        },
        {
            "type": "route",
            "name": "get_users",
            "path": "/api/users",
            "methods": ["GET"],
            "file_path": "api/users.py"
        },
        {
            "type": "class",
            "name": "UserService",
            "docstring": "User management service",
            "methods": ["get", "create", "update"],
            "file_path": "services/user.py"
        }
    ]

    for comp in test_components:
        store.store_component(comp)

    return store


class TestDocumentationGeneratorUnit:
    """Unit tests for DocumentationGenerator."""

    def test_initialization(self, memory_store_with_data, mock_ollama):
        """Test generator initialization."""
        gen = DocumentationGenerator(memory_store_with_data)

        assert gen.memory_store is not None
        assert gen.model == "llama3.2"
        assert gen.temperature == 0.1

    def test_format_component_context_empty(self, memory_store_with_data, mock_ollama):
        """Test formatting empty component list."""
        gen = DocumentationGenerator(memory_store_with_data)

        context = gen._format_component_context([])

        assert context == "No relevant code components found."

    def test_format_component_context_with_components(self, memory_store_with_data, mock_ollama):
        """Test formatting component list."""
        gen = DocumentationGenerator(memory_store_with_data)

        components = [
            {
                "metadata": {
                    "type": "function",
                    "name": "test_func",
                    "file_path": "test.py"
                },
                "document": "Type: function | Name: test_func | File: test.py",
                "score": 0.95
            },
            {
                "metadata": {
                    "type": "class",
                    "name": "TestClass",
                    "file_path": "test.py"
                },
                "document": "Type: class | Name: TestClass",
                "score": 0.80
            }
        ]

        context = gen._format_component_context(components)

        assert "test_func" in context
        assert "TestClass" in context
        assert "FUNCTION" in context
        assert "CLASS" in context
        assert "0.95" in context
        assert "0.80" in context

    def test_build_prompt(self, memory_store_with_data, mock_ollama):
        """Test prompt building."""
        gen = DocumentationGenerator(memory_store_with_data)

        question = "What functions exist?"
        context = "1. [FUNCTION] test_func"

        prompt = gen._build_prompt(question, context)

        assert "Context from codebase:" in prompt
        assert "Question: What functions exist?" in prompt
        assert "1. [FUNCTION] test_func" in prompt
        assert "Answer:" in prompt
        assert "based ONLY on the codebase context" in prompt

    def test_simple_query_calls_llm(self, memory_store_with_data, mock_ollama):
        """Test that simple_query calls the LLM."""
        gen = DocumentationGenerator(memory_store_with_data)

        answer = gen.simple_query("What endpoints exist?")

        # Verify LLM was called
        gen.llm.invoke.assert_called_once()

        # Verify answer is returned
        assert answer == "This is a test response from the LLM."

    def test_simple_query_retrieves_from_memory(self, memory_store_with_data, mock_ollama):
        """Test that simple_query retrieves from memory."""
        gen = DocumentationGenerator(memory_store_with_data)

        # Mock the memory retrieval
        with patch.object(gen.memory_store, 'retrieve_similar') as mock_retrieve:
            mock_retrieve.return_value = [
                {
                    "metadata": {"type": "route", "name": "get_users", "file_path": "api.py"},
                    "document": "Route: /api/users",
                    "score": 0.9
                }
            ]

            answer = gen.simple_query("What endpoints exist?", k=5)

            # Verify retrieval was called
            mock_retrieve.assert_called_once_with("What endpoints exist?", k=5)

            # Verify LLM was called
            assert gen.llm.invoke.called

    def test_get_stats(self, memory_store_with_data, mock_ollama):
        """Test getting statistics."""
        gen = DocumentationGenerator(memory_store_with_data)

        stats = gen.get_stats()

        assert stats["model"] == "llama3.2"
        assert stats["temperature"] == 0.1
        assert stats["memory_components"] == 3
        assert "function" in stats["component_types"]
        assert "route" in stats["component_types"]
        assert "class" in stats["component_types"]

    def test_custom_temperature(self, memory_store_with_data, mock_ollama):
        """Test custom temperature setting."""
        gen = DocumentationGenerator(
            memory_store_with_data,
            temperature=0.5
        )

        assert gen.temperature == 0.5

    def test_custom_model(self, memory_store_with_data, mock_ollama):
        """Test custom model setting."""
        gen = DocumentationGenerator(
            memory_store_with_data,
            model="llama3.2"
        )

        assert gen.model == "llama3.2"


class TestComponentSummaryUnit:
    """Unit tests for generate_component_summary."""

    def test_generate_summary_found(self, memory_store_with_data, mock_ollama):
        """Test generating summary for existing component."""
        gen = DocumentationGenerator(memory_store_with_data)

        summary = gen.generate_component_summary("UserService")

        assert summary
        # LLM should be called
        assert gen.llm.invoke.called

    def test_generate_summary_not_found(self, memory_store_with_data, mock_ollama):
        """Test generating summary for non-existent component."""
        gen = DocumentationGenerator(memory_store_with_data)

        # Mock retrieve to return no exact match
        with patch.object(gen.memory_store, 'retrieve_similar') as mock_retrieve:
            mock_retrieve.return_value = [
                {
                    "metadata": {"type": "class", "name": "OtherClass", "file_path": "other.py"},
                    "document": "Some other class",
                    "score": 0.5
                }
            ]

            summary = gen.generate_component_summary("NonExistent")

            assert "not found" in summary.lower()


if __name__ == "__main__":
    print("Running unit tests for DocumentationGenerator...")
    print("These tests use mocked LLM and don't require Ollama.\n")

    pytest.main([__file__, "-v"])
