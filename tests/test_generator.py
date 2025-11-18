"""Tests for DocumentationGenerator."""

import pytest
import tempfile
import shutil
from pathlib import Path
import requests
from src.core.generator import DocumentationGenerator
from src.core.memory import MemoryStore
from src.core.analyzer import CodebaseAnalyzer


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
def memory_store_with_sample_data(temp_data_dir):
    """Create MemoryStore with sample Flask/FastAPI data."""
    # Initialize memory store
    store = MemoryStore(persist_directory=temp_data_dir)

    # Get path to sample codebase
    project_root = Path(__file__).parent.parent
    sample_dir = project_root / "sample_codebase"

    # Initialize analyzer
    analyzer = CodebaseAnalyzer(str(sample_dir))

    # Parse and store Flask app
    flask_parsed = analyzer.parse_file("flask_app.py")
    if flask_parsed:
        # Store functions
        for func in flask_parsed["functions"]:
            component = {
                "type": "function",
                "name": func["name"],
                "docstring": func.get("docstring"),
                "params": func.get("params", []),
                "decorators": func.get("decorators", []),
                "file_path": flask_parsed["path"]
            }
            store.store_component(component)

        # Store classes
        for cls in flask_parsed["classes"]:
            component = {
                "type": "class",
                "name": cls["name"],
                "docstring": cls.get("docstring"),
                "methods": cls.get("methods", []),
                "file_path": flask_parsed["path"]
            }
            store.store_component(component)

        # Store routes
        for route in flask_parsed["routes"]:
            component = {
                "type": "route",
                "name": route["handler"],
                "path": route["path"],
                "methods": route.get("methods", []),
                "file_path": flask_parsed["path"]
            }
            store.store_component(component)

    # Parse and store FastAPI app
    fastapi_parsed = analyzer.parse_file("fastapi_app.py")
    if fastapi_parsed:
        # Store functions
        for func in fastapi_parsed["functions"]:
            component = {
                "type": "function",
                "name": func["name"],
                "docstring": func.get("docstring"),
                "params": func.get("params", []),
                "decorators": func.get("decorators", []),
                "file_path": fastapi_parsed["path"]
            }
            store.store_component(component)

        # Store classes
        for cls in fastapi_parsed["classes"]:
            component = {
                "type": "class",
                "name": cls["name"],
                "docstring": cls.get("docstring"),
                "methods": cls.get("methods", []),
                "file_path": fastapi_parsed["path"]
            }
            store.store_component(component)

        # Store routes
        for route in fastapi_parsed["routes"]:
            component = {
                "type": "route",
                "name": route["handler"],
                "path": route["path"],
                "methods": route.get("methods", []),
                "file_path": fastapi_parsed["path"]
            }
            store.store_component(component)

    yield store


@pytest.fixture
def generator(memory_store_with_sample_data):
    """Create DocumentationGenerator with sample data."""
    return DocumentationGenerator(memory_store_with_sample_data)


class TestDocumentationGeneratorInit:
    """Test DocumentationGenerator initialization."""

    def test_init_default_config(self, memory_store_with_sample_data):
        """Test initialization with default configuration."""
        gen = DocumentationGenerator(memory_store_with_sample_data)

        assert gen.memory_store is not None
        assert gen.model == "llama3.2"
        assert gen.base_url == "http://localhost:11434"
        assert gen.temperature == 0.1
        assert gen.llm is not None

    def test_init_custom_config(self, memory_store_with_sample_data):
        """Test initialization with custom configuration."""
        gen = DocumentationGenerator(
            memory_store_with_sample_data,
            model="llama3.2",
            temperature=0.3
        )

        assert gen.model == "llama3.2"
        assert gen.temperature == 0.3

    def test_get_stats(self, generator):
        """Test getting generator statistics."""
        stats = generator.get_stats()

        assert "model" in stats
        assert "temperature" in stats
        assert "memory_components" in stats
        assert "component_types" in stats

        assert stats["model"] == "llama3.2"
        assert stats["memory_components"] > 0


class TestSimpleQuery:
    """Test simple_query method."""

    def test_query_user_endpoints(self, generator):
        """Test querying for user management endpoints."""
        question = "What API endpoints exist for user management?"

        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print('='*70)

        answer = generator.simple_query(question, k=5, verbose=True)

        print(f"\n{'='*70}")
        print("Answer:")
        print('='*70)
        print(answer)
        print('='*70 + '\n')

        # Verify answer is not empty
        assert answer
        assert len(answer) > 0

        # Answer should mention routes or endpoints
        answer_lower = answer.lower()
        assert any(word in answer_lower for word in ["endpoint", "route", "api", "user"])

    def test_query_classes(self, generator):
        """Test querying for classes in the codebase."""
        question = "What classes are defined in the codebase?"

        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print('='*70)

        answer = generator.simple_query(question, k=5)

        print(f"\n{'='*70}")
        print("Answer:")
        print('='*70)
        print(answer)
        print('='*70 + '\n')

        assert answer
        assert len(answer) > 0

        # Should mention classes
        answer_lower = answer.lower()
        assert any(word in answer_lower for word in ["class", "userservice", "database"])

    def test_query_authentication(self, generator):
        """Test querying for authentication functionality."""
        question = "How does user authentication work?"

        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print('='*70)

        answer = generator.simple_query(question, k=3)

        print(f"\n{'='*70}")
        print("Answer:")
        print('='*70)
        print(answer)
        print('='*70 + '\n')

        assert answer
        assert len(answer) > 0

    def test_query_no_results(self, generator):
        """Test querying with no relevant results."""
        question = "How does the quantum flux capacitor work?"

        answer = generator.simple_query(question, k=3)

        assert answer
        # Should indicate it couldn't find relevant info
        assert len(answer) > 0

    def test_query_k_parameter(self, generator):
        """Test different k values for retrieval."""
        question = "What functions are available?"

        # Query with different k values
        answer_k1 = generator.simple_query(question, k=1)
        answer_k5 = generator.simple_query(question, k=5)

        assert answer_k1
        assert answer_k5

        # Both should be valid answers
        assert len(answer_k1) > 0
        assert len(answer_k5) > 0


class TestComponentSummary:
    """Test generate_component_summary method."""

    def test_generate_summary_existing_component(self, generator):
        """Test generating summary for existing component."""
        component_name = "UserService"

        print(f"\n{'='*70}")
        print(f"Generating summary for: {component_name}")
        print('='*70)

        summary = generator.generate_component_summary(component_name)

        print(f"\n{'='*70}")
        print("Summary:")
        print('='*70)
        print(summary)
        print('='*70 + '\n')

        assert summary
        assert len(summary) > 0
        assert "UserService" in summary or "user" in summary.lower()

    def test_generate_summary_nonexistent_component(self, generator):
        """Test generating summary for non-existent component."""
        component_name = "NonExistentClass"

        summary = generator.generate_component_summary(component_name)

        assert summary
        assert "not found" in summary.lower()


class TestFormatComponentContext:
    """Test _format_component_context method."""

    def test_format_empty_components(self, generator):
        """Test formatting with no components."""
        context = generator._format_component_context([])

        assert context == "No relevant code components found."

    def test_format_with_components(self, generator):
        """Test formatting with components."""
        components = [
            {
                "metadata": {
                    "type": "function",
                    "name": "test_func",
                    "file_path": "test.py"
                },
                "document": "Type: function | Name: test_func",
                "score": 0.95
            }
        ]

        context = generator._format_component_context(components)

        assert "test_func" in context
        assert "FUNCTION" in context.upper()
        assert "test.py" in context
        assert "0.95" in context


class TestBuildPrompt:
    """Test _build_prompt method."""

    def test_build_prompt_structure(self, generator):
        """Test prompt structure."""
        question = "What endpoints exist?"
        context = "1. [ROUTE] get_users"

        prompt = generator._build_prompt(question, context)

        assert "Context from codebase:" in prompt
        assert "Question:" in prompt
        assert question in prompt
        assert context in prompt
        assert "Answer:" in prompt


if __name__ == "__main__":
    """Run manual integration test."""
    print("\n" + "=" * 70)
    print("DocumentationGenerator Integration Test")
    print("=" * 70 + "\n")

    if not is_ollama_available():
        print("❌ ERROR: Ollama server is not running!")
        print("\n   To run this test:")
        print("   1. Start Ollama: ollama serve")
        print("   2. Ensure llama3.2 is available: ollama pull llama3.2")
        print()
        exit(1)

    print("✓ Ollama server is running\n")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}\n")

    try:
        # Initialize components
        print("-" * 70)
        print("1. Loading Sample Codebase into Memory")
        print("-" * 70)

        store = MemoryStore(persist_directory=temp_dir)
        project_root = Path(__file__).parent.parent
        sample_dir = project_root / "sample_codebase"
        analyzer = CodebaseAnalyzer(str(sample_dir))

        # Load Flask app
        flask_parsed = analyzer.parse_file("flask_app.py")
        if flask_parsed:
            print(f"✓ Parsed flask_app.py")
            for func in flask_parsed["functions"]:
                store.store_component({
                    "type": "function",
                    "name": func["name"],
                    "docstring": func.get("docstring"),
                    "params": func.get("params", []),
                    "decorators": func.get("decorators", []),
                    "file_path": flask_parsed["path"]
                })
            for route in flask_parsed["routes"]:
                store.store_component({
                    "type": "route",
                    "name": route["handler"],
                    "path": route["path"],
                    "methods": route.get("methods", []),
                    "file_path": flask_parsed["path"]
                })

        # Load FastAPI app
        fastapi_parsed = analyzer.parse_file("fastapi_app.py")
        if fastapi_parsed:
            print(f"✓ Parsed fastapi_app.py")
            for func in fastapi_parsed["functions"]:
                store.store_component({
                    "type": "function",
                    "name": func["name"],
                    "docstring": func.get("docstring"),
                    "params": func.get("params", []),
                    "decorators": func.get("decorators", []),
                    "file_path": fastapi_parsed["path"]
                })
            for route in fastapi_parsed["routes"]:
                store.store_component({
                    "type": "route",
                    "name": route["handler"],
                    "path": route["path"],
                    "methods": route.get("methods", []),
                    "file_path": fastapi_parsed["path"]
                })

        stats = store.get_collection_stats()
        print(f"\n✓ Loaded {stats['total_components']} components into memory")
        print(f"  Types: {stats['types']}\n")

        # Initialize generator
        print("-" * 70)
        print("2. Initializing DocumentationGenerator")
        print("-" * 70)

        generator = DocumentationGenerator(store)
        print(f"✓ Initialized with model: {generator.model}")
        print(f"✓ Temperature: {generator.temperature}\n")

        # Run queries
        print("-" * 70)
        print("3. Running Queries")
        print("-" * 70)

        queries = [
            "What API endpoints exist for user management?",
            "What classes are defined in the codebase?",
            "How does user creation work?",
        ]

        for i, question in enumerate(queries, 1):
            print(f"\n{'='*70}")
            print(f"Query {i}: {question}")
            print('='*70)

            answer = generator.simple_query(question, k=5)

            print(f"\nAnswer:")
            print("-" * 70)
            print(answer)
            print("-" * 70)

        print("\n" + "=" * 70)
        print("✓ All queries completed successfully!")
        print("=" * 70)

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\n✓ Cleaned up: {temp_dir}\n")
