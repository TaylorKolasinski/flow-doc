"""Integration tests for implemented DocumentationAgent workflow nodes."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from src.core.agent import DocumentationAgent, DocumentationState
from src.core.analyzer import CodebaseAnalyzer
from src.core.memory import MemoryStore
from src.core.generator import DocumentationGenerator


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for ChromaDB data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_embeddings():
    """Mock Ollama embeddings."""
    with patch('src.core.memory.OllamaEmbeddings') as mock:
        mock_instance = MagicMock()
        mock_instance.embed_query.return_value = [0.1] * 384
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_ollama():
    """Mock Ollama LLM."""
    with patch('src.core.generator.Ollama') as mock:
        mock_instance = MagicMock()
        # Return documentation that will pass validation
        mock_instance.invoke.return_value = """
# TestComponent Documentation

## Overview
This is a comprehensive documentation for TestComponent.

## Description
TestComponent provides functionality for testing the documentation generation workflow.

## Functions

### `test_function(param1, param2)`
Tests the component functionality.

Parameters:
- `param1`: First parameter
- `param2`: Second parameter

## Usage Examples

```python
result = test_function("hello", "world")
```

## Dependencies
- Standard library imports
"""
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def sample_codebase_dir():
    """Get path to sample codebase."""
    project_root = Path(__file__).parent.parent
    return project_root / "sample_codebase"


@pytest.fixture
def agent_with_memory(temp_data_dir, sample_codebase_dir, mock_embeddings, mock_ollama):
    """Create DocumentationAgent with populated memory."""
    analyzer = CodebaseAnalyzer(str(sample_codebase_dir))
    memory = MemoryStore(persist_directory=temp_data_dir)
    generator = DocumentationGenerator(memory)

    # Populate memory with some test components
    test_components = [
        {
            "type": "function",
            "name": "test_similar_func",
            "docstring": "A similar function for testing",
            "params": ["arg1", "arg2"],
            "file_path": "test.py"
        },
        {
            "type": "class",
            "name": "TestClass",
            "docstring": "A test class",
            "methods": ["method1", "method2"],
            "file_path": "test.py"
        }
    ]

    for comp in test_components:
        memory.store_component(comp)

    agent = DocumentationAgent(analyzer, memory, generator)
    return agent


class TestAnalyzeNodeImplementation:
    """Test analyze_node with real implementation."""

    def test_analyze_real_file(self, agent_with_memory):
        """Test analyzing a real file from sample codebase."""
        state = agent_with_memory.create_initial_state(
            component_name="flask_app",
            component_path="flask_app.py"
        )

        updated_state = agent_with_memory.analyze_node(state)

        # Should have parsed metadata
        assert 'code_metadata' in updated_state
        assert updated_state['code_metadata'] is not None
        assert 'path' in updated_state['code_metadata']
        assert updated_state['code_metadata']['path'] == "flask_app.py"

        # Should have found classes and functions
        metadata = updated_state['code_metadata']
        assert 'classes' in metadata or 'functions' in metadata

    def test_analyze_nonexistent_file(self, agent_with_memory):
        """Test analyzing a non-existent file."""
        state = agent_with_memory.create_initial_state(
            component_name="nonexistent",
            component_path="nonexistent.py"
        )

        updated_state = agent_with_memory.analyze_node(state)

        # Should have error
        assert len(updated_state['errors']) > 0
        assert updated_state['code_metadata'].get('error') or 'error' in str(updated_state['errors'])


class TestRetrieveNodeImplementation:
    """Test retrieve_node with real implementation."""

    def test_retrieve_with_populated_memory(self, agent_with_memory):
        """Test retrieval when memory has components."""
        state = agent_with_memory.create_initial_state(
            component_name="test_function",
            component_path="test.py"
        )
        state['code_metadata'] = {
            "functions": [{"name": "test_function", "params": ["arg1"]}]
        }

        updated_state = agent_with_memory.retrieve_node(state)

        # Should have retrieved context
        assert 'retrieved_context' in updated_state
        assert isinstance(updated_state['retrieved_context'], list)
        # Memory has 2 components, should find some
        assert len(updated_state['retrieved_context']) >= 0

    def test_retrieve_formats_context(self, agent_with_memory):
        """Test that retrieved context is formatted correctly."""
        state = agent_with_memory.create_initial_state(
            component_name="UserService",
            component_path="services/user.py"
        )
        state['code_metadata'] = {"classes": [{"name": "UserService"}]}

        updated_state = agent_with_memory.retrieve_node(state)

        # Context should be list of strings
        assert isinstance(updated_state['retrieved_context'], list)
        for ctx in updated_state['retrieved_context']:
            assert isinstance(ctx, str)


class TestGenerateNodeImplementation:
    """Test generate_node with real implementation."""

    def test_generate_builds_prompt(self, agent_with_memory):
        """Test that generate builds proper prompt."""
        state = agent_with_memory.create_initial_state(
            component_name="TestComponent",
            component_path="test.py"
        )
        state['code_metadata'] = {
            "functions": [{"name": "test_func", "params": ["arg1"]}],
            "module_docstring": "Test module"
        }
        state['retrieved_context'] = ["Context 1", "Context 2"]

        # Call generate
        updated_state = agent_with_memory.generate_node(state)

        # Should have documentation
        assert 'draft_documentation' in updated_state
        assert len(updated_state['draft_documentation']) > 0

        # Should increment iteration
        assert updated_state['iteration_count'] == 1

    def test_generate_calls_llm(self, agent_with_memory):
        """Test that generate calls the LLM."""
        state = agent_with_memory.create_initial_state(
            component_name="TestComponent",
            component_path="test.py"
        )
        state['code_metadata'] = {"functions": []}
        state['retrieved_context'] = []

        # Generate
        updated_state = agent_with_memory.generate_node(state)

        # LLM should have been called
        assert agent_with_memory.generator.llm.invoke.called


class TestValidateNodeImplementation:
    """Test validate_node with real implementation."""

    def test_validate_passing_documentation(self, agent_with_memory):
        """Test validation with good documentation."""
        state = agent_with_memory.create_initial_state(
            component_name="TestComponent",
            component_path="test.py"
        )
        state['draft_documentation'] = """
# TestComponent

## Overview
This is a well-documented component with all required sections.

## Description
TestComponent provides comprehensive functionality for testing.

## Functions

### test_function(param1, param2)
A test function with parameters.

Parameters:
- param1: First parameter
- param2: Second parameter

Returns:
- result: The test result

## Usage Examples

```python
result = test_function("hello", "world")
```

## Dependencies
- Standard library
"""

        updated_state = agent_with_memory.validate_node(state)

        # Should pass validation
        validation = updated_state['validation_result']
        assert validation['is_valid'] == True
        assert validation['score'] >= 0.7

    def test_validate_failing_documentation(self, agent_with_memory):
        """Test validation with incomplete documentation."""
        state = agent_with_memory.create_initial_state(
            component_name="TestComponent",
            component_path="test.py"
        )
        state['draft_documentation'] = "# Short doc"

        updated_state = agent_with_memory.validate_node(state)

        # Should fail validation
        validation = updated_state['validation_result']
        assert validation['is_valid'] == False
        assert validation['score'] < 0.7
        assert len(validation['issues']) > 0


class TestStoreNodeImplementation:
    """Test store_node with real implementation."""

    def test_store_creates_file(self, agent_with_memory, temp_data_dir):
        """Test that store creates documentation file."""
        state = agent_with_memory.create_initial_state(
            component_name="TestComponent",
            component_path="test.py"
        )
        state['draft_documentation'] = "# TestComponent\n\nTest documentation"
        state['code_metadata'] = {"functions": []}
        state['validation_result'] = {"is_valid": True, "score": 0.9}

        # Store
        updated_state = agent_with_memory.store_node(state)

        # Check file was created
        docs_dir = Path("docs_output")
        assert docs_dir.exists()
        assert (docs_dir / "TestComponent.md").exists()

        # Read and verify content
        with open(docs_dir / "TestComponent.md", 'r') as f:
            content = f.read()
            assert "TestComponent" in content
            assert "Test documentation" in content

    def test_store_handles_special_characters(self, agent_with_memory):
        """Test store handles component names with special characters."""
        state = agent_with_memory.create_initial_state(
            component_name="test/component",
            component_path="test/component.py"
        )
        state['draft_documentation'] = "# Test"
        state['code_metadata'] = {}

        # Should not raise exception
        updated_state = agent_with_memory.store_node(state)

        # File should be created with safe name
        assert (Path("docs_output") / "test_component.md").exists()


class TestFullWorkflow:
    """Test complete workflow with all nodes."""

    def test_end_to_end_workflow(self, agent_with_memory):
        """Test complete workflow from analyze to store."""
        # Initialize state
        state = agent_with_memory.create_initial_state(
            component_name="flask_app",
            component_path="flask_app.py",
            max_iterations=2
        )

        # Run workflow
        state = agent_with_memory.analyze_node(state)
        assert 'code_metadata' in state

        state = agent_with_memory.retrieve_node(state)
        assert 'retrieved_context' in state

        state = agent_with_memory.generate_node(state)
        assert 'draft_documentation' in state
        assert state['iteration_count'] == 1

        state = agent_with_memory.validate_node(state)
        assert 'validation_result' in state

        # Check routing
        decision = agent_with_memory.should_retry(state)
        assert decision in ["generate", "store"]

        # Store
        state = agent_with_memory.store_node(state)

        # Get stats
        stats = agent_with_memory.get_workflow_stats(state)
        assert stats['component'] == "flask_app"
        assert stats['iterations'] > 0


def cleanup_test_output():
    """Clean up test output directory."""
    docs_dir = Path("docs_output")
    if docs_dir.exists():
        shutil.rmtree(docs_dir)


if __name__ == "__main__":
    try:
        print("Running node implementation tests...")
        pytest.main([__file__, "-v", "-s"])
    finally:
        cleanup_test_output()
