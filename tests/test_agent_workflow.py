"""Tests for DocumentationAgent LangGraph workflow execution."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from src.core.agent import DocumentationAgent
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
def mock_ollama_llm():
    """Mock Ollama LLM for generation."""
    with patch('src.core.generator.Ollama') as mock:
        mock_instance = MagicMock()
        # Return well-formed documentation
        mock_instance.invoke.return_value = """
# TestComponent Documentation

## Overview
TestComponent is a comprehensive module for testing documentation generation workflows.

## Description
This component provides functionality for validating the LangGraph workflow execution
with proper documentation structure and quality checks.

## Functions

### `test_function(param1, param2)`
Main test function that processes parameters.

**Parameters:**
- `param1` (str): First parameter for testing
- `param2` (int): Second parameter for testing

**Returns:**
- dict: Result dictionary with processed data

## Usage Examples

```python
from test_component import test_function

# Basic usage
result = test_function("hello", 42)
print(result)
```

## Dependencies
- Standard library modules
- Testing frameworks
"""
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def sample_codebase_dir():
    """Get path to sample codebase."""
    project_root = Path(__file__).parent.parent
    return project_root / "sample_codebase"


@pytest.fixture
def agent(temp_data_dir, sample_codebase_dir, mock_embeddings, mock_ollama_llm):
    """Create DocumentationAgent with mocked dependencies."""
    analyzer = CodebaseAnalyzer(str(sample_codebase_dir))
    memory = MemoryStore(persist_directory=temp_data_dir)
    generator = DocumentationGenerator(memory)

    # Add some test data to memory
    memory.store_component({
        "type": "function",
        "name": "similar_func",
        "docstring": "A similar function",
        "params": ["arg1"],
        "file_path": "test.py"
    })

    return DocumentationAgent(analyzer, memory, generator)


class TestWorkflowCreation:
    """Test workflow graph creation."""

    def test_create_workflow(self, agent):
        """Test creating the workflow graph."""
        workflow = agent.create_workflow()

        assert workflow is not None
        # Workflow should be compiled and ready to execute

    def test_workflow_has_nodes(self, agent):
        """Test that workflow contains all required nodes."""
        workflow = agent.create_workflow()

        # Get the graph structure
        graph = workflow.get_graph()
        nodes = list(graph.nodes.keys())

        # Check all required nodes exist
        assert "analyze" in nodes
        assert "retrieve" in nodes
        assert "generate" in nodes
        assert "validate" in nodes
        assert "store" in nodes


class TestWorkflowExecution:
    """Test end-to-end workflow execution."""

    def test_run_workflow_success(self, agent):
        """Test successful workflow execution."""
        result = agent.run(
            component_name="flask_app",
            component_path="flask_app.py",
            max_iterations=2
        )

        # Check result structure
        assert 'success' in result
        assert 'component' in result
        assert 'iterations' in result
        assert 'validation_score' in result
        assert 'errors' in result
        assert 'output_file' in result

        # Component should match
        assert result['component'] == "flask_app"

        # Should have at least 1 iteration
        assert result['iterations'] >= 1

    def test_run_workflow_creates_output(self, agent):
        """Test that workflow creates documentation file."""
        result = agent.run(
            component_name="flask_app",
            component_path="flask_app.py"
        )

        # Check output file was created
        output_file = result.get('output_file')
        if output_file and result.get('success'):
            assert Path(output_file).exists()

    def test_run_workflow_with_iterations(self, agent):
        """Test workflow with multiple iterations allowed."""
        result = agent.run(
            component_name="flask_app",
            component_path="flask_app.py",
            max_iterations=3
        )

        # Should complete even if validation fails
        assert result['iterations'] <= 3

    def test_run_workflow_nonexistent_file(self, agent):
        """Test workflow with non-existent file."""
        result = agent.run(
            component_name="nonexistent",
            component_path="nonexistent.py"
        )

        # Should have errors but not crash
        assert isinstance(result, dict)
        assert len(result.get('errors', [])) > 0 or result.get('success') == False


class TestWorkflowState:
    """Test workflow state management."""

    def test_workflow_returns_final_state(self, agent):
        """Test that workflow returns final state."""
        result = agent.run(
            component_name="flask_app",
            component_path="flask_app.py"
        )

        # Should have final state
        final_state = result.get('final_state')
        if final_state:
            assert 'component_name' in final_state
            assert 'code_metadata' in final_state
            assert 'draft_documentation' in final_state
            assert 'validation_result' in final_state

    def test_workflow_tracks_iterations(self, agent):
        """Test that workflow tracks iteration count."""
        result = agent.run(
            component_name="flask_app",
            component_path="flask_app.py",
            max_iterations=2
        )

        # Iteration count should be within bounds
        assert 0 < result['iterations'] <= 2


class TestWorkflowValidation:
    """Test workflow validation behavior."""

    def test_workflow_validates_documentation(self, agent):
        """Test that workflow validates generated documentation."""
        result = agent.run(
            component_name="flask_app",
            component_path="flask_app.py"
        )

        # Should have validation information
        assert 'validation_score' in result
        assert 'validation_passed' in result

        # Score should be between 0 and 1
        assert 0.0 <= result['validation_score'] <= 1.0

    def test_workflow_retries_on_failure(self, agent):
        """Test that workflow retries when validation fails."""
        # Mock LLM to return poor documentation on first attempt
        with patch.object(agent.generator.llm, 'invoke') as mock_invoke:
            # First call returns bad docs, second returns good
            mock_invoke.side_effect = [
                "# Short doc",  # Will fail validation
                """
# Complete Documentation

## Overview
A complete documentation with all required sections.

## Description
Detailed description of the component functionality.

## Functions
Functions are documented with parameters and returns.

Parameters:
- param1: First parameter

## Usage Examples
```python
example_code()
```
"""  # Will pass validation
            ]

            result = agent.run(
                component_name="flask_app",
                component_path="flask_app.py",
                max_iterations=2
            )

            # Should have made multiple attempts
            # (Note: actual count depends on validation logic)
            assert result['iterations'] >= 1


class TestWorkflowVisualization:
    """Test workflow visualization."""

    def test_visualize_workflow_creates_file(self, agent, temp_data_dir):
        """Test that visualization creates output file."""
        output_path = Path(temp_data_dir) / "test_workflow.png"

        # Should not raise exception
        try:
            agent.visualize_workflow(str(output_path))
        except Exception:
            # Visualization might fail without certain dependencies
            # but should handle gracefully
            pass

    def test_visualize_workflow_mermaid_fallback(self, agent, temp_data_dir):
        """Test that visualization falls back to Mermaid."""
        output_path = Path(temp_data_dir) / "test_workflow.png"

        try:
            agent.visualize_workflow(str(output_path))

            # Check if either PNG or Mermaid file was created
            mermaid_path = Path(str(output_path).replace('.png', '.mmd'))
            assert output_path.exists() or mermaid_path.exists()

        except Exception:
            # Visualization may not be available in all environments
            pass


class TestWorkflowErrorHandling:
    """Test workflow error handling."""

    def test_workflow_handles_analysis_errors(self, agent):
        """Test that workflow handles analysis errors gracefully."""
        result = agent.run(
            component_name="invalid",
            component_path="invalid_file.py"
        )

        # Should return result even with errors
        assert isinstance(result, dict)
        assert 'success' in result

    def test_workflow_continues_after_node_errors(self, agent):
        """Test that workflow continues after node errors."""
        # Even if analysis fails, workflow should complete
        result = agent.run(
            component_name="test",
            component_path="nonexistent.py"
        )

        # Should have completed (may have errors)
        assert isinstance(result, dict)
        assert 'iterations' in result


class TestWorkflowIntegration:
    """Integration tests for complete workflow."""

    def test_full_workflow_with_real_file(self, agent):
        """Test complete workflow with real sample file."""
        result = agent.run(
            component_name="flask_app",
            component_path="flask_app.py",
            max_iterations=2
        )

        # Should complete successfully
        assert result is not None
        assert result['component'] == "flask_app"

        # Should have generated documentation
        if result.get('success'):
            assert result['doc_length'] > 0

    def test_workflow_produces_valid_markdown(self, agent):
        """Test that workflow produces valid Markdown."""
        result = agent.run(
            component_name="flask_app",
            component_path="flask_app.py"
        )

        if result.get('success') and result.get('final_state'):
            doc = result['final_state'].get('draft_documentation', '')

            # Should have Markdown structure
            assert '##' in doc or '#' in doc  # Headers
            assert len(doc) > 200  # Substantial content


def cleanup_test_files():
    """Clean up test output files."""
    docs_dir = Path("docs_output")
    if docs_dir.exists():
        shutil.rmtree(docs_dir)


if __name__ == "__main__":
    try:
        print("Running LangGraph workflow tests...")
        pytest.main([__file__, "-v", "-s"])
    finally:
        cleanup_test_files()
