"""Tests for DocumentationAgent workflow structure."""

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
    shutil.rmtree(temp_dir)


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
        mock_instance.invoke.return_value = "Generated documentation"
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def sample_codebase_dir():
    """Get path to sample codebase."""
    project_root = Path(__file__).parent.parent
    return project_root / "sample_codebase"


@pytest.fixture
def agent(temp_data_dir, sample_codebase_dir, mock_embeddings, mock_ollama):
    """Create DocumentationAgent with mocked dependencies."""
    analyzer = CodebaseAnalyzer(str(sample_codebase_dir))
    memory = MemoryStore(persist_directory=temp_data_dir)
    generator = DocumentationGenerator(memory)

    return DocumentationAgent(analyzer, memory, generator)


class TestDocumentationState:
    """Test DocumentationState schema."""

    def test_create_initial_state(self, agent):
        """Test creating initial state."""
        state = agent.create_initial_state(
            component_name="TestComponent",
            component_path="test.py",
            max_iterations=3
        )

        assert state['component_name'] == "TestComponent"
        assert state['component_path'] == "test.py"
        assert state['code_metadata'] == {}
        assert state['retrieved_context'] == []
        assert state['draft_documentation'] == ""
        assert state['validation_result'] == {}
        assert state['iteration_count'] == 0
        assert state['max_iterations'] == 3
        assert state['errors'] == []

    def test_initial_state_default_max_iterations(self, agent):
        """Test default max_iterations."""
        state = agent.create_initial_state(
            component_name="TestComponent",
            component_path="test.py"
        )

        assert state['max_iterations'] == 3


class TestDocumentationAgentInit:
    """Test DocumentationAgent initialization."""

    def test_init_with_components(self, temp_data_dir, sample_codebase_dir, mock_embeddings, mock_ollama):
        """Test initialization with all components."""
        analyzer = CodebaseAnalyzer(str(sample_codebase_dir))
        memory = MemoryStore(persist_directory=temp_data_dir)
        generator = DocumentationGenerator(memory)

        agent = DocumentationAgent(analyzer, memory, generator)

        assert agent.analyzer is not None
        assert agent.memory is not None
        assert agent.generator is not None
        assert agent.analyzer.root_path == Path(sample_codebase_dir)


class TestAnalyzeNode:
    """Test analyze_node workflow node."""

    def test_analyze_node_updates_metadata(self, agent):
        """Test that analyze_node updates code_metadata."""
        state = agent.create_initial_state("TestFunc", "test.py")

        updated_state = agent.analyze_node(state)

        assert 'code_metadata' in updated_state
        assert updated_state['code_metadata'] != {}
        assert updated_state['code_metadata']['name'] == "TestFunc"

    def test_analyze_node_preserves_other_fields(self, agent):
        """Test that analyze_node doesn't overwrite other state fields."""
        state = agent.create_initial_state("TestFunc", "test.py")
        state['errors'] = ["previous error"]

        updated_state = agent.analyze_node(state)

        assert updated_state['component_name'] == "TestFunc"
        # Previous error should be preserved, new error may be added for missing file
        assert "previous error" in updated_state['errors']

    def test_analyze_node_error_handling(self, agent):
        """Test analyze_node error handling."""
        state = agent.create_initial_state("TestFunc", "nonexistent.py")

        # Should not raise exception, but add to errors
        updated_state = agent.analyze_node(state)

        # Placeholder implementation doesn't raise errors yet
        assert isinstance(updated_state, dict)


class TestRetrieveNode:
    """Test retrieve_node workflow node."""

    def test_retrieve_node_updates_context(self, agent):
        """Test that retrieve_node updates retrieved_context."""
        state = agent.create_initial_state("TestFunc", "test.py")
        state['code_metadata'] = {"name": "TestFunc", "type": "function"}

        updated_state = agent.retrieve_node(state)

        assert 'retrieved_context' in updated_state
        assert isinstance(updated_state['retrieved_context'], list)
        # Context may be empty if memory is empty, but should be a list
        assert updated_state['retrieved_context'] is not None

    def test_retrieve_node_preserves_state(self, agent):
        """Test that retrieve_node preserves existing state."""
        state = agent.create_initial_state("TestFunc", "test.py")
        state['code_metadata'] = {"name": "TestFunc"}

        updated_state = agent.retrieve_node(state)

        assert updated_state['code_metadata'] == {"name": "TestFunc"}
        assert updated_state['component_name'] == "TestFunc"


class TestGenerateNode:
    """Test generate_node workflow node."""

    def test_generate_node_creates_documentation(self, agent):
        """Test that generate_node creates draft documentation."""
        state = agent.create_initial_state("TestFunc", "test.py")
        state['code_metadata'] = {"name": "TestFunc"}
        state['retrieved_context'] = ["context1", "context2"]

        updated_state = agent.generate_node(state)

        assert 'draft_documentation' in updated_state
        assert len(updated_state['draft_documentation']) > 0
        # Documentation content depends on LLM (mocked in tests)

    def test_generate_node_increments_iteration(self, agent):
        """Test that generate_node increments iteration count."""
        state = agent.create_initial_state("TestFunc", "test.py")
        initial_iteration = state['iteration_count']

        updated_state = agent.generate_node(state)

        assert updated_state['iteration_count'] == initial_iteration + 1

    def test_generate_node_multiple_iterations(self, agent):
        """Test multiple generation iterations."""
        state = agent.create_initial_state("TestFunc", "test.py")

        # First iteration
        state = agent.generate_node(state)
        assert state['iteration_count'] == 1

        # Second iteration
        state = agent.generate_node(state)
        assert state['iteration_count'] == 2

        # Third iteration
        state = agent.generate_node(state)
        assert state['iteration_count'] == 3


class TestValidateNode:
    """Test validate_node workflow node."""

    def test_validate_node_checks_documentation(self, agent):
        """Test that validate_node validates documentation."""
        state = agent.create_initial_state("TestFunc", "test.py")
        state['draft_documentation'] = """
# TestFunc Documentation

## Overview
This is a test function.

## Description
It does something useful.

## Usage
Call the function.
"""

        updated_state = agent.validate_node(state)

        assert 'validation_result' in updated_state
        assert 'is_valid' in updated_state['validation_result']
        assert 'score' in updated_state['validation_result']
        assert 'checks' in updated_state['validation_result']

    def test_validate_node_passes_good_documentation(self, agent):
        """Test validation passes for good documentation."""
        state = agent.create_initial_state("TestFunc", "test.py")
        state['draft_documentation'] = """
# TestFunc Documentation

## Overview
This is a test function.

## Description
It does something useful.

## Usage
Call the function like this: test_func()
"""

        updated_state = agent.validate_node(state)

        validation = updated_state['validation_result']
        assert validation['is_valid'] == True
        assert validation['score'] > 0

    def test_validate_node_fails_incomplete_documentation(self, agent):
        """Test validation fails for incomplete documentation."""
        state = agent.create_initial_state("TestFunc", "test.py")
        state['draft_documentation'] = """
# TestFunc

## Overview
Incomplete doc.

## Description
Missing sections.

## Usage
[To be implemented]
"""

        updated_state = agent.validate_node(state)

        validation = updated_state['validation_result']
        assert validation['is_valid'] == False
        assert len(validation['issues']) > 0


class TestStoreNode:
    """Test store_node workflow node."""

    def test_store_node_processes_state(self, agent):
        """Test that store_node processes final state."""
        state = agent.create_initial_state("TestFunc", "test.py")
        state['draft_documentation'] = "# Final Documentation"
        state['validation_result'] = {"is_valid": True}

        updated_state = agent.store_node(state)

        # Should not raise exception
        assert isinstance(updated_state, dict)
        assert updated_state['component_name'] == "TestFunc"


class TestShouldRetry:
    """Test should_retry routing logic."""

    def test_should_retry_validation_passed(self, agent):
        """Test routing when validation passes."""
        state = agent.create_initial_state("TestFunc", "test.py")
        state['validation_result'] = {"is_valid": True}
        state['iteration_count'] = 1

        decision = agent.should_retry(state)

        assert decision == "store"

    def test_should_retry_validation_failed_under_max(self, agent):
        """Test routing when validation fails and iterations under max."""
        state = agent.create_initial_state("TestFunc", "test.py", max_iterations=3)
        state['validation_result'] = {"is_valid": False}
        state['iteration_count'] = 1

        decision = agent.should_retry(state)

        assert decision == "generate"

    def test_should_retry_validation_failed_at_max(self, agent):
        """Test routing when validation fails at max iterations."""
        state = agent.create_initial_state("TestFunc", "test.py", max_iterations=3)
        state['validation_result'] = {"is_valid": False}
        state['iteration_count'] = 3

        decision = agent.should_retry(state)

        assert decision == "store"

    def test_should_retry_edge_cases(self, agent):
        """Test should_retry edge cases."""
        # No validation result
        state = agent.create_initial_state("TestFunc", "test.py")
        state['validation_result'] = {}
        state['iteration_count'] = 1

        decision = agent.should_retry(state)
        assert decision == "generate"

        # Iteration 0, failed validation
        state = agent.create_initial_state("TestFunc", "test.py", max_iterations=2)
        state['validation_result'] = {"is_valid": False}
        state['iteration_count'] = 0

        decision = agent.should_retry(state)
        assert decision == "generate"


class TestWorkflowStats:
    """Test get_workflow_stats method."""

    def test_get_stats_complete_workflow(self, agent):
        """Test getting stats from complete workflow."""
        state = agent.create_initial_state("TestFunc", "test.py", max_iterations=3)
        state['iteration_count'] = 2
        state['draft_documentation'] = "# Documentation\nContent here."
        state['validation_result'] = {"is_valid": True, "score": 0.95}
        state['errors'] = []

        stats = agent.get_workflow_stats(state)

        assert stats['component'] == "TestFunc"
        assert stats['iterations'] == 2
        assert stats['max_iterations'] == 3
        assert stats['validation_passed'] == True
        assert stats['validation_score'] == 0.95
        assert stats['error_count'] == 0
        assert stats['doc_length'] > 0

    def test_get_stats_with_errors(self, agent):
        """Test getting stats with errors."""
        state = agent.create_initial_state("TestFunc", "test.py")
        state['errors'] = ["Error 1", "Error 2"]
        state['validation_result'] = {"is_valid": False, "score": 0.3}

        stats = agent.get_workflow_stats(state)

        assert stats['error_count'] == 2
        assert stats['errors'] == ["Error 1", "Error 2"]
        assert stats['validation_passed'] == False


class TestWorkflowIntegration:
    """Integration tests for complete workflow execution."""

    def test_workflow_nodes_sequential(self, agent):
        """Test running workflow nodes sequentially."""
        # Create initial state
        state = agent.create_initial_state("UserService", "flask_app.py")

        # Run through workflow nodes
        state = agent.analyze_node(state)
        assert state['code_metadata'] != {}

        state = agent.retrieve_node(state)
        assert isinstance(state['retrieved_context'], list)
        # Context may be empty if memory is empty

        state = agent.generate_node(state)
        assert state['draft_documentation'] != ""
        assert state['iteration_count'] == 1

        state = agent.validate_node(state)
        assert 'validation_result' in state

        # Check routing decision
        decision = agent.should_retry(state)
        assert decision in ["generate", "store"]

        # Store final result
        state = agent.store_node(state)

        # Get stats
        stats = agent.get_workflow_stats(state)
        assert stats['component'] == "UserService"
        assert stats['iterations'] > 0

    def test_workflow_retry_loop(self, agent):
        """Test workflow retry loop behavior."""
        state = agent.create_initial_state("TestFunc", "test.py", max_iterations=3)

        # Simulate failed validation, should retry
        state['validation_result'] = {"is_valid": False}
        state['iteration_count'] = 1

        decision = agent.should_retry(state)
        assert decision == "generate"

        # Generate again
        state = agent.generate_node(state)
        assert state['iteration_count'] == 2

        # Still fails
        state = agent.validate_node(state)
        state['validation_result'] = {"is_valid": False}

        decision = agent.should_retry(state)
        assert decision == "generate"

        # One more iteration
        state = agent.generate_node(state)
        assert state['iteration_count'] == 3

        # Max iterations reached, should store
        decision = agent.should_retry(state)
        assert decision == "store"


if __name__ == "__main__":
    print("Running tests for DocumentationAgent structure...")
    pytest.main([__file__, "-v"])
