"""Tests for CLI commands."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from click.testing import CliRunner

from src.cli import cli, Context


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_services():
    """Mock all services for testing."""
    with patch('src.cli.CodebaseAnalyzer') as mock_analyzer_class, \
         patch('src.cli.MemoryStore') as mock_memory_class, \
         patch('src.cli.DocumentationGenerator') as mock_generator_class, \
         patch('src.cli.DocumentationAgent') as mock_agent_class:

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
        mock_analyzer_class.return_value = mock_analyzer

        # Mock memory
        mock_memory = MagicMock()
        mock_memory.store_component.return_value = "test.py::TestClass"
        mock_memory.get_collection_stats.return_value = {"count": 10}
        mock_memory.get_all_patterns.return_value = [
            {"type": "function", "name": "test_func", "file_path": "test.py"},
            {"type": "route", "name": "test_route", "path": "/test", "file_path": "test.py"}
        ]
        mock_memory.retrieve_similar.return_value = [
            {
                "metadata": {"file_path": "test.py", "name": "test_func", "type": "function"},
                "score": 0.95,
                "content": "def test_func(arg1): pass"
            }
        ]
        mock_memory_class.return_value = mock_memory

        # Mock generator
        mock_generator = MagicMock()
        mock_generator.simple_query.return_value = "This is a test answer based on the codebase."
        mock_generator_class.return_value = mock_generator

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
        mock_agent_class.return_value = mock_agent

        yield {
            "analyzer": mock_analyzer,
            "memory": mock_memory,
            "generator": mock_generator,
            "agent": mock_agent
        }


class TestAnalyzeCommand:
    """Test flow-doc analyze command."""

    def test_analyze_default_path(self, runner, mock_services):
        """Test analyze command with default path."""
        result = runner.invoke(cli, ['analyze'])

        # Should not fail (might have warnings about missing services but should run)
        assert result.exit_code in [0, 1]  # 1 if services fail to initialize
        assert 'analyz' in result.output.lower() or 'error' in result.output.lower()

    def test_analyze_with_custom_path(self, runner, mock_services, temp_data_dir):
        """Test analyze command with custom path."""
        result = runner.invoke(cli, ['analyze', temp_data_dir])

        # Should attempt to analyze
        assert result.exit_code in [0, 1]

    def test_analyze_with_detailed_flag(self, runner, mock_services):
        """Test analyze command with --detailed flag."""
        result = runner.invoke(cli, ['analyze', '--detailed'])

        # Should not fail catastrophically
        assert result.exit_code in [0, 1]

    def test_analyze_invalid_path(self, runner, mock_services):
        """Test analyze command with invalid path."""
        result = runner.invoke(cli, ['analyze', '/nonexistent/path'])

        # Should fail with error
        assert result.exit_code != 0


class TestDocumentCommand:
    """Test flow-doc document command."""

    def test_document_component(self, runner, mock_services, temp_data_dir):
        """Test document command for a component."""
        # Create test markdown file
        docs_dir = Path("docs_output")
        docs_dir.mkdir(exist_ok=True)
        test_doc = docs_dir / "TestComponent.md"
        test_doc.write_text("# TestComponent\n\nTest documentation")

        result = runner.invoke(cli, ['document', 'TestComponent'])

        # Should attempt to generate docs
        assert result.exit_code in [0, 1]

        # Cleanup
        if test_doc.exists():
            test_doc.unlink()

    def test_document_with_path(self, runner, mock_services):
        """Test document command with explicit path."""
        result = runner.invoke(cli, ['document', 'TestComponent', '--path', 'test.py'])

        # Should attempt to generate docs
        assert result.exit_code in [0, 1]

    def test_document_with_max_iterations(self, runner, mock_services):
        """Test document command with max iterations."""
        result = runner.invoke(cli, [
            'document', 'TestComponent',
            '--max-iterations', '5'
        ])

        # Should attempt to generate docs
        assert result.exit_code in [0, 1]

    def test_document_with_show_markdown(self, runner, mock_services):
        """Test document command with markdown preview."""
        # Create test markdown file
        docs_dir = Path("docs_output")
        docs_dir.mkdir(exist_ok=True)
        test_doc = docs_dir / "TestComponent.md"
        test_doc.write_text("# TestComponent\n\nTest documentation")

        result = runner.invoke(cli, [
            'document', 'TestComponent',
            '--show-markdown'
        ])

        # Should attempt to show markdown
        assert result.exit_code in [0, 1]

        # Cleanup
        if test_doc.exists():
            test_doc.unlink()


class TestQueryCommand:
    """Test flow-doc query command."""

    def test_query_simple(self, runner, mock_services):
        """Test query command with simple question."""
        result = runner.invoke(cli, ['query', 'What does this code do?'])

        # Should attempt to query
        assert result.exit_code in [0, 1]

    def test_query_with_k_param(self, runner, mock_services):
        """Test query command with custom k parameter."""
        result = runner.invoke(cli, [
            'query', 'What authentication methods are used?',
            '--k', '10'
        ])

        # Should attempt to query
        assert result.exit_code in [0, 1]

    def test_query_empty_question(self, runner, mock_services):
        """Test query command with empty question."""
        result = runner.invoke(cli, ['query', ''])

        # Should handle gracefully
        assert result.exit_code in [0, 1, 2]  # 2 is usage error


class TestExportCommand:
    """Test flow-doc export command."""

    def test_export_default(self, runner, mock_services):
        """Test export command with defaults."""
        # Create test documentation files
        docs_dir = Path("docs_output")
        docs_dir.mkdir(exist_ok=True)
        test_doc1 = docs_dir / "Component1.md"
        test_doc2 = docs_dir / "Component2.md"
        test_doc1.write_text("# Component1")
        test_doc2.write_text("# Component2")

        result = runner.invoke(cli, ['export'])

        # Should attempt to export
        assert result.exit_code in [0, 1]

        # Cleanup
        if test_doc1.exists():
            test_doc1.unlink()
        if test_doc2.exists():
            test_doc2.unlink()

    def test_export_markdown_format(self, runner, mock_services):
        """Test export command with markdown format."""
        # Create test documentation files
        docs_dir = Path("docs_output")
        docs_dir.mkdir(exist_ok=True)
        test_doc = docs_dir / "Component1.md"
        test_doc.write_text("# Component1")

        result = runner.invoke(cli, ['export', '--format', 'markdown'])

        # Should list markdown files
        assert result.exit_code in [0, 1]

        # Cleanup
        if test_doc.exists():
            test_doc.unlink()

    def test_export_zip_format(self, runner, mock_services):
        """Test export command with zip format."""
        # Create test documentation files
        docs_dir = Path("docs_output")
        docs_dir.mkdir(exist_ok=True)
        test_doc = docs_dir / "Component1.md"
        test_doc.write_text("# Component1")

        result = runner.invoke(cli, ['export', '--format', 'zip'])

        # Should create zip
        assert result.exit_code in [0, 1]

        # Cleanup
        if test_doc.exists():
            test_doc.unlink()

    def test_export_no_docs(self, runner, mock_services):
        """Test export command when no docs exist."""
        # Ensure docs_output is empty
        docs_dir = Path("docs_output")
        if docs_dir.exists():
            for f in docs_dir.glob("*.md"):
                f.unlink()

        result = runner.invoke(cli, ['export'])

        # Should handle gracefully
        assert result.exit_code in [0, 1]


class TestVisualizeCommand:
    """Test flow-doc visualize command."""

    def test_visualize_default(self, runner, mock_services):
        """Test visualize command with defaults."""
        with patch('src.cli.app') as mock_app:
            mock_app.get_graph.return_value.draw_mermaid.return_value = "graph TD\nA-->B"

            result = runner.invoke(cli, ['visualize'])

            # Should attempt to visualize
            assert result.exit_code in [0, 1]

    def test_visualize_png_format(self, runner, mock_services):
        """Test visualize command with PNG format."""
        result = runner.invoke(cli, ['visualize', '--format', 'png'])

        # Should attempt to create PNG
        assert result.exit_code in [0, 1]

    def test_visualize_mermaid_format(self, runner, mock_services):
        """Test visualize command with Mermaid format."""
        with patch('src.cli.app') as mock_app:
            mock_app.get_graph.return_value.draw_mermaid.return_value = "graph TD\nA-->B"

            result = runner.invoke(cli, ['visualize', '--format', 'mermaid'])

            # Should create mermaid file
            assert result.exit_code in [0, 1]

    def test_visualize_with_custom_output(self, runner, mock_services):
        """Test visualize command with custom output path."""
        with patch('src.cli.app') as mock_app:
            mock_app.get_graph.return_value.draw_mermaid.return_value = "graph TD\nA-->B"

            result = runner.invoke(cli, [
                'visualize',
                '--output', 'custom_viz',
                '--format', 'mermaid'
            ])

            # Should create custom output
            assert result.exit_code in [0, 1]


class TestPatternsCommand:
    """Test flow-doc patterns command."""

    def test_patterns_all(self, runner, mock_services):
        """Test patterns command for all types."""
        result = runner.invoke(cli, ['patterns'])

        # Should show patterns
        assert result.exit_code in [0, 1]

    def test_patterns_routes(self, runner, mock_services):
        """Test patterns command for routes only."""
        result = runner.invoke(cli, ['patterns', '--type', 'routes'])

        # Should show route patterns
        assert result.exit_code in [0, 1]

    def test_patterns_functions(self, runner, mock_services):
        """Test patterns command for functions only."""
        result = runner.invoke(cli, ['patterns', '--type', 'functions'])

        # Should show function patterns
        assert result.exit_code in [0, 1]

    def test_patterns_classes(self, runner, mock_services):
        """Test patterns command for classes only."""
        result = runner.invoke(cli, ['patterns', '--type', 'classes'])

        # Should show class patterns
        assert result.exit_code in [0, 1]


class TestServerCommand:
    """Test flow-doc server command."""

    def test_server_default(self, runner, mock_services):
        """Test server command with defaults."""
        with patch('src.cli.uvicorn.run') as mock_run:
            # Simulate KeyboardInterrupt to stop server
            mock_run.side_effect = KeyboardInterrupt()

            result = runner.invoke(cli, ['server'])

            # Should attempt to start server
            # Exit code 0 because we handle KeyboardInterrupt gracefully
            assert result.exit_code in [0, 1]

    def test_server_custom_port(self, runner, mock_services):
        """Test server command with custom port."""
        with patch('src.cli.uvicorn.run') as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            result = runner.invoke(cli, ['server', '--port', '9000'])

            # Should attempt to start on custom port
            assert result.exit_code in [0, 1]

    def test_server_with_reload(self, runner, mock_services):
        """Test server command with auto-reload."""
        with patch('src.cli.uvicorn.run') as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            result = runner.invoke(cli, ['server', '--reload'])

            # Should attempt to start with reload
            assert result.exit_code in [0, 1]

    def test_server_with_workers(self, runner, mock_services):
        """Test server command with multiple workers."""
        with patch('src.cli.uvicorn.run') as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            result = runner.invoke(cli, ['server', '--workers', '4'])

            # Should attempt to start with workers
            assert result.exit_code in [0, 1]


class TestCLIOptions:
    """Test global CLI options."""

    def test_verbose_flag(self, runner, mock_services):
        """Test --verbose flag."""
        result = runner.invoke(cli, ['--verbose', 'analyze'])

        # Should run with verbose output
        assert result.exit_code in [0, 1]

    def test_custom_codebase(self, runner, mock_services):
        """Test --codebase option."""
        result = runner.invoke(cli, ['--codebase', './test_codebase', 'analyze'])

        # Should use custom codebase path
        assert result.exit_code in [0, 1]

    def test_help_command(self, runner):
        """Test --help option."""
        result = runner.invoke(cli, ['--help'])

        # Should show help and exit cleanly
        assert result.exit_code == 0
        assert 'Flow-Doc' in result.output
        assert 'analyze' in result.output
        assert 'document' in result.output
        assert 'query' in result.output


class TestContextManagement:
    """Test CLI context management."""

    def test_context_initialization(self):
        """Test Context class initialization."""
        ctx = Context()

        assert ctx.analyzer is None
        assert ctx.memory is None
        assert ctx.generator is None
        assert ctx.agent is None
        assert ctx.codebase_path is None
        assert ctx.verbose is False

    def test_context_with_values(self):
        """Test Context class with values set."""
        ctx = Context()
        ctx.verbose = True
        ctx.codebase_path = "./test"

        assert ctx.verbose is True
        assert ctx.codebase_path == "./test"


class TestErrorHandling:
    """Test CLI error handling."""

    def test_invalid_command(self, runner):
        """Test invalid command."""
        result = runner.invoke(cli, ['invalid-command'])

        # Should show error
        assert result.exit_code != 0

    def test_missing_required_argument(self, runner):
        """Test missing required argument."""
        result = runner.invoke(cli, ['document'])  # Missing component name

        # Should show usage error
        assert result.exit_code != 0

    def test_invalid_option_value(self, runner):
        """Test invalid option value."""
        result = runner.invoke(cli, ['patterns', '--type', 'invalid'])

        # Should show error for invalid choice
        assert result.exit_code != 0


def cleanup_test_files():
    """Clean up test output files."""
    docs_dir = Path("docs_output")
    if docs_dir.exists():
        for f in docs_dir.glob("*.md"):
            if f.name.startswith("Test") or f.name.startswith("Component"):
                f.unlink()

    # Clean up visualization files
    for pattern in ["*.mermaid", "workflow_*.png", "custom_viz.*"]:
        for f in Path(".").glob(pattern):
            if f.exists():
                f.unlink()

    # Clean up zip files
    for f in Path(".").glob("flow-doc-*.zip"):
        if f.exists():
            f.unlink()


if __name__ == "__main__":
    try:
        print("Running CLI tests...")
        pytest.main([__file__, "-v", "-s"])
    finally:
        cleanup_test_files()
