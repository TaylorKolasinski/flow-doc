"""Tests for incremental analysis and change detection."""

import pytest
import tempfile
import shutil
import time
import json
from pathlib import Path
from src.core.analyzer import CodebaseAnalyzer


@pytest.fixture
def temp_codebase():
    """Create a temporary codebase for testing."""
    temp_dir = tempfile.mkdtemp()

    # Create some test files
    test_file1 = Path(temp_dir) / "test1.py"
    test_file1.write_text("""
def hello():
    '''Say hello'''
    return 'Hello'

class Greeter:
    '''A greeter class'''
    def greet(self):
        return 'Hi'
""")

    test_file2 = Path(temp_dir) / "test2.py"
    test_file2.write_text("""
def goodbye():
    '''Say goodbye'''
    return 'Goodbye'
""")

    # Wait a moment to ensure different timestamps
    time.sleep(0.1)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestHashComputation:
    """Test file hash computation."""

    def test_compute_hash_returns_sha256(self, temp_codebase):
        """Test that compute_hash returns a valid SHA-256 hash."""
        analyzer = CodebaseAnalyzer(temp_codebase)
        hash_val = analyzer.compute_hash("test1.py")

        # SHA-256 hashes are 64 characters long
        assert len(hash_val) == 64
        assert all(c in '0123456789abcdef' for c in hash_val)

    def test_compute_hash_consistent(self, temp_codebase):
        """Test that compute_hash returns same hash for same content."""
        analyzer = CodebaseAnalyzer(temp_codebase)
        hash1 = analyzer.compute_hash("test1.py")
        hash2 = analyzer.compute_hash("test1.py")

        assert hash1 == hash2

    def test_compute_hash_different_files(self, temp_codebase):
        """Test that different files have different hashes."""
        analyzer = CodebaseAnalyzer(temp_codebase)
        hash1 = analyzer.compute_hash("test1.py")
        hash2 = analyzer.compute_hash("test2.py")

        assert hash1 != hash2

    def test_compute_hash_nonexistent_file(self, temp_codebase):
        """Test that compute_hash handles nonexistent files gracefully."""
        analyzer = CodebaseAnalyzer(temp_codebase)
        hash_val = analyzer.compute_hash("nonexistent.py")

        assert hash_val == ""


class TestChangeDetection:
    """Test change detection functionality."""

    def test_detect_changes_first_run(self, temp_codebase):
        """Test that first run detects all files as new."""
        analyzer = CodebaseAnalyzer(temp_codebase)
        changes = analyzer.detect_changes()

        # All files should be detected as new
        assert len(changes) == 2
        assert all(c['status'] == 'new' for c in changes)

        # Check that file paths are correct
        paths = {c['path'] for c in changes}
        assert 'test1.py' in paths
        assert 'test2.py' in paths

    def test_detect_changes_no_changes(self, temp_codebase):
        """Test that no changes are detected when files haven't changed."""
        analyzer = CodebaseAnalyzer(temp_codebase)

        # First run - establishes baseline
        changes1 = analyzer.detect_changes()
        assert len(changes1) == 2

        # Second run - should detect no changes
        analyzer2 = CodebaseAnalyzer(temp_codebase)
        changes2 = analyzer2.detect_changes()
        assert len(changes2) == 0

    def test_detect_changes_modified_file(self, temp_codebase):
        """Test detection of modified files."""
        analyzer = CodebaseAnalyzer(temp_codebase)

        # Establish baseline
        analyzer.detect_changes()

        # Modify a file
        time.sleep(0.1)  # Ensure different timestamp
        test_file = Path(temp_codebase) / "test1.py"
        test_file.write_text("""
def hello():
    '''Say hello - modified'''
    return 'Hello World'
""")

        # Detect changes
        analyzer2 = CodebaseAnalyzer(temp_codebase)
        changes = analyzer2.detect_changes()

        assert len(changes) == 1
        assert changes[0]['status'] == 'modified'
        assert changes[0]['path'] == 'test1.py'
        assert 'old_hash' in changes[0]
        assert 'hash' in changes[0]
        assert changes[0]['old_hash'] != changes[0]['hash']

    def test_detect_changes_new_file(self, temp_codebase):
        """Test detection of new files."""
        analyzer = CodebaseAnalyzer(temp_codebase)

        # Establish baseline
        analyzer.detect_changes()

        # Add a new file
        new_file = Path(temp_codebase) / "test3.py"
        new_file.write_text("""
def new_function():
    return 'New'
""")

        # Detect changes
        analyzer2 = CodebaseAnalyzer(temp_codebase)
        changes = analyzer2.detect_changes()

        assert len(changes) == 1
        assert changes[0]['status'] == 'new'
        assert changes[0]['path'] == 'test3.py'

    def test_detect_changes_deleted_file(self, temp_codebase):
        """Test detection of deleted files."""
        analyzer = CodebaseAnalyzer(temp_codebase)

        # Establish baseline
        analyzer.detect_changes()

        # Delete a file
        test_file = Path(temp_codebase) / "test1.py"
        test_file.unlink()

        # Detect changes
        analyzer2 = CodebaseAnalyzer(temp_codebase)
        changes = analyzer2.detect_changes()

        assert len(changes) == 1
        assert changes[0]['status'] == 'deleted'
        assert changes[0]['path'] == 'test1.py'

    def test_hash_storage_created(self, temp_codebase):
        """Test that hash storage file is created."""
        analyzer = CodebaseAnalyzer(temp_codebase)
        analyzer.detect_changes()

        assert analyzer.hash_storage_file.exists()

        # Verify JSON structure
        with open(analyzer.hash_storage_file, 'r') as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert 'test1.py' in data
        assert 'test2.py' in data

        # Verify hash entry structure
        for path, info in data.items():
            assert 'hash' in info
            assert 'last_checked' in info
            assert 'last_modified' in info


class TestChangedComponents:
    """Test changed component detection."""

    def test_get_changed_components_new_files(self, temp_codebase):
        """Test getting changed components from new files."""
        analyzer = CodebaseAnalyzer(temp_codebase)

        # First run - all files are new
        changed_components = analyzer.get_changed_components()

        # Should detect components from test1.py and test2.py
        assert len(changed_components) > 0

        # Check for specific components
        component_names = [c.split('::')[-1] for c in changed_components]
        assert 'hello' in component_names
        assert 'Greeter' in component_names
        assert 'goodbye' in component_names

    def test_get_changed_components_modified_file(self, temp_codebase):
        """Test getting changed components after file modification."""
        analyzer = CodebaseAnalyzer(temp_codebase)

        # Establish baseline
        analyzer.detect_changes()

        # Modify a file
        time.sleep(0.1)
        test_file = Path(temp_codebase) / "test1.py"
        test_file.write_text("""
def hello():
    '''Say hello - modified'''
    return 'Hello World'

def new_function():
    '''A new function'''
    return 'New'

class Greeter:
    '''A greeter class'''
    def greet(self):
        return 'Hi'

    def farewell(self):
        return 'Bye'
""")

        # Get changed components
        analyzer2 = CodebaseAnalyzer(temp_codebase)
        changed_components = analyzer2.get_changed_components()

        # Should only include components from test1.py (the modified file)
        assert len(changed_components) > 0
        assert all('test1.py' in c for c in changed_components)

        # Check that new components are detected
        component_names = [c.split('::')[-1] for c in changed_components]
        assert 'new_function' in component_names
        assert 'hello' in component_names
        assert 'Greeter' in component_names

    def test_get_changed_components_no_changes(self, temp_codebase):
        """Test that no components are returned when nothing changed."""
        analyzer = CodebaseAnalyzer(temp_codebase)

        # Establish baseline
        analyzer.detect_changes()

        # No modifications
        analyzer2 = CodebaseAnalyzer(temp_codebase)
        changed_components = analyzer2.get_changed_components()

        assert len(changed_components) == 0

    def test_component_id_format(self, temp_codebase):
        """Test that component IDs have correct format."""
        analyzer = CodebaseAnalyzer(temp_codebase)
        changed_components = analyzer.get_changed_components()

        for component_id in changed_components:
            # Should be in format "file::component"
            assert '::' in component_id
            parts = component_id.split('::')
            assert len(parts) == 2
            assert parts[0].endswith('.py')
            assert len(parts[1]) > 0


class TestIncrementalWorkflow:
    """Test incremental workflow integration."""

    def test_incremental_flag_initialization(self, temp_codebase):
        """Test that incremental flag is properly initialized."""
        from src.core.memory import MemoryStore
        from src.core.generator import DocumentationGenerator
        from src.core.agent import DocumentationAgent

        analyzer = CodebaseAnalyzer(temp_codebase)
        memory = MemoryStore()
        generator = DocumentationGenerator(memory)
        agent = DocumentationAgent(analyzer, memory, generator)

        # Run with incremental=False (should work normally)
        result = agent.run(
            component_name="hello",
            component_path="test1.py",
            max_iterations=1,
            incremental=False
        )

        assert 'skipped' in result
        assert result['skipped'] is False

    def test_incremental_skips_unchanged(self, temp_codebase):
        """Test that incremental mode skips unchanged components."""
        from src.core.memory import MemoryStore
        from src.core.generator import DocumentationGenerator
        from src.core.agent import DocumentationAgent

        analyzer = CodebaseAnalyzer(temp_codebase)

        # Establish baseline
        analyzer.detect_changes()

        # Create agent
        memory = MemoryStore()
        generator = DocumentationGenerator(memory)
        agent = DocumentationAgent(analyzer, memory, generator)

        # Run with incremental=True on unchanged component
        result = agent.run(
            component_name="hello",
            component_path="test1.py",
            max_iterations=1,
            incremental=True
        )

        # Should be skipped
        assert result.get('skipped') is True
        assert result['iterations'] == 0
        assert 'reason' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
