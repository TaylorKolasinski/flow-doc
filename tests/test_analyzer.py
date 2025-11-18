"""Tests for CodebaseAnalyzer functionality."""

import pytest
import tempfile
from pathlib import Path
from src.core.analyzer import CodebaseAnalyzer


class TestCodebaseAnalyzer:
    """Test suite for CodebaseAnalyzer class."""

    def test_init_valid_directory(self):
        """Test initialization with valid directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CodebaseAnalyzer(tmpdir)
            assert analyzer.root_path == Path(tmpdir)

    def test_init_invalid_path(self):
        """Test initialization with non-existent path."""
        with pytest.raises(ValueError, match="Path does not exist"):
            CodebaseAnalyzer("/this/path/does/not/exist")

    def test_init_file_not_directory(self):
        """Test initialization with file instead of directory."""
        with tempfile.NamedTemporaryFile() as tmpfile:
            with pytest.raises(NotADirectoryError, match="not a directory"):
                CodebaseAnalyzer(tmpfile.name)

    def test_scan_empty_directory(self):
        """Test scanning an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CodebaseAnalyzer(tmpdir)
            files = analyzer.scan_directory()
            assert files == []

    def test_scan_directory_with_python_files(self):
        """Test scanning directory with Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test Python files
            (tmppath / "test1.py").write_text("print('hello')")
            (tmppath / "test2.py").write_text("print('world')")

            # Create subdirectory with Python file
            subdir = tmppath / "subdir"
            subdir.mkdir()
            (subdir / "test3.py").write_text("print('nested')")

            analyzer = CodebaseAnalyzer(tmpdir)
            files = analyzer.scan_directory()

            assert len(files) == 3
            assert all(f["name"].endswith(".py") for f in files)
            assert all("path" in f for f in files)
            assert all("size" in f for f in files)
            assert all("modified" in f for f in files)

    def test_scan_skips_non_python_files(self):
        """Test that non-Python files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create various file types
            (tmppath / "test.py").write_text("print('python')")
            (tmppath / "readme.md").write_text("# Readme")
            (tmppath / "data.json").write_text("{}")
            (tmppath / "script.sh").write_text("#!/bin/bash")

            analyzer = CodebaseAnalyzer(tmpdir)
            files = analyzer.scan_directory()

            assert len(files) == 1
            assert files[0]["name"] == "test.py"

    def test_scan_skips_excluded_directories(self):
        """Test that excluded directories are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create Python files in excluded directories
            for dirname in ["__pycache__", ".git", "venv", "node_modules"]:
                dirpath = tmppath / dirname
                dirpath.mkdir()
                (dirpath / "test.py").write_text("print('skip me')")

            # Create one file in root
            (tmppath / "main.py").write_text("print('keep me')")

            analyzer = CodebaseAnalyzer(tmpdir)
            files = analyzer.scan_directory()

            assert len(files) == 1
            assert files[0]["name"] == "main.py"

    def test_file_metadata_structure(self):
        """Test that file metadata has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            test_file = tmppath / "test.py"
            test_content = "print('test')"
            test_file.write_text(test_content)

            analyzer = CodebaseAnalyzer(tmpdir)
            files = analyzer.scan_directory()

            assert len(files) == 1
            metadata = files[0]

            # Check required keys
            assert "path" in metadata
            assert "name" in metadata
            assert "size" in metadata
            assert "modified" in metadata

            # Check values
            assert metadata["name"] == "test.py"
            assert metadata["path"] == "test.py"
            assert metadata["size"] == len(test_content)
            assert isinstance(metadata["modified"], float)

    def test_get_summary(self):
        """Test get_summary method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create files of different sizes
            (tmppath / "small.py").write_text("x=1")
            (tmppath / "large.py").write_text("x=1\n" * 100)

            analyzer = CodebaseAnalyzer(tmpdir)
            summary = analyzer.get_summary()

            assert summary["total_files"] == 2
            assert summary["total_size"] > 0
            assert summary["average_size"] > 0
            assert summary["largest_file"]["path"] == "large.py"
            assert summary["smallest_file"]["path"] == "small.py"

    def test_get_summary_empty_directory(self):
        """Test get_summary on empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CodebaseAnalyzer(tmpdir)
            summary = analyzer.get_summary()

            assert summary["total_files"] == 0
            assert summary["total_size"] == 0
            assert summary["average_size"] == 0
            assert summary["largest_file"] is None
            assert summary["smallest_file"] is None


class TestCodebaseAnalyzerWithRealProject:
    """Test analyzer on the actual project structure."""

    def test_scan_current_project(self):
        """Test scanning the current Flow-Doc project."""
        # Use the src directory as test target
        project_root = Path(__file__).parent.parent
        src_dir = project_root / "src"

        if not src_dir.exists():
            pytest.skip("src directory not found")

        analyzer = CodebaseAnalyzer(str(src_dir))
        files = analyzer.scan_directory()

        # Verify at least some Python files are found
        assert len(files) > 0, "Should find at least one Python file in src/"

        print(f"\n{'='*60}")
        print(f"Found {len(files)} Python files in src/:")
        print(f"{'='*60}")

        for file_info in sorted(files, key=lambda x: x["path"]):
            size_kb = file_info["size"] / 1024
            print(f"  {file_info['path']:<40} ({size_kb:.2f} KB)")

        print(f"{'='*60}\n")

        # Verify structure of returned data
        for f in files:
            assert "path" in f
            assert "name" in f
            assert "size" in f
            assert "modified" in f
            assert f["name"].endswith(".py")

    def test_scan_project_summary(self):
        """Test getting summary of the current project."""
        project_root = Path(__file__).parent.parent
        src_dir = project_root / "src"

        if not src_dir.exists():
            pytest.skip("src directory not found")

        analyzer = CodebaseAnalyzer(str(src_dir))
        summary = analyzer.get_summary()

        print(f"\n{'='*60}")
        print("Project Summary:")
        print(f"{'='*60}")
        print(f"  Total files: {summary['total_files']}")
        print(f"  Total size: {summary['total_size'] / 1024:.2f} KB")
        print(f"  Average size: {summary['average_size'] / 1024:.2f} KB")

        if summary["largest_file"]:
            print(f"  Largest file: {summary['largest_file']['path']} "
                  f"({summary['largest_file']['size'] / 1024:.2f} KB)")

        if summary["smallest_file"]:
            print(f"  Smallest file: {summary['smallest_file']['path']} "
                  f"({summary['smallest_file']['size'] / 1024:.2f} KB)")

        print(f"{'='*60}\n")

        assert summary["total_files"] > 0


class TestASTParser:
    """Test suite for AST parsing functionality."""

    def test_parse_simple_file(self):
        """Test parsing a simple Python file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            test_file = tmppath / "simple.py"
            test_file.write_text("""
def hello(name):
    \"\"\"Say hello.\"\"\"
    return f"Hello, {name}"
""")

            analyzer = CodebaseAnalyzer(tmpdir)
            result = analyzer.parse_file("simple.py")

            assert result is not None
            assert result["path"] == "simple.py"
            assert len(result["functions"]) == 1
            assert result["functions"][0]["name"] == "hello"
            assert result["functions"][0]["params"] == ["name"]
            assert "Say hello" in result["functions"][0]["docstring"]

    def test_parse_file_with_imports(self):
        """Test parsing file with imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            test_file = tmppath / "imports.py"
            test_file.write_text("""
import os
import sys
from pathlib import Path
from typing import List, Dict

def test():
    pass
""")

            analyzer = CodebaseAnalyzer(tmpdir)
            result = analyzer.parse_file("imports.py")

            assert result is not None
            assert "os" in result["imports"]
            assert "sys" in result["imports"]
            assert "pathlib" in result["imports"]
            assert "typing" in result["imports"]

    def test_parse_file_with_class(self):
        """Test parsing file with classes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            test_file = tmppath / "classes.py"
            test_file.write_text("""
class UserService:
    \"\"\"Service for users.\"\"\"

    def __init__(self):
        pass

    def get_user(self, user_id):
        pass

    def create_user(self, name, email):
        \"\"\"Create a user.\"\"\"
        pass
""")

            analyzer = CodebaseAnalyzer(tmpdir)
            result = analyzer.parse_file("classes.py")

            assert result is not None
            assert len(result["classes"]) == 1
            assert result["classes"][0]["name"] == "UserService"
            assert "Service for users" in result["classes"][0]["docstring"]
            assert "__init__" in result["classes"][0]["methods"]
            assert "get_user" in result["classes"][0]["methods"]
            assert "create_user" in result["classes"][0]["methods"]

    def test_parse_file_with_decorators(self):
        """Test parsing file with decorators."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            test_file = tmppath / "decorators.py"
            test_file.write_text("""
class MyClass:
    @staticmethod
    def static_method():
        pass

    @classmethod
    def class_method(cls):
        pass

@property
def my_property():
    pass
""")

            analyzer = CodebaseAnalyzer(tmpdir)
            result = analyzer.parse_file("decorators.py")

            assert result is not None
            assert "property" in result["functions"][0]["decorators"]

    def test_parse_invalid_file(self):
        """Test parsing invalid Python file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            test_file = tmppath / "invalid.py"
            test_file.write_text("this is not valid python ][{")

            analyzer = CodebaseAnalyzer(tmpdir)
            result = analyzer.parse_file("invalid.py")

            assert result is None

    def test_parse_nonexistent_file(self):
        """Test parsing non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CodebaseAnalyzer(tmpdir)
            result = analyzer.parse_file("does_not_exist.py")

            assert result is None

    def test_parse_flask_routes(self):
        """Test parsing Flask routes from sample codebase."""
        project_root = Path(__file__).parent.parent
        sample_file = project_root / "sample_codebase" / "flask_app.py"

        if not sample_file.exists():
            pytest.skip("sample_codebase/flask_app.py not found")

        analyzer = CodebaseAnalyzer(str(project_root / "sample_codebase"))
        result = analyzer.parse_file("flask_app.py")

        assert result is not None

        print(f"\n{'='*60}")
        print("Flask App Analysis:")
        print(f"{'='*60}")
        print(f"\nFile: {result['path']}")
        print(f"\nModule docstring: {result['module_docstring']}")

        print(f"\nImports ({len(result['imports'])}):")
        for imp in result["imports"][:10]:
            print(f"  - {imp}")

        print(f"\nClasses ({len(result['classes'])}):")
        for cls in result["classes"]:
            print(f"  - {cls['name']}")
            print(f"    Methods: {', '.join(cls['methods'])}")
            if cls['docstring']:
                print(f"    Doc: {cls['docstring'][:60]}...")

        print(f"\nFunctions ({len(result['functions'])}):")
        for func in result["functions"]:
            params_str = ', '.join(func['params'])
            decorators_str = ', '.join(func['decorators']) if func['decorators'] else 'None'
            print(f"  - {func['name']}({params_str})")
            print(f"    Decorators: {decorators_str}")
            if func['docstring']:
                print(f"    Doc: {func['docstring'][:60]}...")

        print(f"\nRoutes ({len(result['routes'])}):")
        for route in result["routes"]:
            methods = ', '.join(route['methods']) if route['methods'] else 'ANY'
            print(f"  - {methods:6} {route['path']:30} -> {route['handler']}")

        print(f"{'='*60}\n")

        # Assertions
        assert len(result["classes"]) >= 1
        assert any(cls["name"] == "UserService" for cls in result["classes"])
        assert len(result["routes"]) >= 3
        assert any(route["path"] == "/" for route in result["routes"])

    def test_parse_fastapi_routes(self):
        """Test parsing FastAPI routes from sample codebase."""
        project_root = Path(__file__).parent.parent
        sample_file = project_root / "sample_codebase" / "fastapi_app.py"

        if not sample_file.exists():
            pytest.skip("sample_codebase/fastapi_app.py not found")

        analyzer = CodebaseAnalyzer(str(project_root / "sample_codebase"))
        result = analyzer.parse_file("fastapi_app.py")

        assert result is not None

        print(f"\n{'='*60}")
        print("FastAPI App Analysis:")
        print(f"{'='*60}")
        print(f"\nFile: {result['path']}")
        print(f"\nModule docstring: {result['module_docstring']}")

        print(f"\nImports ({len(result['imports'])}):")
        for imp in result["imports"][:10]:
            print(f"  - {imp}")

        print(f"\nClasses ({len(result['classes'])}):")
        for cls in result["classes"]:
            print(f"  - {cls['name']}")
            print(f"    Methods: {', '.join(cls['methods'][:5])}")
            if cls['docstring']:
                print(f"    Doc: {cls['docstring'][:60]}...")

        print(f"\nFunctions ({len(result['functions'])}):")
        for func in result["functions"]:
            params_str = ', '.join(func['params'][:3])
            if len(func['params']) > 3:
                params_str += "..."
            decorators_str = ', '.join(func['decorators']) if func['decorators'] else 'None'
            print(f"  - {func['name']}({params_str})")
            print(f"    Decorators: {decorators_str}")

        print(f"\nRoutes ({len(result['routes'])}):")
        for route in result["routes"]:
            methods = ', '.join(route['methods']) if route['methods'] else 'ANY'
            print(f"  - {methods:6} {route['path']:30} -> {route['handler']}")

        print(f"{'='*60}\n")

        # Assertions
        assert len(result["classes"]) >= 2
        assert len(result["routes"]) >= 5
        assert any(route["methods"] == ["GET"] for route in result["routes"])
        assert any(route["methods"] == ["POST"] for route in result["routes"])


if __name__ == "__main__":
    # Run the real project tests when executed directly
    print("Running CodebaseAnalyzer tests on current project...\n")

    test_suite = TestCodebaseAnalyzerWithRealProject()
    test_suite.test_scan_current_project()
    test_suite.test_scan_project_summary()

    print("\nRunning AST Parser tests...\n")
    ast_test_suite = TestASTParser()
    ast_test_suite.test_parse_flask_routes()
    ast_test_suite.test_parse_fastapi_routes()

    print("\nAll tests completed successfully!")
