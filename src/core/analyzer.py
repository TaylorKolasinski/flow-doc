"""File system analyzer for Python codebases."""

import ast
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class CodebaseAnalyzer:
    """Analyzes a Python codebase by scanning directories and collecting file metadata."""

    # Directories to skip during scanning
    SKIP_DIRS = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "env",
        "ENV",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        ".nox",
        "build",
        "dist",
        "*.egg-info",
        ".eggs",
    }

    def __init__(self, root_path: str) -> None:
        """
        Initialize the codebase analyzer.

        Args:
            root_path: Root directory of the codebase to analyze

        Raises:
            ValueError: If root_path is invalid or doesn't exist
            NotADirectoryError: If root_path is not a directory
        """
        self.root_path = Path(root_path)

        if not self.root_path.exists():
            raise ValueError(f"Path does not exist: {root_path}")

        if not self.root_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {root_path}")

        # Initialize hash storage path (stored in codebase root)
        self.hash_storage_dir = self.root_path / ".flow-doc"
        self.hash_storage_file = self.hash_storage_dir / "file_hashes.json"
        self.hash_storage_dir.mkdir(exist_ok=True)

        # Track changed components
        self._changed_components: List[str] = []

        logger.info(f"Initialized CodebaseAnalyzer for: {self.root_path}")

    def _should_skip_directory(self, dir_path: Path) -> bool:
        """
        Check if a directory should be skipped during scanning.

        Args:
            dir_path: Directory path to check

        Returns:
            True if directory should be skipped, False otherwise
        """
        dir_name = dir_path.name
        return dir_name in self.SKIP_DIRS or dir_name.startswith(".")

    def _is_in_skipped_directory(self, file_path: Path) -> bool:
        """
        Check if a file is inside a directory that should be skipped.

        Args:
            file_path: Path to check

        Returns:
            True if file is in a skipped directory, False otherwise
        """
        try:
            relative_path = file_path.relative_to(self.root_path)
            # Check all parent directories
            for parent in relative_path.parents:
                if parent != Path(".") and self._should_skip_directory(self.root_path / parent):
                    return True
            return False
        except ValueError:
            # Path is not relative to root
            return False

    def _get_file_metadata(self, file_path: Path) -> Optional[Dict[str, any]]:
        """
        Extract metadata from a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            Dictionary containing file metadata, or None if error occurs
        """
        try:
            stat = file_path.stat()
            relative_path = file_path.relative_to(self.root_path)

            return {
                "path": str(relative_path),
                "name": file_path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
        except PermissionError:
            logger.warning(f"Permission denied accessing file: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error reading file metadata for {file_path}: {e}")
            return None

    def scan_directory(self) -> List[Dict[str, any]]:
        """
        Recursively scan the codebase directory for Python files.

        Returns:
            List of dictionaries containing metadata for each Python file found.
            Each dictionary contains:
                - path: Relative path from root
                - name: Filename
                - size: File size in bytes
                - modified: Last modified timestamp

        Example:
            >>> analyzer = CodebaseAnalyzer("/path/to/project")
            >>> files = analyzer.scan_directory()
            >>> print(files[0])
            {
                'path': 'src/main.py',
                'name': 'main.py',
                'size': 1024,
                'modified': 1699123456.789
            }
        """
        python_files = []
        files_scanned = 0
        files_skipped = 0

        logger.info(f"Starting scan of directory: {self.root_path}")

        try:
            for item in self.root_path.rglob("*"):
                try:
                    # Skip if item is in an excluded directory
                    if self._is_in_skipped_directory(item):
                        logger.debug(f"Skipping item in excluded directory: {item}")
                        continue

                    # Process only Python files
                    if item.is_file() and item.suffix == ".py":
                        files_scanned += 1
                        metadata = self._get_file_metadata(item)

                        if metadata:
                            python_files.append(metadata)
                            logger.debug(f"Added file: {metadata['path']}")
                        else:
                            files_skipped += 1

                except PermissionError:
                    logger.warning(f"Permission denied accessing: {item}")
                    files_skipped += 1
                    continue
                except Exception as e:
                    logger.error(f"Error processing {item}: {e}")
                    files_skipped += 1
                    continue

        except PermissionError:
            logger.error(f"Permission denied accessing root directory: {self.root_path}")
            raise
        except Exception as e:
            logger.error(f"Error during directory scan: {e}")
            raise

        logger.info(
            f"Scan complete. Found {len(python_files)} Python files "
            f"({files_scanned} scanned, {files_skipped} skipped)"
        )

        return python_files

    def _extract_decorator_info(self, decorator: ast.AST) -> Dict[str, Any]:
        """
        Extract information from a decorator node.

        Args:
            decorator: AST decorator node

        Returns:
            Dictionary with decorator name and arguments
        """
        try:
            if isinstance(decorator, ast.Name):
                return {"name": decorator.id, "args": []}
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    name = decorator.func.id
                elif isinstance(decorator.func, ast.Attribute):
                    name = ast.unparse(decorator.func)
                else:
                    name = ast.unparse(decorator.func)

                # Extract arguments
                args = []
                for arg in decorator.args:
                    try:
                        args.append(ast.unparse(arg))
                    except:
                        args.append(str(arg))

                return {"name": name, "args": args}
            elif isinstance(decorator, ast.Attribute):
                return {"name": ast.unparse(decorator), "args": []}
            else:
                return {"name": ast.unparse(decorator), "args": []}
        except Exception as e:
            logger.debug(f"Error extracting decorator info: {e}")
            return {"name": "unknown", "args": []}

    def _is_route_decorator(self, decorator_info: Dict[str, Any]) -> bool:
        """
        Check if a decorator is a Flask/FastAPI route decorator.

        Args:
            decorator_info: Decorator information dictionary

        Returns:
            True if decorator is a route decorator
        """
        route_patterns = [
            "route",  # Flask @app.route
            "get", "post", "put", "delete", "patch",  # FastAPI @app.get, @router.get
            "api_route",  # FastAPI @app.api_route
        ]
        name = decorator_info["name"].lower()
        return any(pattern in name for pattern in route_patterns)

    def _extract_route_info(self, decorator_info: Dict[str, Any], func_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract route information from a decorator.

        Args:
            decorator_info: Decorator information
            func_name: Name of the decorated function

        Returns:
            Dictionary with route information or None
        """
        if not self._is_route_decorator(decorator_info):
            return None

        route_info = {
            "handler": func_name,
            "path": None,
            "methods": []
        }

        # Extract HTTP method from decorator name
        name_lower = decorator_info["name"].lower()
        for method in ["get", "post", "put", "delete", "patch", "options", "head"]:
            if method in name_lower:
                route_info["methods"] = [method.upper()]
                break

        # Extract path from arguments
        if decorator_info["args"]:
            # First argument is usually the path
            path_arg = decorator_info["args"][0]
            # Remove quotes if present
            route_info["path"] = path_arg.strip('"\'')

        # Check for methods in keyword arguments (Flask style)
        if "methods" not in route_info or not route_info["methods"]:
            for arg in decorator_info["args"]:
                if "methods" in str(arg).lower():
                    # Try to extract methods list
                    try:
                        if "[" in arg and "]" in arg:
                            methods_str = arg[arg.index("[")+1:arg.index("]")]
                            route_info["methods"] = [m.strip().strip('"\'').upper()
                                                     for m in methods_str.split(",")]
                    except:
                        pass

        return route_info if route_info["path"] else None

    def _get_docstring(self, node: Union[ast.FunctionDef, ast.ClassDef, ast.Module]) -> Optional[str]:
        """
        Extract docstring from a node.

        Args:
            node: AST node (function, class, or module)

        Returns:
            Docstring or None
        """
        try:
            docstring = ast.get_docstring(node)
            return docstring if docstring else None
        except:
            return None

    def _extract_function_params(self, func: ast.FunctionDef) -> List[str]:
        """
        Extract parameter names from a function.

        Args:
            func: Function definition node

        Returns:
            List of parameter names
        """
        params = []
        for arg in func.args.args:
            params.append(arg.arg)
        return params

    def parse_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse a Python file using AST to extract code structure.

        Args:
            file_path: Path to the Python file (can be relative to root or absolute)

        Returns:
            Dictionary containing parsed information:
            {
                "path": str,
                "imports": List[str],
                "classes": List[Dict],
                "functions": List[Dict],
                "routes": List[Dict],
                "module_docstring": Optional[str]
            }
            Returns None if file cannot be parsed.

        Example:
            >>> analyzer = CodebaseAnalyzer("/path/to/project")
            >>> info = analyzer.parse_file("src/main.py")
            >>> print(info["classes"])
            [{"name": "UserService", "methods": ["get_user"], "docstring": "..."}]
        """
        # Handle both absolute and relative paths
        path = Path(file_path)
        if not path.is_absolute():
            path = self.root_path / path

        if not path.exists():
            logger.error(f"File does not exist: {path}")
            return None

        if not path.is_file():
            logger.error(f"Path is not a file: {path}")
            return None

        try:
            # Read file content
            with open(path, "r", encoding="utf-8") as f:
                source_code = f.read()

            # Parse AST
            tree = ast.parse(source_code, filename=str(path))

            # Extract relative path
            try:
                relative_path = path.relative_to(self.root_path)
            except ValueError:
                relative_path = path

            # Initialize result structure
            result = {
                "path": str(relative_path),
                "imports": [],
                "classes": [],
                "functions": [],
                "routes": [],
                "module_docstring": self._get_docstring(tree)
            }

            # Walk through AST nodes
            for node in ast.walk(tree):
                # Extract imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        result["imports"].append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        result["imports"].append(node.module)

            # Process top-level nodes only
            for node in tree.body:
                # Extract classes
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "methods": [],
                        "docstring": self._get_docstring(node),
                        "decorators": []
                    }

                    # Extract decorators
                    for decorator in node.decorator_list:
                        dec_info = self._extract_decorator_info(decorator)
                        class_info["decorators"].append(dec_info["name"])

                    # Extract methods (both sync and async)
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            class_info["methods"].append(item.name)

                    result["classes"].append(class_info)

                # Extract top-level functions (both sync and async)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_info = {
                        "name": node.name,
                        "params": self._extract_function_params(node),
                        "decorators": [],
                        "docstring": self._get_docstring(node),
                        "is_async": isinstance(node, ast.AsyncFunctionDef)
                    }

                    # Extract decorators and check for routes
                    for decorator in node.decorator_list:
                        dec_info = self._extract_decorator_info(decorator)
                        func_info["decorators"].append(dec_info["name"])

                        # Check if this is a route decorator
                        route_info = self._extract_route_info(dec_info, node.name)
                        if route_info:
                            result["routes"].append(route_info)

                    result["functions"].append(func_info)

            logger.debug(f"Successfully parsed file: {path}")
            return result

        except SyntaxError as e:
            logger.error(f"Syntax error parsing {path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing file {path}: {e}")
            return None

    def compute_hash(self, file_path: str) -> str:
        """
        Generate SHA-256 hash of file contents.

        Args:
            file_path: Path to the file (can be relative to root or absolute)

        Returns:
            SHA-256 hash as hex string

        Example:
            >>> analyzer = CodebaseAnalyzer("/path/to/project")
            >>> hash_val = analyzer.compute_hash("src/main.py")
            >>> print(hash_val)
            'a1b2c3d4e5f6...'
        """
        # Handle both absolute and relative paths
        path = Path(file_path)
        if not path.is_absolute():
            path = self.root_path / path

        try:
            with open(path, "rb") as f:
                file_hash = hashlib.sha256()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
                return file_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {path}: {e}")
            return ""

    def _load_stored_hashes(self) -> Dict[str, Dict[str, str]]:
        """
        Load stored file hashes from JSON file.

        Returns:
            Dictionary mapping file paths to hash metadata
        """
        if not self.hash_storage_file.exists():
            return {}

        try:
            with open(self.hash_storage_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding hash storage file: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading hash storage: {e}")
            return {}

    def _save_hashes(self, hashes: Dict[str, Dict[str, str]]) -> None:
        """
        Save file hashes to JSON file.

        Args:
            hashes: Dictionary of file path to hash metadata
        """
        try:
            with open(self.hash_storage_file, "w") as f:
                json.dump(hashes, f, indent=2)
            logger.debug(f"Saved hashes for {len(hashes)} files")
        except Exception as e:
            logger.error(f"Error saving hashes: {e}")

    def detect_changes(self) -> List[Dict[str, str]]:
        """
        Compare current file hashes with stored hashes to detect changes.

        Returns:
            List of changed files with metadata:
            [
                {
                    "path": "src/main.py",
                    "status": "modified",  # or "new" or "deleted"
                    "hash": "abc123...",
                    "old_hash": "def456...",  # only for modified files
                    "last_modified": "2025-01-15T10:30:00"
                }
            ]

        Example:
            >>> analyzer = CodebaseAnalyzer("/path/to/project")
            >>> changes = analyzer.detect_changes()
            >>> for change in changes:
            ...     print(f"{change['path']}: {change['status']}")
            src/main.py: modified
            src/new_file.py: new
        """
        changes = []
        stored_hashes = self._load_stored_hashes()
        current_files = self.scan_directory()
        current_hashes = {}

        # Compute hashes for all current files
        for file_info in current_files:
            file_path = file_info["path"]
            current_hash = self.compute_hash(file_path)

            if not current_hash:
                continue

            # Get file modification time
            full_path = self.root_path / file_path
            last_modified = datetime.fromtimestamp(full_path.stat().st_mtime).isoformat()

            current_hashes[file_path] = {
                "hash": current_hash,
                "last_checked": datetime.now().isoformat(),
                "last_modified": last_modified
            }

            # Check if file is new or modified
            if file_path not in stored_hashes:
                changes.append({
                    "path": file_path,
                    "status": "new",
                    "hash": current_hash,
                    "last_modified": last_modified
                })
                logger.info(f"New file detected: {file_path}")
            elif stored_hashes[file_path]["hash"] != current_hash:
                changes.append({
                    "path": file_path,
                    "status": "modified",
                    "hash": current_hash,
                    "old_hash": stored_hashes[file_path]["hash"],
                    "last_modified": last_modified
                })
                logger.info(f"Modified file detected: {file_path}")

        # Check for deleted files
        current_paths = {f["path"] for f in current_files}
        for stored_path, stored_info in stored_hashes.items():
            if stored_path not in current_paths:
                changes.append({
                    "path": stored_path,
                    "status": "deleted",
                    "hash": stored_info["hash"],
                    "last_modified": stored_info.get("last_modified", "unknown")
                })
                logger.info(f"Deleted file detected: {stored_path}")

        # Update stored hashes
        if changes:
            # Remove deleted files from storage
            for change in changes:
                if change["status"] == "deleted":
                    stored_hashes.pop(change["path"], None)

            # Update with current hashes
            stored_hashes.update(current_hashes)
            self._save_hashes(stored_hashes)

        logger.info(f"Change detection complete: {len(changes)} changes detected")
        return changes

    def get_changed_components(self) -> List[str]:
        """
        Get list of component names that have changed since last analysis.

        Returns only components (classes, functions, routes) from changed files.

        Returns:
            List of component identifiers in format "file::component"

        Example:
            >>> analyzer = CodebaseAnalyzer("/path/to/project")
            >>> changed = analyzer.get_changed_components()
            >>> print(changed)
            ['src/main.py::UserService', 'src/main.py::login', 'src/auth.py::verify_token']
        """
        if self._changed_components:
            return self._changed_components

        changes = self.detect_changes()
        changed_components = []

        # Process only new and modified files (skip deleted)
        for change in changes:
            if change["status"] in ["new", "modified"]:
                file_path = change["path"]
                parsed_info = self.parse_file(file_path)

                if not parsed_info:
                    continue

                # Extract component names from parsed file
                # Classes
                for class_info in parsed_info.get("classes", []):
                    component_id = f"{file_path}::{class_info['name']}"
                    changed_components.append(component_id)

                # Functions (top-level only)
                for func_info in parsed_info.get("functions", []):
                    component_id = f"{file_path}::{func_info['name']}"
                    changed_components.append(component_id)

                # Routes
                for route_info in parsed_info.get("routes", []):
                    component_id = f"{file_path}::{route_info['handler']}"
                    if component_id not in changed_components:
                        changed_components.append(component_id)

        self._changed_components = changed_components
        logger.info(f"Found {len(changed_components)} changed components")
        return changed_components

    def get_summary(self) -> Dict[str, any]:
        """
        Get a summary of the scanned codebase.

        Returns:
            Dictionary containing summary statistics
        """
        files = self.scan_directory()

        if not files:
            return {
                "total_files": 0,
                "total_size": 0,
                "average_size": 0,
                "largest_file": None,
                "smallest_file": None,
            }

        total_size = sum(f["size"] for f in files)
        largest = max(files, key=lambda f: f["size"])
        smallest = min(files, key=lambda f: f["size"])

        return {
            "total_files": len(files),
            "total_size": total_size,
            "average_size": total_size // len(files) if files else 0,
            "largest_file": {"path": largest["path"], "size": largest["size"]},
            "smallest_file": {"path": smallest["path"], "size": smallest["size"]},
        }
