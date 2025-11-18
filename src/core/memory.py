"""ChromaDB-based semantic memory for code patterns."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)


class MemoryStore:
    """Manages semantic storage and retrieval of code patterns using ChromaDB."""

    # Configuration
    DEFAULT_PERSIST_DIR = "./data"
    DEFAULT_COLLECTION_NAME = "codebase_patterns"
    DEFAULT_EMBEDDING_MODEL = "llama3.2"
    OLLAMA_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
    ):
        """
        Initialize the MemoryStore with ChromaDB.

        Args:
            persist_directory: Directory for ChromaDB persistence (default: ./data)
            collection_name: Name of the ChromaDB collection (default: codebase_patterns)
            embedding_model: Ollama embedding model name (default: llama3.2)
            ollama_base_url: Ollama server URL (default: http://localhost:11434)
        """
        self.persist_directory = persist_directory or self.DEFAULT_PERSIST_DIR
        self.collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        self.embedding_model = embedding_model or self.DEFAULT_EMBEDDING_MODEL
        self.ollama_base_url = ollama_base_url or self.OLLAMA_BASE_URL

        # Ensure persistence directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize Ollama embeddings
        try:
            self.embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.ollama_base_url,
            )
            logger.info(f"Initialized Ollama embeddings with model: {self.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embeddings: {e}")
            raise

        # Initialize ChromaDB client with persistence
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            logger.info(f"Initialized ChromaDB at: {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Code patterns and components from analyzed codebases"}
            )
            logger.info(f"Using collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}")
            raise

    def _generate_text_representation(self, component_data: Dict[str, Any]) -> str:
        """
        Generate a text representation of a component for embedding.

        Args:
            component_data: Parsed component metadata

        Returns:
            Text representation combining key component information
        """
        parts = []

        # Component type
        comp_type = component_data.get("type", "component")
        parts.append(f"Type: {comp_type}")

        # Component name
        name = component_data.get("name", "unknown")
        parts.append(f"Name: {name}")

        # Docstring
        docstring = component_data.get("docstring")
        if docstring:
            parts.append(f"Description: {docstring}")

        # For classes: include methods
        if comp_type == "class":
            methods = component_data.get("methods", [])
            if methods:
                parts.append(f"Methods: {', '.join(methods)}")

        # For functions: include parameters
        if comp_type == "function":
            params = component_data.get("params", [])
            if params:
                parts.append(f"Parameters: {', '.join(params)}")

            # Include decorators
            decorators = component_data.get("decorators", [])
            if decorators:
                parts.append(f"Decorators: {', '.join(decorators)}")

        # For routes: include path and methods
        if comp_type == "route":
            route_path = component_data.get("path", "")
            http_methods = component_data.get("methods", [])
            if route_path:
                parts.append(f"Route: {route_path}")
            if http_methods:
                parts.append(f"HTTP Methods: {', '.join(http_methods)}")

        # File path
        file_path = component_data.get("file_path")
        if file_path:
            parts.append(f"File: {file_path}")

        return " | ".join(parts)

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using Ollama.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def store_component(self, component_data: Dict[str, Any]) -> str:
        """
        Store a code component in ChromaDB with semantic embedding.

        Args:
            component_data: Parsed component metadata with fields:
                - type: "class", "function", "route", etc.
                - name: Component name
                - docstring: Optional documentation
                - methods/params/decorators: Type-specific fields
                - file_path: Source file path

        Returns:
            Document ID of the stored component

        Example:
            >>> memory = MemoryStore()
            >>> component = {
            ...     "type": "function",
            ...     "name": "get_user",
            ...     "docstring": "Retrieve user by ID",
            ...     "params": ["user_id"],
            ...     "decorators": ["app.get"],
            ...     "file_path": "api/users.py"
            ... }
            >>> doc_id = memory.store_component(component)
        """
        try:
            # Generate text representation
            text = self._generate_text_representation(component_data)

            # Generate embedding
            embedding = self._generate_embedding(text)

            # Generate document ID
            name = component_data.get("name", "unknown")
            file_path = component_data.get("file_path", "unknown")
            doc_id = f"{file_path}::{name}"

            # Prepare metadata (ChromaDB requires all values to be str, int, float, or bool)
            metadata = {
                "type": component_data.get("type", "unknown"),
                "name": name,
                "file_path": file_path,
            }

            # Add optional fields to metadata
            if "docstring" in component_data and component_data["docstring"]:
                metadata["docstring"] = str(component_data["docstring"])[:1000]  # Limit length

            # For routes, add specific metadata
            if component_data.get("type") == "route":
                if "path" in component_data:
                    metadata["route_path"] = component_data["path"]
                if "methods" in component_data:
                    metadata["http_methods"] = ",".join(component_data["methods"])

            # Store in ChromaDB
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata]
            )

            logger.info(f"Stored component: {doc_id}")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to store component: {e}")
            raise

    def retrieve_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve semantically similar components using vector search.

        Args:
            query: Natural language query or code description
            k: Number of results to return (default: 5)

        Returns:
            List of similar components with metadata and similarity scores

        Example:
            >>> memory = MemoryStore()
            >>> results = memory.retrieve_similar("user authentication functions", k=3)
            >>> for result in results:
            ...     print(f"{result['name']}: {result['score']}")
        """
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            similar_components = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    component = {
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
                    }
                    similar_components.append(component)

            logger.info(f"Retrieved {len(similar_components)} similar components for query: {query}")
            return similar_components

        except Exception as e:
            logger.error(f"Failed to retrieve similar components: {e}")
            raise

    def get_all_patterns(self) -> List[str]:
        """
        Extract unique patterns and conventions from stored components.

        Returns:
            List of identified patterns

        Example:
            >>> memory = MemoryStore()
            >>> patterns = memory.get_all_patterns()
            >>> print(patterns)
            ["route naming: /api/{resource}", "async functions with database operations"]
        """
        try:
            # Get all documents from collection
            all_docs = self.collection.get(
                include=["metadatas", "documents"]
            )

            patterns = []

            if not all_docs["ids"]:
                logger.info("No components stored yet")
                return patterns

            # Extract route patterns
            route_paths = set()
            for metadata in all_docs["metadatas"]:
                if metadata.get("type") == "route" and "route_path" in metadata:
                    route_paths.add(metadata["route_path"])

            if route_paths:
                # Analyze route naming conventions
                if any("/api/" in path for path in route_paths):
                    patterns.append("route naming: /api/{resource} pattern")
                if any("{" in path and "}" in path for path in route_paths):
                    patterns.append("route naming: uses path parameters")

            # Extract decorator patterns
            decorator_patterns = set()
            for doc in all_docs["documents"]:
                if "Decorators:" in doc:
                    # Extract decorator info
                    if "app.get" in doc or "app.post" in doc:
                        decorator_patterns.add("FastAPI route decorators")
                    if "app.route" in doc:
                        decorator_patterns.add("Flask route decorators")
                    if "@staticmethod" in doc:
                        decorator_patterns.add("static methods")

            patterns.extend(decorator_patterns)

            # Extract class patterns
            class_count = sum(1 for m in all_docs["metadatas"] if m.get("type") == "class")
            if class_count > 0:
                patterns.append(f"object-oriented design: {class_count} classes")

            # Extract async patterns
            async_count = sum(1 for doc in all_docs["documents"] if "async" in doc.lower())
            if async_count > 0:
                patterns.append(f"async programming: {async_count} async functions")

            # Extract error handling patterns
            error_handling = sum(1 for doc in all_docs["documents"] if "error" in doc.lower() or "exception" in doc.lower())
            if error_handling > 2:
                patterns.append("error handling: try-except pattern used")

            logger.info(f"Identified {len(patterns)} patterns")
            return patterns

        except Exception as e:
            logger.error(f"Failed to extract patterns: {e}")
            raise

    def clear_collection(self):
        """Clear all documents from the collection (useful for testing)."""
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Code patterns and components from analyzed codebases"}
            )
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise

    def get_all_components(self) -> List[Dict[str, Any]]:
        """
        Get all stored components with their metadata.

        Returns:
            List of component dictionaries suitable for visualization

        Example:
            >>> memory = MemoryStore()
            >>> components = memory.get_all_components()
            >>> print(f"Found {len(components)} components")
        """
        try:
            # Get all documents from collection
            all_docs = self.collection.get(
                include=["metadatas", "documents"]
            )

            if not all_docs["ids"]:
                logger.info("No components stored yet")
                return []

            components = []
            for i, doc_id in enumerate(all_docs["ids"]):
                metadata = all_docs["metadatas"][i]
                document = all_docs["documents"][i]

                # Build component dict
                component = {
                    "id": doc_id,
                    "type": metadata.get("type", "unknown"),
                    "name": metadata.get("name", "unknown"),
                    "file_path": metadata.get("file_path", "unknown"),
                    "docstring": document,
                    "size": metadata.get("size", 0),
                }

                # Add type-specific fields
                if metadata.get("type") == "class":
                    component["methods"] = metadata.get("methods", [])
                elif metadata.get("type") == "function":
                    component["params"] = metadata.get("params", [])
                elif metadata.get("type") == "route":
                    component["route_path"] = metadata.get("route_path", "")
                    # http_methods is stored as comma-separated string
                    http_methods_str = metadata.get("http_methods", "")
                    component["methods"] = http_methods_str.split(",") if http_methods_str else []
                    component["path"] = metadata.get("route_path", "")

                components.append(component)

            logger.info(f"Retrieved {len(components)} components from memory")
            return components

        except Exception as e:
            logger.error(f"Failed to retrieve all components: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the stored components.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()

            # Get all metadata to compute stats
            all_docs = self.collection.get(include=["metadatas"])

            stats = {
                "total_components": count,
                "types": {},
            }

            # Count by type
            for metadata in all_docs["metadatas"]:
                comp_type = metadata.get("type", "unknown")
                stats["types"][comp_type] = stats["types"].get(comp_type, 0) + 1

            logger.info(f"Collection stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise
