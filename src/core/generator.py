"""Documentation generator using LLM and semantic memory."""

import logging
from typing import Optional, List, Dict, Any
try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    from langchain_community.llms import Ollama
from src.core.memory import MemoryStore

logger = logging.getLogger(__name__)


class DocumentationGenerator:
    """
    Generates documentation and answers questions about codebases using RAG.

    Retrieval Augmented Generation (RAG):
    1. Retrieve relevant code components from memory
    2. Build context with retrieved components
    3. Generate answer using LLM with context
    """

    DEFAULT_MODEL = "llama3.2"
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TEMPERATURE = 0.1  # Low temperature for more focused, factual responses

    def __init__(
        self,
        memory_store: MemoryStore,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize the documentation generator.

        Args:
            memory_store: MemoryStore instance with indexed codebase
            model: Ollama model name (default: llama3.2)
            base_url: Ollama server URL (default: http://localhost:11434)
            temperature: LLM temperature for generation (default: 0.1)
        """
        self.memory_store = memory_store
        self.model = model or self.DEFAULT_MODEL
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.temperature = temperature if temperature is not None else self.DEFAULT_TEMPERATURE

        # Initialize Ollama LLM
        try:
            self.llm = Ollama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
            )
            logger.info(f"Initialized Ollama LLM with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            raise

    def _format_component_context(self, components: List[Dict[str, Any]]) -> str:
        """
        Format retrieved components into readable context.

        Args:
            components: List of retrieved components from memory

        Returns:
            Formatted context string
        """
        if not components:
            return "No relevant code components found."

        context_parts = []
        for i, comp in enumerate(components, 1):
            metadata = comp["metadata"]
            document = comp["document"]
            score = comp.get("score", 0.0)

            # Format component info
            comp_type = metadata.get("type", "unknown")
            name = metadata.get("name", "unknown")
            file_path = metadata.get("file_path", "unknown")

            part = f"{i}. [{comp_type.upper()}] {name}\n"
            part += f"   File: {file_path}\n"
            part += f"   Relevance: {score:.2f}\n"
            part += f"   Details: {document}\n"

            context_parts.append(part)

        return "\n".join(context_parts)

    def simple_query(self, question: str, k: int = 3, verbose: bool = False) -> str:
        """
        Answer a question about the codebase using RAG.

        Process:
        1. Retrieve top-k similar components from memory
        2. Build context from retrieved components
        3. Generate answer using LLM with context

        Args:
            question: Natural language question about the codebase
            k: Number of components to retrieve (default: 3)
            verbose: If True, log the full prompt and context

        Returns:
            Generated answer as string

        Example:
            >>> generator = DocumentationGenerator(memory_store)
            >>> answer = generator.simple_query("What API endpoints exist for users?")
            >>> print(answer)
        """
        try:
            logger.info(f"Processing query: {question}")

            # Step 1: Retrieve relevant components
            logger.debug(f"Retrieving top-{k} similar components...")
            components = self.memory_store.retrieve_similar(question, k=k)

            if not components:
                logger.warning("No components retrieved from memory")
                return "I couldn't find any relevant code components to answer this question."

            logger.debug(f"Retrieved {len(components)} components")

            # Step 2: Format context
            context = self._format_component_context(components)

            # Step 3: Build prompt
            prompt = self._build_prompt(question, context)

            if verbose:
                logger.info("=" * 70)
                logger.info("FULL PROMPT:")
                logger.info("=" * 70)
                logger.info(prompt)
                logger.info("=" * 70)

            # Step 4: Generate answer
            logger.debug("Generating answer with LLM...")
            response = self.llm.invoke(prompt)

            logger.info(f"Generated answer ({len(response)} chars)")
            return response.strip()

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build the prompt for the LLM.

        Args:
            question: User's question
            context: Formatted context from retrieved components

        Returns:
            Complete prompt string
        """
        prompt = f"""You are a helpful code documentation assistant. You answer questions about a codebase based on the provided context.

Context from codebase:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the codebase context provided above
- Be specific and reference file paths when relevant
- If the context doesn't contain enough information, say so
- Keep your answer clear and concise
- For API endpoints, list them in a structured format

Answer:"""

        return prompt

    def generate_component_summary(self, component_name: str) -> str:
        """
        Generate a detailed summary for a specific component.

        Args:
            component_name: Name of the component to summarize

        Returns:
            Generated summary
        """
        try:
            # Retrieve the specific component
            components = self.memory_store.retrieve_similar(
                f"component named {component_name}", k=5
            )

            # Find exact match
            exact_match = None
            for comp in components:
                if comp["metadata"]["name"] == component_name:
                    exact_match = comp
                    break

            if not exact_match:
                return f"Component '{component_name}' not found in the codebase."

            # Build detailed context
            metadata = exact_match["metadata"]
            comp_type = metadata.get("type", "unknown")

            prompt = f"""Generate a comprehensive documentation summary for this {comp_type}:

{exact_match['document']}

Include:
- Purpose and functionality
- Parameters/methods (if applicable)
- Usage example (if you can infer from context)
- File location

Documentation:"""

            response = self.llm.invoke(prompt)
            return response.strip()

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the generator and memory store.

        Returns:
            Dictionary with statistics
        """
        memory_stats = self.memory_store.get_collection_stats()

        return {
            "model": self.model,
            "temperature": self.temperature,
            "memory_components": memory_stats["total_components"],
            "component_types": memory_stats["types"],
        }
