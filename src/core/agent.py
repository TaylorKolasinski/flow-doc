"""LangGraph-based agentic workflow for documentation generation."""

import logging
import json
from pathlib import Path
from typing import TypedDict, List, Dict, Optional, Any, Literal
from langgraph.graph import StateGraph, END
from src.core.analyzer import CodebaseAnalyzer
from src.core.memory import MemoryStore
from src.core.generator import DocumentationGenerator
from src.core.pattern_detector import PatternDetector

logger = logging.getLogger(__name__)


class DocumentationState(TypedDict):
    """
    State schema for the documentation generation workflow.

    This state is passed between workflow nodes and updated at each step.
    """
    # Input
    component_name: str  # Name of the component to document
    component_path: str  # File path of the component

    # Analysis phase
    code_metadata: Dict[str, Any]  # Parsed code metadata from analyzer

    # Pattern detection phase
    pattern_suggestions: List[str]  # Pattern deviation suggestions
    check_patterns: bool  # Whether to check patterns

    # Retrieval phase
    retrieved_context: List[str]  # Similar components from memory

    # Generation phase
    draft_documentation: str  # Generated documentation

    # Validation phase
    validation_result: Dict[str, Any]  # Validation results

    # Control flow
    iteration_count: int  # Current iteration number
    max_iterations: int  # Maximum allowed iterations
    errors: List[str]  # Accumulated errors


class DocumentationAgent:
    """
    Multi-agent documentation generator using LangGraph workflow.

    The agent orchestrates a workflow with the following stages:
    1. Analyze: Parse code metadata using AST
    2. Retrieve: Find similar components from memory
    3. Generate: Create documentation using LLM
    4. Validate: Check documentation quality
    5. Store: Save to memory or retry if validation fails

    Workflow:
    ┌─────────┐     ┌──────────┐     ┌──────────┐
    │ Analyze │ --> │ Retrieve │ --> │ Generate │
    └─────────┘     └──────────┘     └──────────┘
                                           │
                                           ↓
                                     ┌──────────┐
                                     │ Validate │
                                     └──────────┘
                                           │
                         ┌─────────────────┴──────────────┐
                         ↓                                ↓
                    [Pass/MaxIter]                    [Fail]
                         ↓                                ↓
                   ┌─────────┐                      [Retry Generate]
                   │  Store  │                            │
                   └─────────┘                            │
                         │                                │
                         ↓                                │
                      [Done] <───────────────────────────┘
    """

    def __init__(
        self,
        analyzer: CodebaseAnalyzer,
        memory: MemoryStore,
        generator: DocumentationGenerator,
        pattern_detector: Optional[PatternDetector] = None,
    ):
        """
        Initialize the documentation agent.

        Args:
            analyzer: CodebaseAnalyzer for parsing code
            memory: MemoryStore for semantic storage and retrieval
            generator: DocumentationGenerator for LLM-based generation
            pattern_detector: Optional PatternDetector for checking code patterns
        """
        self.analyzer = analyzer
        self.memory = memory
        self.generator = generator
        self.pattern_detector = pattern_detector or PatternDetector()

        logger.info("Initialized DocumentationAgent")
        logger.debug(f"  - Analyzer root: {analyzer.root_path}")
        logger.debug(f"  - Memory collection: {memory.collection_name}")
        logger.debug(f"  - Generator model: {generator.model}")

    def analyze_node(self, state: DocumentationState) -> DocumentationState:
        """
        Analyze code component and extract metadata.

        This node:
        1. Uses CodebaseAnalyzer to parse the component file
        2. Extracts code metadata (functions, classes, routes, etc.)
        3. Updates state with parsed metadata

        Args:
            state: Current workflow state

        Returns:
            Updated state with code_metadata populated

        Raises:
            Exception if analysis fails
        """
        logger.info(f"[ANALYZE] Processing: {state['component_path']}")

        try:
            # Parse the component file
            parsed = self.analyzer.parse_file(state['component_path'])

            if parsed is None:
                logger.warning(f"[ANALYZE] Could not parse {state['component_path']}")
                state['errors'].append(f"Failed to parse file: {state['component_path']}")
                state['code_metadata'] = {
                    "path": state['component_path'],
                    "name": state['component_name'],
                    "error": "Parse failed"
                }
                return state

            # Update state with parsed metadata
            state['code_metadata'] = parsed

            # Log extracted information
            logger.debug(f"[ANALYZE] Extracted metadata for {state['component_name']}")
            logger.debug(f"  - Classes: {len(parsed.get('classes', []))}")
            logger.debug(f"  - Functions: {len(parsed.get('functions', []))}")
            logger.debug(f"  - Routes: {len(parsed.get('routes', []))}")

            return state

        except Exception as e:
            logger.error(f"[ANALYZE] Failed: {e}")
            state['errors'].append(f"Analysis failed: {str(e)}")
            state['code_metadata'] = {"error": str(e)}
            return state

    def check_patterns_node(self, state: DocumentationState) -> DocumentationState:
        """
        Check component against codebase patterns and detect deviations.

        This node:
        1. Retrieves all components from memory to analyze patterns
        2. Detects common patterns across the codebase
        3. Compares current component against those patterns
        4. Generates suggestions for deviations
        5. Updates state with pattern suggestions

        Args:
            state: Current workflow state with code_metadata

        Returns:
            Updated state with pattern_suggestions populated
        """
        # Skip if pattern checking is disabled
        if not state.get('check_patterns', False):
            logger.debug("[PATTERNS] Pattern checking disabled, skipping")
            state['pattern_suggestions'] = []
            return state

        logger.info(f"[PATTERNS] Checking patterns for {state['component_name']}")

        try:
            # Get all components from memory for pattern analysis
            all_results = self.memory.retrieve_similar("", k=100)

            # Convert memory results to component format
            all_components = []
            for result in all_results:
                doc = result.get('document', '')
                metadata = result.get('metadata', {})
                # Parse metadata from document
                comp = {
                    'name': metadata.get('name', ''),
                    'type': metadata.get('type', ''),
                    'file_path': metadata.get('file_path', ''),
                    'docstring': doc[:200] if doc else '',  # First 200 chars as docstring
                }
                all_components.append(comp)

            # Analyze patterns across all components
            logger.debug(f"[PATTERNS] Analyzing patterns from {len(all_components)} components")
            patterns = self.pattern_detector.analyze_patterns(all_components)

            logger.debug(f"[PATTERNS] Consistency score: {patterns['consistency_score']:.1%}")

            # Get current component from code_metadata
            metadata = state.get('code_metadata', {})
            current_component = {
                'name': state['component_name'],
                'type': metadata.get('type', 'unknown'),
                'file_path': state['component_path'],
                'docstring': metadata.get('docstring', ''),
                'methods': metadata.get('methods', []),
                'imports': metadata.get('imports', []),
                'decorators': metadata.get('decorators', []),
                'params': metadata.get('params', []),
                'returns': metadata.get('returns', ''),
                'raises': metadata.get('raises', []),
            }

            # Detect deviations
            suggestions = self.pattern_detector.detect_deviations(
                current_component,
                patterns
            )

            state['pattern_suggestions'] = suggestions

            if suggestions:
                logger.info(f"[PATTERNS] ⚠ Found {len(suggestions)} pattern deviations")
                for i, suggestion in enumerate(suggestions[:3], 1):
                    logger.debug(f"  {i}. {suggestion[:100]}...")
            else:
                logger.info("[PATTERNS] ✓ No pattern deviations found")

            return state

        except Exception as e:
            logger.error(f"[PATTERNS] Failed: {e}")
            state['errors'].append(f"Pattern checking failed: {str(e)}")
            # Don't fail the workflow, just skip pattern checking
            state['pattern_suggestions'] = []
            return state

    def retrieve_node(self, state: DocumentationState) -> DocumentationState:
        """
        Retrieve similar components from semantic memory.

        This node:
        1. Queries MemoryStore for components similar to current one
        2. Formats retrieved components as context
        3. Updates state with retrieved context

        Args:
            state: Current workflow state with code_metadata

        Returns:
            Updated state with retrieved_context populated
        """
        logger.info(f"[RETRIEVE] Searching for similar components to {state['component_name']}")

        try:
            # Build query from component name and metadata
            metadata = state.get('code_metadata', {})
            comp_type = metadata.get('type', 'component')

            # Create descriptive query
            query_parts = [state['component_name']]

            # Add type information
            if metadata.get('classes'):
                query_parts.append('class')
            if metadata.get('functions'):
                query_parts.append('function')
            if metadata.get('routes'):
                query_parts.append('route endpoint')

            # Add docstring if available
            if metadata.get('module_docstring'):
                query_parts.append(metadata['module_docstring'][:100])

            query = ' '.join(query_parts)

            logger.debug(f"[RETRIEVE] Query: {query}")

            # Retrieve similar components from memory
            results = self.memory.retrieve_similar(query, k=5)

            # Format results as context strings
            context = []
            for i, result in enumerate(results, 1):
                doc = result.get('document', '')
                metadata_result = result.get('metadata', {})
                score = result.get('score', 0.0)

                context_str = f"Similar Component {i} (relevance: {score:.2f}):\n{doc}"
                context.append(context_str)

            state['retrieved_context'] = context

            logger.debug(f"[RETRIEVE] Found {len(context)} similar components")
            return state

        except Exception as e:
            logger.error(f"[RETRIEVE] Failed: {e}")
            state['errors'].append(f"Retrieval failed: {str(e)}")
            # Continue with empty context
            state['retrieved_context'] = []
            return state

    def generate_node(self, state: DocumentationState) -> DocumentationState:
        """
        Generate documentation using LLM with context.

        This node:
        1. Builds prompt with code metadata and retrieved context
        2. Calls LLM to generate documentation
        3. Updates state with draft documentation
        4. Increments iteration count

        Args:
            state: Current workflow state with metadata and context

        Returns:
            Updated state with draft_documentation and incremented iteration_count
        """
        logger.info(
            f"[GENERATE] Creating documentation (iteration {state['iteration_count'] + 1}/"
            f"{state['max_iterations']})"
        )

        try:
            # Build documentation prompt
            prompt = self._build_documentation_prompt(state)

            logger.debug(f"[GENERATE] Prompt length: {len(prompt)} chars")

            # Generate documentation using LLM
            documentation = self.generator.llm.invoke(prompt)

            # Update state
            state['draft_documentation'] = documentation.strip()
            state['iteration_count'] += 1

            logger.debug(
                f"[GENERATE] Generated {len(state['draft_documentation'])} chars "
                f"of documentation"
            )
            return state

        except Exception as e:
            logger.error(f"[GENERATE] Failed: {e}")
            state['errors'].append(f"Generation failed: {str(e)}")
            state['iteration_count'] += 1
            # Provide fallback documentation
            state['draft_documentation'] = f"# {state['component_name']}\n\nError generating documentation: {str(e)}"
            return state

    def validate_node(self, state: DocumentationState) -> DocumentationState:
        """
        Validate the generated documentation quality.

        This node:
        1. Checks documentation for completeness
        2. Validates against quality criteria
        3. Updates state with validation results

        Validation criteria:
        - Has overview section
        - Has description
        - Has usage examples (if applicable)
        - Minimum length requirements
        - No placeholder text

        Args:
            state: Current workflow state with draft_documentation

        Returns:
            Updated state with validation_result
        """
        logger.info("[VALIDATE] Checking documentation quality")

        try:
            doc = state['draft_documentation']
            metadata = state.get('code_metadata', {})
            issues = []
            checks = {}

            # Check: Minimum length (200 characters)
            checks['min_length'] = len(doc) >= 200
            if not checks['min_length']:
                issues.append("Documentation too short (minimum 200 characters)")

            # Check: Has overview section
            checks['has_overview'] = "## Overview" in doc or "# Overview" in doc
            if not checks['has_overview']:
                issues.append("Missing Overview section")

            # Check: Has description or details
            checks['has_description'] = any(x in doc for x in ["## Description", "## Details", "## Summary"])
            if not checks['has_description']:
                issues.append("Missing Description/Details section")

            # Check: No placeholder text
            placeholders = ["[To be implemented]", "[TODO]", "[PLACEHOLDER]", "TODO:", "FIXME:"]
            checks['no_placeholders'] = not any(p in doc for p in placeholders)
            if not checks['no_placeholders']:
                issues.append("Contains placeholder text")

            # Check: Has API endpoints documented (if routes exist)
            if metadata.get('routes'):
                checks['has_endpoints'] = any(x in doc for x in ["## API Endpoints", "## Endpoints", "## Routes"])
                if not checks['has_endpoints']:
                    issues.append("Component has routes but no API Endpoints section")
            else:
                checks['has_endpoints'] = True  # Not applicable

            # Check: Has parameters documented (if functions exist)
            if metadata.get('functions'):
                checks['has_parameters'] = any(x in doc.lower() for x in ["parameter", "param", "argument", "arg"])
                if not checks['has_parameters']:
                    issues.append("Component has functions but parameters not documented")
            else:
                checks['has_parameters'] = True  # Not applicable

            # Calculate score (percentage of checks passed)
            total_checks = len(checks)
            passed_checks = sum(1 for v in checks.values() if v)
            score = passed_checks / total_checks if total_checks > 0 else 0.0

            # Validation passes if score >= 0.7
            is_valid = score >= 0.7

            validation = {
                "is_valid": is_valid,
                "score": score,
                "checks": checks,
                "issues": issues
            }

            state['validation_result'] = validation

            if is_valid:
                logger.info(f"[VALIDATE] ✓ Passed (score: {score:.2f})")
            else:
                logger.warning(
                    f"[VALIDATE] ✗ Failed (score: {score:.2f}): {', '.join(issues)}"
                )

            return state

        except Exception as e:
            logger.error(f"[VALIDATE] Failed: {e}")
            state['errors'].append(f"Validation failed: {str(e)}")
            state['validation_result'] = {
                "is_valid": False,
                "score": 0.0,
                "checks": {},
                "issues": [str(e)]
            }
            return state

    def store_node(self, state: DocumentationState) -> DocumentationState:
        """
        Store the final documentation.

        This node:
        1. Saves documentation to output
        2. Optionally stores in memory for future reference
        3. Marks workflow as complete

        Args:
            state: Current workflow state with validated documentation

        Returns:
            Final state with completion status
        """
        logger.info("[STORE] Saving documentation")

        try:
            # Create docs_output directory if it doesn't exist
            docs_dir = Path("docs_output")
            docs_dir.mkdir(exist_ok=True)

            # Save documentation to file
            component_name = state['component_name'].replace('/', '_').replace('\\', '_')
            filename = f"{component_name}.md"
            filepath = docs_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(state['draft_documentation'])

            logger.info(f"[STORE] ✓ Saved documentation to {filepath}")

            # Store component in memory for future retrieval
            try:
                metadata = state.get('code_metadata', {})

                # Determine component type
                comp_type = "component"
                if metadata.get('classes'):
                    comp_type = "class"
                elif metadata.get('functions'):
                    comp_type = "function"
                elif metadata.get('routes'):
                    comp_type = "route"

                # Create memory component
                memory_component = {
                    "type": comp_type,
                    "name": state['component_name'],
                    "docstring": state['draft_documentation'][:500],  # First 500 chars
                    "file_path": state['component_path']
                }

                # Add type-specific fields
                if metadata.get('classes'):
                    memory_component['methods'] = [
                        cls.get('methods', []) for cls in metadata['classes']
                    ][0] if metadata['classes'] else []

                if metadata.get('functions'):
                    memory_component['params'] = [
                        func.get('params', []) for func in metadata['functions']
                    ][0] if metadata['functions'] else []

                # Store in memory
                self.memory.store_component(memory_component)
                logger.debug(f"[STORE] ✓ Stored in memory")

            except Exception as mem_error:
                logger.warning(f"[STORE] Failed to store in memory: {mem_error}")
                # Continue even if memory storage fails

            # Log final stats
            validation = state.get('validation_result', {})
            logger.info(
                f"[STORE] Final stats: {state['iteration_count']} iterations, "
                f"validation score: {validation.get('score', 0.0):.2f}, "
                f"{len(state['errors'])} errors"
            )

            return state

        except Exception as e:
            logger.error(f"[STORE] Failed: {e}")
            state['errors'].append(f"Storage failed: {str(e)}")
            return state

    def should_retry(self, state: DocumentationState) -> Literal["generate", "store"]:
        """
        Determine whether to retry generation or proceed to storage.

        Decision logic:
        - If validation passed: proceed to "store"
        - If validation failed AND iterations < max: retry "generate"
        - If validation failed AND iterations >= max: proceed to "store" (best effort)

        Args:
            state: Current workflow state with validation results

        Returns:
            "generate" to retry generation, or "store" to proceed to storage
        """
        validation_passed = state['validation_result'].get('is_valid', False)
        current_iteration = state['iteration_count']
        max_iterations = state['max_iterations']

        if validation_passed:
            logger.info(
                f"[ROUTER] Validation passed -> proceeding to STORE"
            )
            return "store"

        if current_iteration < max_iterations:
            logger.warning(
                f"[ROUTER] Validation failed (iteration {current_iteration}/"
                f"{max_iterations}) -> retrying GENERATE"
            )
            return "generate"

        logger.warning(
            f"[ROUTER] Max iterations reached ({max_iterations}) -> "
            f"proceeding to STORE with best effort"
        )
        return "store"

    def create_initial_state(
        self,
        component_name: str,
        component_path: str,
        max_iterations: int = 3,
        check_patterns: bool = False
    ) -> DocumentationState:
        """
        Create initial state for the workflow.

        Args:
            component_name: Name of the component to document
            component_path: File path of the component
            max_iterations: Maximum generation attempts (default: 3)
            check_patterns: Whether to check for pattern deviations (default: False)

        Returns:
            Initial DocumentationState
        """
        return DocumentationState(
            component_name=component_name,
            component_path=component_path,
            code_metadata={},
            pattern_suggestions=[],
            check_patterns=check_patterns,
            retrieved_context=[],
            draft_documentation="",
            validation_result={},
            iteration_count=0,
            max_iterations=max_iterations,
            errors=[]
        )

    def _build_documentation_prompt(self, state: DocumentationState) -> str:
        """
        Build the documentation generation prompt.

        Args:
            state: Current workflow state

        Returns:
            Formatted prompt string
        """
        metadata = state.get('code_metadata', {})
        context = state.get('retrieved_context', [])

        # Format code metadata
        metadata_str = self._format_code_metadata(metadata)

        # Format retrieved context
        context_str = "\n\n".join(context) if context else "No similar components found."

        # Format pattern suggestions
        pattern_suggestions = state.get('pattern_suggestions', [])
        patterns_str = ""
        if pattern_suggestions:
            patterns_str = "\n\nPattern Deviations Detected:\n"
            patterns_str += "The following deviations from codebase patterns were found:\n"
            for i, suggestion in enumerate(pattern_suggestions, 1):
                patterns_str += f"{i}. {suggestion}\n"
            patterns_str += "\nPlease address these pattern deviations in a dedicated section of the documentation."

        # Build prompt
        prompt = f"""You are documenting a Python component from a Flask/FastAPI codebase.

Component: {state['component_name']}
Path: {state['component_path']}

Code Analysis:
{metadata_str}

Similar Components in Codebase:
{context_str}{patterns_str}

Generate comprehensive Markdown documentation including:
- ## Overview: High-level description of what this component does
- ## API Endpoints: (if routes exist) List all endpoints with methods and descriptions
- ## Functions/Methods: Document key functions with parameters and return values
- ## Usage Examples: Show practical usage with code snippets
- ## Dependencies: Note any imports or dependencies
- ## Code Quality Notes: (if pattern deviations exist) Document pattern deviations and improvement suggestions

Requirements:
1. Be specific and reference actual code patterns from the analysis
2. Use proper Markdown formatting with headers and code blocks
3. Include inline code references with backticks
4. NO placeholder text like "[To be implemented]" or "TODO"
5. Focus on practical usage and real examples
6. If pattern deviations were detected, include them in a "Code Quality Notes" section with actionable improvement suggestions
7. If this is iteration {state['iteration_count'] + 1}, improve upon the previous attempt

Documentation:"""

        return prompt

    def _format_code_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Format code metadata for the prompt.

        Args:
            metadata: Parsed code metadata

        Returns:
            Formatted string
        """
        lines = []

        # Module docstring
        if metadata.get('module_docstring'):
            lines.append(f"Module: {metadata['module_docstring']}")

        # Imports
        if metadata.get('imports'):
            lines.append(f"\nImports: {', '.join(metadata['imports'][:10])}")

        # Classes
        if metadata.get('classes'):
            lines.append("\nClasses:")
            for cls in metadata['classes']:
                lines.append(f"  - {cls['name']}")
                if cls.get('docstring'):
                    lines.append(f"    Doc: {cls['docstring'][:100]}")
                if cls.get('methods'):
                    lines.append(f"    Methods: {', '.join(cls['methods'][:10])}")

        # Functions
        if metadata.get('functions'):
            lines.append("\nFunctions:")
            for func in metadata['functions']:
                params = ', '.join(func.get('params', []))
                lines.append(f"  - {func['name']}({params})")
                if func.get('docstring'):
                    lines.append(f"    Doc: {func['docstring'][:100]}")
                if func.get('decorators'):
                    lines.append(f"    Decorators: {', '.join(func['decorators'])}")

        # Routes
        if metadata.get('routes'):
            lines.append("\nAPI Routes:")
            for route in metadata['routes']:
                methods = ', '.join(route.get('methods', ['ANY']))
                lines.append(f"  - {methods} {route.get('path', 'unknown')} -> {route.get('handler', 'unknown')}")

        return '\n'.join(lines) if lines else "No metadata available"

    def get_workflow_stats(self, state: DocumentationState) -> Dict[str, Any]:
        """
        Get statistics about the workflow execution.

        Args:
            state: Final workflow state

        Returns:
            Dictionary with workflow statistics
        """
        validation = state.get('validation_result', {})

        return {
            "component": state['component_name'],
            "iterations": state['iteration_count'],
            "max_iterations": state['max_iterations'],
            "validation_passed": validation.get('is_valid', False),
            "validation_score": validation.get('score', 0.0),
            "error_count": len(state['errors']),
            "errors": state['errors'],
            "doc_length": len(state.get('draft_documentation', '')),
        }

    def create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow for documentation generation.

        The workflow follows this structure:
        1. START → analyze (parse code with AST)
        2. analyze → check_patterns (detect pattern deviations)
        3. check_patterns → retrieve (find similar components)
        4. retrieve → generate (create documentation)
        5. generate → validate (check quality)
        6. validate → [conditional]:
           - If retry needed: → generate (iterative refinement)
           - If done: → store (save results)
        7. store → END

        Returns:
            Compiled StateGraph ready for execution
        """
        logger.info("Creating LangGraph workflow")

        # Create StateGraph with DocumentationState schema
        workflow = StateGraph(DocumentationState)

        # Add nodes
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("pattern_check", self.check_patterns_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("validate", self.validate_node)
        workflow.add_node("store", self.store_node)

        logger.debug("Added workflow nodes: analyze, pattern_check, retrieve, generate, validate, store")

        # Add edges (linear flow until validation)
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "pattern_check")
        workflow.add_edge("pattern_check", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "validate")

        # Add conditional edge from validate
        # Routes to either "generate" (retry) or "store" (done)
        workflow.add_conditional_edges(
            "validate",
            self.should_retry,
            {
                "generate": "generate",  # Retry generation
                "store": "store"  # Proceed to storage
            }
        )

        # Add edge from store to END
        workflow.add_edge("store", END)

        logger.debug("Added workflow edges with conditional routing")

        # Compile the graph
        compiled_workflow = workflow.compile()

        logger.info("Workflow compiled successfully")

        return compiled_workflow

    def run(
        self,
        component_name: str,
        component_path: str,
        max_iterations: int = 3,
        incremental: bool = False,
        check_patterns: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the documentation generation workflow.

        Args:
            component_name: Name of the component to document
            component_path: File path of the component
            max_iterations: Maximum generation attempts (default: 3)
            incremental: If True, only document components that have changed (default: False)
            check_patterns: If True, check component against codebase patterns (default: False)

        Returns:
            Dictionary with final state and execution results

        Example:
            >>> agent = DocumentationAgent(analyzer, memory, generator)
            >>> result = agent.run("UserService", "services/user.py")
            >>> print(f"Success: {result['success']}")
            >>> print(f"Iterations: {result['iterations']}")

            >>> # Incremental mode
            >>> result = agent.run("UserService", "services/user.py", incremental=True)
            >>> if result.get('skipped'):
            ...     print("Component unchanged, skipped documentation")

            >>> # Pattern checking mode
            >>> result = agent.run("UserService", "services/user.py", check_patterns=True)
            >>> if result.get('pattern_suggestions'):
            ...     print("Pattern deviations found and included in documentation")
        """
        logger.info(f"Starting workflow for {component_name} at {component_path}")

        try:
            # Check if incremental mode is enabled
            if incremental:
                logger.info("[INCREMENTAL] Checking for changes...")
                changed_components = self.analyzer.get_changed_components()

                # Build component identifier (file::component)
                component_id = f"{component_path}::{component_name}"

                # Check if this component has changed
                if component_id not in changed_components:
                    logger.info(f"[INCREMENTAL] ⊘ Skipping {component_name} - no changes detected")

                    # Return skipped result
                    safe_name = component_name.replace('/', '_').replace('\\', '_')
                    return {
                        "success": True,
                        "skipped": True,
                        "component": component_name,
                        "iterations": 0,
                        "validation_score": 1.0,  # Assume existing docs are valid
                        "validation_passed": True,
                        "errors": [],
                        "doc_length": 0,
                        "output_file": f"docs_output/{safe_name}.md",
                        "final_state": None,
                        "reason": "No changes detected in component"
                    }
                else:
                    logger.info(f"[INCREMENTAL] ✓ Component {component_name} has changes - proceeding with documentation")

            # Create initial state
            initial_state = self.create_initial_state(
                component_name=component_name,
                component_path=component_path,
                max_iterations=max_iterations,
                check_patterns=check_patterns
            )

            # Create and compile workflow
            workflow = self.create_workflow()

            # Execute the workflow
            logger.info("Executing workflow...")
            final_state = workflow.invoke(initial_state)

            # Get statistics
            stats = self.get_workflow_stats(final_state)

            # Determine success
            validation = final_state.get('validation_result', {})
            success = validation.get('is_valid', False) or len(final_state.get('errors', [])) == 0

            # Build result
            safe_name = component_name.replace('/', '_').replace('\\', '_')
            result = {
                "success": success,
                "skipped": False,
                "component": component_name,
                "iterations": stats['iterations'],
                "validation_score": stats['validation_score'],
                "validation_passed": stats['validation_passed'],
                "errors": stats['errors'],
                "doc_length": stats['doc_length'],
                "output_file": f"docs_output/{safe_name}.md",
                "final_state": final_state
            }

            if success:
                logger.info(
                    f"✓ Workflow completed successfully for {component_name} "
                    f"({stats['iterations']} iterations, score: {stats['validation_score']:.2f})"
                )
            else:
                logger.warning(
                    f"✗ Workflow completed with issues for {component_name} "
                    f"({len(stats['errors'])} errors)"
                )

            return result

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)

            # Return error result
            return {
                "success": False,
                "skipped": False,
                "component": component_name,
                "iterations": 0,
                "validation_score": 0.0,
                "validation_passed": False,
                "errors": [str(e)],
                "doc_length": 0,
                "output_file": None,
                "final_state": None,
                "exception": str(e)
            }

    def visualize_workflow(self, output_path: str = "workflow.png") -> None:
        """
        Generate a visualization of the workflow graph.

        Creates a PNG image showing the workflow structure with nodes and edges.

        Args:
            output_path: Path to save the visualization (default: "workflow.png")

        Example:
            >>> agent = DocumentationAgent(analyzer, memory, generator)
            >>> agent.visualize_workflow("docs/workflow.png")
        """
        try:
            logger.info(f"Generating workflow visualization: {output_path}")

            # Create workflow
            workflow = self.create_workflow()

            # Get the graph representation
            # LangGraph's compiled graphs have a get_graph method
            try:
                # Try to get the Mermaid representation
                graph_repr = workflow.get_graph()

                # If we have a graph object, try to generate image
                try:
                    # Some versions of LangGraph support direct PNG export
                    from IPython.display import Image as IPythonImage
                    png_data = graph_repr.draw_png()

                    output_file = Path(output_path)
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    with open(output_file, 'wb') as f:
                        f.write(png_data)

                    logger.info(f"✓ Workflow visualization saved to {output_path}")

                except Exception as png_error:
                    logger.warning(f"Could not generate PNG: {png_error}")

                    # Fallback: save Mermaid diagram as text
                    mermaid_path = output_path.replace('.png', '.mmd')
                    with open(mermaid_path, 'w') as f:
                        f.write(graph_repr.draw_mermaid())

                    logger.info(f"✓ Mermaid diagram saved to {mermaid_path}")
                    logger.info("  Convert to PNG with: mmdc -i workflow.mmd -o workflow.png")

            except Exception as graph_error:
                logger.warning(f"Could not visualize graph: {graph_error}")
                logger.info("Workflow structure:")
                logger.info("  START → analyze → retrieve → generate → validate")
                logger.info("                                              ↓")
                logger.info("                                         [conditional]")
                logger.info("                                              ↓")
                logger.info("                                    ┌─────────┴─────────┐")
                logger.info("                                    ↓                   ↓")
                logger.info("                              [retry: generate]    [done: store]")
                logger.info("                                    │                   ↓")
                logger.info("                                    └──────────────→  END")

        except Exception as e:
            logger.error(f"Failed to visualize workflow: {e}")
            raise
