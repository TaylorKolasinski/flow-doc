# Prompts Used

## Architecture Planning with Claude

The project began with a take-home assessment that included a comprehensive technical specification for an AI-powered documentation assistant. Rather than immediately proceeding to implementation, the specification was first reviewed collaboratively with Claude to develop a structured, incremental development approach suitable for Claude Code.

### Original Assessment Specification

The technical specification outlined the following system requirements:

**System Architecture:**
- AI-powered documentation assistant for Flask/FastAPI codebases
- AST-based code analysis for pattern extraction
- ChromaDB vector memory for semantic search
- LangGraph agentic workflow for documentation generation
- Ollama (llama3.2) for local LLM inference
- FastAPI + CLI dual interface

**Component Architecture:**
```
Interface Layer (FastAPI + CLI)
    ↓
Service Orchestration (LangGraph Agent)
    ↓
Core Services:
- Analyzer (AST parsing, pattern extraction)
- Memory (ChromaDB embeddings, context retrieval)
- Generator (Ollama LLM, prompt templates)
- Visualizer (NetworkX graphs, Plotly charts)
```

**Core Features Required:**
1. **Code Analysis Engine** - Recursive traversal, AST parsing, pattern extraction (naming conventions, error handling, architectural patterns, component dependencies)
2. **Memory System** - ChromaDB vector store with persistence, semantic embedding via Ollama, context retrieval (top-k similarity), pattern frequency tracking
3. **LangGraph Agentic Workflow** - State machine: START → Analyze → Retrieve → Generate → Validate → [retry loop on failure] → Store → END
4. **Documentation Generator** - Context-aware prompt construction, structured output validation, pattern suggestion integration, Markdown/HTML export
5. **Natural Language Query Interface** - Semantic search over codebase, RAG (Retrieval-Augmented Generation), conversational API/architectural Q&A
6. **Component Visualization** - Dependency graph (NetworkX), interactive charts (Plotly), HTML export with zoom/pan

**Bonus Features:**
- Diff detection (hash-based change tracking, selective re-documentation)
- Pattern suggestions (deviation detection, consistency enforcement)
- Component visualization (interactive HTML graphs)

**Technical Stack:** Python 3.9+, FastAPI, Click, LangGraph, LangChain, Ollama (llama3.2), ChromaDB, AST module, NetworkX, Plotly

**Implementation Phases Outlined:**
1. **Phase 1: Foundation** - Project scaffold, analyzer, ChromaDB, simple RAG
2. **Phase 2: Agentic Workflow** - LangGraph state machine, validation/retry, API/CLI
3. **Phase 3: Bonus Features** - Diff detection, pattern suggestions, visualization
4. **Phase 4: Polish** - Testing, documentation, sample outputs

### Planning Approach

The specification was decomposed with Claude into an incremental, testable development strategy. The key planning principle was to avoid building the entire system at once, instead creating specific prompts for Claude Code that could be executed and validated independently before proceeding to the next component.

The planning session resulted in a 4-phase approach with 8 main implementation prompts, each designed with:
- Clear input/output specifications
- Explicit data structures and return types
- Error handling requirements
- Test validation criteria
- Integration points with other components

This collaborative planning transformed the assessment specification into an executable roadmap suitable for systematic development with Claude Code.

---

## Original Development Prompts

The following prompts document the systematic approach used in the initial development of Flow-Doc, showing how the project was built from scratch through well-structured, incremental prompts.

### Prompt 1: Project Scaffolding

**Goal:** Set up the complete project structure with all necessary files and dependencies

**Prompt:**
```markdown
I'm building Flow-Doc, an AI-powered documentation assistant for Flask/FastAPI codebases.

Tech Stack:
- Python 3.9+
- Ollama (llama3.2) for local LLM
- LangGraph for agentic workflow
- ChromaDB for vector memory
- FastAPI for API + Click for CLI
- AST for code parsing
- NetworkX + Plotly for visualization

Project structure should be:
flow-doc/
├── src/
│   ├── api/
│   ├── core/
│   ├── cli.py
│   └── main.py
├── sample_codebase/  # Already exists locally
├── docs_output/
├── data/  # ChromaDB persistence
├── tests/
├── requirements.txt
├── README.md
└── prompts_used.md

Please create:
1. The complete project structure with empty files
2. requirements.txt with all dependencies pinned
3. Basic README.md with project overview
4. .gitignore for Python projects

Don't implement any logic yet - just scaffold the structure.
```

**Result:** Complete project structure with proper organization, all dependencies specified, basic documentation framework

**Why it worked:**
- Clear tech stack specification
- Explicit file structure
- Separation of concerns (API vs core vs CLI)
- Specific instruction to NOT implement logic yet

---

### Prompt 2: AST-Based Code Parsing

**Goal:** Implement Python code analysis using AST to extract functions, classes, routes, and metadata

**Prompt:**
```markdown
Extend CodebaseAnalyzer in src/core/analyzer.py to parse Python files using AST.

Add a new method: `parse_file(self, file_path: str) -> Dict`

This should extract:
1. **Imports**: All import statements (import x, from y import z)
2. **Classes**: Class names and their methods
3. **Functions**: Top-level function names and their parameters
4. **Decorators**: Any decorators used (@app.route, @staticmethod, etc)
5. **Docstrings**: Module, class, and function docstrings

Return structure:
{
  "path": "...",
  "imports": ["fastapi", "typing", ...],
  "classes": [
    {
      "name": "UserService",
      "methods": ["get_user", "create_user"],
      "docstring": "..."
    }
  ],
  "functions": [
    {
      "name": "calculate_total",
      "params": ["price", "tax"],
      "decorators": ["@staticmethod"],
      "docstring": "..."
    }
  ],
  "routes": [  # Special: detect Flask/FastAPI routes
    {
      "path": "/api/users",
      "method": "GET",
      "handler": "get_users"
    }
  ]
}

Handle AST parsing errors gracefully - if a file can't be parsed, log the error and return None.

Update the test to parse one file from sample_codebase and print the extracted metadata.
```

**Result:** Working AST parser that extracts all relevant code components with proper error handling

**Why it worked:**
- Explicit return structure with examples
- Special handling for Flask/FastAPI routes (domain-specific)
- Clear error handling requirements
- Test specification included

---

### Prompt 3: ChromaDB Vector Storage

**Goal:** Integrate ChromaDB for semantic storage and retrieval of code components

**Prompt:**
```markdown
Create src/core/memory.py with a MemoryStore class for ChromaDB integration.

Requirements:
1. Initialize ChromaDB with persistent storage in data/ directory
2. Create a collection named "codebase_patterns"
3. Implement these methods:

   a) `store_component(self, component_data: Dict) -> str`
      - Takes parsed code metadata
      - Generates text representation for embedding
      - Stores in ChromaDB with metadata
      - Returns document ID

   b) `retrieve_similar(self, query: str, k: int = 5) -> List[Dict]`
      - Semantic search using Ollama embeddings
      - Returns top-k similar components

   c) `get_all_patterns(self) -> List[str]`
      - Returns unique patterns/conventions found
      - Examples: "route naming: /api/{resource}", "error handling: try-except"

4. For embeddings, use Ollama's embedding model:
   - Model: "llama3.2"
   - Integration via langchain_community.embeddings.OllamaEmbeddings

5. Text representation for embedding should combine:
   - Component name
   - Docstring
   - Method/function names
   - Route patterns (if applicable)

Add configuration for:
- ChromaDB persistence directory
- Collection name
- Embedding model name

Create tests/test_memory.py that:
- Stores 2-3 mock components
- Retrieves similar items with a query
- Verifies persistence (restart and reload)
```

**Result:** Working vector database integration with semantic search capabilities

**Why it worked:**
- Method-by-method specification
- Clear embedding strategy (what text to combine)
- Specific technology choices (Ollama embeddings)
- Persistence requirements explicit
- Test requirements included

---

### Prompt 4: LangGraph State Schema

**Goal:** Define the state machine structure for the documentation workflow

**Prompt:**
````markdown
Create src/core/agent.py to implement the LangGraph agentic workflow.

First, define the state schema using TypedDict:
```python
from typing import TypedDict, List, Dict, Optional

class DocumentationState(TypedDict):
    component_name: str
    component_path: str
    code_metadata: Dict  # From analyzer
    retrieved_context: List[str]
    draft_documentation: str
    validation_result: Dict
    iteration_count: int
    max_iterations: int
    errors: List[str]
```

Then create a `DocumentationAgent` class:
- `__init__(self, analyzer: CodebaseAnalyzer, memory: MemoryStore, generator: DocumentationGenerator)`
- Store references to core services

Create placeholder methods for each workflow node (we'll implement next):
- `analyze_node(state: DocumentationState) -> DocumentationState`
- `retrieve_node(state: DocumentationState) -> DocumentationState`
- `generate_node(state: DocumentationState) -> DocumentationState`
- `validate_node(state: DocumentationState) -> DocumentationState`
- `store_node(state: DocumentationState) -> DocumentationState`

Add a method to check if we should retry:
- `should_retry(state: DocumentationState) -> str`
  - Returns "generate" if validation failed and iterations < max
  - Returns "store" if validation passed or max iterations reached

Don't implement the workflow graph yet - just the structure.
````

**Result:** Complete state schema and agent class structure ready for node implementation

**Why it worked:**
- State schema defined first (foundation)
- Placeholder methods allow incremental implementation
- Clear separation of concerns (each node is independent)
- Retry logic specified upfront

---

### Prompt 5: LangGraph Workflow Assembly

**Goal:** Connect all workflow nodes into a complete LangGraph state machine

**Prompt:**
```markdown
Complete the DocumentationAgent in src/core/agent.py by assembling the LangGraph workflow.

Add method: `create_workflow(self) -> StateGraph`

Build the graph:
1. Create StateGraph with DocumentationState
2. Add all nodes (analyze, retrieve, generate, validate, store)
3. Add edges:
   - START → analyze
   - analyze → retrieve
   - retrieve → generate
   - generate → validate
   - validate → conditional edge using should_retry:
     * If retry needed → generate
     * If done → store
   - store → END

4. Compile the graph

Add method: `run(self, component_name: str, component_path: str) -> Dict`
- Initialize state with component info
- Execute the workflow
- Return final state

Add method: `visualize_workflow(self, output_path: str = "workflow.png")`
- Generate PNG of the workflow graph
- Use LangGraph's built-in visualization

Handle:
- Max iterations (default 3)
- Workflow errors (catch and log)
- Return both success and failure states

Use LangGraph's StateGraph and conditional edges.
```

**Result:** Complete working multi-agent workflow with validation and retry logic

**Why it worked:**
- Step-by-step graph construction
- Conditional edge logic clearly specified
- Error handling requirements explicit
- Visualization capability included

---

### Prompt 6: Pattern Detection

**Goal:** Implement codebase-wide pattern analysis and deviation detection

**Prompt:**
```markdown
Create src/core/pattern_detector.py for pattern analysis.

Class: PatternDetector

1. `analyze_patterns(self, all_components: List[Dict]) -> Dict[str, Any]`
   Extract common patterns:
   - Route naming conventions (e.g., "/api/{resource}")
   - Error handling strategies (try-except, raise, return codes)
   - Authentication patterns (decorators, middleware)
   - Naming conventions (snake_case, PascalCase)
   - Import organization

   Return:
   {
     "route_patterns": ["/api/v1/{resource}", ...],
     "error_handling": {"try_except": 15, "raise": 8},
     "auth_decorators": ["@require_auth", "@admin_only"],
     "naming_convention": "snake_case",
     "consistency_score": 0.85
   }

2. `detect_deviations(self, component: Dict, patterns: Dict) -> List[str]`
   Compare component against detected patterns
   Return list of suggestions:
   - "Route doesn't follow /api/v1/{resource} pattern"
   - "Missing error handling (other components use try-except)"
   - "Function names should use snake_case"

3. `generate_recommendations(self, patterns: Dict) -> str`
   Generate markdown report with:
   - Pattern summary
   - Consistency metrics
   - Improvement suggestions

Integrate into DocumentationAgent:
- Add validation check for pattern compliance
- Include suggestions in generated documentation
- Add flag: `check_patterns` in workflow

Add endpoint: GET /patterns/suggestions/{component}
Add CLI: flow-doc patterns check COMPONENT
```

**Result:** Working pattern detection system that identifies inconsistencies and provides actionable suggestions

**Why it worked:**
- Specific pattern categories defined
- Clear return structures
- Integration points specified
- Actionable output format

---

### Prompt 7: Dependency Visualization

**Goal:** Create interactive and static visualizations of code dependencies

**Prompt:**
```markdown
Create src/core/visualizer.py for dependency visualization.

Class: CodebaseVisualizer

1. `build_dependency_graph(self, components: List[Dict]) -> nx.DiGraph`
   - Create NetworkX directed graph
   - Nodes: components (files, classes, functions)
   - Edges: imports, function calls, inheritance
   - Node attributes: type, size (LOC), docstring_quality
   - Edge attributes: import_type (direct, from)

2. `detect_circular_dependencies(self, graph: nx.DiGraph) -> List[List[str]]`
   - Find cycles in graph
   - Return list of circular dependency chains

3. `generate_plotly_interactive(self, graph: nx.DiGraph, output_path: str)`
   - Create interactive HTML visualization
   - Use Plotly for interactivity (zoom, pan, hover)
   - Color-code by:
     * Node type (file, class, function)
     * Documentation quality (green=good, yellow=partial, red=missing)
   - Show metadata on hover
   - Save to output_path

4. `generate_static_image(self, graph: nx.DiGraph, output_path: str)`
   - Generate PNG using matplotlib
   - Hierarchical layout
   - Save to output_path

5. `export_mermaid(self, graph: nx.DiGraph) -> str`
   - Convert to Mermaid diagram syntax
   - Return string (can be rendered in markdown)

Add to API:
- GET /visualize/interactive → HTML file
- GET /visualize/static → PNG file
- GET /visualize/mermaid → Mermaid syntax

Add to CLI:
- flow-doc visualize --format [interactive|static|mermaid]
- Opens in browser if interactive
```

**Result:** Complete visualization system with multiple output formats

**Why it worked:**
- Multiple output formats specified
- Clear graph structure definition
- Specific libraries mentioned
- Integration endpoints defined

---

### Prompt 8: FastAPI Endpoints

**Goal:** Implement REST API with all core endpoints

**Prompt:**
```markdown
Implement FastAPI endpoints in src/api/app.py and src/api/routes.py.

First, create Pydantic models in src/api/models.py:
- AnalyzeRequest: {"path": str}
- DocumentRequest: {"component": str, "path": str}
- QueryRequest: {"question": str}
- AnalyzeResponse: {"files_found": int, "components": List[str]}
- DocumentResponse: {"success": bool, "markdown": str, "path": str}
- QueryResponse: {"answer": str, "sources": List[str]}

Then implement endpoints in routes.py:

POST /analyze
- Input: codebase path
- Analyze all Python files
- Store in memory
- Return summary

POST /document/{component}
- Input: component name
- Run agentic workflow
- Return generated docs

POST /query
- Input: natural language question
- RAG query over codebase
- Return answer with sources

GET /export
- Export all documentation as ZIP
- Include all docs_output/ files

GET /visualize
- Return component dependency graph
- JSON format

GET /patterns
- Return detected patterns from memory
- Group by type (routes, error handling, etc)

GET /health
- Check Ollama connection
- Check ChromaDB connection
- Return status

In app.py:
- Initialize FastAPI app
- Set up CORS
- Include routes
- Add startup event to initialize services
- Add lifespan context manager for cleanup

Use dependency injection for services (analyzer, memory, agent).
```

**Result:** Complete REST API with all core functionality

**Why it worked:**
- Pydantic models defined first
- Each endpoint specified with input/output
- Clear service initialization
- Dependency injection pattern specified

---

*Document created: 2024-01-15*
*Project: Flow-Doc AI Documentation Assistant*
