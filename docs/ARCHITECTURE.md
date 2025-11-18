# Architecture

## System Overview

Flow-Doc uses a multi-agent workflow to analyze Python codebases and generate documentation. The system consists of three main layers:

1. **Analysis Layer**: AST parsing and code extraction
2. **Processing Layer**: LangGraph workflow orchestration with LLM generation
3. **Storage Layer**: ChromaDB vector database for semantic search

```
┌─────────────────────────────────────────────┐
│              User Interfaces                 │
│  CLI (Click)  |  REST API (FastAPI)         │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│          DocumentationAgent                  │
│         (LangGraph Workflow)                 │
│                                              │
│  Analyze → Check Patterns → Retrieve →      │
│  Generate → Validate → Store                 │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│            Core Services                     │
│  Analyzer | Memory | Generator | Patterns   │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│         External Services                    │
│     Ollama (LLM) | ChromaDB (Vectors)       │
└─────────────────────────────────────────────┘
```

## Core Components

### 1. CodebaseAnalyzer (`src/core/analyzer.py`)

Parses Python files using AST module to extract:
- Functions (params, return types, decorators, docstrings)
- Classes (methods, bases, attributes)
- Routes (Flask/FastAPI decorators, HTTP methods, paths)
- Metadata (imports, complexity, LOC)

**Key Method**: `parse_file(file_path)` → Returns structured component data

### 2. MemoryStore (`src/core/memory.py`)

ChromaDB wrapper for vector storage:
- Stores code components with embeddings (Ollama)
- Semantic search via `retrieve_similar(query, k)`
- Persistent storage with metadata filtering

### 3. DocumentationAgent (`src/core/agent.py`)

LangGraph workflow orchestrator with 6 nodes:

**Node 1: Analyze** - Parse code with AST
**Node 2: Check Patterns** - Detect deviations from codebase patterns
**Node 3: Retrieve** - Semantic search for similar components
**Node 4: Generate** - LLM generates documentation with context
**Node 5: Validate** - Check completeness, accuracy, format
**Node 6: Store** - Save to file and memory

**Workflow State**:
```python
{
    "component_name": str,
    "component_path": str,
    "code_metadata": Dict,
    "pattern_suggestions": List[str],
    "retrieved_context": List[Dict],
    "draft_documentation": str,
    "validation_result": Dict,
    "iteration_count": int,
    "max_iterations": int
}
```

**Control Flow**: After validation, either store (if passed) or retry generation (if failed and under max iterations)

### 4. PatternDetector (`src/core/pattern_detector.py`)

Analyzes codebase patterns:
- Route patterns (versioning, prefixes, HTTP methods)
- Naming conventions (snake_case, camelCase, PascalCase)
- Error handling (try/except, raises clauses)
- Documentation coverage (docstring presence/quality)

Returns consistency score (0.0-1.0) and deviation suggestions.

### 5. DocumentationGenerator (`src/core/generator.py`)

LLM-based documentation generation:
- Builds structured prompts with code + context + patterns
- Invokes Ollama with low temperature (0.1) for factual output
- Supports RAG queries for codebase Q&A

### 6. CodebaseVisualizer (`src/core/visualizer.py`)

Dependency graph generation:
- Builds NetworkX directed graph from imports
- Detects circular dependencies
- Exports to HTML (Plotly), PNG (Matplotlib), Mermaid

## LangGraph Workflow

The documentation generation follows a stateful workflow:

```
START → Analyze → Check Patterns → Retrieve → Generate
                                                    ↓
                                                Validate
                                                    ↓
                                        ┌───────────┴──────────┐
                                        ↓                      ↓
                                   [Valid/MaxIter]        [Invalid]
                                        ↓                      ↓
                                      Store              Retry Generate
                                        ↓                      ↓
                                       END ←──────────────────┘
```

**Key Features**:
- State persists across all nodes
- Validation drives retry logic (up to max_iterations)
- Pattern suggestions included in generation prompt
- Context from similar components improves consistency

## Key Design Decisions

### 1. Local-First Architecture

**Decision**: Use Ollama instead of cloud APIs (OpenAI, Anthropic)

**Rationale**: Privacy, no API costs, offline capability

**Tradeoff**: Requires local GPU/CPU resources, slower than cloud

### 2. Multi-Agent Workflow (LangGraph)

**Decision**: Use LangGraph for workflow orchestration instead of simple pipeline

**Rationale**:
- Stateful execution with retry logic
- Conditional branching based on validation
- Easy to extend with new nodes

**Alternative Considered**: Simple function pipeline, but lacked retry and state management

### 3. Vector Storage for Context

**Decision**: Store all components in ChromaDB for semantic search

**Rationale**:
- Retrieve similar components as examples for LLM
- Enables RAG-based codebase Q&A
- Maintains consistency across documentation

**Implementation**: Each component stored with Ollama embeddings

### 4. Incremental Analysis

**Decision**: Track file checksums (SHA-256) to detect changes

**Rationale**: Large codebases don't need full re-analysis on every run

**Implementation**: Store checksums in JSON, compare on each analysis, only process changed files

### 5. Pattern Detection Integration

**Decision**: Add pattern checking as separate workflow node

**Rationale**:
- Separate concern from code analysis
- Optional (can be disabled per-request)
- Suggestions included in LLM prompt to improve generated docs

## Data Flow

**Analysis Flow**:
```
Python Files → AST Parser → Component Metadata → MemoryStore → ChromaDB
```

**Generation Flow**:
```
Component Name → Analyze → Pattern Check → Retrieve Context
                                                ↓
                                          Build Prompt
                                                ↓
                                           Ollama LLM
                                                ↓
                                          Validate → Store/Retry
```

**Query Flow**:
```
Question → Semantic Search → Retrieve Top-K → Build Prompt → LLM → Answer
```

## Technology Choices

- **LangChain/LangGraph**: Multi-agent workflows with state management
- **ChromaDB**: Local vector database with good Python integration
- **Ollama**: Local LLM inference with OpenAI-compatible API
- **FastAPI**: Modern async web framework with automatic OpenAPI docs
- **Click**: CLI framework with good argument parsing and help generation
- **NetworkX**: Graph algorithms library for dependency analysis
- **Plotly**: Interactive visualizations without server requirement

## Scalability Considerations

**Current Limitations**:
- Single-threaded workflow execution
- In-memory component storage during analysis
- No distributed ChromaDB support

**Scaling Strategies**:
- Parallel workflow execution for batch generation
- Batch embedding generation (reduce Ollama calls)
- Incremental analysis for large codebases
- Distributed ChromaDB for enterprise scale
