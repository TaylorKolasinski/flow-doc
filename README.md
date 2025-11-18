# Flow-Doc

AI-powered documentation assistant for Python codebases. Analyzes Flask/FastAPI applications, generates documentation using local LLMs, detects code patterns, and visualizes dependencies.

## Features

- AST-based code analysis and metadata extraction
- Local LLM documentation generation (Ollama)
- LangGraph multi-agent workflow with validation
- ChromaDB vector storage for semantic search
- Pattern detection and consistency checking
- Dependency graph visualization
- Incremental analysis (only processes changed files)
- REST API and CLI interfaces

## Installation

**Prerequisites**: Python 3.9+, Ollama

**1. Install Ollama and pull model**
```bash
# Install Ollama (see https://ollama.ai/)
ollama pull llama3.2
ollama serve
```

**2. Clone repository**
```bash
git clone <repository-url>
cd flow-doc
```

**3. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Verify installation**
```bash
flow-doc --help
```

## Quick Start

**1. Analyze the sample codebase**
```bash
flow-doc analyze flow_sample_codebase
```
Expected: Should find ~55 components (17 files, 21 classes, 21 functions, 13 routes)

**2. Check code patterns**
```bash
flow-doc patterns check create_order
```
Expected: Shows consistency score and any pattern deviations

**3. Generate documentation**
```bash
flow-doc document create_order
```
Expected: Creates docs_output/create_order.md with component documentation

**4. Query the codebase**
```bash
flow-doc query "What API endpoints exist?"
```
Expected: Lists all API routes with descriptions

**5. Visualize dependencies**
```bash
flow-doc visualize
```
Expected: Creates codebase_dependencies.html (interactive dependency graph)

**6. Test the API (optional)**
```bash
flow-doc server
# In another terminal:
curl http://localhost:8000/docs
```

## Project Structure

```
flow-doc/
├── src/
│   ├── core/
│   │   ├── analyzer.py          # AST parsing and code extraction
│   │   ├── memory.py             # ChromaDB vector storage
│   │   ├── generator.py          # LLM documentation generation
│   │   ├── agent.py              # LangGraph workflow orchestration
│   │   ├── pattern_detector.py  # Code pattern analysis
│   │   └── visualizer.py         # Dependency graph visualization
│   ├── api/
│   │   ├── app.py                # FastAPI application
│   │   ├── routes.py             # API endpoints
│   │   └── models.py             # Pydantic request/response models
│   └── cli.py                    # Click CLI interface
├── tests/
│   ├── conftest.py               # Shared test fixtures
│   ├── test_visualizer.py        # Visualization tests (25 tests)
│   └── test_pattern_detector.py  # Pattern detection tests (29 tests)
├── docs/
│   └── ARCHITECTURE.md           # System design and workflow
├── sample_codebase/
│   ├── flask_app.py              # Sample Flask application
│   └── fastapi_app.py            # Sample FastAPI application
├── docs_output/                  # Generated documentation
├── .github/workflows/test.yml    # CI/CD pipeline
└── prompts_used.md               # Implementation approach
```

## Technology Stack

**Core**:
- Python 3.9+
- LangChain + LangGraph (multi-agent workflows)
- Ollama (local LLM inference)
- ChromaDB (vector database)

**Analysis**:
- AST module (code parsing)
- NetworkX (dependency graphs)

**Visualization**:
- Plotly (interactive graphs)
- Matplotlib (static diagrams)

**API/CLI**:
- FastAPI (REST API)
- Click (CLI framework)
- Rich (terminal formatting)

**Testing**:
- pytest + pytest-cov (54 tests, >80% coverage)
- GitHub Actions (multi-version CI/CD)

## CLI Commands

**Analysis**
- `flow-doc analyze <path>` - Analyze codebase and extract components
- `flow-doc analyze <path> --incremental` - Only process changed files

**Documentation Generation**
- `flow-doc generate <component> <file>` - Generate docs for specific component
- `flow-doc generate-all` - Generate docs for all components
- `flow-doc generate-all --filter type:route` - Filter by component type

**Pattern Detection**
- `flow-doc patterns` - Show codebase pattern analysis
- `flow-doc patterns check <component>` - Check specific component for deviations

**Visualization**
- `flow-doc visualize` - Generate interactive dependency graph (HTML)
- `flow-doc visualize --format png` - Export as PNG image
- `flow-doc visualize --format mermaid` - Export as Mermaid diagram
- `flow-doc visualize --open-browser` - Open HTML visualization in browser

**API Server**
- `flow-doc serve` - Start FastAPI server on port 8000

## API Endpoints

Start server: `flow-doc serve`

**Core endpoints**:
- `GET /health` - System health check
- `POST /analyze` - Analyze codebase
- `POST /document` - Generate documentation
- `POST /query` - RAG query over codebase
- `GET /patterns` - Pattern analysis
- `GET /patterns/suggestions/{component}` - Get pattern suggestions
- `GET /visualize` - Dependency graph data

Documentation: `http://localhost:8000/docs`

## Configuration

**Environment Variables**:
- `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL` - Model to use (default: llama3.2)
- `CHROMA_PERSIST_DIRECTORY` - ChromaDB data directory (default: ./data)

**LLM Settings** (`src/core/generator.py`):
```python
DEFAULT_MODEL = "llama3.2"
DEFAULT_TEMPERATURE = 0.1  # Low for factual documentation
```

## Testing

Run the existing test suite:
```bash
pytest tests/ -v
```

Expected: 54 tests should pass (25 visualizer tests, 29 pattern detector tests)

## Implementation Details

This project demonstrates:

1. **Multi-Agent Workflow**: LangGraph orchestrates a 6-node workflow (analyze, check patterns, retrieve context, generate, validate, store)

2. **Pattern Detection**: Analyzes route patterns, naming conventions, error handling, documentation coverage, and provides deviation suggestions

3. **RAG Architecture**: ChromaDB + Ollama embeddings for semantic code search and context-aware documentation generation

4. **Incremental Analysis**: SHA-256 checksums track file changes to avoid re-processing unchanged code

5. **Validation Loop**: Generated documentation is validated and regenerated (up to max iterations) until quality standards are met

See `prompts_used.md` for detailed implementation approach and prompt engineering techniques used.
