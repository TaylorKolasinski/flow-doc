#!/usr/bin/env python3
"""
Flow-Doc CLI - Command-line interface for automated documentation generation.

This module provides a comprehensive CLI using Click for all Flow-Doc functionality.
"""

import sys
import os
import zipfile
import webbrowser
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich import box

from src.core.analyzer import CodebaseAnalyzer
from src.core.memory import MemoryStore
from src.core.generator import DocumentationGenerator
from src.core.agent import DocumentationAgent
from src.core.pattern_detector import PatternDetector

# Import uvicorn for server command (optional dependency)
try:
    import uvicorn
except ImportError:
    uvicorn = None

# Initialize rich console
console = Console()

# Global context for sharing services between commands
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


class Context:
    """Shared context for CLI commands."""

    def __init__(self):
        self.analyzer: Optional[CodebaseAnalyzer] = None
        self.memory: Optional[MemoryStore] = None
        self.generator: Optional[DocumentationGenerator] = None
        self.agent: Optional[DocumentationAgent] = None
        self.codebase_path: Optional[str] = None
        self.verbose: bool = False


pass_context = click.make_pass_decorator(Context, ensure=True)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--codebase', '-c', default='./flow_sample_codebase', help='Codebase path')
@pass_context
def cli(ctx: Context, verbose: bool, codebase: str):
    """
    Flow-Doc: AI-Powered Documentation Assistant

    Automatically generate comprehensive documentation for your Flask/FastAPI
    codebases using LangGraph, Ollama, and ChromaDB.

    Examples:

        flow-doc analyze ./my-project

        flow-doc document UserService

        flow-doc query "What authentication is used?"

        flow-doc export --format zip
    """
    ctx.verbose = verbose
    ctx.codebase_path = codebase


# Alias for backward compatibility with tests
app = cli


# ============================================================================
# ANALYZE COMMAND
# ============================================================================

@cli.command()
@click.argument('path', default='.', type=click.Path(exists=True))
@click.option('--detailed', '-d', is_flag=True, help='Show detailed component information')
@click.option('--incremental', '-i', is_flag=True, help='Only analyze changed files')
@pass_context
def analyze(ctx: Context, path: str, detailed: bool, incremental: bool):
    """
    Analyze a codebase and store components in memory.

    Scans all Python files, extracts functions, classes, and routes,
    and stores them in ChromaDB with semantic embeddings.

    Examples:

        flow-doc analyze

        flow-doc analyze ./my-project

        flow-doc analyze . --detailed

        flow-doc analyze . --incremental
    """
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Analyzing Codebase{' (Incremental)' if incremental else ''}[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    try:
        # Initialize services
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing services...", total=None)

            ctx.analyzer = CodebaseAnalyzer(path)
            ctx.memory = MemoryStore(persist_directory="./data")
            progress.update(task, description="✓ Services initialized")

        # Detect changes if incremental mode
        files_to_process = []
        if incremental:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Detecting changes...", total=None)
                changes = ctx.analyzer.detect_changes()
                progress.update(task, description=f"✓ Detected {len(changes)} changes")

            # Display changes
            if changes:
                console.print()
                console.print("[bold]File Changes:[/bold]")
                for change in changes[:10]:
                    status = change['status']
                    path_str = change['path']
                    if status == "new":
                        console.print(f"  [green]+ {path_str}[/green] (new)")
                    elif status == "modified":
                        console.print(f"  [yellow]~ {path_str}[/yellow] (modified)")
                    elif status == "deleted":
                        console.print(f"  [red]- {path_str}[/red] (deleted)")

                if len(changes) > 10:
                    console.print(f"  ... and {len(changes) - 10} more")
                console.print()

                # Only process changed files (new and modified, not deleted)
                changed_paths = {c['path'] for c in changes if c['status'] in ['new', 'modified']}
                all_files = ctx.analyzer.scan_directory()
                files_to_process = [f for f in all_files if f['path'] in changed_paths]

                if not files_to_process:
                    console.print("[yellow]No files to analyze (all changes were deletions)[/yellow]")
                    return
            else:
                console.print()
                console.print("[green]✓ No changes detected - codebase up to date![/green]")
                console.print()
                return
        else:
            # Full analysis - scan all files
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Scanning directory...", total=None)
                files_to_process = ctx.analyzer.scan_directory()
                progress.update(task, description=f"✓ Found {len(files_to_process)} Python files")

        files = files_to_process

        # Parse files with progress bar
        components = []
        components_by_type = {"class": 0, "function": 0, "route": 0}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Parsing files...", total=len(files))

            for file_info in files:
                try:
                    parsed = ctx.analyzer.parse_file(file_info['path'])
                    if not parsed:
                        progress.advance(task)
                        continue

                    # Extract and store components
                    for cls in parsed.get('classes', []):
                        component = {
                            "type": "class",
                            "name": cls['name'],
                            "file_path": file_info['path'],
                            "methods": cls.get('methods', []),
                            "docstring": cls.get('docstring', '')
                        }
                        ctx.memory.store_component(component)
                        components.append(component)
                        components_by_type['class'] += 1

                    for func in parsed.get('functions', []):
                        component = {
                            "type": "function",
                            "name": func['name'],
                            "file_path": file_info['path'],
                            "params": func.get('params', []),
                            "docstring": func.get('docstring', '')
                        }
                        ctx.memory.store_component(component)
                        components.append(component)
                        components_by_type['function'] += 1

                    for route in parsed.get('routes', []):
                        component = {
                            "type": "route",
                            "name": route['handler'],
                            "file_path": file_info['path'],
                            "path": route['path'],
                            "methods": route.get('methods', [])
                        }
                        ctx.memory.store_component(component)
                        components.append(component)
                        components_by_type['route'] += 1

                except Exception as e:
                    if ctx.verbose:
                        console.print(f"[yellow]Warning: Failed to parse {file_info['path']}: {e}[/yellow]")

                progress.advance(task)

        # Display summary
        console.print()
        table = Table(title="Analysis Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Count", style="green", justify="right")

        table.add_row("Python Files", str(len(files)))
        table.add_row("Total Components", str(len(components)))
        table.add_row("├─ Classes", str(components_by_type['class']))
        table.add_row("├─ Functions", str(components_by_type['function']))
        table.add_row("└─ Routes", str(components_by_type['route']))

        console.print(table)

        # Detailed view
        if detailed and components:
            console.print()
            console.print("[bold]Component Details:[/bold]")
            console.print()

            for comp in components[:10]:  # Show first 10
                comp_type = comp['type']
                comp_name = comp['name']
                comp_file = comp['file_path']

                if comp_type == "class":
                    methods = ', '.join(comp.get('methods', [])[:3])
                    console.print(f"[cyan]class[/cyan] {comp_name} ({comp_file})")
                    if methods:
                        console.print(f"  └─ methods: {methods}")
                elif comp_type == "function":
                    params = ', '.join(comp.get('params', []))
                    console.print(f"[green]def[/green] {comp_name}({params}) ({comp_file})")
                elif comp_type == "route":
                    methods = ', '.join(comp.get('methods', ['GET']))
                    console.print(f"[yellow]route[/yellow] {methods} {comp.get('path')} → {comp_name} ({comp_file})")

            if len(components) > 10:
                console.print(f"\n... and {len(components) - 10} more")

        console.print()
        console.print("[bold green]✓ Analysis complete![/bold green]")
        console.print()

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        if ctx.verbose:
            console.print_exception()
        sys.exit(1)


# ============================================================================
# DOCUMENT COMMAND
# ============================================================================

@cli.command()
@click.argument('component')
@click.option('--path', '-p', default=None, help='Component file path')
@click.option('--output-dir', '-o', default='docs_output', help='Output directory')
@click.option('--max-iterations', '-i', default=3, type=int, help='Maximum refinement iterations')
@click.option('--show-markdown', '-m', is_flag=True, help='Display generated markdown')
@click.option('--incremental', is_flag=True, help='Skip if component unchanged')
@pass_context
def document(ctx: Context, component: str, path: Optional[str], output_dir: str, max_iterations: int, show_markdown: bool, incremental: bool):
    """
    Generate documentation for a specific component.

    Uses the LangGraph agentic workflow to generate high-quality documentation
    with iterative refinement and quality validation.

    Examples:

        flow-doc document UserService

        flow-doc document flask_app --path flask_app.py

        flow-doc document MyClass --max-iterations 5 --show-markdown

        flow-doc document MyClass --incremental
    """
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Generating Documentation: {component}[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    try:
        # Initialize services first (needed to query memory)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing workflow...", total=None)

            ctx.analyzer = CodebaseAnalyzer(ctx.codebase_path)
            ctx.memory = MemoryStore(persist_directory="./data")
            ctx.generator = DocumentationGenerator(ctx.memory)
            ctx.agent = DocumentationAgent(ctx.analyzer, ctx.memory, ctx.generator)

            progress.update(task, description="✓ Workflow initialized")

        # Auto-detect path if not provided
        if not path:
            # First, try to find the component in memory
            console.print(f"[dim]Searching for {component} in memory...[/dim]")

            # Get all components from memory
            all_components = ctx.memory.get_all_components()

            # Search for matching component by name
            matching_component = None
            for comp in all_components:
                if comp.get('name') == component:
                    matching_component = comp
                    break

            if matching_component and matching_component.get('file_path'):
                # Use the file path from memory
                path = matching_component['file_path']
                console.print(f"[dim]Found in memory: {path}[/dim]")
            else:
                # Fallback: Try common patterns
                console.print(f"[yellow]Component not found in memory. Trying common path patterns...[/yellow]")
                possible_paths = [
                    f"{component}.py",
                    f"{component.lower()}.py",
                    f"src/{component}.py",
                    f"src/{component.lower()}.py",
                ]

                for p in possible_paths:
                    check_path = Path(ctx.codebase_path) / p
                    if check_path.exists() or Path(p).exists():
                        path = p
                        break

                if not path:
                    console.print("[yellow]Path not found. Using component name as filename.[/yellow]")
                    path = f"{component}.py"

        # Run workflow
        console.print(f"[dim]Component:[/dim] {component}")
        console.print(f"[dim]Path:[/dim] {path}")
        console.print(f"[dim]Max Iterations:[/dim] {max_iterations}")
        console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running workflow...", total=None)

            result = ctx.agent.run(
                component_name=component,
                component_path=path,
                max_iterations=max_iterations,
                incremental=incremental
            )

            progress.update(task, description="✓ Workflow complete")

        # Check if skipped
        if result.get('skipped'):
            console.print()
            console.print(f"[yellow]⊘ Skipped {component} - no changes detected[/yellow]")
            console.print(f"[dim]Reason: {result.get('reason', 'Unknown')}[/dim]")
            console.print()
            return

        # Display results
        console.print()
        status_color = "green" if result['success'] else "yellow"
        status_icon = "✓" if result['success'] else "⚠"

        table = Table(title="Generation Results", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style=status_color)

        table.add_row("Status", f"{status_icon} {'Success' if result['success'] else 'Completed with issues'}")
        table.add_row("Iterations", str(result['iterations']))
        table.add_row("Validation Score", f"{result['validation_score']:.2f}")
        table.add_row("Validation Passed", "Yes" if result['validation_passed'] else "No")
        table.add_row("Output File", result['output_file'])
        table.add_row("Doc Length", f"{result['doc_length']} characters")

        if result['errors']:
            table.add_row("Errors", str(len(result['errors'])))

        console.print(table)

        # Show errors if any
        if result['errors']:
            console.print()
            console.print("[bold yellow]Errors:[/bold yellow]")
            for i, error in enumerate(result['errors'], 1):
                console.print(f"  {i}. {error}")

        # Show markdown if requested
        if show_markdown and result['success']:
            output_path = Path(result['output_file'])
            if output_path.exists():
                console.print()
                console.print(Panel.fit("[bold]Generated Documentation[/bold]", border_style="cyan"))
                console.print()

                with open(output_path, 'r') as f:
                    content = f.read()

                # Show first 500 characters
                preview = content[:500]
                if len(content) > 500:
                    preview += "\n\n[... truncated ...]"

                md = Markdown(preview)
                console.print(md)

        console.print()
        console.print(f"[bold green]✓ Documentation saved to: {result['output_file']}[/bold green]")
        console.print()

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        if ctx.verbose:
            console.print_exception()
        sys.exit(1)


# ============================================================================
# QUERY COMMAND
# ============================================================================

@cli.command()
@click.argument('question')
@click.option('--k', '-k', default=5, type=int, help='Number of similar components to retrieve')
@pass_context
def query(ctx: Context, question: str, k: int):
    """
    Ask natural language questions about the codebase.

    Uses RAG (Retrieval Augmented Generation) to answer questions based on
    components stored in memory.

    Examples:

        flow-doc query "What authentication methods are used?"

        flow-doc query "How does the UserService work?" --k 10

        flow-doc query "List all API endpoints"
    """
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Querying Codebase[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    try:
        # Initialize services
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing...", total=None)

            ctx.memory = MemoryStore(persist_directory="./data")
            ctx.generator = DocumentationGenerator(ctx.memory)

            progress.update(task, description="✓ Services initialized")

        # Display question
        console.print(f"[bold]Question:[/bold] {question}")
        console.print()

        # Generate answer
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating answer...", total=None)

            answer = ctx.generator.simple_query(question, k=k)
            similar = ctx.memory.retrieve_similar(question, k=k)

            progress.update(task, description="✓ Answer generated")

        # Display answer
        console.print(Panel(
            Markdown(answer),
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))

        # Display sources
        if similar:
            console.print()
            console.print("[bold]Sources:[/bold]")
            console.print()

            for i, comp in enumerate(similar, 1):
                metadata = comp.get('metadata', {})
                file_path = metadata.get('file_path', 'unknown')
                name = metadata.get('name', 'unknown')
                comp_type = metadata.get('type', 'unknown')
                score = comp.get('score', 0.0)

                # Color by type
                type_colors = {
                    'class': 'cyan',
                    'function': 'green',
                    'route': 'yellow'
                }
                color = type_colors.get(comp_type, 'white')

                console.print(f"  {i}. [{color}]{comp_type}[/{color}] {name} ({file_path}) - Score: {score:.2f}")

        console.print()

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        if ctx.verbose:
            console.print_exception()
        sys.exit(1)


# ============================================================================
# EXPORT COMMAND
# ============================================================================

@cli.command()
@click.option('--format', '-f', 'fmt', type=click.Choice(['markdown', 'zip'], case_sensitive=False), default='zip', help='Export format')
@click.option('--output', '-o', default=None, help='Output file path')
@pass_context
def export(ctx: Context, fmt: str, output: Optional[str]):
    """
    Export all generated documentation.

    Exports all documentation files from docs_output/ directory.

    Examples:

        flow-doc export

        flow-doc export --format markdown

        flow-doc export --format zip --output my-docs.zip
    """
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Exporting Documentation[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    try:
        docs_dir = Path("docs_output")

        if not docs_dir.exists():
            console.print("[bold red]✗ Error:[/bold red] No documentation found.")
            console.print("[dim]Run 'flow-doc document' to generate documentation first.[/dim]")
            sys.exit(1)

        md_files = list(docs_dir.glob("*.md"))

        if not md_files:
            console.print("[bold red]✗ Error:[/bold red] No documentation files found.")
            sys.exit(1)

        console.print(f"Found {len(md_files)} documentation files")
        console.print()

        if fmt == 'markdown':
            # Just list the files
            table = Table(title="Documentation Files", box=box.ROUNDED)
            table.add_column("File", style="cyan")
            table.add_column("Size", style="green", justify="right")

            for md_file in md_files:
                size = md_file.stat().st_size
                size_str = f"{size:,} bytes"
                table.add_row(md_file.name, size_str)

            console.print(table)
            console.print()
            console.print(f"[bold]Location:[/bold] {docs_dir.absolute()}")

        elif fmt == 'zip':
            # Create ZIP file
            if not output:
                output = "flow-doc-documentation.zip"

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Creating ZIP archive...", total=len(md_files))

                with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for md_file in md_files:
                        zipf.write(md_file, arcname=md_file.name)
                        progress.advance(task)

            console.print()
            console.print(f"[bold green]✓ Exported to:[/bold green] {output}")

            # Show file size
            output_path = Path(output)
            size = output_path.stat().st_size
            size_kb = size / 1024
            console.print(f"[dim]Size: {size_kb:.2f} KB ({len(md_files)} files)[/dim]")

        console.print()

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        if ctx.verbose:
            console.print_exception()
        sys.exit(1)


# ============================================================================
# VISUALIZE COMMAND
# ============================================================================

@cli.command()
@click.option('--output', '-o', default='codebase_dependencies', help='Output file (without extension)')
@click.option('--format', '-f', 'fmt', type=click.Choice(['html', 'png', 'mermaid'], case_sensitive=False), default='html', help='Output format')
@click.option('--open-browser', '-b', is_flag=True, help='Open visualization in browser')
@pass_context
def visualize(ctx: Context, output: str, fmt: str, open_browser: bool):
    """
    Generate codebase dependency graph visualization.

    Creates a visual representation of component dependencies showing:
    - Routes, classes, and functions as nodes
    - Import relationships as edges
    - Color-coding by component type
    - Documentation quality indicators

    Examples:

        flow-doc visualize

        flow-doc visualize --format png --output my-deps

        flow-doc visualize --format mermaid

        flow-doc visualize --open-browser
    """
    from src.core.visualizer import CodebaseVisualizer
    import webbrowser

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Generating Dependency Visualization[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    try:
        # Initialize services
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing...", total=None)

            ctx.memory = MemoryStore(persist_directory="./data")
            visualizer = CodebaseVisualizer()

            progress.update(task, description="✓ Services initialized")

        # Get all components from memory
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading components...", total=None)

            components = ctx.memory.get_all_components()

            if not components:
                console.print()
                console.print("[yellow]⚠ No components found in memory.[/yellow]")
                console.print("[dim]Run 'flow-doc analyze <path>' first to analyze a codebase.[/dim]")
                console.print()
                sys.exit(1)

            progress.update(task, description=f"✓ Loaded {len(components)} components")

        # Build dependency graph
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Building dependency graph...", total=None)

            graph = visualizer.build_dependency_graph(components)

            progress.update(task, description=f"✓ Graph built ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")

        # Generate visualization based on format
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating visualization...", total=None)

            if fmt == 'html':
                output_file = f"{output}.html"
                visualizer.generate_plotly_interactive(graph, output_file)
            elif fmt == 'png':
                output_file = f"{output}.png"
                visualizer.generate_static_image(graph, output_file)
            elif fmt == 'mermaid':
                output_file = f"{output}.mmd"
                mermaid_diagram = visualizer.export_mermaid(graph)
                Path(output_file).write_text(mermaid_diagram)

            progress.update(task, description="✓ Visualization created")

        console.print()
        console.print(f"[bold green]✓ Visualization saved to:[/bold green] {output_file}")

        # Show file info
        if Path(output_file).exists():
            size = Path(output_file).stat().st_size
            console.print(f"[dim]Size: {size:,} bytes[/dim]")
            console.print(f"[dim]Format: {fmt.upper()}[/dim]")
            console.print(f"[dim]Components: {len(components)}[/dim]")

        # Open in browser if requested
        if open_browser and fmt == 'html':
            console.print()
            console.print("[cyan]Opening in browser...[/cyan]")
            webbrowser.open(f"file://{Path(output_file).absolute()}")

        console.print()

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        if ctx.verbose:
            console.print_exception()
        sys.exit(1)


# ============================================================================
# PATTERNS COMMAND (moved to group below at line 934)
# ============================================================================

# Removed duplicate - see patterns group below at line ~860


# ============================================================================
# SERVER COMMAND
# ============================================================================

@cli.command()
@click.option('--host', '-h', default='0.0.0.0', help='Host to bind to')
@click.option('--port', '-p', default=8000, type=int, help='Port to bind to')
@click.option('--reload', '-r', is_flag=True, help='Enable auto-reload')
@click.option('--workers', '-w', default=1, type=int, help='Number of workers')
@pass_context
def server(ctx: Context, host: str, port: int, reload: bool, workers: int):
    """
    Start the FastAPI server.

    Launches the Flow-Doc REST API server for programmatic access.

    Examples:

        flow-doc server

        flow-doc server --port 9000

        flow-doc server --reload

        flow-doc server --workers 4
    """
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Starting Flow-Doc API Server[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    try:
        if uvicorn is None:
            console.print("[bold red]Error:[/bold red] uvicorn is not installed")
            console.print("[dim]Install it with: pip install uvicorn[standard][/dim]")
            sys.exit(1)

        console.print(f"[bold]Host:[/bold] {host}")
        console.print(f"[bold]Port:[/bold] {port}")
        console.print(f"[bold]Workers:[/bold] {workers}")
        console.print(f"[bold]Reload:[/bold] {'Enabled' if reload else 'Disabled'}")
        console.print()

        console.print(f"[dim]Documentation:[/dim] http://{host}:{port}/docs")
        console.print(f"[dim]Health Check:[/dim] http://{host}:{port}/api/v1/health")
        console.print()

        console.print("[green]Starting server... Press Ctrl+C to stop[/green]")
        console.print()

        uvicorn.run(
            "src.api.app:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level="info"
        )

    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Server stopped by user[/yellow]")
        console.print()
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        if ctx.verbose:
            console.print_exception()
        sys.exit(1)


# ============================================================================
# PATTERNS COMMAND GROUP
# ============================================================================

@cli.group(invoke_without_command=True)
@click.option('--type', '-t', 'pattern_type', type=click.Choice(['routes', 'functions', 'classes', 'all'], case_sensitive=False), default='all', help='Pattern type to display')
@click.pass_context
def patterns(click_ctx, pattern_type: str):
    """
    Analyze and check code patterns.

    When called without a subcommand, shows a summary of detected patterns.
    Commands for detecting code patterns and finding deviations from
    codebase consistency standards.

    Examples:
        flow-doc patterns                    # Show all patterns
        flow-doc patterns --type routes      # Show only route patterns
        flow-doc patterns check UserService  # Check specific component
    """
    from src.core.pattern_detector import PatternDetector
    from rich.table import Table

    # If a subcommand is being invoked, don't run this code
    if click_ctx.invoked_subcommand is not None:
        return

    # Show pattern summary when no subcommand is provided
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Codebase Pattern Analysis[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    try:
        # Initialize services
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing patterns...", total=None)

            memory = MemoryStore(persist_directory="./data")
            pattern_detector = PatternDetector()

            # Get all components from memory
            components = memory.get_all_components()

            if not components:
                progress.stop()
                console.print()
                console.print("[yellow]⚠ No components found in memory.[/yellow]")
                console.print("[dim]Run 'flow-doc analyze <path>' first to analyze a codebase.[/dim]")
                console.print()
                sys.exit(1)

            # Analyze patterns
            patterns_data = pattern_detector.analyze_patterns(components)

            progress.update(task, description="✓ Analysis complete")

        console.print()

        # Display overall statistics
        stats_table = Table(title="📊 Overall Statistics", show_header=True, header_style="bold cyan")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total Components", str(patterns_data['total_components']))
        components_by_type = patterns_data.get('components_by_type', {})
        stats_table.add_row("├─ Classes", str(components_by_type.get('class', 0)))
        stats_table.add_row("├─ Functions", str(components_by_type.get('function', 0)))
        stats_table.add_row("└─ Routes", str(components_by_type.get('route', 0)))

        # Color-code consistency score
        consistency = patterns_data['consistency_score']
        if consistency >= 0.8:
            consistency_color = "green"
        elif consistency >= 0.6:
            consistency_color = "yellow"
        else:
            consistency_color = "red"
        stats_table.add_row("Consistency Score", f"[{consistency_color}]{consistency:.1%}[/{consistency_color}]")

        console.print(stats_table)
        console.print()

        # Route Patterns (if type is 'routes' or 'all')
        if pattern_type in ['routes', 'all']:
            route_patterns = patterns_data['route_patterns']
            if route_patterns.get('total_routes', 0) > 0:
                route_table = Table(title="🛣️  Route Patterns", show_header=True, header_style="bold cyan")
                route_table.add_column("Metric", style="cyan")
                route_table.add_column("Value", style="green")

                route_table.add_row("Total Routes", str(route_patterns['total_routes']))
                route_table.add_row("Uses /api/ Prefix", "✓ Yes" if route_patterns.get('api_prefix') else "✗ No")
                route_table.add_row("Versioned APIs", "✓ Yes" if route_patterns.get('versioned') else "✗ No")

                # HTTP methods distribution
                methods = route_patterns.get('methods_used', {})
                if methods:
                    method_str = ", ".join([f"{k}: {v}" for k, v in sorted(methods.items())])
                    route_table.add_row("HTTP Methods", method_str)

                # Common patterns
                common_patterns = route_patterns.get('patterns', [])
                if common_patterns:
                    route_table.add_row("Common Patterns", common_patterns[0] if common_patterns else "None")
                    for pattern in common_patterns[1:3]:
                        route_table.add_row("", pattern)

                console.print(route_table)
                console.print()

        # Naming Conventions (if type is 'functions', 'classes', or 'all')
        if pattern_type in ['functions', 'classes', 'all']:
            naming = patterns_data['naming_conventions']
            naming_table = Table(title="📝 Naming Conventions", show_header=True, header_style="bold cyan")
            naming_table.add_column("Type", style="cyan")
            naming_table.add_column("Convention", style="green")
            naming_table.add_column("Consistency", style="yellow")

            if pattern_type in ['functions', 'all']:
                func_consistency = naming['function_consistency']
                func_color = "green" if func_consistency >= 0.8 else "yellow" if func_consistency >= 0.6 else "red"
                naming_table.add_row(
                    "Functions",
                    naming['function_convention'],
                    f"[{func_color}]{func_consistency:.1%}[/{func_color}]"
                )

            if pattern_type in ['classes', 'all']:
                class_consistency = naming['class_consistency']
                class_color = "green" if class_consistency >= 0.8 else "yellow" if class_consistency >= 0.6 else "red"
                naming_table.add_row(
                    "Classes",
                    naming['class_convention'],
                    f"[{class_color}]{class_consistency:.1%}[/{class_color}]"
                )

            console.print(naming_table)
            console.print()

        # Documentation Coverage (always show)
        if pattern_type == 'all':
            docs = patterns_data['documentation_patterns']
            doc_table = Table(title="📚 Documentation Coverage", show_header=True, header_style="bold cyan")
            doc_table.add_column("Metric", style="cyan")
            doc_table.add_column("Value", style="green")

            doc_rate = docs['documentation_rate']
            doc_color = "green" if doc_rate >= 80 else "yellow" if doc_rate >= 50 else "red"

            doc_table.add_row("Total Components", str(docs['total_components']))
            doc_table.add_row("With Docstrings", f"{docs['documented_count']} ({doc_rate:.1f}%)")
            doc_table.add_row("Coverage", f"[{doc_color}]{doc_rate:.1f}%[/{doc_color}]")
            doc_table.add_row("Dominant Style", docs.get('dominant_style', 'none'))

            if docs['average_length'] > 0:
                doc_table.add_row("Avg Length", f"{docs['average_length']:.0f} chars")

            console.print(doc_table)
            console.print()

        # Error Handling (always show)
        if pattern_type == 'all':
            error_handling = patterns_data['error_handling']
            error_table = Table(title="⚠️  Error Handling", show_header=True, header_style="bold cyan")
            error_table.add_column("Pattern", style="cyan")
            error_table.add_column("Count", style="green")

            error_table.add_row("Try-Except Blocks", str(error_handling.get('try_except', 0)))
            error_table.add_row("Raise Statements", str(error_handling.get('raise', 0)))
            error_table.add_row("Return Codes", str(error_handling.get('return_codes', 0)))
            error_table.add_row("Logging", str(error_handling.get('logging', 0)))

            console.print(error_table)
            console.print()

        # Summary message
        console.print(Panel(
            f"[bold green]✓[/bold green] Pattern analysis complete\n"
            f"[dim]Use 'flow-doc patterns check <component>' to check a specific component[/dim]",
            border_style="green"
        ))
        console.print()

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        # Check if we have verbose from the parent context
        verbose = getattr(click_ctx.obj, 'verbose', False) if click_ctx.obj else False
        if verbose:
            console.print_exception()
        sys.exit(1)


@patterns.command(name='check')
@click.argument('component')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed pattern analysis')
@pass_context
def patterns_check(ctx: Context, component: str, verbose: bool):
    """
    Check a component for pattern deviations.

    Analyzes the codebase patterns and compares the specified component
    against common patterns to identify potential improvements.

    COMPONENT can be in format "component_name" or "file_path::component_name"

    Examples:

        flow-doc patterns check UserService

        flow-doc patterns check services/user.py::UserService

        flow-doc patterns check get_users --verbose
    """
    console.print(Panel.fit(
        f"[bold cyan]Pattern Analysis[/bold cyan]\n"
        f"Component: {component}",
        border_style="cyan"
    ))
    console.print()

    try:
        # Initialize services
        with console.status("[bold cyan]Initializing services...[/bold cyan]"):
            ctx.analyzer = CodebaseAnalyzer(ctx.codebase_path)
            try:
                ctx.memory = MemoryStore(persist_directory="./data")
            except KeyError as e:
                if "'_type'" in str(e):
                    console.print()
                    console.print("[bold red]✗ ChromaDB collection error[/bold red]")
                    console.print()
                    console.print("[yellow]The ChromaDB collection has incompatible metadata.[/yellow]")
                    console.print("[yellow]This usually happens after upgrading ChromaDB versions.[/yellow]")
                    console.print()
                    console.print("[bold]To fix this, run these commands:[/bold]")
                    console.print("  [cyan]rm -rf ./data/chroma.sqlite3[/cyan]")
                    console.print("  [cyan]rm -rf ./data/*/[/cyan]  [dim]# Remove ChromaDB collection data[/dim]")
                    console.print()
                    console.print("Then re-run '[cyan]flow-doc analyze[/cyan]' to rebuild the collection.")
                    console.print()
                    sys.exit(1)
                else:
                    raise
            pattern_detector = PatternDetector()

        console.print("[green]✓[/green] Services initialized")
        console.print()

        # Parse component identifier
        if "::" in component:
            file_path, component_name = component.split("::", 1)
        else:
            component_name = component
            file_path = None

        # Get all components from memory
        with console.status("[bold cyan]Retrieving components from memory...[/bold cyan]"):
            all_results = ctx.memory.retrieve_similar("", k=100)

        console.print(f"[green]✓[/green] Retrieved {len(all_results)} components")
        console.print()

        # Convert to component format
        all_components = []
        target_component = None

        for result in all_results:
            doc = result.get('document', '')
            metadata = result.get('metadata', {})
            comp = {
                'name': metadata.get('name', ''),
                'type': metadata.get('type', ''),
                'file_path': metadata.get('file_path', ''),
                'docstring': doc[:200] if doc else '',
                'methods': metadata.get('methods', []),
                'imports': metadata.get('imports', []),
                'decorators': metadata.get('decorators', []),
                'params': metadata.get('params', []),
                'returns': metadata.get('returns', ''),
                'raises': metadata.get('raises', []),
            }
            all_components.append(comp)

            # Find target component
            if comp['name'] == component_name:
                if file_path is None or comp['file_path'] == file_path:
                    target_component = comp

        if target_component is None:
            console.print(f"[bold red]✗ Component '{component}' not found[/bold red]")
            console.print()
            console.print("[dim]Make sure you have run 'flow-doc analyze' first[/dim]")
            sys.exit(1)

        # Analyze patterns
        with console.status("[bold cyan]Analyzing codebase patterns...[/bold cyan]"):
            patterns_data = pattern_detector.analyze_patterns(all_components)

        consistency_score = patterns_data.get('consistency_score', 0.0)

        console.print(f"[green]✓[/green] Pattern analysis complete")
        console.print(f"  Codebase consistency: {consistency_score:.1%}")
        console.print()

        # Detect deviations
        suggestions = pattern_detector.detect_deviations(target_component, patterns_data)

        # Display results
        console.print(Panel.fit(
            f"[bold]Component:[/bold] {component_name}\n"
            f"[bold]File:[/bold] {target_component['file_path']}\n"
            f"[bold]Type:[/bold] {target_component['type']}\n"
            f"[bold]Consistency Score:[/bold] {consistency_score:.1%}",
            title="Component Info",
            border_style="blue"
        ))
        console.print()

        if suggestions:
            console.print(f"[yellow]⚠ Found {len(suggestions)} pattern deviations:[/yellow]")
            console.print()

            for i, suggestion in enumerate(suggestions, 1):
                console.print(f"  [yellow]{i}.[/yellow] {suggestion}")

            console.print()
            console.print("[dim]These suggestions can help improve code consistency[/dim]")
        else:
            console.print("[green]✓ No pattern deviations found![/green]")
            console.print()
            console.print("[dim]This component follows codebase patterns well[/dim]")

        # Show verbose pattern details if requested
        if verbose and suggestions:
            console.print()
            console.print(Panel.fit(
                "[bold]Pattern Details[/bold]",
                border_style="cyan"
            ))
            console.print()

            # Show naming conventions
            naming = patterns_data.get('naming_conventions', {})
            console.print(f"[bold]Naming Conventions:[/bold]")
            console.print(f"  Functions: {naming.get('function_convention', 'N/A')} "
                         f"({naming.get('function_consistency', 0):.0%} consistent)")
            console.print(f"  Classes: {naming.get('class_convention', 'N/A')} "
                         f"({naming.get('class_consistency', 0):.0%} consistent)")
            console.print()

            # Show route patterns
            route_patterns = patterns_data.get('route_patterns', {})
            if route_patterns.get('total_routes', 0) > 0:
                console.print(f"[bold]Route Patterns:[/bold]")
                console.print(f"  Versioned: {route_patterns.get('versioned', False)}")
                console.print(f"  API Prefix: {route_patterns.get('api_prefix', False)}")
                console.print(f"  Consistency: {route_patterns.get('consistency', 0):.0%}")
                console.print()

            # Show error handling
            error_handling = patterns_data.get('error_handling', {})
            console.print(f"[bold]Error Handling:[/bold]")
            console.print(f"  Coverage: {error_handling.get('percentage_with_handling', 0):.0%}")
            console.print()

        console.print()

    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]✗ Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        if ctx.verbose or verbose:
            console.print_exception()
        sys.exit(1)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the CLI."""
    cli(obj=Context())


if __name__ == '__main__':
    main()
