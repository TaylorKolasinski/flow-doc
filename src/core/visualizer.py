"""Dependency visualization for Python codebases."""

import logging
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

logger = logging.getLogger(__name__)


class CodebaseVisualizer:
    """
    Visualize codebase dependencies and relationships.

    Creates dependency graphs showing imports, function calls, and inheritance
    relationships. Supports multiple output formats: interactive HTML, static PNG,
    and Mermaid diagrams.

    Example:
        >>> visualizer = CodebaseVisualizer()
        >>> components = [...]  # List of parsed components
        >>> graph = visualizer.build_dependency_graph(components)
        >>> visualizer.generate_plotly_interactive(graph, "deps.html")
    """

    def __init__(self):
        """Initialize the visualizer."""
        logger.info("Initialized CodebaseVisualizer")

    def build_dependency_graph(self, components: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Create a directed dependency graph from components.

        Builds a NetworkX graph where:
        - Nodes represent files, classes, and functions
        - Edges represent dependencies (imports, calls, inheritance)

        Args:
            components: List of component dictionaries with metadata

        Returns:
            NetworkX directed graph with node and edge attributes

        Example:
            >>> components = [
            ...     {
            ...         "type": "class",
            ...         "name": "UserService",
            ...         "file_path": "services/user.py",
            ...         "methods": ["get_user", "create_user"],
            ...         "docstring": "User service class"
            ...     }
            ... ]
            >>> graph = visualizer.build_dependency_graph(components)
            >>> print(f"Nodes: {graph.number_of_nodes()}")
        """
        logger.info(f"Building dependency graph from {len(components)} components")

        graph = nx.DiGraph()

        # Track files and their components for building relationships
        files = {}
        file_imports = {}
        component_map = {}

        # First pass: Add all nodes
        for component in components:
            comp_type = component.get('type', 'unknown')
            name = component.get('name', 'unknown')
            file_path = component.get('file_path', 'unknown')

            # Create unique node ID
            if comp_type == 'file':
                node_id = file_path
            else:
                node_id = f"{file_path}::{name}"

            # Calculate documentation quality
            docstring = component.get('docstring', '')
            if docstring and len(docstring) > 100:
                doc_quality = 'good'
            elif docstring and len(docstring) > 20:
                doc_quality = 'partial'
            else:
                doc_quality = 'missing'

            # Add node with attributes
            graph.add_node(
                node_id,
                label=name,
                type=comp_type,
                file_path=file_path,
                docstring=docstring,
                doc_quality=doc_quality,
                size=component.get('size', 0),
                methods=component.get('methods', []),
                params=component.get('params', [])
            )

            # Track for relationship building
            component_map[node_id] = component

            if file_path not in files:
                files[file_path] = []
            files[file_path].append(node_id)

        # Second pass: Add edges based on relationships
        for component in components:
            file_path = component.get('file_path', 'unknown')
            comp_type = component.get('type', 'unknown')
            name = component.get('name', 'unknown')

            node_id = f"{file_path}::{name}" if comp_type != 'file' else file_path

            # Add import edges
            imports = component.get('imports', [])
            for imp in imports:
                # Try to find the imported module
                import_node = self._find_import_node(imp, files)
                if import_node:
                    graph.add_edge(
                        node_id,
                        import_node,
                        relationship='imports',
                        import_type='direct'
                    )

            # Add class inheritance edges
            if comp_type == 'class':
                bases = component.get('bases', [])
                for base in bases:
                    base_node = self._find_class_node(base, component_map)
                    if base_node:
                        graph.add_edge(
                            node_id,
                            base_node,
                            relationship='inherits',
                            import_type='inheritance'
                        )

            # Add function call edges (if we have call graph data)
            calls = component.get('calls', [])
            for call in calls:
                call_node = self._find_function_node(call, component_map)
                if call_node:
                    graph.add_edge(
                        node_id,
                        call_node,
                        relationship='calls',
                        import_type='function_call'
                    )

        logger.info(
            f"Built graph with {graph.number_of_nodes()} nodes "
            f"and {graph.number_of_edges()} edges"
        )

        return graph

    def _find_import_node(self, import_name: str, files: Dict[str, List[str]]) -> Optional[str]:
        """
        Find the node corresponding to an import statement.

        Args:
            import_name: Name of the import (e.g., 'os', 'src.utils')
            files: Dictionary mapping file paths to node IDs

        Returns:
            Node ID if found, None otherwise
        """
        # Convert import name to potential file path
        # e.g., 'src.utils' -> 'src/utils.py'
        potential_path = import_name.replace('.', '/') + '.py'

        # Check if this file exists in our codebase
        for file_path in files.keys():
            if file_path.endswith(potential_path) or file_path == potential_path:
                return file_path

        return None

    def _find_class_node(self, class_name: str, component_map: Dict[str, Dict]) -> Optional[str]:
        """
        Find the node corresponding to a class name.

        Args:
            class_name: Name of the class
            component_map: Dictionary mapping node IDs to components

        Returns:
            Node ID if found, None otherwise
        """
        for node_id, component in component_map.items():
            if component.get('type') == 'class' and component.get('name') == class_name:
                return node_id
        return None

    def _find_function_node(self, func_name: str, component_map: Dict[str, Dict]) -> Optional[str]:
        """
        Find the node corresponding to a function name.

        Args:
            func_name: Name of the function
            component_map: Dictionary mapping node IDs to components

        Returns:
            Node ID if found, None otherwise
        """
        for node_id, component in component_map.items():
            if component.get('type') == 'function' and component.get('name') == func_name:
                return node_id
        return None

    def detect_circular_dependencies(self, graph: nx.DiGraph) -> List[List[str]]:
        """
        Detect circular dependencies in the dependency graph.

        Finds all cycles (circular import chains) in the graph.

        Args:
            graph: Dependency graph

        Returns:
            List of cycles, where each cycle is a list of node IDs

        Example:
            >>> cycles = visualizer.detect_circular_dependencies(graph)
            >>> for cycle in cycles:
            ...     print(f"Circular dependency: {' -> '.join(cycle)}")
            Circular dependency: module_a.py -> module_b.py -> module_c.py -> module_a.py
        """
        logger.info("Detecting circular dependencies")

        try:
            cycles = list(nx.simple_cycles(graph))
            logger.info(f"Found {len(cycles)} circular dependencies")

            # Sort cycles by length (longest first)
            cycles.sort(key=len, reverse=True)

            # Log warning if cycles found
            if cycles:
                logger.warning(f"⚠ Detected {len(cycles)} circular dependencies")
                for i, cycle in enumerate(cycles[:5], 1):  # Log first 5
                    cycle_str = ' -> '.join(cycle) + f' -> {cycle[0]}'
                    logger.warning(f"  {i}. {cycle_str}")

            return cycles

        except Exception as e:
            logger.error(f"Error detecting cycles: {e}")
            return []

    def generate_plotly_interactive(
        self,
        graph: nx.DiGraph,
        output_path: str,
        title: str = "Codebase Dependency Graph"
    ) -> None:
        """
        Generate an interactive HTML visualization using Plotly.

        Creates an interactive graph with:
        - Zoom and pan controls
        - Hover tooltips with node metadata
        - Color-coding by node type and documentation quality
        - Force-directed layout

        Args:
            graph: Dependency graph to visualize
            output_path: Path to save the HTML file
            title: Title for the visualization

        Example:
            >>> visualizer.generate_plotly_interactive(graph, "deps.html")
            >>> # Opens deps.html in browser to see interactive graph
        """
        logger.info(f"Generating interactive Plotly visualization: {output_path}")

        if graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, cannot generate visualization")
            return

        try:
            # Use spring layout for positioning
            pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)

            # Prepare edge traces
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )

            for edge in graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += tuple([x0, x1, None])
                edge_trace['y'] += tuple([y0, y1, None])

            # Prepare node traces (separate by type for different colors)
            node_traces = []
            type_colors = {
                'file': '#1f77b4',      # Blue
                'class': '#ff7f0e',     # Orange
                'function': '#2ca02c',  # Green
                'route': '#d62728',     # Red
                'unknown': '#7f7f7f'    # Gray
            }

            quality_colors = {
                'good': '#2ca02c',      # Green
                'partial': '#ff7f0e',   # Orange
                'missing': '#d62728'    # Red
            }

            # Group nodes by type
            nodes_by_type = {}
            for node in graph.nodes():
                node_type = graph.nodes[node].get('type', 'unknown')
                if node_type not in nodes_by_type:
                    nodes_by_type[node_type] = []
                nodes_by_type[node_type].append(node)

            # Create trace for each type
            for node_type, nodes in nodes_by_type.items():
                node_x = []
                node_y = []
                node_text = []
                node_colors = []

                for node in nodes:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)

                    # Build hover text
                    attrs = graph.nodes[node]
                    hover_text = f"<b>{attrs.get('label', node)}</b><br>"
                    hover_text += f"Type: {attrs.get('type', 'unknown')}<br>"
                    hover_text += f"File: {attrs.get('file_path', 'unknown')}<br>"

                    doc_quality = attrs.get('doc_quality', 'missing')
                    hover_text += f"Documentation: {doc_quality}<br>"

                    if attrs.get('methods'):
                        methods = ', '.join(attrs['methods'][:3])
                        hover_text += f"Methods: {methods}<br>"

                    if attrs.get('params'):
                        params = ', '.join(attrs['params'][:3])
                        hover_text += f"Params: {params}<br>"

                    node_text.append(hover_text)

                    # Color by documentation quality
                    node_colors.append(quality_colors.get(doc_quality, '#7f7f7f'))

                node_trace = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    text=node_text,
                    name=node_type.capitalize(),
                    marker=dict(
                        size=10,
                        color=node_colors,
                        line_width=2,
                        line_color='white'
                    )
                )

                node_traces.append(node_trace)

            # Create figure
            fig = go.Figure(
                data=[edge_trace] + node_traces,
                layout=go.Layout(
                    title=dict(
                        text=title,
                        font=dict(size=20)
                    ),
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[
                        dict(
                            text="Color indicates documentation quality: "
                                 "<span style='color:green'>Good</span> | "
                                 "<span style='color:orange'>Partial</span> | "
                                 "<span style='color:red'>Missing</span>",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=-0.1,
                            xanchor='center',
                            yanchor='bottom'
                        )
                    ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white'
                )
            )

            # Save to HTML
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_file))

            logger.info(f"✓ Interactive visualization saved to {output_path}")

        except Exception as e:
            logger.error(f"Error generating Plotly visualization: {e}", exc_info=True)
            raise

    def generate_static_image(
        self,
        graph: nx.DiGraph,
        output_path: str,
        title: str = "Codebase Dependency Graph"
    ) -> None:
        """
        Generate a static PNG image using matplotlib.

        Creates a hierarchical layout visualization suitable for documentation.

        Args:
            graph: Dependency graph to visualize
            output_path: Path to save the PNG file
            title: Title for the visualization

        Example:
            >>> visualizer.generate_static_image(graph, "deps.png")
        """
        logger.info(f"Generating static PNG visualization: {output_path}")

        if graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, cannot generate visualization")
            return

        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            # Create figure
            fig, ax = plt.subplots(figsize=(16, 12))

            # Use hierarchical layout if possible, fall back to spring
            try:
                pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
            except:
                pos = nx.spring_layout(graph, seed=42)

            # Define colors
            type_colors = {
                'file': '#1f77b4',
                'class': '#ff7f0e',
                'function': '#2ca02c',
                'route': '#d62728',
                'unknown': '#7f7f7f'
            }

            quality_colors = {
                'good': '#2ca02c',
                'partial': '#ff7f0e',
                'missing': '#d62728'
            }

            # Get node colors based on documentation quality
            node_colors = []
            for node in graph.nodes():
                doc_quality = graph.nodes[node].get('doc_quality', 'missing')
                node_colors.append(quality_colors.get(doc_quality, '#7f7f7f'))

            # Draw edges
            nx.draw_networkx_edges(
                graph,
                pos,
                edge_color='#888888',
                alpha=0.5,
                arrows=True,
                arrowsize=10,
                ax=ax
            )

            # Draw nodes
            nx.draw_networkx_nodes(
                graph,
                pos,
                node_color=node_colors,
                node_size=500,
                alpha=0.9,
                ax=ax
            )

            # Draw labels
            labels = {node: graph.nodes[node].get('label', node) for node in graph.nodes()}
            nx.draw_networkx_labels(
                graph,
                pos,
                labels,
                font_size=8,
                font_weight='bold',
                ax=ax
            )

            # Add legend for documentation quality
            legend_elements = [
                mpatches.Patch(color='#2ca02c', label='Good Documentation'),
                mpatches.Patch(color='#ff7f0e', label='Partial Documentation'),
                mpatches.Patch(color='#d62728', label='Missing Documentation')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            # Set title and layout
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.axis('off')

            plt.tight_layout()

            # Save figure
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_file), dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✓ Static visualization saved to {output_path}")

        except ImportError:
            logger.error("matplotlib is required for static image generation")
            logger.info("Install with: pip install matplotlib")
            raise
        except Exception as e:
            logger.error(f"Error generating static visualization: {e}", exc_info=True)
            raise

    def export_mermaid(
        self,
        graph: nx.DiGraph,
        max_nodes: int = 50
    ) -> str:
        """
        Export graph as Mermaid diagram syntax.

        Converts the dependency graph to Mermaid markdown format,
        which can be rendered in GitHub, GitLab, and documentation tools.

        Args:
            graph: Dependency graph to convert
            max_nodes: Maximum nodes to include (to avoid huge diagrams)

        Returns:
            Mermaid diagram as string

        Example:
            >>> mermaid = visualizer.export_mermaid(graph)
            >>> print(mermaid)
            graph TD
                A[module_a.py] --> B[module_b.py]
                B --> C[module_c.py]
            >>> # Can be embedded in markdown files
        """
        logger.info("Exporting graph as Mermaid diagram")

        if graph.number_of_nodes() == 0:
            return "graph TD\n    A[Empty Graph]"

        lines = ["graph TD"]

        # Limit nodes if graph is too large
        nodes = list(graph.nodes())[:max_nodes]
        if len(graph.nodes()) > max_nodes:
            logger.warning(
                f"Graph has {len(graph.nodes())} nodes, limiting to {max_nodes} "
                "for Mermaid export"
            )

        # Create node ID mapping (Mermaid needs simple IDs)
        node_id_map = {}
        for i, node in enumerate(nodes):
            node_id_map[node] = f"N{i}"

        # Add nodes with labels
        for node in nodes:
            node_id = node_id_map[node]
            label = graph.nodes[node].get('label', node)
            node_type = graph.nodes[node].get('type', 'unknown')
            doc_quality = graph.nodes[node].get('doc_quality', 'missing')

            # Choose shape based on type
            if node_type == 'class':
                shape_start, shape_end = "[", "]"  # Rectangle
            elif node_type == 'function':
                shape_start, shape_end = "(", ")"  # Rounded
            elif node_type == 'route':
                shape_start, shape_end = "{", "}"  # Diamond
            else:
                shape_start, shape_end = "[[", "]]"  # Subroutine

            # Add documentation quality indicator
            quality_indicator = {
                'good': '✓',
                'partial': '⚠',
                'missing': '✗'
            }.get(doc_quality, '?')

            node_line = f"    {node_id}{shape_start}\"{label} {quality_indicator}\"{shape_end}"
            lines.append(node_line)

        # Add edges
        for edge in graph.edges():
            if edge[0] in node_id_map and edge[1] in node_id_map:
                source_id = node_id_map[edge[0]]
                target_id = node_id_map[edge[1]]

                # Get edge type for label
                relationship = graph.edges[edge].get('relationship', '')
                edge_label = relationship if relationship else ''

                if edge_label:
                    edge_line = f"    {source_id} -->|{edge_label}| {target_id}"
                else:
                    edge_line = f"    {source_id} --> {target_id}"

                lines.append(edge_line)

        # Add style classes
        lines.append("")
        lines.append("    classDef good fill:#2ca02c,stroke:#1e7d1e,color:#fff")
        lines.append("    classDef partial fill:#ff7f0e,stroke:#cc6600,color:#fff")
        lines.append("    classDef missing fill:#d62728,stroke:#a01f20,color:#fff")

        mermaid_diagram = "\n".join(lines)

        logger.info(f"✓ Exported Mermaid diagram with {len(nodes)} nodes")

        return mermaid_diagram

    def get_statistics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Calculate graph statistics.

        Args:
            graph: Dependency graph

        Returns:
            Dictionary with graph metrics

        Example:
            >>> stats = visualizer.get_statistics(graph)
            >>> print(f"Total nodes: {stats['total_nodes']}")
            >>> print(f"Average connections: {stats['avg_degree']:.2f}")
        """
        if graph.number_of_nodes() == 0:
            return {
                'total_nodes': 0,
                'total_edges': 0,
                'avg_degree': 0,
                'density': 0,
                'circular_dependencies': 0,
                'nodes_by_type': {},
                'documentation_coverage': {}
            }

        # Basic metrics
        stats = {
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
            'density': nx.density(graph),
            'circular_dependencies': len(self.detect_circular_dependencies(graph))
        }

        # Count by type
        nodes_by_type = {}
        doc_quality = {'good': 0, 'partial': 0, 'missing': 0}

        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'unknown')
            nodes_by_type[node_type] = nodes_by_type.get(node_type, 0) + 1

            quality = graph.nodes[node].get('doc_quality', 'missing')
            doc_quality[quality] = doc_quality.get(quality, 0) + 1

        stats['nodes_by_type'] = nodes_by_type
        stats['documentation_coverage'] = {
            'good': doc_quality['good'],
            'partial': doc_quality['partial'],
            'missing': doc_quality['missing'],
            'percentage_documented': (
                (doc_quality['good'] + doc_quality['partial']) / stats['total_nodes'] * 100
                if stats['total_nodes'] > 0 else 0
            )
        }

        return stats
