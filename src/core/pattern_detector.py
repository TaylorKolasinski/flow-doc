"""Pattern detection and analysis for Python codebases."""

import re
import logging
from typing import List, Dict, Any, Set, Tuple
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class PatternDetector:
    """
    Detect and analyze code patterns across a codebase.

    Identifies common patterns including:
    - Route naming conventions
    - Error handling strategies
    - Authentication patterns
    - Naming conventions
    - Import organization

    Example:
        >>> detector = PatternDetector()
        >>> patterns = detector.analyze_patterns(components)
        >>> deviations = detector.detect_deviations(component, patterns)
        >>> report = detector.generate_recommendations(patterns)
    """

    def __init__(self):
        """Initialize the pattern detector."""
        logger.info("Initialized PatternDetector")

    def analyze_patterns(self, all_components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract common patterns from all components.

        Analyzes the entire codebase to identify:
        - Route patterns and conventions
        - Error handling approaches
        - Authentication/authorization patterns
        - Naming conventions
        - Import organization
        - Documentation patterns

        Args:
            all_components: List of all parsed components

        Returns:
            Dictionary containing detected patterns and statistics

        Example:
            >>> patterns = detector.analyze_patterns(components)
            >>> print(patterns['consistency_score'])
            0.85
            >>> print(patterns['route_patterns'])
            ['/api/v1/{resource}', '/api/v1/{resource}/{id}']
        """
        logger.info(f"Analyzing patterns across {len(all_components)} components")

        patterns = {
            'route_patterns': self._analyze_route_patterns(all_components),
            'error_handling': self._analyze_error_handling(all_components),
            'auth_patterns': self._analyze_auth_patterns(all_components),
            'naming_conventions': self._analyze_naming_conventions(all_components),
            'import_organization': self._analyze_imports(all_components),
            'documentation_patterns': self._analyze_documentation(all_components),
            'consistency_score': 0.0,
            'total_components': len(all_components),
            'components_by_type': self._count_by_type(all_components)
        }

        # Calculate overall consistency score
        patterns['consistency_score'] = self._calculate_consistency_score(patterns)

        logger.info(f"Pattern analysis complete (consistency: {patterns['consistency_score']:.2f})")

        return patterns

    def _analyze_route_patterns(self, components: List[Dict]) -> Dict[str, Any]:
        """
        Analyze API route patterns.

        Identifies:
        - Common path structures
        - Versioning patterns (/api/v1, /v2, etc.)
        - Resource naming
        - HTTP methods usage

        Args:
            components: List of components

        Returns:
            Dictionary with route pattern analysis
        """
        routes = [c for c in components if c.get('type') == 'route']

        if not routes:
            return {
                'patterns': [],
                'versioned': False,
                'methods_used': {},
                'consistency': 0.0
            }

        paths = []
        methods_count = Counter()
        versioned_count = 0
        api_prefix_count = 0

        for route in routes:
            path = route.get('path', '')
            methods = route.get('methods', [])

            if path:
                paths.append(path)

                # Check for versioning
                if re.search(r'/v\d+/', path):
                    versioned_count += 1

                # Check for API prefix
                if path.startswith('/api/'):
                    api_prefix_count += 1

            for method in methods:
                methods_count[method] += 1

        # Extract common patterns
        pattern_templates = self._extract_route_templates(paths)

        # Calculate consistency
        consistency = 0.0
        if routes:
            version_consistency = versioned_count / len(routes)
            prefix_consistency = api_prefix_count / len(routes)
            consistency = (version_consistency + prefix_consistency) / 2

        return {
            'patterns': pattern_templates,
            'versioned': versioned_count > len(routes) / 2,
            'api_prefix': api_prefix_count > len(routes) / 2,
            'methods_used': dict(methods_count),
            'total_routes': len(routes),
            'consistency': consistency
        }

    def _extract_route_templates(self, paths: List[str]) -> List[str]:
        """
        Extract route templates from actual paths.

        Converts:
        - /api/users/123 -> /api/users/{id}
        - /api/v1/products/456 -> /api/v1/products/{id}

        Args:
            paths: List of route paths

        Returns:
            List of template patterns
        """
        templates = []

        for path in paths:
            # Replace numeric IDs with {id}
            template = re.sub(r'/\d+', '/{id}', path)

            # Replace UUIDs with {id}
            template = re.sub(
                r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
                '/{id}',
                template
            )

            templates.append(template)

        # Count occurrences and return most common
        template_counts = Counter(templates)
        return [t for t, _ in template_counts.most_common(10)]

    def _analyze_error_handling(self, components: List[Dict]) -> Dict[str, Any]:
        """
        Analyze error handling patterns.

        Detects:
        - Try-except usage
        - Raise statements
        - Error return codes
        - Logging patterns

        Args:
            components: List of components

        Returns:
            Dictionary with error handling statistics
        """
        functions = [c for c in components if c.get('type') in ['function', 'class']]

        if not functions:
            return {
                'try_except': 0,
                'raise': 0,
                'return_codes': 0,
                'logging': 0,
                'percentage_with_handling': 0.0
            }

        try_except_count = 0
        raise_count = 0
        return_code_count = 0
        logging_count = 0

        # Analyze decorators and docstrings for patterns
        for comp in functions:
            decorators = comp.get('decorators', [])
            docstring = comp.get('docstring', '')

            # Check decorators for error handling
            if any('error' in str(d).lower() or 'exception' in str(d).lower() for d in decorators):
                try_except_count += 1

            # Check docstring for error handling mentions
            if docstring:
                doc_lower = docstring.lower()
                if 'raises' in doc_lower or 'raise' in doc_lower:
                    raise_count += 1
                if 'returns' in doc_lower and ('error' in doc_lower or 'status' in doc_lower):
                    return_code_count += 1
                if 'log' in doc_lower:
                    logging_count += 1

        total = len(functions)
        handled = try_except_count + raise_count + return_code_count

        return {
            'try_except': try_except_count,
            'raise': raise_count,
            'return_codes': return_code_count,
            'logging': logging_count,
            'total_components': total,
            'percentage_with_handling': (handled / total * 100) if total > 0 else 0.0
        }

    def _analyze_auth_patterns(self, components: List[Dict]) -> Dict[str, Any]:
        """
        Analyze authentication and authorization patterns.

        Detects:
        - Auth decorators
        - Permission checks
        - Token validation

        Args:
            components: List of components

        Returns:
            Dictionary with authentication pattern statistics
        """
        auth_decorators = []
        auth_middleware = []
        permission_checks = 0

        for comp in components:
            decorators = comp.get('decorators', [])

            for dec in decorators:
                dec_str = str(dec).lower()

                # Check for auth decorators
                if any(keyword in dec_str for keyword in ['auth', 'login', 'require', 'permission', 'role']):
                    auth_decorators.append(str(dec))

            # Check docstring for permission mentions
            docstring = comp.get('docstring', '')
            if docstring:
                doc_lower = docstring.lower()
                if any(word in doc_lower for word in ['permission', 'authorized', 'authenticated', 'role']):
                    permission_checks += 1

        # Count unique decorator patterns
        decorator_counts = Counter(auth_decorators)

        return {
            'auth_decorators': [d for d, _ in decorator_counts.most_common(10)],
            'decorator_usage_count': len(auth_decorators),
            'permission_checks': permission_checks,
            'has_auth_pattern': len(auth_decorators) > 0
        }

    def _analyze_naming_conventions(self, components: List[Dict]) -> Dict[str, Any]:
        """
        Analyze naming conventions across the codebase.

        Detects:
        - snake_case vs camelCase vs PascalCase
        - Consistency in function/class naming
        - Abbreviations and patterns

        Args:
            components: List of components

        Returns:
            Dictionary with naming convention statistics
        """
        function_names = []
        class_names = []
        variable_patterns = []

        for comp in components:
            comp_type = comp.get('type')
            name = comp.get('name', '')

            if comp_type == 'function':
                function_names.append(name)
            elif comp_type == 'class':
                class_names.append(name)

        # Analyze function naming
        function_conventions = self._detect_case_convention(function_names)

        # Analyze class naming
        class_conventions = self._detect_case_convention(class_names)

        return {
            'function_convention': function_conventions['dominant'],
            'function_consistency': function_conventions['consistency'],
            'class_convention': class_conventions['dominant'],
            'class_consistency': class_conventions['consistency'],
            'overall_consistency': (
                function_conventions['consistency'] + class_conventions['consistency']
            ) / 2
        }

    def _detect_case_convention(self, names: List[str]) -> Dict[str, Any]:
        """
        Detect the dominant naming convention in a list of names.

        Args:
            names: List of identifiers

        Returns:
            Dictionary with convention analysis
        """
        if not names:
            return {
                'dominant': 'unknown',
                'consistency': 0.0,
                'breakdown': {}
            }

        conventions = {
            'snake_case': 0,
            'camelCase': 0,
            'PascalCase': 0,
            'UPPER_CASE': 0
        }

        for name in names:
            if '_' in name:
                if name.isupper():
                    conventions['UPPER_CASE'] += 1
                else:
                    conventions['snake_case'] += 1
            elif name[0].isupper():
                conventions['PascalCase'] += 1
            elif any(c.isupper() for c in name[1:]):
                conventions['camelCase'] += 1
            else:
                conventions['snake_case'] += 1  # Default assumption

        total = len(names)
        dominant = max(conventions, key=conventions.get)
        consistency = conventions[dominant] / total if total > 0 else 0.0

        return {
            'dominant': dominant,
            'consistency': consistency,
            'breakdown': {k: v for k, v in conventions.items() if v > 0}
        }

    def _analyze_imports(self, components: List[Dict]) -> Dict[str, Any]:
        """
        Analyze import organization patterns.

        Detects:
        - Relative vs absolute imports
        - Import ordering
        - Common dependencies

        Args:
            components: List of components

        Returns:
            Dictionary with import pattern statistics
        """
        all_imports = []
        relative_count = 0
        absolute_count = 0

        for comp in components:
            imports = comp.get('imports', [])
            all_imports.extend(imports)

            for imp in imports:
                if imp.startswith('.'):
                    relative_count += 1
                else:
                    absolute_count += 1

        # Count most common imports
        import_counts = Counter(all_imports)
        common_imports = [imp for imp, _ in import_counts.most_common(20)]

        # Categorize imports
        stdlib = []
        third_party = []
        local = []

        for imp in common_imports:
            if imp.startswith('.') or 'src' in imp or 'app' in imp:
                local.append(imp)
            elif imp in ['os', 'sys', 're', 'json', 'datetime', 'pathlib', 'typing']:
                stdlib.append(imp)
            else:
                third_party.append(imp)

        total_imports = len(all_imports)
        prefer_absolute = absolute_count > relative_count if total_imports > 0 else True

        return {
            'total_imports': total_imports,
            'unique_imports': len(set(all_imports)),
            'prefer_absolute': prefer_absolute,
            'relative_imports': relative_count,
            'absolute_imports': absolute_count,
            'common_imports': common_imports[:10],
            'stdlib_usage': stdlib[:5],
            'third_party': third_party[:5],
            'local_imports': local[:5]
        }

    def _analyze_documentation(self, components: List[Dict]) -> Dict[str, Any]:
        """
        Analyze documentation patterns.

        Detects:
        - Docstring presence
        - Documentation style (Google, NumPy, etc.)
        - Documentation completeness

        Args:
            components: List of components

        Returns:
            Dictionary with documentation statistics
        """
        total = len(components)
        with_docstring = 0
        docstring_lengths = []
        styles = {'google': 0, 'numpy': 0, 'sphinx': 0, 'basic': 0}

        for comp in components:
            docstring = comp.get('docstring', '')

            if docstring:
                with_docstring += 1
                docstring_lengths.append(len(docstring))

                # Detect style
                if 'Args:' in docstring and 'Returns:' in docstring:
                    styles['google'] += 1
                elif 'Parameters' in docstring and '----------' in docstring:
                    styles['numpy'] += 1
                elif ':param' in docstring or ':return:' in docstring:
                    styles['sphinx'] += 1
                else:
                    styles['basic'] += 1

        avg_length = sum(docstring_lengths) / len(docstring_lengths) if docstring_lengths else 0
        dominant_style = max(styles, key=styles.get) if with_docstring > 0 else 'none'

        return {
            'total_components': total,
            'documented_count': with_docstring,
            'documentation_rate': (with_docstring / total * 100) if total > 0 else 0.0,
            'average_length': avg_length,
            'dominant_style': dominant_style,
            'style_breakdown': {k: v for k, v in styles.items() if v > 0}
        }

    def _count_by_type(self, components: List[Dict]) -> Dict[str, int]:
        """Count components by type."""
        counts = Counter(c.get('type', 'unknown') for c in components)
        return dict(counts)

    def _calculate_consistency_score(self, patterns: Dict[str, Any]) -> float:
        """
        Calculate overall consistency score based on all patterns.

        Args:
            patterns: Dictionary of detected patterns

        Returns:
            Consistency score between 0.0 and 1.0
        """
        scores = []

        # Route consistency
        route_patterns = patterns.get('route_patterns', {})
        if route_patterns.get('total_routes', 0) > 0:
            scores.append(route_patterns.get('consistency', 0.0))

        # Naming consistency
        naming = patterns.get('naming_conventions', {})
        scores.append(naming.get('overall_consistency', 0.0))

        # Documentation rate
        docs = patterns.get('documentation_patterns', {})
        doc_rate = docs.get('documentation_rate', 0.0) / 100.0
        scores.append(doc_rate)

        # Calculate average
        return sum(scores) / len(scores) if scores else 0.0

    def detect_deviations(
        self,
        component: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> List[str]:
        """
        Compare a component against detected patterns and find deviations.

        Args:
            component: Single component to check
            patterns: Detected patterns from analyze_patterns()

        Returns:
            List of suggestion strings

        Example:
            >>> suggestions = detector.detect_deviations(component, patterns)
            >>> for suggestion in suggestions:
            ...     print(f"- {suggestion}")
            - Route doesn't follow /api/v1/{resource} pattern
            - Missing error handling (80% of components use try-except)
            - Function name should use snake_case (90% consistency)
        """
        suggestions = []
        comp_type = component.get('type', 'unknown')
        name = component.get('name', '')

        # Check route patterns
        if comp_type == 'route':
            suggestions.extend(self._check_route_deviations(component, patterns))

        # Check naming conventions
        suggestions.extend(self._check_naming_deviations(component, patterns))

        # Check documentation
        suggestions.extend(self._check_documentation_deviations(component, patterns))

        # Check error handling
        suggestions.extend(self._check_error_handling_deviations(component, patterns))

        # Check authentication
        suggestions.extend(self._check_auth_deviations(component, patterns))

        return suggestions

    def _check_route_deviations(
        self,
        component: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> List[str]:
        """Check route-specific deviations."""
        suggestions = []
        route_patterns = patterns.get('route_patterns', {})

        path = component.get('path', '')

        if not path:
            return suggestions

        # Check versioning
        if route_patterns.get('versioned') and not re.search(r'/v\d+/', path):
            suggestions.append(
                "Route doesn't include version (e.g., /api/v1/). "
                f"{route_patterns.get('total_routes', 0)} other routes use versioning."
            )

        # Check API prefix
        if route_patterns.get('api_prefix') and not path.startswith('/api/'):
            suggestions.append(
                "Route doesn't start with /api/ prefix. "
                "This is the standard pattern in this codebase."
            )

        # Check against common patterns
        common_patterns = route_patterns.get('patterns', [])
        if common_patterns:
            # Normalize path for comparison
            normalized = re.sub(r'/\d+', '/{id}', path)

            if normalized not in common_patterns:
                suggestions.append(
                    f"Route pattern '{path}' doesn't match common patterns. "
                    f"Common patterns: {', '.join(common_patterns[:3])}"
                )

        return suggestions

    def _check_naming_deviations(
        self,
        component: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> List[str]:
        """Check naming convention deviations."""
        suggestions = []
        naming = patterns.get('naming_conventions', {})
        comp_type = component.get('type')
        name = component.get('name', '')

        if not name:
            return suggestions

        if comp_type == 'function':
            expected = naming.get('function_convention', 'snake_case')
            consistency = naming.get('function_consistency', 0.0)

            if not self._matches_convention(name, expected):
                suggestions.append(
                    f"Function name '{name}' doesn't follow {expected} convention "
                    f"({consistency*100:.0f}% of functions use {expected})."
                )

        elif comp_type == 'class':
            expected = naming.get('class_convention', 'PascalCase')
            consistency = naming.get('class_consistency', 0.0)

            if not self._matches_convention(name, expected):
                suggestions.append(
                    f"Class name '{name}' doesn't follow {expected} convention "
                    f"({consistency*100:.0f}% of classes use {expected})."
                )

        return suggestions

    def _matches_convention(self, name: str, convention: str) -> bool:
        """Check if a name matches a naming convention."""
        if convention == 'snake_case':
            return '_' in name or name.islower()
        elif convention == 'camelCase':
            return name[0].islower() and any(c.isupper() for c in name[1:])
        elif convention == 'PascalCase':
            return name[0].isupper()
        elif convention == 'UPPER_CASE':
            return name.isupper()
        return True

    def _check_documentation_deviations(
        self,
        component: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> List[str]:
        """Check documentation deviations."""
        suggestions = []
        docs = patterns.get('documentation_patterns', {})

        docstring = component.get('docstring', '')
        doc_rate = docs.get('documentation_rate', 0.0)

        # Check if missing documentation when others have it
        if not docstring and doc_rate > 50.0:
            suggestions.append(
                f"Missing docstring. {doc_rate:.0f}% of components are documented. "
                "Add a docstring explaining the purpose and usage."
            )

        # Check documentation style
        if docstring:
            dominant_style = docs.get('dominant_style', 'none')

            if dominant_style != 'basic':
                # Check if component follows dominant style
                style_match = False

                if dominant_style == 'google' and 'Args:' in docstring:
                    style_match = True
                elif dominant_style == 'numpy' and 'Parameters' in docstring:
                    style_match = True
                elif dominant_style == 'sphinx' and ':param' in docstring:
                    style_match = True

                if not style_match and len(docstring) > 50:
                    suggestions.append(
                        f"Docstring doesn't follow {dominant_style} style. "
                        "Consider using the same format as other components for consistency."
                    )

        return suggestions

    def _check_error_handling_deviations(
        self,
        component: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> List[str]:
        """Check error handling deviations."""
        suggestions = []
        error_handling = patterns.get('error_handling', {})

        comp_type = component.get('type')

        if comp_type not in ['function', 'class']:
            return suggestions

        docstring = component.get('docstring', '')
        decorators = component.get('decorators', [])

        # Check if component mentions error handling
        has_error_handling = (
            'raises' in docstring.lower() or
            'raise' in docstring.lower() or
            'exception' in docstring.lower() or
            any('error' in str(d).lower() for d in decorators)
        )

        handling_rate = error_handling.get('percentage_with_handling', 0.0)

        if not has_error_handling and handling_rate > 50.0:
            suggestions.append(
                f"Missing error handling documentation. {handling_rate:.0f}% of components "
                "document their error handling. Consider adding Raises section to docstring."
            )

        return suggestions

    def _check_auth_deviations(
        self,
        component: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> List[str]:
        """Check authentication pattern deviations."""
        suggestions = []
        auth_patterns = patterns.get('auth_patterns', {})

        comp_type = component.get('type')

        # Check if route should have auth
        if comp_type == 'route' and auth_patterns.get('has_auth_pattern'):
            decorators = component.get('decorators', [])
            path = component.get('path', '')

            has_auth = any(
                'auth' in str(d).lower() or 'login' in str(d).lower() or 'require' in str(d).lower()
                for d in decorators
            )

            # Check if it's a potentially sensitive endpoint
            sensitive_patterns = ['/admin', '/delete', '/update', '/create', '/edit']
            is_sensitive = any(pattern in path.lower() for pattern in sensitive_patterns)

            if is_sensitive and not has_auth:
                common_decorators = auth_patterns.get('auth_decorators', [])
                if common_decorators:
                    suggestions.append(
                        f"Potentially sensitive route '{path}' missing authentication decorator. "
                        f"Common decorators used: {', '.join(common_decorators[:3])}"
                    )

        return suggestions

    def generate_recommendations(self, patterns: Dict[str, Any]) -> str:
        """
        Generate a markdown report with pattern analysis and recommendations.

        Args:
            patterns: Detected patterns from analyze_patterns()

        Returns:
            Markdown-formatted report string

        Example:
            >>> report = detector.generate_recommendations(patterns)
            >>> print(report)
            # Code Pattern Analysis Report
            ...
            >>> with open('patterns.md', 'w') as f:
            ...     f.write(report)
        """
        lines = ["# Code Pattern Analysis Report", ""]

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"**Overall Consistency Score: {patterns['consistency_score']:.1%}**")
        lines.append("")
        lines.append(f"Analyzed {patterns['total_components']} components across the codebase.")
        lines.append("")

        # Components breakdown
        lines.append("### Components by Type")
        lines.append("")
        for comp_type, count in patterns['components_by_type'].items():
            lines.append(f"- **{comp_type}**: {count}")
        lines.append("")

        # Route Patterns
        route_patterns = patterns.get('route_patterns', {})
        if route_patterns.get('total_routes', 0) > 0:
            lines.append("## API Route Patterns")
            lines.append("")
            lines.append(f"**Total Routes**: {route_patterns['total_routes']}")
            lines.append(f"**Versioned**: {'Yes' if route_patterns.get('versioned') else 'No'}")
            lines.append(f"**API Prefix**: {'Yes' if route_patterns.get('api_prefix') else 'No'}")
            lines.append(f"**Consistency**: {route_patterns.get('consistency', 0)*100:.0f}%")
            lines.append("")

            common_patterns = route_patterns.get('patterns', [])
            if common_patterns:
                lines.append("### Common Route Patterns")
                lines.append("")
                for pattern in common_patterns[:5]:
                    lines.append(f"- `{pattern}`")
                lines.append("")

            methods = route_patterns.get('methods_used', {})
            if methods:
                lines.append("### HTTP Methods Usage")
                lines.append("")
                for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"- **{method}**: {count}")
                lines.append("")

        # Naming Conventions
        naming = patterns.get('naming_conventions', {})
        lines.append("## Naming Conventions")
        lines.append("")
        lines.append(f"**Functions**: {naming.get('function_convention', 'unknown')} "
                    f"({naming.get('function_consistency', 0)*100:.0f}% consistent)")
        lines.append(f"**Classes**: {naming.get('class_convention', 'unknown')} "
                    f"({naming.get('class_consistency', 0)*100:.0f}% consistent)")
        lines.append("")

        # Error Handling
        error_handling = patterns.get('error_handling', {})
        lines.append("## Error Handling")
        lines.append("")
        lines.append(f"**Components with error handling**: "
                    f"{error_handling.get('percentage_with_handling', 0):.0f}%")
        lines.append("")
        lines.append("### Error Handling Approaches")
        lines.append("")
        lines.append(f"- **Try-Except**: {error_handling.get('try_except', 0)} components")
        lines.append(f"- **Raises**: {error_handling.get('raise', 0)} components")
        lines.append(f"- **Return Codes**: {error_handling.get('return_codes', 0)} components")
        lines.append(f"- **Logging**: {error_handling.get('logging', 0)} components")
        lines.append("")

        # Authentication
        auth_patterns = patterns.get('auth_patterns', {})
        if auth_patterns.get('has_auth_pattern'):
            lines.append("## Authentication & Authorization")
            lines.append("")
            lines.append(f"**Components with auth**: {auth_patterns.get('decorator_usage_count', 0)}")
            lines.append(f"**Permission checks**: {auth_patterns.get('permission_checks', 0)}")
            lines.append("")

            decorators = auth_patterns.get('auth_decorators', [])
            if decorators:
                lines.append("### Common Auth Decorators")
                lines.append("")
                for dec in decorators[:5]:
                    lines.append(f"- `{dec}`")
                lines.append("")

        # Documentation
        docs = patterns.get('documentation_patterns', {})
        lines.append("## Documentation")
        lines.append("")
        lines.append(f"**Documentation Rate**: {docs.get('documentation_rate', 0):.1f}%")
        lines.append(f"**Average Docstring Length**: {docs.get('average_length', 0):.0f} characters")
        lines.append(f"**Dominant Style**: {docs.get('dominant_style', 'none')}")
        lines.append("")

        # Import Organization
        imports = patterns.get('import_organization', {})
        lines.append("## Import Organization")
        lines.append("")
        lines.append(f"**Total Imports**: {imports.get('total_imports', 0)}")
        lines.append(f"**Unique Imports**: {imports.get('unique_imports', 0)}")
        lines.append(f"**Prefer Absolute**: {'Yes' if imports.get('prefer_absolute') else 'No'}")
        lines.append("")

        common_imports = imports.get('common_imports', [])
        if common_imports:
            lines.append("### Most Common Dependencies")
            lines.append("")
            for imp in common_imports[:10]:
                lines.append(f"- `{imp}`")
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")

        recommendations = self._generate_improvement_suggestions(patterns)
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"### {i}. {rec['title']}")
                lines.append("")
                lines.append(rec['description'])
                lines.append("")
                if rec.get('action'):
                    lines.append(f"**Action**: {rec['action']}")
                    lines.append("")
        else:
            lines.append(" No major issues found. Codebase follows consistent patterns.")
            lines.append("")

        return "\n".join(lines)

    def _generate_improvement_suggestions(
        self,
        patterns: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate specific improvement suggestions based on patterns."""
        suggestions = []

        # Check documentation rate
        docs = patterns.get('documentation_patterns', {})
        doc_rate = docs.get('documentation_rate', 0)

        if doc_rate < 50:
            suggestions.append({
                'title': 'Improve Documentation Coverage',
                'description': f"Only {doc_rate:.0f}% of components have documentation. "
                              "Well-documented code is easier to maintain and understand.",
                'action': "Add docstrings to classes and functions explaining their purpose, "
                         "parameters, return values, and potential exceptions."
            })

        # Check naming consistency
        naming = patterns.get('naming_conventions', {})
        overall_naming = naming.get('overall_consistency', 0)

        if overall_naming < 0.8:
            suggestions.append({
                'title': 'Standardize Naming Conventions',
                'description': f"Naming consistency is at {overall_naming*100:.0f}%. "
                              "Inconsistent naming makes code harder to read.",
                'action': f"Use {naming.get('function_convention', 'snake_case')} for functions "
                         f"and {naming.get('class_convention', 'PascalCase')} for classes consistently."
            })

        # Check error handling
        error_handling = patterns.get('error_handling', {})
        handling_rate = error_handling.get('percentage_with_handling', 0)

        if handling_rate < 30:
            suggestions.append({
                'title': 'Add Error Handling',
                'description': f"Only {handling_rate:.0f}% of components document error handling. "
                              "Proper error handling improves reliability.",
                'action': "Add try-except blocks where appropriate and document raised exceptions "
                         "in docstrings."
            })

        # Check route consistency
        route_patterns = patterns.get('route_patterns', {})
        route_consistency = route_patterns.get('consistency', 0)

        if route_patterns.get('total_routes', 0) > 0 and route_consistency < 0.7:
            suggestions.append({
                'title': 'Standardize API Routes',
                'description': f"Route patterns are {route_consistency*100:.0f}% consistent. "
                              "Inconsistent API patterns confuse users.",
                'action': "Follow a consistent pattern like /api/v1/{resource} or /api/v1/{resource}/{id}"
            })

        return suggestions
