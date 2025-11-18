"""
Tests for PatternDetector - Code pattern analysis and deviation detection.

This module tests:
- Route pattern analysis (versioning, prefixes, HTTP methods)
- Error handling pattern detection
- Authentication pattern identification
- Naming convention analysis
- Import organization patterns
- Documentation coverage analysis
- Consistency scoring
- Deviation detection and suggestions
"""

import pytest

from src.core.pattern_detector import PatternDetector


# ============================================================================
# Initialization Tests
# ============================================================================

def test_pattern_detector_initialization():
    """Test PatternDetector initializes correctly."""
    detector = PatternDetector()

    assert detector is not None


# ============================================================================
# Route Pattern Tests
# ============================================================================

def test_analyze_route_patterns_detects_versioning():
    """Test detecting versioned routes."""
    components = [
        {"type": "route", "path": "/api/v1/users", "methods": ["GET"]},
        {"type": "route", "path": "/api/v1/posts", "methods": ["GET"]},
        {"type": "route", "path": "/api/v2/users", "methods": ["GET"]}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    route_patterns = patterns["route_patterns"]
    assert route_patterns["versioned"] is True
    assert route_patterns["total_routes"] == 3


def test_analyze_route_patterns_detects_api_prefix():
    """Test detecting /api/ prefix."""
    components = [
        {"type": "route", "path": "/api/users", "methods": ["GET"]},
        {"type": "route", "path": "/api/posts", "methods": ["GET"]}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    route_patterns = patterns["route_patterns"]
    assert route_patterns["api_prefix"] is True


def test_analyze_route_patterns_detects_methods():
    """Test detecting HTTP methods usage."""
    components = [
        {"type": "route", "path": "/users", "methods": ["GET"]},
        {"type": "route", "path": "/users", "methods": ["POST"]},
        {"type": "route", "path": "/users", "methods": ["GET", "PUT"]}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    methods_used = patterns["route_patterns"]["methods_used"]
    assert methods_used["GET"] >= 2
    assert methods_used["POST"] >= 1
    assert methods_used["PUT"] >= 1


def test_analyze_route_patterns_empty_routes():
    """Test route analysis with no routes."""
    components = [
        {"type": "function", "name": "test"}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    route_patterns = patterns["route_patterns"]
    # When no routes, total_routes might not be present
    assert route_patterns.get("total_routes", 0) == 0


# ============================================================================
# Error Handling Pattern Tests
# ============================================================================

def test_analyze_error_handling_detects_try_except():
    """Test detecting try-except blocks."""
    components = [
        {"type": "function", "name": "func1", "raises": ["ValueError"], "docstring": "Raises ValueError if invalid"},
        {"type": "function", "name": "func2", "raises": ["HTTPException"], "docstring": "Raises HTTPException on error"}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    error_handling = patterns["error_handling"]
    # Error handling is detected from docstring mentions
    assert error_handling["raise"] >= 2
    assert error_handling["percentage_with_handling"] > 0


def test_analyze_error_handling_no_error_handling():
    """Test components with no error handling."""
    components = [
        {"type": "function", "name": "func1", "raises": []},
        {"type": "function", "name": "func2"}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    error_handling = patterns["error_handling"]
    assert error_handling["percentage_with_handling"] == 0.0


# ============================================================================
# Authentication Pattern Tests
# ============================================================================

def test_analyze_auth_patterns_detects_decorators():
    """Test detecting auth decorators."""
    components = [
        {"type": "route", "name": "get_users", "decorators": ["@require_auth"]},
        {"type": "route", "name": "create_user", "decorators": ["@require_auth", "@admin_only"]},
        {"type": "route", "name": "public_endpoint", "decorators": []}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    auth_patterns = patterns["auth_patterns"]
    assert "@require_auth" in auth_patterns["auth_decorators"]
    assert auth_patterns["decorator_usage_count"] >= 2
    assert auth_patterns["has_auth_pattern"] is True


def test_analyze_auth_patterns_no_auth():
    """Test components with no auth patterns."""
    components = [
        {"type": "function", "name": "func1", "decorators": []},
        {"type": "function", "name": "func2", "decorators": []}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    auth_patterns = patterns["auth_patterns"]
    assert auth_patterns["has_auth_pattern"] is False


# ============================================================================
# Naming Convention Tests
# ============================================================================

def test_analyze_naming_conventions_snake_case():
    """Test detecting snake_case naming."""
    components = [
        {"type": "function", "name": "get_user"},
        {"type": "function", "name": "create_user"},
        {"type": "function", "name": "delete_user"}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    naming = patterns["naming_conventions"]
    assert naming["function_convention"] == "snake_case"
    assert naming["function_consistency"] > 0.9  # 90%+ consistent


def test_analyze_naming_conventions_camel_case():
    """Test detecting camelCase naming."""
    components = [
        {"type": "function", "name": "getUser"},
        {"type": "function", "name": "createUser"},
        {"type": "function", "name": "deleteUser"}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    naming = patterns["naming_conventions"]
    assert naming["function_convention"] == "camelCase"


def test_analyze_naming_conventions_pascal_case_classes():
    """Test detecting PascalCase for classes."""
    components = [
        {"type": "class", "name": "UserService"},
        {"type": "class", "name": "AuthService"},
        {"type": "class", "name": "DataValidator"}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    naming = patterns["naming_conventions"]
    assert naming["class_convention"] == "PascalCase"
    assert naming["class_consistency"] > 0.9


def test_analyze_naming_conventions_mixed_styles():
    """Test handling mixed naming styles."""
    components = [
        {"type": "function", "name": "get_user"},  # snake_case
        {"type": "function", "name": "getUser"},   # camelCase
        {"type": "function", "name": "GetUser"}    # PascalCase
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    naming = patterns["naming_conventions"]
    # Should detect dominant pattern
    assert naming["function_consistency"] < 1.0  # Not 100% consistent


# ============================================================================
# Import Organization Tests
# ============================================================================

def test_analyze_imports_detects_patterns():
    """Test analyzing import patterns."""
    components = [
        {"type": "function", "name": "f1", "imports": ["os", "sys"]},
        {"type": "function", "name": "f2", "imports": ["typing", "pathlib"]},
        {"type": "function", "name": "f3", "imports": ["fastapi", "pydantic"]}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    imports = patterns["import_organization"]
    assert imports["total_imports"] >= 6
    assert imports["unique_imports"] >= 6


def test_analyze_imports_detects_common_modules():
    """Test detecting commonly used modules."""
    components = [
        {"type": "function", "name": "f1", "imports": ["os", "sys"]},
        {"type": "function", "name": "f2", "imports": ["os", "pathlib"]},
        {"type": "function", "name": "f3", "imports": ["os", "typing"]}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    imports = patterns["import_organization"]
    assert "os" in imports["common_imports"]


# ============================================================================
# Documentation Pattern Tests
# ============================================================================

def test_analyze_documentation_calculates_coverage():
    """Test documentation coverage calculation."""
    components = [
        {"type": "function", "name": "f1", "docstring": "Good documentation here"},
        {"type": "function", "name": "f2", "docstring": "Also documented"},
        {"type": "function", "name": "f3", "docstring": ""},
        {"type": "function", "name": "f4", "docstring": None}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    docs = patterns["documentation_patterns"]
    assert docs["total_components"] == 4
    assert docs["documented_count"] == 2
    assert docs["documentation_rate"] == 50.0


def test_analyze_documentation_all_documented():
    """Test when all components are documented."""
    components = [
        {"type": "function", "name": "f1", "docstring": "Doc 1"},
        {"type": "function", "name": "f2", "docstring": "Doc 2"}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    docs = patterns["documentation_patterns"]
    assert docs["documentation_rate"] == 100.0


def test_analyze_documentation_none_documented():
    """Test when no components are documented."""
    components = [
        {"type": "function", "name": "f1", "docstring": ""},
        {"type": "function", "name": "f2"}
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    docs = patterns["documentation_patterns"]
    assert docs["documentation_rate"] == 0.0


# ============================================================================
# Consistency Score Tests
# ============================================================================

def test_consistency_score_calculation(sample_components_list):
    """Test overall consistency score calculation."""
    detector = PatternDetector()
    patterns = detector.analyze_patterns(sample_components_list)

    assert "consistency_score" in patterns
    assert 0.0 <= patterns["consistency_score"] <= 1.0


def test_consistency_score_perfect_codebase():
    """Test consistency score with perfectly consistent codebase."""
    components = [
        {
            "type": "route",
            "name": "get_users",
            "path": "/api/v1/users",
            "methods": ["GET"],
            "docstring": "Get all users",
            "decorators": ["@require_auth"]
        },
        {
            "type": "route",
            "name": "create_user",
            "path": "/api/v1/users",
            "methods": ["POST"],
            "docstring": "Create a user",
            "decorators": ["@require_auth"]
        }
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    # Should have high consistency
    assert patterns["consistency_score"] > 0.5


# ============================================================================
# Deviation Detection Tests
# ============================================================================

def test_detect_deviations_finds_route_issues():
    """Test detecting route pattern deviations."""
    # Establish pattern with versioned routes
    all_components = [
        {"type": "route", "path": "/api/v1/users", "methods": ["GET"]},
        {"type": "route", "path": "/api/v1/posts", "methods": ["GET"]},
        {"type": "route", "path": "/api/v1/comments", "methods": ["GET"]}
    ]

    # Component that deviates
    component = {
        "type": "route",
        "name": "legacy_endpoint",
        "path": "/legacy/data",  # No /api/ prefix, no version
        "methods": ["GET"],
        "docstring": ""
    }

    detector = PatternDetector()
    patterns = detector.analyze_patterns(all_components)
    suggestions = detector.detect_deviations(component, patterns)

    # Should suggest adding /api/ prefix and versioning
    assert len(suggestions) > 0
    assert any("api" in s.lower() or "version" in s.lower() for s in suggestions)


def test_detect_deviations_finds_naming_issues():
    """Test detecting naming convention deviations."""
    all_components = [
        {"type": "function", "name": "get_user"},
        {"type": "function", "name": "create_user"},
        {"type": "function", "name": "delete_user"}
    ]

    # Component with different naming style
    component = {
        "type": "function",
        "name": "getPost",  # camelCase instead of snake_case
        "docstring": "Get post"
    }

    detector = PatternDetector()
    patterns = detector.analyze_patterns(all_components)
    suggestions = detector.detect_deviations(component, patterns)

    # Should suggest using snake_case
    assert len(suggestions) > 0
    assert any("snake_case" in s.lower() for s in suggestions)


def test_detect_deviations_finds_documentation_issues():
    """Test detecting missing documentation."""
    all_components = [
        {"type": "function", "name": "f1", "docstring": "Documented"},
        {"type": "function", "name": "f2", "docstring": "Also documented"},
        {"type": "function", "name": "f3", "docstring": "Well documented"}
    ]

    # Component missing docs
    component = {
        "type": "function",
        "name": "undocumented_function",
        "docstring": ""
    }

    detector = PatternDetector()
    patterns = detector.analyze_patterns(all_components)
    suggestions = detector.detect_deviations(component, patterns)

    # Should suggest adding documentation
    assert len(suggestions) > 0
    assert any("docstring" in s.lower() or "document" in s.lower() for s in suggestions)


def test_detect_deviations_no_issues():
    """Test when component follows all patterns."""
    all_components = [
        {
            "type": "route",
            "name": "get_users",
            "path": "/api/v1/users",
            "methods": ["GET"],
            "docstring": "Get all users",
            "decorators": ["@require_auth"]
        }
    ]

    # Perfectly conforming component
    component = {
        "type": "route",
        "name": "get_posts",
        "path": "/api/v1/posts",
        "methods": ["GET"],
        "docstring": "Get all posts",
        "decorators": ["@require_auth"]
    }

    detector = PatternDetector()
    patterns = detector.analyze_patterns(all_components)
    suggestions = detector.detect_deviations(component, patterns)

    # Should have no or minimal suggestions
    assert len(suggestions) <= 1  # May have minor suggestions


# ============================================================================
# Report Generation Tests
# ============================================================================

def test_generate_recommendations_creates_markdown(sample_components_list):
    """Test generating markdown recommendations report."""
    detector = PatternDetector()
    patterns = detector.analyze_patterns(sample_components_list)

    report = detector.generate_recommendations(patterns)

    assert isinstance(report, str)
    assert len(report) > 0
    assert "# Code Pattern Analysis Report" in report


def test_generate_recommendations_includes_sections(sample_components_list):
    """Test report includes all required sections."""
    detector = PatternDetector()
    patterns = detector.analyze_patterns(sample_components_list)

    report = detector.generate_recommendations(patterns)

    # Check for key sections
    assert "## Executive Summary" in report
    assert "Consistency Score" in report or "consistency" in report.lower()
    assert "#" in report  # Has markdown headers


def test_generate_recommendations_empty_patterns():
    """Test report generation with empty patterns."""
    detector = PatternDetector()
    patterns = detector.analyze_patterns([])

    report = detector.generate_recommendations(patterns)

    assert isinstance(report, str)
    assert len(report) > 0


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_pattern_analysis_workflow(sample_components_list):
    """Test complete pattern analysis workflow."""
    detector = PatternDetector()

    # Analyze patterns
    patterns = detector.analyze_patterns(sample_components_list)
    assert patterns is not None
    assert "consistency_score" in patterns

    # Detect deviations for each component
    for component in sample_components_list:
        suggestions = detector.detect_deviations(component, patterns)
        assert isinstance(suggestions, list)

    # Generate report
    report = detector.generate_recommendations(patterns)
    assert len(report) > 0

    # All operations completed successfully
    assert True


def test_pattern_detection_with_diverse_components():
    """Test pattern detection with various component types."""
    components = [
        # Routes with different patterns
        {"type": "route", "name": "r1", "path": "/api/v1/users", "methods": ["GET"], "docstring": "Doc"},
        {"type": "route", "name": "r2", "path": "/api/v2/posts", "methods": ["POST"], "docstring": "Doc"},
        # Functions with different styles
        {"type": "function", "name": "get_user", "docstring": "Snake case"},
        {"type": "function", "name": "getPost", "docstring": "Camel case"},
        # Classes
        {"type": "class", "name": "UserService", "docstring": "Service class"},
        {"type": "class", "name": "auth_helper", "docstring": ""},  # Wrong casing
    ]

    detector = PatternDetector()
    patterns = detector.analyze_patterns(components)

    assert patterns["route_patterns"]["total_routes"] == 2
    assert patterns["documentation_patterns"]["total_components"] == 6
    assert 0.0 <= patterns["consistency_score"] <= 1.0
