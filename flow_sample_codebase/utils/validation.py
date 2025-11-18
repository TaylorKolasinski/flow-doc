"""
Validation utilities for business logic.
"""

from typing import Optional
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    pass


def validate_positive_number(value: float, field_name: str) -> None:
    """
    Validate that a number is positive.
    
    Args:
        value: Number to validate
        field_name: Name of the field for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if value <= 0:
        raise ValidationError(f"{field_name} must be positive")


def validate_not_empty(value: Optional[str], field_name: str) -> None:
    """
    Validate that a string is not empty.
    
    Args:
        value: String to validate
        field_name: Name of the field for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if not value or not value.strip():
        raise ValidationError(f"{field_name} cannot be empty")


def validate_stock_availability(required: int, available: int, product_name: str) -> None:
    """
    Validate that sufficient stock is available.
    
    Args:
        required: Quantity required
        available: Quantity available
        product_name: Name of the product
        
    Raises:
        HTTPException: If insufficient stock
    """
    if required > available:
        logger.warning(f"Insufficient stock for {product_name}: required {required}, available {available}")
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient stock for {product_name}. Available: {available}"
        )
