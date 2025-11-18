"""
Product data models and schemas.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime
from decimal import Decimal


class ProductBase(BaseModel):
    """Base product schema with common fields."""
    name: str = Field(..., min_length=1, max_length=200, description="Product name")
    description: Optional[str] = Field(None, max_length=1000, description="Product description")
    price: Decimal = Field(..., gt=0, description="Product price (must be positive)")
    sku: str = Field(..., min_length=3, max_length=50, description="Stock Keeping Unit")
    category: str = Field(..., description="Product category")
    stock_quantity: int = Field(default=0, ge=0, description="Available stock quantity")


class ProductCreate(ProductBase):
    """Schema for creating a new product."""
    
    @validator('sku')
    def sku_alphanumeric(cls, v):
        """Validate SKU is alphanumeric with hyphens."""
        if not all(c.isalnum() or c == '-' for c in v):
            raise ValueError('SKU must be alphanumeric with optional hyphens')
        return v.upper()


class ProductUpdate(BaseModel):
    """Schema for updating an existing product."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    price: Optional[Decimal] = Field(None, gt=0)
    category: Optional[str] = None
    stock_quantity: Optional[int] = Field(None, ge=0)


class ProductResponse(ProductBase):
    """Schema for product responses."""
    id: str = Field(..., description="Unique product identifier")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }


class ProductListResponse(BaseModel):
    """Schema for paginated product list responses."""
    products: list[ProductResponse]
    total: int
    page: int
    page_size: int
    has_more: bool
