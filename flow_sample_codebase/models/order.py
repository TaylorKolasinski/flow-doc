"""
Order data models and schemas.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from decimal import Decimal
from enum import Enum


class OrderStatus(str, Enum):
    """Possible order statuses."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class OrderItemCreate(BaseModel):
    """Schema for order line items."""
    product_id: str = Field(..., description="Product identifier")
    quantity: int = Field(..., gt=0, description="Quantity ordered")
    price: Decimal = Field(..., gt=0, description="Price per unit at time of order")


class OrderItemResponse(OrderItemCreate):
    """Schema for order item responses."""
    id: str
    product_name: str
    subtotal: Decimal
    
    class Config:
        json_encoders = {
            Decimal: lambda v: float(v)
        }


class OrderCreate(BaseModel):
    """Schema for creating a new order."""
    user_id: str = Field(..., description="User placing the order")
    items: List[OrderItemCreate] = Field(..., min_items=1, description="Order items")
    shipping_address: str = Field(..., min_length=10, description="Shipping address")
    notes: Optional[str] = Field(None, max_length=500, description="Order notes")
    
    @validator('items')
    def validate_items(cls, v):
        """Ensure at least one item in order."""
        if not v:
            raise ValueError('Order must contain at least one item')
        return v


class OrderStatusUpdate(BaseModel):
    """Schema for updating order status."""
    status: OrderStatus = Field(..., description="New order status")
    notes: Optional[str] = Field(None, max_length=500, description="Status update notes")


class OrderResponse(BaseModel):
    """Schema for order responses."""
    id: str
    user_id: str
    items: List[OrderItemResponse]
    total_amount: Decimal
    status: OrderStatus
    shipping_address: str
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
