"""
Order service layer - business logic for order operations.
"""

from typing import Dict, Any
from fastapi import HTTPException
from decimal import Decimal
import logging

from models.order import OrderCreate, OrderResponse, OrderStatus, OrderStatusUpdate, OrderItemResponse
from utils.database import get_db
from utils.validation import validate_stock_availability

logger = logging.getLogger(__name__)


class OrderService:
    """Service for managing order operations."""
    
    def __init__(self):
        self.db = get_db()
    
    async def create_order(self, order_data: OrderCreate) -> OrderResponse:
        """
        Create a new order.
        
        Validates product availability and creates order with line items.
        
        Args:
            order_data: Order creation data
            
        Returns:
            Created order
            
        Raises:
            HTTPException: If products not found or insufficient stock
        """
        # Validate all products exist and have sufficient stock
        validated_items = []
        total_amount = Decimal(0)
        
        for item in order_data.items:
            product = self.db.products.get(item.product_id)
            
            if not product:
                raise HTTPException(
                    status_code=404,
                    detail=f"Product {item.product_id} not found"
                )
            
            # Check stock availability
            validate_stock_availability(
                required=item.quantity,
                available=product["stock_quantity"],
                product_name=product["name"]
            )
            
            # Calculate subtotal
            subtotal = Decimal(str(item.price)) * item.quantity
            total_amount += subtotal
            
            validated_items.append({
                "id": self.db.generate_id(),
                "product_id": item.product_id,
                "product_name": product["name"],
                "quantity": item.quantity,
                "price": float(item.price),
                "subtotal": float(subtotal)
            })
            
            # Reduce stock
            product["stock_quantity"] -= item.quantity
        
        # Create order
        order_id = self.db.generate_id()
        now = self.db.get_timestamp()
        
        order = {
            "id": order_id,
            "user_id": order_data.user_id,
            "items": validated_items,
            "total_amount": float(total_amount),
            "status": OrderStatus.PENDING,
            "shipping_address": order_data.shipping_address,
            "notes": order_data.notes,
            "created_at": now,
            "updated_at": now
        }
        
        self.db.orders[order_id] = order
        
        logger.info(f"Created order: {order_id} for user {order_data.user_id}")
        
        # Convert items to response format
        response_items = [OrderItemResponse(**item) for item in validated_items]
        
        return OrderResponse(
            id=order["id"],
            user_id=order["user_id"],
            items=response_items,
            total_amount=Decimal(str(order["total_amount"])),
            status=order["status"],
            shipping_address=order["shipping_address"],
            notes=order["notes"],
            created_at=order["created_at"],
            updated_at=order["updated_at"]
        )
    
    async def get_order(self, order_id: str) -> OrderResponse:
        """
        Retrieve an order by ID.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Order data
            
        Raises:
            HTTPException: If order not found
        """
        order = self.db.orders.get(order_id)
        
        if not order:
            raise HTTPException(
                status_code=404,
                detail=f"Order {order_id} not found"
            )
        
        response_items = [OrderItemResponse(**item) for item in order["items"]]
        
        return OrderResponse(
            id=order["id"],
            user_id=order["user_id"],
            items=response_items,
            total_amount=Decimal(str(order["total_amount"])),
            status=order["status"],
            shipping_address=order["shipping_address"],
            notes=order["notes"],
            created_at=order["created_at"],
            updated_at=order["updated_at"]
        )
    
    async def update_order_status(
        self, 
        order_id: str, 
        status_update: OrderStatusUpdate
    ) -> OrderResponse:
        """
        Update order status.
        
        Handles order lifecycle transitions (pending -> confirmed -> processing -> shipped -> delivered).
        
        Args:
            order_id: Order identifier
            status_update: New status and optional notes
            
        Returns:
            Updated order
            
        Raises:
            HTTPException: If order not found or invalid status transition
        """
        order = self.db.orders.get(order_id)
        
        if not order:
            # Note: This uses 400 instead of 404 (intentional inconsistency for exercise)
            raise HTTPException(
                status_code=400,
                detail="Order not found"
            )
        
        # Validate status transition (simplified)
        current_status = OrderStatus(order["status"])
        new_status = status_update.status
        
        # Can't change status of cancelled or delivered orders
        if current_status in [OrderStatus.CANCELLED, OrderStatus.DELIVERED]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot update status of {current_status} order"
            )
        
        # Update status
        order["status"] = new_status
        order["updated_at"] = self.db.get_timestamp()
        
        if status_update.notes:
            # Append to existing notes
            existing_notes = order.get("notes", "")
            order["notes"] = f"{existing_notes}\n{status_update.notes}".strip()
        
        logger.info(f"Updated order {order_id} status: {current_status} -> {new_status}")
        
        response_items = [OrderItemResponse(**item) for item in order["items"]]
        
        return OrderResponse(
            id=order["id"],
            user_id=order["user_id"],
            items=response_items,
            total_amount=Decimal(str(order["total_amount"])),
            status=order["status"],
            shipping_address=order["shipping_address"],
            notes=order["notes"],
            created_at=order["created_at"],
            updated_at=order["updated_at"]
        )


# Singleton instance
order_service = OrderService()
