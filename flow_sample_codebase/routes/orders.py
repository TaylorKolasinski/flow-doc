"""
Order API routes.
"""

from fastapi import APIRouter, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials
import logging

from models.order import OrderCreate, OrderResponse, OrderStatusUpdate
from services.order_service import order_service
from utils.auth import security

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/orders", response_model=OrderResponse, status_code=201)
async def create_order(
    order: OrderCreate,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Create a new order.
    
    Requires authentication. Validates product availability, calculates totals,
    and reserves inventory. The order is created in PENDING status.
    
    Validates:
    - All products exist
    - Sufficient stock is available
    - Prices are current
    
    On success, reduces product inventory and returns the created order.
    """
    # Auth check
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        result = await order_service.create_order(order)
        return {
            "data": result,
            "message": "Order created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise HTTPException(status_code=500, detail="Failed to create order")


@router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: str,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Retrieve order details by ID.
    
    Requires authentication. Returns complete order information including
    all line items, current status, and shipping details.
    """
    # Auth check
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    result = await order_service.get_order(order_id)
    return {"data": result, "message": "Order retrieved"}


@router.patch("/orders/{order_id}/status", response_model=OrderResponse)
async def update_order_status(
    order_id: str,
    status_update: OrderStatusUpdate,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Update the status of an order.
    
    Requires authentication. Updates order through its lifecycle:
    pending -> confirmed -> processing -> shipped -> delivered
    
    Cannot update status of cancelled or already delivered orders.
    Optional notes can be added to document the status change.
    """
    # Auth check
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    result = await order_service.update_order_status(order_id, status_update)
    return {
        "data": result,
        "message": f"Order status updated to {status_update.status}"
    }
