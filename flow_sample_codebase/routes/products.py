"""
Product API routes.
"""

from fastapi import APIRouter, HTTPException, Security, Query
from fastapi.security import HTTPAuthorizationCredentials
from typing import Optional
import logging

from models.product import ProductCreate, ProductUpdate, ProductResponse, ProductListResponse
from services.product_service import product_service
from utils.auth import security, require_auth

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/products", response_model=ProductListResponse)
async def list_products(
    category: Optional[str] = Query(None, description="Filter by category"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page")
):
    """
    List all products with optional filtering and pagination.
    
    Returns a paginated list of products. Can be filtered by category.
    Supports pagination to handle large product catalogs efficiently.
    """
    try:
        result = await product_service.list_products(
            category=category,
            page=page,
            page_size=page_size
        )
        return ProductListResponse(**result)
    except Exception as e:
        logger.error(f"Error listing products: {e}")
        raise HTTPException(status_code=500, detail="Failed to list products")


@router.get("/products/{product_id}", response_model=ProductResponse)
async def get_product(product_id: str):
    """
    Retrieve a specific product by ID.
    
    Returns detailed information about a single product including
    current stock levels, pricing, and description.
    """
    return await product_service.get_product(product_id)


@router.post("/products", response_model=ProductResponse, status_code=201)
async def create_product(
    product: ProductCreate,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Create a new product in the catalog.
    
    Requires authentication. Validates product data including SKU uniqueness,
    price validity, and required fields. Returns the created product with
    generated ID and timestamps.
    """
    # Auth check
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        result = await product_service.create_product(product)
        return {"data": result, "message": "Product created successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating product: {e}")
        raise HTTPException(status_code=500, detail="Failed to create product")


@router.put("/products/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: str,
    product: ProductUpdate,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Update an existing product.
    
    Requires authentication. Allows partial updates - only provided fields
    will be modified. SKU cannot be changed through this endpoint.
    """
    # Auth check
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    result = await product_service.update_product(product_id, product)
    return {"data": result, "message": "Product updated successfully"}


@router.delete("/products/{product_id}", status_code=204)
async def delete_product(
    product_id: str,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Delete a product from the catalog.
    
    Requires authentication. Permanently removes the product.
    This operation cannot be undone.
    """
    # Auth check
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    await product_service.delete_product(product_id)
    return None
