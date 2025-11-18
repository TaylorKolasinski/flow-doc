"""
Product service layer - business logic for product operations.
"""

from typing import List, Optional, Dict, Any
from fastapi import HTTPException
from datetime import datetime
import logging

from models.product import ProductCreate, ProductUpdate, ProductResponse
from utils.database import get_db
from utils.validation import ValidationError, validate_positive_number

logger = logging.getLogger(__name__)


class ProductService:
    """Service for managing product operations."""
    
    def __init__(self):
        self.db = get_db()
    
    async def create_product(self, product_data: ProductCreate) -> ProductResponse:
        """
        Create a new product in the catalog.
        
        Args:
            product_data: Product creation data
            
        Returns:
            Created product
            
        Raises:
            HTTPException: If SKU already exists
        """
        # Check if SKU already exists
        if product_data.sku in self.db.product_skus:
            logger.warning(f"Attempt to create product with duplicate SKU: {product_data.sku}")
            raise HTTPException(
                status_code=409,
                detail=f"Product with SKU {product_data.sku} already exists"
            )
        
        # Create product
        product_id = self.db.generate_id()
        now = self.db.get_timestamp()
        
        product = {
            "id": product_id,
            "name": product_data.name,
            "description": product_data.description,
            "price": float(product_data.price),
            "sku": product_data.sku,
            "category": product_data.category,
            "stock_quantity": product_data.stock_quantity,
            "created_at": now,
            "updated_at": now
        }
        
        self.db.products[product_id] = product
        self.db.product_skus[product_data.sku] = product_id
        
        logger.info(f"Created product: {product_id} ({product_data.sku})")
        
        return ProductResponse(**product)
    
    async def get_product(self, product_id: str) -> ProductResponse:
        """
        Retrieve a product by ID.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Product data
            
        Raises:
            HTTPException: If product not found
        """
        product = self.db.products.get(product_id)
        
        if not product:
            logger.warning(f"Product not found: {product_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Product {product_id} not found"
            )
        
        return ProductResponse(**product)
    
    async def list_products(
        self, 
        category: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        List products with optional filtering and pagination.
        
        Args:
            category: Filter by category
            page: Page number (1-indexed)
            page_size: Items per page
            
        Returns:
            Paginated product list
        """
        products = list(self.db.products.values())
        
        # Filter by category if provided
        if category:
            products = [p for p in products if p["category"] == category]
        
        # Pagination
        total = len(products)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        paginated_products = products[start_idx:end_idx]
        
        return {
            "products": [ProductResponse(**p) for p in paginated_products],
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": end_idx < total
        }
    
    async def update_product(self, product_id: str, update_data: ProductUpdate) -> ProductResponse:
        """
        Update an existing product.
        
        Args:
            product_id: Product identifier
            update_data: Fields to update
            
        Returns:
            Updated product
            
        Raises:
            HTTPException: If product not found
        """
        product = self.db.products.get(product_id)
        
        if not product:
            logger.warning(f"Product not found for update: {product_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Product {product_id} not found"
            )
        
        # Update only provided fields
        update_dict = update_data.dict(exclude_unset=True)
        
        for field, value in update_dict.items():
            if value is not None:
                product[field] = float(value) if field == "price" else value
        
        product["updated_at"] = self.db.get_timestamp()
        
        logger.info(f"Updated product: {product_id}")
        
        return ProductResponse(**product)
    
    async def delete_product(self, product_id: str) -> None:
        """
        Delete a product from the catalog.
        
        Args:
            product_id: Product identifier
            
        Raises:
            HTTPException: If product not found
        """
        product = self.db.products.get(product_id)
        
        if not product:
            # Note: This uses 400 instead of 404 (intentional inconsistency for exercise)
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete product {product_id}: not found"
            )
        
        # Remove from SKU mapping
        sku = product["sku"]
        if sku in self.db.product_skus:
            del self.db.product_skus[sku]
        
        del self.db.products[product_id]
        
        logger.info(f"Deleted product: {product_id}")


# Singleton instance
product_service = ProductService()
