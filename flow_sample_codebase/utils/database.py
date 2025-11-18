"""
Database utilities and mock data storage.

This is a simplified in-memory database for the sample API.
In production, this would use SQLAlchemy, MongoDB, etc.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


class InMemoryDB:
    """Simple in-memory database for demonstration purposes."""
    
    def __init__(self):
        self.products: Dict[str, Dict] = {}
        self.orders: Dict[str, Dict] = {}
        self.users: Dict[str, Dict] = {}
        self.product_skus: Dict[str, str] = {}  # SKU -> ID mapping
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with some sample data."""
        logger.info("Initializing sample data")
        
        # Sample products
        sample_products = [
            {
                "id": str(uuid.uuid4()),
                "name": "Wireless Headphones",
                "description": "Premium noise-cancelling headphones",
                "price": 199.99,
                "sku": "WH-1000",
                "category": "Electronics",
                "stock_quantity": 50,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            },
            {
                "id": str(uuid.uuid4()),
                "name": "USB-C Cable",
                "description": "Fast charging USB-C cable",
                "price": 19.99,
                "sku": "CABLE-USBC-2M",
                "category": "Accessories",
                "stock_quantity": 200,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        ]
        
        for product in sample_products:
            self.products[product["id"]] = product
            self.product_skus[product["sku"]] = product["id"]
    
    def generate_id(self) -> str:
        """Generate a unique identifier."""
        return str(uuid.uuid4())
    
    def get_timestamp(self) -> datetime:
        """Get current timestamp."""
        return datetime.utcnow()


# Global database instance
db = InMemoryDB()


def get_db() -> InMemoryDB:
    """Get database instance."""
    return db
