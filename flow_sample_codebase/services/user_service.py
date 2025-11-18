"""
User service layer - business logic for user operations.
"""

from typing import Optional
from fastapi import HTTPException
import hashlib
import logging

from models.user import UserCreate, UserResponse, UserUpdate
from utils.database import get_db

logger = logging.getLogger(__name__)


class UserService:
    """Service for managing user operations."""
    
    def __init__(self):
        self.db = get_db()
    
    def _hash_password(self, password: str) -> str:
        """
        Hash a password (simplified for demo).
        
        In production, use bcrypt or argon2.
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """
        Create a new user account.
        
        Args:
            user_data: User registration data
            
        Returns:
            Created user (without password)
            
        Raises:
            HTTPException: If email already exists
        """
        # Check if email already exists
        for user in self.db.users.values():
            if user["email"] == user_data.email:
                raise HTTPException(
                    status_code=409,
                    detail="Email already registered"
                )
        
        # Create user
        user_id = self.db.generate_id()
        now = self.db.get_timestamp()
        
        user = {
            "id": user_id,
            "email": user_data.email,
            "password_hash": self._hash_password(user_data.password),
            "first_name": user_data.first_name,
            "last_name": user_data.last_name,
            "phone": user_data.phone,
            "is_active": True,
            "created_at": now
        }
        
        self.db.users[user_id] = user
        
        logger.info(f"Created user: {user_id} ({user_data.email})")
        
        return UserResponse(
            id=user["id"],
            email=user["email"],
            first_name=user["first_name"],
            last_name=user["last_name"],
            phone=user["phone"],
            created_at=user["created_at"],
            is_active=user["is_active"]
        )
    
    async def get_user(self, user_id: str) -> UserResponse:
        """
        Retrieve a user by ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            User data (without password)
            
        Raises:
            HTTPException: If user not found
        """
        user = self.db.users.get(user_id)
        
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found"
            )
        
        return UserResponse(
            id=user["id"],
            email=user["email"],
            first_name=user["first_name"],
            last_name=user["last_name"],
            phone=user["phone"],
            created_at=user["created_at"],
            is_active=user["is_active"]
        )
    
    async def update_user(self, user_id: str, update_data: UserUpdate) -> UserResponse:
        """
        Update user information.
        
        Args:
            user_id: User identifier
            update_data: Fields to update
            
        Returns:
            Updated user
            
        Raises:
            HTTPException: If user not found
        """
        user = self.db.users.get(user_id)
        
        if not user:
            # Note: Returns different error message format (intentional inconsistency)
            raise HTTPException(
                status_code=404,
                detail={"error": "User not found", "user_id": user_id}
            )
        
        # Update only provided fields
        update_dict = update_data.dict(exclude_unset=True)
        
        for field, value in update_dict.items():
            if value is not None:
                user[field] = value
        
        logger.info(f"Updated user: {user_id}")
        
        return UserResponse(
            id=user["id"],
            email=user["email"],
            first_name=user["first_name"],
            last_name=user["last_name"],
            phone=user["phone"],
            created_at=user["created_at"],
            is_active=user["is_active"]
        )


# Singleton instance
user_service = UserService()
