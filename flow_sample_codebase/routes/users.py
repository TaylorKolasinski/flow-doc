"""
User API routes.
"""

from fastapi import APIRouter, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials
import logging

from models.user import UserCreate, UserResponse, UserUpdate
from services.user_service import user_service
from utils.auth import security

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate):
    """
    Register a new user account.
    
    Creates a new user with validated email and password.
    Password must meet security requirements:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    
    Returns the created user (password excluded).
    """
    try:
        result = await user_service.create_user(user)
        return {
            "data": result,
            "message": "User registered successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Retrieve user information by ID.
    
    Requires authentication. Returns user profile data excluding
    sensitive information like password hash.
    """
    # Auth check
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    result = await user_service.get_user(user_id)
    return {"data": result}


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_update: UserUpdate,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Update user profile information.
    
    Requires authentication. Allows updating name and phone number.
    Email and password changes require separate endpoints (not implemented
    in this demo).
    """
    # Auth check
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    result = await user_service.update_user(user_id, user_update)
    return {
        "data": result,
        "message": "User updated"
    }
