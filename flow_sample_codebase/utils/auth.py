"""
Authentication and authorization utilities.
"""

from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from functools import wraps
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer()


def require_auth(func):
    """
    Decorator to require authentication for endpoints.
    
    This is a simplified auth decorator for the sample API.
    In production, this would validate JWT tokens, check expiration, etc.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract credentials from kwargs (injected by FastAPI dependency)
        credentials = kwargs.get('credentials')
        
        if not credentials:
            logger.warning("Authentication required but no credentials provided")
            raise HTTPException(
                status_code=401,
                detail="Authentication required"
            )
        
        # Simplified token validation (just check it exists)
        if not credentials.credentials:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )
        
        # In production: validate JWT, check expiration, load user context
        logger.info(f"Authenticated request to {func.__name__}")
        
        return await func(*args, **kwargs)
    
    return wrapper


def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Extract current user from authentication credentials.
    
    Returns:
        User ID extracted from token
    """
    # Simplified - in production would decode JWT and return user object
    return "user_123"


def check_permission(resource: str, action: str):
    """
    Decorator to check specific permissions for an action.
    
    Args:
        resource: Resource type (e.g., 'product', 'order')
        action: Action type (e.g., 'read', 'write', 'delete')
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Simplified permission check
            logger.info(f"Checking permission: {action} on {resource}")
            
            # In production: check user roles, permissions, etc.
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
