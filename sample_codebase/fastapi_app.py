"""Sample FastAPI application for testing."""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import asyncio

app = FastAPI(title="Sample API")


class User(BaseModel):
    """User model."""
    id: int
    name: str
    email: str
    active: bool = True


class UserCreate(BaseModel):
    """Schema for creating users."""
    name: str
    email: str


class Database:
    """Mock database class."""

    def __init__(self):
        """Initialize database."""
        self.users: List[User] = []

    async def get_user(self, user_id: int) -> Optional[User]:
        """Retrieve user by ID."""
        for user in self.users:
            if user.id == user_id:
                return user
        return None

    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        user_id = len(self.users) + 1
        user = User(id=user_id, **user_data.dict())
        self.users.append(user)
        return user


db = Database()


def get_db():
    """Dependency for database access."""
    return db


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to FastAPI"}


@app.get("/api/users", response_model=List[User])
async def get_users(skip: int = 0, limit: int = 10, database: Database = Depends(get_db)):
    """Get all users with pagination."""
    return database.users[skip:skip + limit]


@app.get("/api/users/{user_id}", response_model=User)
async def get_user(user_id: int, database: Database = Depends(get_db)):
    """Get user by ID."""
    user = await database.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/api/users", response_model=User, status_code=201)
async def create_user(user: UserCreate, database: Database = Depends(get_db)):
    """Create a new user."""
    return await database.create_user(user)


@app.put("/api/users/{user_id}", response_model=User)
async def update_user(user_id: int, user_data: UserCreate):
    """Update an existing user."""
    user = await db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.name = user_data.name
    user.email = user_data.email
    return user


@app.delete("/api/users/{user_id}")
async def delete_user(user_id: int):
    """Delete a user."""
    user = await db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.users.remove(user)
    return {"message": "User deleted"}


async def background_task():
    """Background task example."""
    await asyncio.sleep(10)
    print("Task completed")
