"""Sample Flask application for testing."""

from flask import Flask, request, jsonify
from typing import Dict, List
import os

app = Flask(__name__)


class UserService:
    """Service for managing user operations."""

    def __init__(self):
        self.users = {}

    def get_user(self, user_id: int):
        """Get a user by ID."""
        return self.users.get(user_id)

    def create_user(self, name: str, email: str):
        """Create a new user."""
        user_id = len(self.users) + 1
        self.users[user_id] = {"name": name, "email": email}
        return user_id

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        return "@" in email


user_service = UserService()


@app.route("/")
def index():
    """Homepage route."""
    return "Welcome to Flask App"


@app.route("/api/users", methods=["GET"])
def get_users():
    """Get all users."""
    return jsonify(user_service.users)


@app.route("/api/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    """Get a specific user."""
    user = user_service.get_user(user_id)
    if user:
        return jsonify(user)
    return jsonify({"error": "User not found"}), 404


@app.route("/api/users", methods=["POST"])
def create_user():
    """Create a new user."""
    data = request.json
    user_id = user_service.create_user(data["name"], data["email"])
    return jsonify({"id": user_id}), 201


def calculate_total(price: float, tax: float) -> float:
    """Calculate total price with tax."""
    return price + (price * tax)


if __name__ == "__main__":
    app.run(debug=True)
