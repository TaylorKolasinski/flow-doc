# E-Commerce Sample API

This is a sample FastAPI microservice provided as the codebase for the **AI-Powered API Documentation Assistant** exercise.

## Overview

A simple e-commerce API with endpoints for managing:
- **Products**: CRUD operations for product catalog
- **Orders**: Order creation and status management
- **Users**: User registration and profile management

## Architecture

```
sample_api/
├── app.py              # Main FastAPI application
├── routes/             # API endpoint definitions
│   ├── products.py
│   ├── orders.py
│   └── users.py
├── models/             # Pydantic schemas
│   ├── product.py
│   ├── order.py
│   └── user.py
├── services/           # Business logic layer
│   ├── product_service.py
│   ├── order_service.py
│   └── user_service.py
└── utils/              # Utilities
    ├── auth.py         # Authentication decorators
    ├── database.py     # In-memory database
    └── validation.py   # Validation utilities
```

## API Patterns

### Standard Response Format
Most endpoints return data in this format:
```json
{
  "data": { ... },
  "message": "Operation successful"
}
```

### Authentication
Protected endpoints require a Bearer token in the Authorization header:
```
Authorization: Bearer <token>
```

### Error Handling
Errors are returned with appropriate HTTP status codes and detail messages.

### Pagination
List endpoints support pagination with `page` and `page_size` query parameters.

## Endpoints

### Products
- `GET /api/products` - List all products (with pagination)
- `GET /api/products/{id}` - Get product by ID
- `POST /api/products` - Create new product (auth required)
- `PUT /api/products/{id}` - Update product (auth required)
- `DELETE /api/products/{id}` - Delete product (auth required)

### Orders
- `POST /api/orders` - Create new order (auth required)
- `GET /api/orders/{id}` - Get order by ID (auth required)
- `PATCH /api/orders/{id}/status` - Update order status (auth required)

### Users
- `POST /api/users` - Register new user
- `GET /api/users/{id}` - Get user by ID (auth required)
- `PUT /api/users/{id}` - Update user profile (auth required)

## Running the API

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py

# Or use uvicorn directly
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

## Interactive Documentation

Once running, access the auto-generated API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Design Patterns to Discover

This codebase contains several intentional patterns and inconsistencies that your documentation assistant should identify:

### Consistent Patterns
- ✅ Service layer separation (routes → services → database)
- ✅ Pydantic model validation
- ✅ Standard response format with `data` and `message` fields
- ✅ Bearer token authentication decorator
- ✅ Logging throughout the application
- ✅ Timestamp tracking (created_at, updated_at)

### Inconsistencies (For AI to Detect)
- ⚠️ **Error Codes**: Some endpoints return 404 for "not found", others return 400
- ⚠️ **Response Format**: Most endpoints wrap responses, some don't
- ⚠️ **Pagination**: Products endpoint has pagination, orders/users don't
- ⚠️ **Error Messages**: User update returns structured error object, others return strings

## Notes for Exercise

This is an intentionally simplified API with:
- In-memory database (data doesn't persist)
- Simplified authentication (no real JWT validation)
- Basic validation (production would be more thorough)
- Some intentional inconsistencies for pattern detection

Your task is to build an AI system that can:
1. Analyze this codebase
2. Extract patterns and conventions
3. Generate comprehensive documentation
4. Identify inconsistencies
5. Answer natural language queries about the API

Good luck! 🚀
