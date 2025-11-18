#!/usr/bin/env python3
"""
Run the Flow-Doc FastAPI server.

Usage:
    python run_api.py              # Run on default port 8000
    python run_api.py --port 9000  # Run on custom port
    python run_api.py --reload     # Run with auto-reload
"""

import sys
import argparse
import uvicorn


def main():
    """Run the FastAPI server."""
    parser = argparse.ArgumentParser(description="Run Flow-Doc API Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Log level (default: info)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )

    args = parser.parse_args()

    print(f"""
╔{'=' * 68}╗
║{' ' * 68}║
║{'Flow-Doc API Server'.center(68)}║
║{' ' * 68}║
╚{'=' * 68}╝

Starting server on http://{args.host}:{args.port}

Documentation: http://{args.host}:{args.port}/docs
Health Check:  http://{args.host}:{args.port}/api/v1/health

Press Ctrl+C to stop the server
{'=' * 70}
    """)

    try:
        uvicorn.run(
            "src.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            workers=args.workers if not args.reload else 1  # Workers incompatible with reload
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
