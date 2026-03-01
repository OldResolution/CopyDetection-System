#!/usr/bin/env python
"""
CopyDetection-System: Entry Point
Runs the Flask web application.
"""
import sys
from src.main import app, run_app

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--production':
        # Production mode
        print("[INFO] Starting in production mode...")
        app.run(debug=False, host='0.0.0.0', port=5001)
    else:
        # Development mode
        print("[INFO] Starting in development mode...")
        run_app()
