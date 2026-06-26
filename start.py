#!/usr/bin/env python
"""
CopyDetection-System: Entry Point
Runs the Flask web application.
"""
import sys
import subprocess
from src.main import run_app

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--production':
        print("[INFO] Starting production server with Gunicorn...")
        raise SystemExit(subprocess.call([
            sys.executable,
            "-m",
            "gunicorn",
            "-w",
            "4",
            "-b",
            "0.0.0.0:5001",
            "src.main:app",
        ]))
    else:
        print("[INFO] Starting local development server...")
        run_app()
