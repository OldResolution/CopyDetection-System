"""
Main entry point for the CopyDetection-System Flask application.
"""
import os
from flask import Flask
from flask_cors import CORS
from src.api.routes import get_detector, register_routes

def create_app():
    """Factory function to create and configure the Flask app."""
    # Get absolute paths for templates and static files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_folder = os.path.join(base_dir, 'ui', 'templates')
    static_folder = os.path.join(base_dir, 'ui', 'static')
    
    app = Flask(__name__, 
                template_folder=template_folder,
                static_folder=static_folder)
    CORS(app)
    
    # Register all API routes
    register_routes(app)
    
    return app

app = create_app()

def run_app():
    """Run the Flask development server."""
    # Initialize detector on startup
    get_detector()
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host='0.0.0.0', port=5001)

if __name__ == '__main__':
    run_app()
