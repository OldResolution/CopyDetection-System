import os

# Database configuration (SQLite + FAISS)
DATABASE_FOLDER = "database"

# Text validation constraints
MIN_TEXT_LENGTH = 50
MAX_PDF_PAGES = 25
MAX_TEXT_LENGTH = 100000 

# ML Model configuration
MODEL_NAME = 'all-MiniLM-L6-v2'

# File upload constraints
MAX_ANALYZE_UPLOAD_SIZE = 16 * 1024 * 1024   # 16 MB

ALLOWED_ANALYZE_MIMETYPES = {
    'application/pdf',
    'text/plain',
}
