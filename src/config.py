import os

# Database configuration (SQLite + FAISS)
DATABASE_FOLDER = os.getenv("DATABASE_FOLDER", "database")
DATA_FOLDER = os.getenv("DATA_FOLDER", "test_data/Excel_Dataset")

# Text validation constraints
MIN_TEXT_LENGTH = 50
MAX_PDF_PAGES = 25
MAX_TEXT_LENGTH = 100000 

# ML Model configuration
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# File upload constraints
MAX_ANALYZE_UPLOAD_SIZE = 16 * 1024 * 1024   # 16 MB

ALLOWED_ANALYZE_MIMETYPES = {
    'application/pdf',
    'text/plain',
}
