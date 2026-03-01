import os

# Reference dataset path (relative to project root)
DEFAULT_REFERENCE_PATH = os.path.join("test_data", "Excel_Dataset", "processed_books_dataset-1.xlsx")

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
