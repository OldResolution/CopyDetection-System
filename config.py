import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DEFAULT_REFERENCE_PATH = r"D:\CopyDetection-System-main\Excel_Dataset\processed_books_dataset-1.xlsx"

MODEL_NAME = "all-MiniLM-L6-v2"

MIN_TEXT_LENGTH         = 50
MAX_TEXT_LENGTH         = 100000
MAX_PDF_PAGES           = 25

MAX_ANALYZE_UPLOAD_SIZE = 16 * 1024 * 1024   # 16 MB

ALLOWED_ANALYZE_MIMETYPES = {
    'application/pdf',
    'text/plain',
}