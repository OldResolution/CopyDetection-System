import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Settings
DEFAULT_REFERENCE_PATH = r"D:\CopyDetection-System-main\Excel_Dataset\processed_books_dataset-1.xlsx"

# Model Settings
MODEL_NAME = "all-MiniLM-L6-v2"

# Constraints
MAX_PDF_PAGES = 25
MAX_TEXT_LENGTH = 100000
MIN_TEXT_LENGTH = 50