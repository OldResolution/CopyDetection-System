import fitz  # PyMuPDF
import re
import traceback
from config import MAX_PDF_PAGES, MAX_TEXT_LENGTH

def clean_extracted_text(text):
    if not text:
        return ""
    # Fix hyphenated words split across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Remove standalone page numbers
    text = re.sub(r'\n\s*\d+\s*\n', ' ', text)
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:MAX_TEXT_LENGTH]

def extract_text_from_pdf(file_storage):
    try:
        file_bytes = file_storage.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        
        max_pages = min(len(doc), MAX_PDF_PAGES)
        print(f"[PDF] Extracting text from {max_pages} pages...")
        
        for page_num in range(max_pages):
            page = doc.load_page(page_num)
            text += page.get_text("text") + " "
        
        doc.close()
        return clean_extracted_text(text)
    except Exception as e:
        print(f"[ERROR] PDF extraction failed: {e}")
        traceback.print_exc()
        return ""