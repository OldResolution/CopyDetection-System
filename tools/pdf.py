import os
import re
import fitz  # PyMuPDF
from pathlib import Path
from ftfy import fix_text
import pytesseract
from PIL import Image
import io
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Optional: pip install tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Optional: pip install opencv-python for image preprocessing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Uncomment and set this if Tesseract is not in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Global logger
logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Comprehensive text cleaning pipeline for PDF extraction.
    Pipeline: Extract → Normalize → Regex Remove → Deduplicate → Filter Short
    """
    
    def __init__(self, min_line_length=5, preserve_isbn=True):
        self.min_line_length = min_line_length
        self.preserve_isbn = preserve_isbn
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile all regex patterns for performance."""
        
        # --- NORMALIZATION PATTERNS ---
        # Hyphenation across lines: "plagiar-\nism" → "plagiarism"
        self.regex_hyphenation = re.compile(r'(\w+)-\s*\n\s*(\w+)')
        # Multiple spaces → single space
        self.regex_multi_space = re.compile(r'[ \t]+')
        # Multiple newlines → double newline (paragraph break)
        self.regex_multi_newline = re.compile(r'\n{3,}')
        # Control characters (except newline)
        self.regex_control_chars = re.compile(r'[\r\f\v\x00-\x08\x0b\x0c\x0e-\x1f]')
        
        # --- CID / ENCODING ERRORS ---
        self.regex_cid = re.compile(r'\(cid:\d+\)')
        
        # --- STRUCTURAL HEADERS ---
        self.regex_headers = re.compile(
            r'^\s*(foreword|prologue|preface|introduction|acknowledgements?|abstract|contents|'
            r'index|bibliography|epilogue|afterword|appendix|glossary|dedication|copyright|'
            r'about\s+the\s+author|author\'?s?\s+note|editor\'?s?\s+note)\s*([:—\-]\s*.+)?\s*$',
            re.IGNORECASE
        )
        
        # --- NUMBERED STRUCTURES (Chapter 1, Part II, etc.) ---
        self.regex_numbered = re.compile(
            r'^\s*(chapter|book|part|vol\.?|volume|section|act|episode|unit)\s+'
            r'(\d+|[IVXLCDM]+|one|two|three|four|five|six|seven|eight|nine|ten|'
            r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|'
            r'twenty-?one|twenty-?two|twenty-?three|twenty-?four|twenty-?five)'
            r'\s*([:—\-]\s*.+|\s{2,}[A-Z].*)?\.?\s*$',
            re.IGNORECASE
        )
        
        # --- TABLE OF CONTENTS ENTRIES ---
        # "Introduction ............ 5" or "Topic     14"
        self.regex_toc_entry = re.compile(r'^.+(\.{2,}|\s{4,})\s*\d+\s*$')
        
        # --- TOC CHAPTER LINES (without page numbers) ---
        # "Chapter 1 The Departure of Boromir" or "Chapter I A Long-expected Party"
        self.regex_toc_chapter = re.compile(
            r'^\s*(Chapter|Part|Book|Volume|Prologue|Epilogue)\s+'
            r'(\d+|[IVXLCDM]+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|'
            r'Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty)'
            r'\s+[A-Z][A-Za-z\s\-\',]+\s*$',
            re.IGNORECASE
        )
        
        # --- TOC HEADERS ---
        self.regex_toc_headers = re.compile(
            r'^\s*(complete\s+)?table\s+of\s+contents\s*$|'
            r'^\s*list\s+of\s+(figures|tables|illustrations)\s*$|'
            r'^\s*concerning\s+.+$|'
            r'^\s*of\s+the\s+.+$|'
            r'^\s*(a\s+)?note\s+on\s+.+$',
            re.IGNORECASE
        )
        
        # --- PAGE NUMBERS (isolated) ---
        self.regex_page_num = re.compile(r'^\s*(\d{1,4}|[ivxlcIVXLC]+)\s*$')
        
        # --- PAGE HEADER/FOOTER PATTERNS ---
        # "Page 42", "- 42 -", "42 | Chapter 1"
        self.regex_page_markers = re.compile(
            r'^\s*(page\s+\d+|\-\s*\d+\s*\-|\d+\s*\|\s*.+|.+\s*\|\s*\d+)\s*$',
            re.IGNORECASE
        )
        
        # --- WATERMARKS/ANNOTATIONS ---
        self.regex_watermarks = re.compile(
            r'^\s*(draft|confidential|do\s+not\s+distribute|sample|preview|'
            r'review\s+copy|uncorrected\s+proof|advance\s+reader|not\s+for\s+sale)\s*$',
            re.IGNORECASE
        )
        
        # --- BOILERPLATE (Copyright, etc.) ---
        self.regex_boilerplate = re.compile(
            r'^\s*(©|copyright|\(c\)|all\s+rights\s+reserved|printed\s+in|'
            r'published\s+by|first\s+edition|second\s+edition|'
            r'library\s+of\s+congress|cataloging|'
            r'this\s+e?-?book\s+is|e-?book\s+edition|digital\s+edition)\s*.{0,100}$',
            re.IGNORECASE
        )
        
        # --- ISBN PATTERN (for extraction, not removal) ---
        self.regex_isbn = re.compile(r'ISBN[-:\s]*([\d\-Xx]+)')
        
        # --- ISOLATED ALL-CAPS TITLES ---
        self.regex_all_caps_title = re.compile(r'^\s*[A-Z][A-Z\s]{2,48}[A-Z]\s*$')
        
        # --- PUNCTUATION NOISE ---
        # Lines that are mostly bullets, dashes, dots, etc.
        self.regex_punctuation_line = re.compile(r'^[\s\-•·●○◦▪▫★☆※†‡§¶\.\*\#\=\~\|\+\_]+$')
        
        # --- FIGURE/TABLE CAPTIONS (often garbled) ---
        self.regex_figure_table = re.compile(
            r'^\s*(fig\.?|figure|table|chart|diagram|illustration|exhibit)\s*[\d\.]+\s*[:.]?\s*$',
            re.IGNORECASE
        )
        
        # --- OCR ERROR PATTERNS ---
        # Common OCR misreads
        self.ocr_corrections = [
            (re.compile(r'\bl\b'), 'I'),  # Isolated 'l' often should be 'I'
            (re.compile(r'(?<=[a-z])0(?=[a-z])'), 'o'),  # '0' between letters → 'o'
            (re.compile(r'(?<=[A-Z])0(?=[A-Z])'), 'O'),  # '0' between caps → 'O'
            (re.compile(r'\brn\b'), 'm'),  # 'rn' misread as 'm'
            (re.compile(r'(?<=\w)l(?=\d)'), '1'),  # 'l' before digit → '1'
            (re.compile(r'(?<=\d)l(?=\w)'), '1'),  # 'l' after digit → '1'
        ]
    
    def normalize(self, text, is_ocr=False):
        """
        Stage 1: Normalize text encoding and formatting.
        """
        if not text:
            return ""
        
        # Fix Unicode mojibake (â€œ → ")
        text = fix_text(text)
        
        # Remove control characters
        text = self.regex_control_chars.sub('', text)
        
        # Fix hyphenation across lines
        text = self.regex_hyphenation.sub(r'\1\2', text)
        
        # Normalize whitespace
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            # Collapse multiple spaces/tabs to single space
            line = self.regex_multi_space.sub(' ', line)
            normalized_lines.append(line.strip())
        
        text = '\n'.join(normalized_lines)
        
        # Collapse multiple blank lines to max 2
        text = self.regex_multi_newline.sub('\n\n', text)
        
        # Apply OCR-specific corrections if needed
        if is_ocr:
            text = self._apply_ocr_corrections(text)
        
        return text
    
    def _apply_ocr_corrections(self, text):
        """Apply common OCR error corrections."""
        # Be conservative - only apply in clear contexts
        # Most corrections are risky, so we keep minimal
        return text
    
    def remove_noise(self, text):
        """
        Stage 2: Remove PDF noise using regex patterns.
        """
        if not text:
            return ""
        
        # Remove CID encoding errors
        text = self.regex_cid.sub('', text)
        
        # Process line by line
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines (will be handled in filter stage)
            if not stripped:
                cleaned_lines.append('')
                continue
            
            # Check against all noise patterns
            if self._is_noise_line(stripped):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _is_noise_line(self, line):
        """Check if a line matches any noise pattern."""
        # Page numbers
        if self.regex_page_num.match(line):
            return True
        
        # Page markers ("Page 42", "- 42 -")
        if self.regex_page_markers.match(line):
            return True
        
        # Structural headers
        if self.regex_headers.match(line):
            return True
        
        # Numbered structures (Chapter 1, Part II)
        if self.regex_numbered.match(line):
            return True
        
        # TOC entries
        if self.regex_toc_entry.match(line):
            return True
        
        # TOC chapter lines (Chapter 1 Title, etc.)
        if self.regex_toc_chapter.match(line):
            return True
        
        # TOC headers
        if self.regex_toc_headers.match(line):
            return True
        
        # Watermarks
        if self.regex_watermarks.match(line):
            return True
        
        # Boilerplate (but preserve ISBN if configured)
        if self.regex_boilerplate.match(line):
            if self.preserve_isbn and self.regex_isbn.search(line):
                return False  # Keep lines with ISBN
            return True
        
        # All-caps isolated titles
        if self.regex_all_caps_title.match(line):
            return True
        
        # Punctuation-only lines
        if self.regex_punctuation_line.match(line):
            return True
        
        # Figure/table captions (often garbled without context)
        if self.regex_figure_table.match(line):
            return True
        
        return False
    
    def deduplicate(self, text, threshold=3):
        """
        Stage 3: Remove repeating lines (headers/footers).
        Lines appearing more than `threshold` times are considered repeated headers/footers.
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        
        # Count line occurrences (normalized for comparison)
        line_counts = {}
        for line in lines:
            normalized = line.strip().lower()
            if normalized and len(normalized) > 3:  # Ignore very short lines
                line_counts[normalized] = line_counts.get(normalized, 0) + 1
        
        # Identify repeated lines (likely headers/footers)
        repeated = {line for line, count in line_counts.items() if count > threshold}
        
        # Filter out repeated lines
        cleaned_lines = []
        for line in lines:
            normalized = line.strip().lower()
            if normalized not in repeated:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def filter_short(self, text, min_length=None):
        """
        Stage 4: Remove short/garbled lines.
        """
        if not text:
            return ""
        
        min_len = min_length or self.min_line_length
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Keep empty lines for paragraph structure (up to 1 consecutive)
            if not stripped:
                # Only keep if previous line wasn't empty
                if filtered_lines and filtered_lines[-1].strip():
                    filtered_lines.append('')
                continue
            
            # Filter by length
            if len(stripped) < min_len:
                continue
            
            # Filter by character content (too few alphanumeric)
            alnum_count = sum(1 for c in stripped if c.isalnum())
            if len(stripped) > 0 and alnum_count / len(stripped) < 0.3:
                continue
            
            filtered_lines.append(line)
        
        # Remove trailing empty lines
        while filtered_lines and not filtered_lines[-1].strip():
            filtered_lines.pop()
        
        return '\n'.join(filtered_lines)
    
    def clean(self, text, is_ocr=False, dedupe_threshold=3):
        """
        Run the full cleaning pipeline.
        
        Args:
            text: Raw extracted text
            is_ocr: Whether text came from OCR (enables OCR-specific cleaning)
            dedupe_threshold: Min occurrences to consider a line repeated
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Pipeline: Normalize → Regex Remove → Deduplicate → Filter Short
        text = self.normalize(text, is_ocr=is_ocr)
        text = self.remove_noise(text)
        text = self.deduplicate(text, threshold=dedupe_threshold)
        text = self.filter_short(text)
        
        return text
    
    def extract_isbn(self, text):
        """Extract ISBN from text if present."""
        match = self.regex_isbn.search(text)
        return match.group(1) if match else None

    def is_likely_prose(self, line):
        """
        Check if a line is likely prose/dialogue content (not noise).
        Used for soft filtering in clean output generation.
        
        Args:
            line: Text line to analyze
        Returns:
            True if line appears to be actual content
        """
        stripped = line.strip()
        
        # Empty lines are not prose
        if not stripped:
            return False
        
        # Too short to be meaningful prose
        if len(stripped) < 15:
            return False
        
        # Check against noise patterns (return False if noise)
        if self.regex_toc_entry.match(stripped):
            return False
        if self.regex_toc_chapter.match(stripped):
            return False
        if self.regex_numbered.match(stripped):
            return False
        if self.regex_headers.match(stripped):
            return False
        if self.regex_page_num.match(stripped):
            return False
        if self.regex_page_markers.match(stripped):
            return False
        if self.regex_watermarks.match(stripped):
            return False
        if self.regex_boilerplate.match(stripped):
            return False
        if self.regex_all_caps_title.match(stripped):
            return False
        if self.regex_punctuation_line.match(stripped):
            return False
        if self.regex_figure_table.match(stripped):
            return False
        if self.regex_toc_headers.match(stripped):
            return False
        
        # Prose characteristics check
        has_sentence_end = bool(re.search(r'[.!?;:,]', stripped))
        letter_count = sum(1 for c in stripped if c.isalpha())
        letter_ratio = letter_count / len(stripped) if stripped else 0
        has_lowercase = any(c.islower() for c in stripped)
        
        # Prose typically has >50% letters and contains lowercase
        if letter_ratio > 0.5 and has_lowercase:
            return True
        
        # Has sentence punctuation and reasonable length
        if has_sentence_end and len(stripped) > 30:
            return True
        
        return False


class BookIngesterStrict:
    def __init__(self, input_folder, output_folder, max_workers=4, skip_existing=True, generate_clean=True):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.skip_existing = skip_existing
        self.generate_clean = generate_clean  # Generate additional clean output file
        
        # Initialize text cleaner with the full pipeline
        self.cleaner = TextCleaner(min_line_length=5, preserve_isbn=True)
        
        # Setup logging
        self._setup_logging()
        
        # Statistics
        self.stats = {
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'total': 0
        }
    
    def _setup_logging(self):
        """Setup logging to file and console."""
        log_file = self.output_folder / f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Clear any existing handlers
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        
        # Console handler (less verbose)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f"Logging initialized. Log file: {log_file}")

    def clean_text(self, text, is_ocr=False):
        """
        Clean extracted text using the full pipeline.
        Pipeline: Normalize → Regex Remove → Deduplicate → Filter Short
        
        Args:
            text: Raw extracted text
            is_ocr: True if text came from OCR (enables OCR-specific cleaning)
        """
        return self.cleaner.clean(text, is_ocr=is_ocr)

    def preprocess_image_for_ocr(self, img):
        """
        Apply OpenCV preprocessing to improve OCR accuracy.
        Pipeline: Grayscale → Denoise → Adaptive Threshold
        
        Args:
            img: PIL Image
        Returns:
            Preprocessed PIL Image
        """
        if not CV2_AVAILABLE:
            return img  # Return original if OpenCV not installed
        
        # Convert PIL Image to numpy array
        img_array = np.array(img)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Adaptive thresholding for better text contrast
        threshold = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to PIL Image
        return Image.fromarray(threshold)

    def ocr_page(self, page):
        """
        Perform OCR on a single page and return the raw text.
        Uses OpenCV preprocessing if available.
        """
        # Convert page to image for OCR
        # Higher DPI (e.g., 300) gives better OCR results
        zoom = 300 / 72  # 300 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert pixmap to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Apply preprocessing if OpenCV is available
        img = self.preprocess_image_for_ocr(img)
        
        # Perform OCR on the image
        return pytesseract.image_to_string(img)

    def extract_metadata(self, doc, pdf_path):
        """
        Extract metadata from PDF properties.
        Returns a dict with title, author, subject, etc.
        """
        metadata = doc.metadata or {}
        
        # Clean and extract metadata
        extracted = {
            'filename': pdf_path.name,
            'title': metadata.get('title', '').strip() or None,
            'author': metadata.get('author', '').strip() or None,
            'subject': metadata.get('subject', '').strip() or None,
            'keywords': metadata.get('keywords', '').strip() or None,
            'creator': metadata.get('creator', '').strip() or None,
            'producer': metadata.get('producer', '').strip() or None,
            'creation_date': metadata.get('creationDate', '').strip() or None,
            'modification_date': metadata.get('modDate', '').strip() or None,
            'page_count': len(doc),
            'file_size_bytes': pdf_path.stat().st_size if pdf_path.exists() else None,
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        return extracted

    def format_metadata_section(self, metadata):
        """
        Format metadata as a text section for the output file.
        """
        lines = [
            "=" * 60,
            "[ METADATA_SECTION - PDF Properties ]",
            "=" * 60,
            ""
        ]
        
        for key, value in metadata.items():
            if value is not None:
                lines.append(f"{key}: {value}")
        
        lines.append("\n")
        return "\n".join(lines)

    def save_metadata_json(self, metadata, pdf_path):
        """
        Save metadata as a separate JSON file.
        """
        json_path = self.output_folder / f"{pdf_path.stem}_metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return json_path

    def extract_text_pymupdf(self, page):
        """
        Extract text from a page using PyMuPDF's built-in text extraction.
        Returns the raw text.
        """
        return page.get_text("text")

    def is_extraction_valid(self, text, min_chars=100, max_garbage_ratio=0.3):
        """
        Check if PyMuPDF extraction produced valid text.
        Returns False if:
        - Text is too short
        - Too many CID errors or garbage characters
        - Too few alphanumeric characters
        """
        if not text or len(text.strip()) < min_chars:
            return False
        
        # Check for CID errors (common in scanned PDFs)
        cid_count = len(re.findall(r'\(cid:\d+\)', text))
        if cid_count > 10:
            return False
        
        # Check ratio of alphanumeric to total characters
        alphanumeric = sum(1 for c in text if c.isalnum() or c.isspace())
        total = len(text)
        if total > 0 and (alphanumeric / total) < (1 - max_garbage_ratio):
            return False
        
        return True

    def is_toc_heavy(self, text, threshold=0.5):
        """
        Check if a text section is predominantly TOC content.
        Used to detect if front/back sections are TOC-heavy.
        
        Args:
            text: Text to analyze
            threshold: Ratio of TOC lines to consider section as TOC-heavy
        Returns:
            True if section appears to be mostly TOC
        """
        if not text or not text.strip():
            return False
        
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return False
        
        toc_line_count = 0
        for line in lines:
            # Check TOC patterns
            if self.cleaner.regex_toc_entry.match(line):
                toc_line_count += 1
            elif self.cleaner.regex_toc_chapter.match(line):
                toc_line_count += 1
            elif self.cleaner.regex_numbered.match(line):
                toc_line_count += 1
            elif self.cleaner.regex_toc_headers.match(line):
                toc_line_count += 1
        
        toc_ratio = toc_line_count / len(lines)
        return toc_ratio >= threshold

    def ocr_entire_book(self, doc, pdf_name):
        """
        Fallback: OCR scan the entire book when PyMuPDF extraction fails.
        """
        logger.info(f"   🔄 Falling back to full OCR scan for {pdf_name}...")
        all_text_parts = []
        
        all_text_parts.append("=" * 60)
        all_text_parts.append("[ FULL_OCR_SCAN - PyMuPDF extraction failed ]")
        all_text_parts.append("[ Entire book scanned via OCR ]")
        all_text_parts.append("=" * 60 + "\n")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            raw_text = self.ocr_page(page)
            clean_text = self.clean_text(raw_text, is_ocr=True)
            
            all_text_parts.append(f"--- PAGE {page_num + 1} (OCR) ---")
            all_text_parts.append(clean_text if clean_text.strip() else "[No text extracted]")
            all_text_parts.append("")
            
            # Progress indicator for large books
            if (page_num + 1) % 50 == 0:
                logger.info(f"   📄 OCR progress: {page_num + 1}/{len(doc)} pages...")
        
        return "\n".join(all_text_parts)

    def generate_clean_output(self, detailed_text):
        """
        Generate a clean output from the detailed extraction.
        Applies soft filtering on ALL sections (front, middle, back) to keep
        prose/dialogue while skipping obvious TOC/chapter titles and metadata.
        
        Args:
            detailed_text: The full detailed extraction text
        Returns:
            Clean text suitable for plagiarism detection reference
        """
        lines = detailed_text.split('\n')
        clean_lines = []
        in_metadata_section = False
        prev_was_empty = False
        
        for line in lines:
            stripped = line.strip()
            
            # Track metadata section (skip entirely)
            if 'METADATA_SECTION' in stripped:
                in_metadata_section = True
                continue
            if in_metadata_section:
                # End metadata section when we hit next section marker
                if stripped.startswith('[') and 'SECTION' in stripped:
                    in_metadata_section = False
                continue
            
            # Skip section markers and page headers
            if stripped.startswith('=' * 10):
                continue
            if stripped.startswith('[') and stripped.endswith(']'):
                continue
            if re.match(r'^---\s*PAGE\s*\d+', stripped):
                continue
            
            # Skip placeholder text
            if stripped == '[No text extracted]':
                continue
            
            # Skip obvious metadata lines
            if any(key in stripped for key in ['filename:', 'title:', 'author:', 'subject:', 
                                                 'keywords:', 'creator:', 'producer:', 'creation_date:',
                                                 'modification_date:', 'page_count:', 'file_size_bytes:',
                                                 'extraction_timestamp:']):
                continue
            
            # Handle empty lines (preserve paragraph breaks)
            if not stripped:
                if not prev_was_empty and clean_lines:
                    clean_lines.append('')
                    prev_was_empty = True
                continue
            
            # Apply soft filtering: keep lines that look like prose
            if self.cleaner.is_likely_prose(stripped):
                clean_lines.append(stripped)
                prev_was_empty = False
        
        # Remove leading/trailing empty lines
        while clean_lines and not clean_lines[0].strip():
            clean_lines.pop(0)
        while clean_lines and not clean_lines[-1].strip():
            clean_lines.pop()
        
        # Join and collapse excessive blank lines
        result = '\n'.join(clean_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()

    def process_pdf(self, pdf_path):
        """
        Process a single PDF file. Returns dict with status info.
        """
        pdf_path = Path(pdf_path)  # Ensure Path object
        result = {'file': pdf_path.name, 'status': 'unknown', 'pages': 0, 'method': None}
        
        # Check if already processed (skip_existing)
        output_path = self.output_folder / f"{pdf_path.stem}.txt"
        if self.skip_existing and output_path.exists():
            logger.info(f"⏭️  Skipping (already exists): {pdf_path.name}")
            result['status'] = 'skipped'
            return result
        
        logger.info(f"📖 Processing: {pdf_path.name}...")
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            result['pages'] = total_pages
            
            if total_pages == 0:
                logger.warning(f"⚠️ Warning: {pdf_path.name} has no pages.")
                result['status'] = 'empty'
                return result
            
            # Extract and save metadata
            metadata = self.extract_metadata(doc, pdf_path)
            self.save_metadata_json(metadata, pdf_path)
            
            # Determine which pages to OCR (first 3 and last 3)
            front_page_count = min(3, total_pages)
            front_pages = list(range(front_page_count))
            
            back_page_count = min(3, total_pages)
            back_pages = list(range(total_pages - back_page_count, total_pages))
            # Remove duplicates for small books
            back_pages = [p for p in back_pages if p not in front_pages]
            
            # Middle pages (between front and back sections)
            ocr_pages = set(front_pages + back_pages)
            middle_pages = [p for p in range(total_pages) if p not in ocr_pages]
            
            all_text_parts = []
            
            # --- METADATA SECTION ---
            all_text_parts.append(self.format_metadata_section(metadata))
            
            # --- OCR FRONT SECTION (First 3 pages) ---
            all_text_parts.append("=" * 60)
            all_text_parts.append("[ OCR_FRONT_SECTION - First 3 Pages ]")
            all_text_parts.append("[ Contains: Title, Author, Copyright, etc. ]")
            all_text_parts.append("=" * 60 + "\n")
            
            for page_num in front_pages:
                page = doc[page_num]
                raw_text = self.ocr_page(page)
                clean_text = self.clean_text(raw_text, is_ocr=True)
                
                all_text_parts.append(f"--- PAGE {page_num + 1} (OCR) ---")
                all_text_parts.append(clean_text if clean_text.strip() else "[No text extracted]")
                all_text_parts.append("")
            
            # --- MIDDLE SECTION (PyMuPDF extraction with per-page OCR fallback) ---
            if middle_pages:
                all_text_parts.append("\n" + "=" * 60)
                all_text_parts.append("[ CONTENT_SECTION - Main Body ]")
                all_text_parts.append("[ Extracted via PyMuPDF with per-page OCR fallback ]")
                all_text_parts.append("=" * 60 + "\n")
                
                ocr_fallback_count = 0
                
                for page_num in middle_pages:
                    page = doc[page_num]
                    try:
                        raw_text = self.extract_text_pymupdf(page)
                        
                        # Per-page validity check - OCR if extraction fails
                        if not self.is_extraction_valid(raw_text, min_chars=50):
                            # OCR fallback for this specific page
                            raw_text = self.ocr_page(page)
                            clean_text = self.clean_text(raw_text, is_ocr=True)
                            ocr_fallback_count += 1
                            all_text_parts.append(f"--- PAGE {page_num + 1} (OCR fallback) ---")
                        else:
                            clean_text = self.clean_text(raw_text, is_ocr=False)
                            all_text_parts.append(f"--- PAGE {page_num + 1} ---")
                        
                        all_text_parts.append(clean_text if clean_text.strip() else "[No text extracted]")
                        all_text_parts.append("")
                        
                    except Exception as e:
                        # Emergency OCR for failed page
                        try:
                            raw_text = self.ocr_page(page)
                            clean_text = self.clean_text(raw_text, is_ocr=True)
                            ocr_fallback_count += 1
                            all_text_parts.append(f"--- PAGE {page_num + 1} (OCR emergency) ---")
                            all_text_parts.append(clean_text if clean_text.strip() else "[No text extracted]")
                        except:
                            all_text_parts.append(f"--- PAGE {page_num + 1} ---")
                            all_text_parts.append(f"[Extraction error: {e}]")
                        all_text_parts.append("")
                
                if ocr_fallback_count > 0:
                    logger.info(f"   🔄 Used OCR fallback for {ocr_fallback_count}/{len(middle_pages)} middle pages")
            
            # --- OCR BACK SECTION (Last 3 pages) ---
            if back_pages:
                all_text_parts.append("\n" + "=" * 60)
                all_text_parts.append("[ OCR_BACK_SECTION - Last 3 Pages ]")
                all_text_parts.append("[ Contains: Index, About Author, Bibliography, etc. ]")
                all_text_parts.append("=" * 60 + "\n")
                
                for page_num in back_pages:
                    page = doc[page_num]
                    raw_text = self.ocr_page(page)
                    clean_text = self.clean_text(raw_text, is_ocr=True)
                    
                    all_text_parts.append(f"--- PAGE {page_num + 1} (OCR) ---")
                    all_text_parts.append(clean_text if clean_text.strip() else "[No text extracted]")
                    all_text_parts.append("")
            
            # --- FINALIZE OUTPUT ---
            final_text = "\n".join(all_text_parts)

            # Defensive Check: Did we get anything?
            if not final_text.strip():
                logger.warning(f"⚠️ Warning: {pdf_path.name} - No text extracted.")
                result['status'] = 'empty'
                return result

            # Save to .txt
            output_filename = pdf_path.stem + ".txt"
            output_path = self.output_folder / output_filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            
            logger.info(f"✅ Saved detailed text (OCR: {len(front_pages)+len(back_pages)} pages, PyMuPDF: {len(middle_pages)} pages) to: {output_filename}")
            result['method'] = 'hybrid'
            
            # Generate clean output if enabled
            if self.generate_clean:
                clean_output = self.generate_clean_output(final_text)
                clean_filename = pdf_path.stem + "_clean.txt"
                clean_path = self.output_folder / clean_filename
                with open(clean_path, "w", encoding="utf-8") as f:
                    f.write(clean_output)
                logger.info(f"✅ Saved clean text to: {clean_filename}")
            
            result['status'] = 'success'
            return result

        except Exception as e:
            logger.error(f"❌ ERROR processing {pdf_path.name}: {e}")
            # Attempt full OCR as last resort
            try:
                logger.info(f"   🔄 Attempting emergency full OCR scan...")
                doc = fitz.open(pdf_path)
                final_text = self.ocr_entire_book(doc, pdf_path.name)
                
                if final_text.strip():
                    output_filename = pdf_path.stem + ".txt"
                    output_path = self.output_folder / output_filename
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(final_text)
                    logger.info(f"✅ Emergency OCR saved to: {output_filename}")
                    result['status'] = 'success'
                    result['method'] = 'emergency_ocr'
                    return result
                else:
                    logger.error(f"❌ Emergency OCR also failed for {pdf_path.name}")
                    result['status'] = 'failed'
                    return result
            except Exception as ocr_e:
                logger.error(f"❌ Emergency OCR failed: {ocr_e}")
                result['status'] = 'failed'
                return result

    def run(self):
        """Run the ingestion pipeline with parallel processing."""
        pdf_files = list(self.input_folder.glob("*.pdf"))
        self.stats['total'] = len(pdf_files)
        
        logger.info(f"="*60)
        logger.info(f"PDF INGESTION STARTED")
        logger.info(f"Input folder: {self.input_folder}")
        logger.info(f"Output folder: {self.output_folder}")
        logger.info(f"Found {len(pdf_files)} PDFs")
        logger.info(f"Max workers: {self.max_workers}")
        logger.info(f"Skip existing: {self.skip_existing}")
        logger.info(f"Generate clean output: {self.generate_clean}")
        logger.info(f"="*60)
        
        if not pdf_files:
            logger.error("❌ No PDFs found! Check your input folder path.")
            return

        # Parallel processing with progress bar
        if self.max_workers > 1 and len(pdf_files) > 1:
            logger.info(f"🚀 Starting parallel processing with {self.max_workers} workers...")
            self._run_parallel(pdf_files)
        else:
            logger.info("🔄 Starting sequential processing...")
            self._run_sequential(pdf_files)
        
        # Print final summary
        self._print_summary()
    
    def _run_sequential(self, pdf_files):
        """Process PDFs one by one with progress bar."""
        if TQDM_AVAILABLE:
            iterator = tqdm(pdf_files, desc="Processing PDFs", unit="file")
        else:
            iterator = pdf_files
        
        for pdf in iterator:
            result = self.process_pdf(pdf)
            self._update_stats(result)
    
    def _run_parallel(self, pdf_files):
        """Process PDFs in parallel using ThreadPoolExecutor."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {executor.submit(self.process_pdf, pdf): pdf for pdf in pdf_files}
            
            # Process completed tasks with progress bar
            if TQDM_AVAILABLE:
                iterator = tqdm(as_completed(future_to_pdf), total=len(pdf_files), desc="Processing PDFs", unit="file")
            else:
                iterator = as_completed(future_to_pdf)
            
            for future in iterator:
                pdf = future_to_pdf[future]
                try:
                    result = future.result()
                    results.append(result)
                    self._update_stats(result)
                except Exception as e:
                    logger.error(f"❌ Unexpected error with {pdf.name}: {e}")
                    self.stats['failed'] += 1
        
        return results
    
    def _update_stats(self, result):
        """Update processing statistics based on result."""
        if result is None:
            self.stats['failed'] += 1
        elif result.get('status') == 'success':
            self.stats['processed'] += 1
        elif result.get('status') == 'skipped':
            self.stats['skipped'] += 1
        else:
            self.stats['failed'] += 1
    
    def _print_summary(self):
        """Print final processing summary."""
        logger.info(f"\n" + "="*60)
        logger.info(f"INGESTION COMPLETE")
        logger.info(f"="*60)
        logger.info(f"Total PDFs found:    {self.stats['total']}")
        logger.info(f"Successfully processed: {self.stats['processed']}")
        logger.info(f"Skipped (existing):     {self.stats['skipped']}")
        logger.info(f"Failed:                 {self.stats['failed']}")
        logger.info(f"="*60)

# --- EXECUTION ---
if __name__ == "__main__":
    ingester = BookIngesterStrict(
        input_folder="SCI_FI", 
        output_folder="Excel_Dataset",
        max_workers=4,
        skip_existing=False,
        generate_clean=True
    )
    ingester.run()  