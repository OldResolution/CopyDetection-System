# Test Data & Reference Materials

This directory contains essential reference data and test materials for the CopyDetection-System:

## Directories

### `Excel_Dataset/`
**Status:** Required for production  
**Contents:** Reference books dataset  
- `processed_books_dataset-1.xlsx` - Main reference dataset containing text content and metadata for 33 books
- Text files and metadata JSON for individual books
- Pre-processed and cleaned text versions

**Purpose:** The Flask app loads this dataset at startup to enable plagiarism detection. Without this data, the system cannot function.

### `SCI_FI/`
**Status:** Optional (for testing/evaluation)  
**Contents:** Sample PDF files for testing  
- 33 science fiction PDF books (scf(1).pdf through scf(33).pdf)
- Titles include Dune, 1984, Foundation, etc.
- Used for validating PDF text extraction and detection accuracy

**Purpose:** For developers testing the PDF upload feature and verifying detection accuracy against known books.

## Usage

1. **Production:** Ensure `Excel_Dataset/` is present. The app will fail to load without `processed_books_dataset-1.xlsx`
2. **Testing:** Use SCI_FI PDFs to test the `/analyze` endpoint with file uploads
3. **Development:** Modify or expand these datasets as needed for testing new features

## Notes

- The `Excel_Dataset/` contains the core reference data that drives plagiarism detection
- Both directories can be backed up and restored independently
- For large-scale deployments, consider optimizing dataset storage or migrating to database solutions
