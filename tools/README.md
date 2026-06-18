# Development Tools & Admin Utilities

This directory contains development and administrative tools for managing the CopyDetection-System database and testing advanced features.

## Files

### `data_store.py`
**Purpose:** Thin CLI wrapper around `src.storage.hybrid_store.HybridDataStore`  
**Use Cases:**
- Building and maintaining vector embeddings for semantic search
- Ingesting new documents into the database
- Chunking and preprocessing text data
- Syncing between SQL and vector databases

**Usage:**
```bash
python -m tools.data_store
```

**Configuration:** Uses `.env` file in this directory

### `db_viewer.py` (310 lines)
**Purpose:** Interactive Streamlit web UI for exploring the database  
**Features:**
- View indexed documents and their statistics
- Browse text chunks
- Inspect FAISS vector index
- Perform semantic searches
- Monitor database sync status

**Usage:**
```bash
streamlit run tools/db_viewer.py
```

**Access:** Opens at `http://localhost:8501`

### `db_info.py` (176 lines)
**Purpose:** CLI tool for displaying database statistics and health information  
**Features:**
- Show SQLite database size and contents
- List all indexed documents
- Display FAISS vector index status
- Check sync alignment between SQLite and FAISS

**Usage:**
```bash
python -m tools.db_info
```

### `pdf.py` (1044 lines)
**Purpose:** Advanced PDF text extraction with OCR fallback support  
**Features:**
- Comprehensive text cleaning (removes headers, footers, page numbers)
- OCR support via pytesseract for scanned PDFs
- Concurrent multi-page processing
- Detailed extraction statistics
- Image preprocessing with OpenCV

**Note:** Currently not integrated into the main app. The simpler `src/common/pdf_processor.py` is used for production.

**Usage:** Can be adapted to replace standard PDF processor if OCR is needed

### `.env`
Configuration file for database utilities (not tracked in git)

### Database location
The canonical database is `../database/` at the repository root:
- `documents.db` - SQLite database
- `faiss_index.bin` - FAISS vector index
- `faiss_ids.pkl` - Vector ID mapping

`tools/database/` is ignored and should not be used for runtime data.

## Setup

1. **Install optional dependencies:**
   ```bash
   pip install streamlit python-dotenv faiss-cpu sentence-transformers
   ```

2. **Configure `.env`:**
   ```env
   DATA_FOLDER=../test_data/Excel_Dataset
   DB_FOLDER=../database
   SQLITE_DB_NAME=documents.db
   FAISS_INDEX_NAME=faiss_index.bin
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   CHUNK_SIZE=500
   CHUNK_OVERLAP=50
   ```

3. **Run database viewer:**
   ```bash
   cd tools
   streamlit run db_viewer.py
   ```

## Notes

- These tools are **not required** for the main Flask app to function
- They are useful for development, testing, and database management
- The main app uses only `src/` files for plagiarism detection
- For production deployments, these tools can be deployed separately or not at all
