# CopyDetection-System

A professional plagiarism and copyright infringement detection system using advanced NLP and semantic analysis. This system detects copied content, paraphrased material, and provides detailed analysis reports with risk scoring.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Documentation](#documentation)
- [Configuration](#configuration)

## 🚀 Features

- **Multi-Phase Plagiarism Detection**: Combines FAISS vector search, n-gram analysis, semantic embeddings, and stylometric features
- **Fast Database Search**: Uses SQLite + FAISS for instant semantic similarity matching across millions of words
- **Risk Classification**: SAFE, MODERATE, HIGH, and CRITICAL risk levels
- **Detailed Reporting**: Comprehensive analysis including:
  - N-gram frequency analysis (bigrams, trigrams, 4-grams)
  - Paraphrase detection with multi-signal analysis
  - Per-sentence risk heatmaps
  - Vocabulary statistics
  - Repeated phrase highlighting
- **REST API**: Flask-based API for integration with other systems
- **Web Interface**: HTML/CSS interface for manual analysis and report viewing
- **Reference Dataset**: 33 reference books with metadata for comparison
- **PDF Support**: Extract and analyze text from PDF documents
- **Development Tools**: Database viewer, CLI statistics tool, advanced PDF extraction

## 📁 Project Structure

```
CopyDetection-System/
├── src/                          # Production code
│   ├── __init__.py
│   ├── config.py                 # Configuration constants
│   ├── main.py                   # Flask app factory
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py             # REST endpoints
│   ├── plagiarism_detection/
│   │   ├── __init__.py
│   │   ├── detector.py           # Legacy Excel-based detector (deprecated)
│   │   └── db_detector.py        # Database detector (SQLite + FAISS) [ACTIVE]
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── generator.py          # Detailed report generation
│   ├── common/
│   │   ├── __init__.py
│   │   ├── text_processor.py     # Text preprocessing & tokenization
│   │   ├── metrics.py            # Similarity & stylometric calculations
│   │   └── pdf_processor.py      # PDF text extraction
│   └── ui/
│       ├── __init__.py
│       └── templates/
│           ├── index.html        # Main analyzer interface
│           └── report_viewer.html # Detailed report viewer
│
├── tools/                        # Development utilities (optional)
│   ├── __init__.py
│   ├── data_store.py            # SQLite + FAISS vector database
│   ├── db_viewer.py             # Streamlit database explorer
│   ├── db_info.py               # CLI database statistics
│   └── pdf.py                   # Advanced PDF extraction with OCR
│
├── test_data/                   # Source data for database population
│   ├── Excel_Dataset/           # 33 reference books (source files)
│   │   └── processed_books_dataset-1.xlsx
│   └── SCI_FI/                  # 33 sample PDF books [OPTIONAL]
│
├── tests/                       # Unit tests (future)
│
├── database/                    # PRIMARY STORAGE: SQLite + FAISS [REQUIRED]
│   ├── documents.db             # Document metadata and text (SQLite)
│   ├── faiss_index.bin          # Vector embeddings (FAISS)
│   └── faiss_ids.pkl            # Chunk ID mappings
│
├── docs/                        # Project documentation
│   ├── ARCHITECTURE.md          # Technical design document
│   ├── SETUP.md                 # Installation & troubleshooting
│   └── API.md                   # REST API documentation
│
├── start.py                     # Application entry point
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🏃 Quick Start (Run locally)

These steps get the app running locally with a populated database suitable for development and testing.

Prerequisites
- Python 3.8+ (3.9/3.10 recommended)
- pip or conda
- Tesseract OCR installed and on `PATH` (required for `pytesseract` when extracting text from scanned PDFs)
- (Windows/FAISS) If `faiss-cpu` fails to install via pip, install `faiss` via Conda: `conda install -c pytorch faiss-cpu -y`

1) Create and activate a virtual environment, then install Python dependencies

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# Additional packages used by the tools/ingest pipeline
pip install pandas sentence-transformers python-dotenv tqdm faiss-cpu
# If faiss-cpu pip package fails on Windows, use conda (recommended):
# conda install -c pytorch faiss-cpu -y
```

2) (Optional) Create a `.env` at the repository root to override ingestion settings (example):

```text
DATA_FOLDER=test_data/Excel_Dataset
DB_FOLDER=database
SQLITE_DB_NAME=documents.db
FAISS_INDEX_NAME=faiss_index.bin
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
SKIP_EXISTING=true
```

3) Populate the hybrid database (SQLite + FAISS)

```bash
cd tools
python data_store.py
```

This will scan `DATA_FOLDER` (defaults to `test_data/Excel_Dataset`) for `*_clean.txt` and matching `*_metadata.json` files, insert documents into `database/documents.db`, and write the FAISS index files into the `database/` folder.

4) Run the application

Development mode (auto-reload, debug):
```bash
cd ..
python start.py
```

Production mode (no debug):
```bash
python start.py --production
```

Default host/port: `0.0.0.0:5001` (web UI at http://localhost:5001)

5) Quick smoke tests

- Health check:
```bash
curl http://localhost:5001/health
```
- Analyze text (JSON):
```bash
curl -X POST http://localhost:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{"essay_text":"Short sample essay text that meets MIN_TEXT_LENGTH"}'
```
- Analyze file upload (text or PDF):
```bash
curl -X POST http://localhost:5001/analyze -F "file=@/path/to/sample.txt"
```

Troubleshooting & notes
- Model downloads: `sentence-transformers` will download `all-MiniLM-L6-v2` on first use (internet required).
- Tesseract OCR: Ensure `tesseract` binary is installed and accessible if OCR is needed.
- FAISS: If you encounter install issues on Windows, prefer Conda as shown above.
- Ingestion may be CPU/memory intensive—reduce `CHUNK_SIZE` or process smaller batches if necessary.

See the "Development" and "Tools" sections below for additional commands (database viewer, stats, etc.).

## 🔌 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Web interface |
| GET | `/health` | Health check |
| POST | `/analyze` | Analyze text for plagiarism |
| POST | `/report` | Generate detailed analysis report |
| GET | `/report-viewer` | View generated reports |

See [API.md](docs/API.md) for detailed endpoint documentation.

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical design covering:
  - Detection pipeline (3-phase analysis)
  - API structure and routing
  - Configuration system
  - Database schema

- **[SETUP.md](docs/SETUP.md)** - Installation guide with:
  - Dependency management
  - Troubleshooting common issues
  - Testing procedures
  - Using development tools

- **[API.md](docs/API.md)** - Complete REST API reference:
  - Request/response schemas
  - Error codes
  - Usage examples
  - Rate limiting (if applicable)

## ⚙️ Configuration

Core settings in [src/config.py](src/config.py):

```python
# Database configuration
DATABASE_FOLDER = "database"  # SQLite + FAISS storage

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model
```

## 🛠️ Development

### Running Development Tools

**Database Viewer (Streamlit):**
```bash
cd tools
streamlit run db_viewer.py
```

**Database Statistics:**
```bash
cd tools
python db_info.py
```

### Running Tests

```bash
pytest tests/
```

## 📊 Detection Pipeline

The system uses a 3-phase approach with database acceleration:

1. **FAISS Semantic Search** - Fast vector similarity search across 66K+ text chunks using pre-computed embeddings
2. **Document Grouping** - Group matching chunks by source document
3. **Detailed Analysis** - N-gram, semantic, and stylometric scoring on matched documents
4. **Risk Scoring** - Weighted combination with classification

Risk levels:
- **SAFE** (0-30%) - No plagiarism detected
- **MODERATE** (30-50%) - Minor similar content
- **HIGH** (50-75%) - Significant plagiarism
- **CRITICAL** (75-100%) - Severe plagiarism

## 🔍 Report Features

Generated reports include:

- **Summary** - Risk level and overall statistics
- **N-gram Analysis** - Most common bigrams, trigrams, 4-grams
- **Repeated Phrases** - Multi-word segment detection
- **Paraphrase Detection** - 4-signal analysis for reworded content
- **Heatmap** - Per-sentence risk visualization
- **Vocabulary Statistics** - Readability metrics and word frequency
- **Sources** - Matching reference documents

## 📦 Dependencies

Key packages (see requirements.txt for complete list):

- **faiss-cpu** (or faiss-gpu) - Fast vector similarity search
- **sentence-transformers** - Semantic embeddings  
- **flask** - Web framework
- **sqlite3** - Document metadata storage (built-in)
- **nltk** - NLP utilities
- **pymupdf** - PDF text extraction
- **numpy** - Numerical computing

## 🐛 Troubleshooting

**Common Issues:**

- **ModuleNotFoundError**: Ensure dependencies are installed with `pip install -r requirements.txt`
- **Reference data not found**: Verify `test_data/Excel_Dataset/processed_books_dataset-1.xlsx` exists
- **Port 5001 already in use**: Change `PORT` in `src/config.py` or stop conflicting process
- **PDF extraction fails**: Some PDFs may require OCR; use `tools/pdf.py` for advanced extraction

See [SETUP.md](docs/SETUP.md) for more help.

## 📝 License

[Add your license here]

## 👥 Contributors

[Add contributor information]

## 📧 Contact

[Add contact information]

---

**Last Updated**: December 2024  
**Project Status**: Production-ready with development tools
