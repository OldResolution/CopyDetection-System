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

- **Multi-Phase Plagiarism Detection**: Combines Jaccard similarity, n-gram analysis, semantic embeddings, and stylometric features
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
│   │   └── detector.py           # Core detection engine (3-phase analysis)
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
├── test_data/                   # Test and reference data
│   ├── Excel_Dataset/           # 33 reference books [REQUIRED]
│   │   └── processed_books_dataset-1.xlsx
│   └── SCI_FI/                  # 33 sample PDF books [OPTIONAL]
│
├── tests/                       # Unit tests (future)
│
├── database/                    # Runtime generated data (git-ignored)
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

## 🏃 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone and navigate to project**
   ```bash
   cd CopyDetection-System
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure test data exists**
   - The system requires reference books in `test_data/Excel_Dataset/processed_books_dataset-1.xlsx`
   - Optional: PDF samples in `test_data/SCI_FI/` for testing

4. **Start the application**
   ```bash
   python start.py
   ```
   - Web interface: http://localhost:5001
   - Health check: http://localhost:5001/health

### Usage Examples

**Via Web Interface:**
1. Navigate to http://localhost:5001
2. Enter text to analyze or upload a PDF
3. Click "Analyze for Plagiarism"
4. Review results and click "Generate Detailed Report"

**Via REST API:**
```bash
# Analyze text
curl -X POST http://localhost:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'

# Generate report
curl -X POST http://localhost:5001/report \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

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
# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model

# Text processing constraints
MIN_TEXT_LENGTH = 50              # Minimum text length for analysis
MAX_TEXT_LENGTH = 100000          # Maximum text length
MAX_PDF_PAGES = 25                # Maximum PDF pages to process

# Detection thresholds
QUICK_FILTER_THRESHOLD = 0.4      # Jaccard threshold for quick filter
DETAILED_MATCH_THRESHOLD = 0.3    # Threshold for detailed analysis

# Reference data
DEFAULT_REFERENCE_PATH = "test_data/Excel_Dataset/processed_books_dataset-1.xlsx"
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

The system uses a 3-phase approach:

1. **Quick Filter** - Jaccard + n-gram similarity to reduce candidates
2. **Detailed Analysis** - Semantic, n-gram, and stylometric scoring
3. **Risk Scoring** - Weighted combination with classification

Risk levels:
- **SAFE** (0-20%) - No plagiarism detected
- **MODERATE** (20-40%) - Minor similar content
- **HIGH** (40-70%) - Significant plagiarism
- **CRITICAL** (70-100%) - Severe plagiarism

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

- **sentence-transformers** - Semantic embeddings
- **flask** - Web framework
- **pandas** - Data manipulation
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
