# CopyDetection-System Documentation

## 📋 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python start.py
   ```

3. **Access the web UI:**
   - Open http://localhost:5001 in your browser

## 📁 Project Structure

```
CopyDetection-System/
├── src/                           # Production source code
│   ├── main.py                    # Flask app entry point
│   ├── config.py                  # Configuration constants
│   ├── api/routes.py              # Flask API endpoints
│   ├── plagiarism_detection/      # Detection engine
│   │   └── detector.py            # AdvancedPlagiarismDetector class
│   ├── reporting/                 # Report generation
│   │   └── generator.py           # Detailed analysis reports
│   ├── common/                    # Shared utilities
│   │   ├── text_processor.py      # Text preprocessing
│   │   ├── metrics.py             # Similarity metrics
│   │   └── pdf_processor.py       # PDF extraction
│   └── ui/templates/              # HTML templates
│       ├── index.html             # Main analyzer UI
│       └── report_viewer.html     # Detailed report viewer
│
├── tools/                         # Development & admin utilities
│   ├── data_store.py              # Hybrid SQLite + FAISS storage
│   ├── db_viewer.py               # Streamlit database explorer
│   ├── db_info.py                 # CLI database statistics
│   └── pdf.py                     # Advanced PDF extraction (optional)
│
├── test_data/                     # Reference data & test files
│   ├── Excel_Dataset/             # Core reference books dataset (REQUIRED)
│   └── SCI_FI/                    # Sample PDF test files
│
├── tests/                         # Unit & integration tests (future)
├── docs/                          # Documentation
├── start.py                       # Application entry point
├── requirements.txt               # Python dependencies
├── README.md                      # Project README
└── .gitignore                     # Git ignore rules
```

## 🔍 How It Works

### Detection Pipeline

The system uses a three-phase approach:

**Phase 1: Quick Filter**
- Jaccard similarity of token sets
- N-gram overlap scoring (bigrams)
- Narrows candidates to top 50 books

**Phase 2: Detailed Analysis**
For each candidate:
1. **Semantic Score** (60% weight)
   - SentenceTransformer embeddings
   - Compares essay to multiple text chunks
   - Chunks from different parts of reference book

2. **N-gram Score** (20% weight)
   - Trigram overlap coefficient
   - Detects copied phrases

3. **Stylometric Score** (20% weight)
   - 7-feature writing style vector:
     - Average word length
     - Average sentence length (log-scaled)
     - Type-Token ratio (vocabulary diversity)
     - Stopword ratio
     - Punctuation ratio
     - Unique word ratio
     - Sentence count (log-scaled)
   - Cosine similarity between essay and book

**Phase 3: Scoring & Risk Classification**
- Combined weighted score (0-1)
- Deduplication: filters duplicate book titles
- Risk levels:
  - **SAFE** (< 0.3): No significant match
  - **MODERATE** (0.3-0.5): Stylistic or semantic similarities
  - **HIGH** (0.5-0.75): Strong evidence of infringement
  - **CRITICAL** (≥ 0.75): High probability of copied content

### REST API Endpoints

#### `POST /analyze`
Submit text/PDF for plagiarism analysis

**Request:**
- File upload (`.pdf` or `.txt`) OR JSON with `essay_text` field

**Response:**
```json
{
  "combined_score": 0.75,
  "risk_level": "HIGH",
  "top_source": "Book Title by Author",
  "plagiarized_books": [
    {
      "book_title": "...",
      "book_author": "...",
      "combined_score": 0.75,
      "risk_level": "HIGH",
      "ngram_score": 0.6,
      "semantic_score": 0.8,
      "stylometric_score": 0.5
    }
  ],
  "feature_names": [...],
  "essay_features": [...],
  "extracted_text": "..."  // For PDF uploads
}
```

#### `POST /report`
Generate detailed analysis report

**Request:**
```json
{
  "text": "original essay text",
  "analysis": { ...analysis response from /analyze... }
}
```

**Response:** Detailed report with:
- N-gram frequency tables
- Highlighted repeated phrases
- Paraphrased segment detection
- Sentence-level risk heatmap
- Vocabulary & readability statistics

#### `GET /`
Serve main analyzer UI (index.html)

#### `GET /health`
Health check with loaded books count

## ⚙️ Configuration

Edit `src/config.py`:

```python
DEFAULT_REFERENCE_PATH = "test_data/Excel_Dataset/processed_books_dataset-1.xlsx"
MIN_TEXT_LENGTH = 50                      # Minimum characters required
MAX_PDF_PAGES = 25                        # Max pages to extract from PDF
MAX_TEXT_LENGTH = 100000                  # Max characters to analyze (100 KB)
MODEL_NAME = 'all-MiniLM-L6-v2'          # Sentence-Transformers model
MAX_ANALYZE_UPLOAD_SIZE = 16 * 1024 * 1024  # 16 MB file limit
```

## 📦 Dependencies

Key packages:
- `Flask` - Web framework
- `sentence-transformers` - Semantic embeddings
- `pandas` - Data handling
- `nltk` - Natural language processing
- `numpy` - Numerical computing
- `PyMuPDF` - PDF text extraction

See `requirements.txt` for full list.

## 🧪 Testing

```bash
# Run tests (future)
pytest tests/

# Test the API
curl -X POST http://localhost:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{"essay_text": "Your text here..."}'
```

## 🛠️ Development Tools

### Database Viewer (Streamlit)
```bash
cd tools
streamlit run db_viewer.py
```
Opens interactive dashboard at http://localhost:8501

### Database Info (CLI)
```bash
python tools/db_info.py
```
Displays SQLite and FAISS statistics

## 📝 API Response Walkthrough

### Analysis Response Structure
```json
{
  "feature_names": [
    "Avg Word Len",           // Average word length in characters
    "Avg Sent Len (Log)",     // Log-scaled average sentence length
    "Type-Token",             // TTR: unique words / total words
    "Stopwords",              // Ratio of stopwords (function words)
    "Punctuation",            // Ratio of punctuation marks
    "Unique Words",           // Ratio of unique words
    "Sent Count (Log)"        // Log-scaled number of sentences
  ],
  "essay_features": [4.2, 1.8, 0.6, 0.3, 0.05, 0.6, 2.1],
  "combined_score": 0.75,    // Final plagiarism likelihood (0-1)
  "risk_level": "HIGH",
  "top_source": "Book Title by Author Name",
  "plagiarized_books": [
    {
      "book_title": "...",
      "book_author": "...",
      "combined_score": 0.75,  // Weighted combination of three signals
      "risk_level": "HIGH",
      "ngram_score": 0.6,      // Phrase overlap (0-1)
      "semantic_score": 0.8,   // Meaning similarity (0-1)
      "stylometric_score": 0.5 // Writing style similarity (0-1)
    }
  ]
}
```

### Report Response Structure
Detailed report includes:
- **Summary**: Overall statistics, readability (Flesch score)
- **N-gram Frequency**: Bigrams, trigrams, 4-grams with densities
- **Repeated Phrases**: Color-coded by frequency tier
- **Paraphrased Segments**: Suspicious sentences with 4-part analysis
  - Vocabulary substitution score
  - Content divergence score
  - Structural patterns (passive voice, nominalization)
  - N-gram novelty vs. neighbors
- **Sentence Risk Heatmap**: Per-sentence risk scoring
- **Vocabulary Stats**: TTR, hapax legomena, readability, top 50 words
- **Matched Books**: Detailed source matching

See `templates/report_viewer.html` for full UI implementation.

## 🚀 Production Deployment

```bash
# Run in production mode
python start.py --production
```

For Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5001 "src.main:app"
```

## 📚 References

- **Sentence-Transformers**: https://www.sbert.net/
- **FAISS**: https://github.com/facebookresearch/faiss
- **NLTK**: https://www.nltk.org/
- **Flask**: https://flask.palletsprojects.com/

## 📄 License

See project documentation for license information.
