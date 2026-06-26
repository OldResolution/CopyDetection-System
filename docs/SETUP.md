# Installation & Setup Guide

## Prerequisites

- Python 3.11 recommended for the pinned ML dependencies
- pip or conda

## Installation Steps

### 1. Clone or Download

```bash
git clone <repository-url>
cd CopyDetection-System
```

### 2. Create Virtual Environment (Optional but recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n copydetection python=3.11
conda activate copydetection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import flask, sentence_transformers, faiss, nltk; print('All dependencies installed')"
```

## Running the Application

### Development Mode

```bash
python start.py
```

The app will:
- Load the canonical SQLite + FAISS database from `database/`
- Initialize the SentenceTransformer model
- Start the Flask server on http://localhost:5001

### First Run

On first run outside Docker, the app may download:
1. The embedding model cache
2. NLTK packages (tokenizer, stopwords)
3. The detector will then read the prebuilt SQLite + FAISS files from `database/`

### Access the Application

Open your browser and visit:
- **Main analyzer:** http://localhost:5001/
- **Health check:** http://localhost:5001/health

## Configuration

Edit `src/config.py` to customize:

```python
DATABASE_FOLDER = "database"
DATA_FOLDER = "test_data/Excel_Dataset"
MIN_TEXT_LENGTH = 50           # Minimum text required (characters)
MAX_PDF_PAGES = 25             # Maximum pages to extract
MAX_TEXT_LENGTH = 100000       # Maximum text to analyze (100 KB)
MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence-Transformers model
```

## Troubleshooting

### Issue: "Database is empty"
**Solution:** Rebuild the canonical store:
```bash
python -m tools.data_store
```

### Issue: NLTK data not found
**Solution:** The app auto-downloads. If it fails:
```bash
python -m nltk.downloader punkt stopwords punkt_tab
```

### Issue: CUDA/GPU errors
**Solution:** The app uses CPU by default. It's fine. If you want GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Port 5001 already in use
**Solution:** For local testing, edit the development server port in `src/main.py`.
For production-style runs, use Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5002 "src.main:app"
```

### Issue: "Model can't download"
**Solution:** Set cache directory:
```bash
export TRANSFORMERS_CACHE=/path/to/cache
python start.py
```

## Testing the API

### Test with curl

```bash
# Plain text analysis
curl -X POST http://localhost:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "essay_text": "The quick brown fox jumps over the lazy dog. This is a test."
  }'

# PDF file upload
curl -X POST http://localhost:5001/analyze \
  -F "file=@path/to/test.pdf"
```

### Test with Python

```python
import requests

# Plain text
response = requests.post(
    'http://localhost:5001/analyze',
    json={'essay_text': 'Your text here...'}
)
print(response.json())

# PDF file
with open('test.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5001/analyze', files=files)
    print(response.json())
```

## Database Tools (Optional)

### View Database with Streamlit

```bash
cd tools
streamlit run db_viewer.py
```

Opens at http://localhost:8501

### Get Database Statistics

```bash
python -m tools.db_info
```

## Project Structure Reference

```
CopyDetection-System/
├── src/                    Production code
│   ├── main.py            Flask app
│   ├── config.py          Settings
│   ├── api/routes.py      Endpoints
│   ├── plagiarism_detection/  Detection engine
│   ├── reporting/         Report generation
│   ├── common/            Utilities
│   └── ui/templates/      HTML templates
├── tools/                 Development tools
├── test_data/            Reference data
├── start.py              Entry point
└── requirements.txt      Dependencies
```

## Next Steps

1. Try the web UI at http://localhost:5001/
2. Upload a PDF or paste text to analyze
3. Review the detection results
4. Click "Generate Detailed Report" for deeper analysis

## Getting Help

- Check the logs for detailed error messages
- Review `docs/ARCHITECTURE.md` for technical details
- Run health check: `curl http://localhost:5001/health`

## Performance Notes

- First run loads all 33 books (~15-30 seconds)
- Analysis takes 5-15 seconds per submission depending on text length
- GPU would speed this up significantly
- For high-volume use, consider caching or batch processing
