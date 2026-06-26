# CopyDetection-System Architecture

## Runtime Shape

The application is a Flask service with a database-backed plagiarism detector.
The current runtime detector is `src/plagiarism_detection/db_detector.py`, not
the deprecated in-memory detector in `src/plagiarism_detection/detector.py`.

```
CopyDetection-System/
|-- src/
|   |-- main.py                         # Flask app factory and local dev server
|   |-- config.py                       # Runtime configuration and env defaults
|   |-- api/routes.py                   # HTTP endpoints
|   |-- plagiarism_detection/
|   |   |-- db_detector.py              # Active detector: SQLite + FAISS
|   |   `-- detector.py                 # Deprecated/reference detector
|   |-- storage/
|   |   `-- hybrid_store.py             # HybridDataStore runtime dependency
|   |-- common/
|   |   |-- metrics.py                  # N-gram and stylometric scoring
|   |   |-- pdf_processor.py            # PDF text extraction
|   |   `-- text_processor.py           # Tokenization and preprocessing
|   |-- reporting/generator.py          # Detailed report generation
|   `-- ui/
|       |-- templates/                  # HTML templates
|       `-- static/                     # Static assets
|-- database/                           # Canonical SQLite + FAISS store
|-- tools/                              # Admin utilities and rebuild commands
|-- tests/                              # Unit and API integration tests
|-- Dockerfile
|-- docker-compose.yml
`-- requirements.txt
```

## Canonical Database Location

`database/` at the repository root is the canonical runtime data store.
It contains:

- `documents.db`: SQLite document, metadata, and chunk records
- `faiss_index.bin`: FAISS vector index
- `faiss_ids.pkl`: FAISS index position to chunk ID mapping

`tools/database/` is intentionally ignored. It was a duplicate generated
location from earlier tooling and should not be used for runtime data.

## Storage Layer

`src/storage/hybrid_store.py` owns `HybridDataStore`.

The store combines:

- SQLite for document metadata, full cleaned text, and chunk text.
- FAISS for normalized sentence-transformer embeddings of chunks.
- A pickle ID map that links FAISS vector positions back to SQLite chunk IDs.

The rebuild command is kept as a thin admin wrapper:

```bash
python -m tools.data_store
```

Configuration comes from environment variables with repo-root defaults:

- `DATA_FOLDER`, default `test_data/Excel_Dataset`
- `DATABASE_FOLDER` or `DB_FOLDER`, default `database`
- `SQLITE_DB_NAME`, default `documents.db`
- `FAISS_INDEX_NAME`, default `faiss_index.bin`
- `FAISS_IDS_NAME`, default `faiss_ids.pkl`
- `EMBEDDING_MODEL`, default `all-MiniLM-L6-v2`
- `CHUNK_SIZE`, default `500`
- `CHUNK_OVERLAP`, default `50`
- `SKIP_EXISTING`, default `true`

## Active Detection Flow

`DatabasePlagiarismDetector.analyze_text()` performs four stages:

1. Extract features from the submitted essay.
   - Raw tokens and stopword-filtered tokens come from `preprocess_text`.
   - Stylometric features come from `extract_stylometric_features`.

2. Run semantic search against FAISS.
   - The first 5000 characters of the essay are embedded.
   - The top 100 matching chunks are retrieved from the vector index.

3. Group matches by source document and run detailed scoring.
   - Full document text is loaded from SQLite.
   - Semantic score is the best FAISS chunk similarity for that document.
   - N-gram score uses trigram overlap between essay tokens and book tokens.
   - Stylometric score uses cosine similarity over the seven-style-feature vector.

4. Weight and classify results.
   - Strong semantic matches emphasize semantic similarity.
   - Strong phrase overlap emphasizes n-gram similarity.
   - Weak semantic and weak n-gram matches are penalized.
   - Matches below `0.3` combined score are filtered out.

Risk levels:

- `SAFE`: below `0.3`
- `MODERATE`: `0.3` to below `0.5`
- `HIGH`: `0.5` to below `0.75`
- `CRITICAL`: `0.75` and above

## API Endpoints

- `GET /`: serves the analyzer UI.
- `GET /health`: initializes the detector and returns database counts.
- `POST /analyze`: accepts JSON `essay_text` or `.txt`/`.pdf` upload, then
  returns analysis plus `extracted_text`.
- `GET /report-viewer`: serves the report UI.
- `POST /report`: builds the detailed report from submitted text and the
  `/analyze` response.

The detector is lazy-loaded in `src/api/routes.py` so tests and lightweight
imports do not load FAISS or the embedding model until analysis is needed.

## Deployment

Local testing can use:

```bash
python start.py
```

Production-style execution should use Gunicorn:

```bash
python start.py --production
```

or directly:

```bash
gunicorn -w 4 -b 0.0.0.0:5001 "src.main:app"
```

The Docker image uses `python:3.11-slim`, installs pinned dependencies,
downloads NLTK data, and preloads the `all-MiniLM-L6-v2` model cache during
build so container cold-starts are not dependent on network downloads.

```bash
docker compose up --build
```

## Tests

The baseline test suite covers:

- Pure metric functions in `src/common/metrics.py`.
- API integration for `/analyze` and `/report` using a small fixture text and
  fake detector.

Run:

```bash
pytest
```
