# API Endpoints Documentation

## Base URL
```
http://localhost:5001
```

## Endpoints

### 1. Home Page / UI
```
GET /
```
Returns the main analyzer HTML interface.

**Response:** HTML page

---

### 2. Analyze Text/PDF
```
POST /analyze
```
Submit text or PDF file for plagiarism analysis.

#### Request Options

**Option A: Plain Text (JSON)**
```bash
curl -X POST http://localhost:5001/analyze \
  -H "Content-Type: application/json" \
 -d '{
  "essay_text": "Your text to analyze here..."
}'
```

**Option B: File Upload (Multipart)**
```bash
curl -X POST http://localhost:5001/analyze \
  -F "file=@document.pdf"
```

#### Constraints
- **Minimum text:** 50 characters
- **Maximum text:** 100,000 characters (100 KB)
- **Supported file types:** PDF, TXT
- **Maximum file size:** 16 MB

#### Response (200 OK)
```json
{
  "feature_names": [
    "Avg Word Len",
    "Avg Sent Len (Log)",
    "Type-Token",
    "Stopwords",
    "Punctuation",
    "Unique Words",
    "Sent Count (Log)"
  ],
  "essay_features": [4.2, 1.8, 0.6, 0.3, 0.05, 0.6, 2.1],
  "combined_score": 0.75,
  "risk_level": "HIGH",
  "top_source": "1984 by George Orwell",
  "plagiarized_books": [
    {
      "book_title": "1984",
      "book_author": "George Orwell",
      "combined_score": 0.75,
      "risk_level": "HIGH",
      "ngram_score": 0.6,
      "semantic_score": 0.8,
      "stylometric_score": 0.5
    }
  ],
  "extracted_text": "The submitted text content..."
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `combined_score` | float (0-1) | Overall plagiarism likelihood |
| `risk_level` | string | SAFE, MODERATE, HIGH, or CRITICAL |
| `top_source` | string | Highest-matching book and author |
| `plagiarized_books` | array | List of matches sorted by score |
| `feature_names` | array | Names of stylometric features analyzed |
| `essay_features` | array | Feature values for submitted text |
| `extracted_text` | string | Text extracted from PDF (if applicable) |

#### Risk Levels

| Level | Score | Meaning |
|-------|-------|---------|
| **SAFE** | < 0.3 | No significant match in database |
| **MODERATE** | 0.3-0.5 | Stylistic or semantic similarities |
| **HIGH** | 0.5-0.75 | Strong evidence of infringement |
| **CRITICAL** | ≥ 0.75 | High probability of plagiarism |

#### Scoring Components

Each matched book includes three similarity scores:

- **ngram_score** (0-1): Ratio of matching three-word phrases
- **semantic_score** (0-1): Meaning similarity via embeddings
- **stylometric_score** (0-1): Writing style similarity

#### Error Responses

**400 Bad Request:**
```json
{
  "error": "Text too short. Min 50 chars required."
}
```

**404 Not Found:**
```json
{
  "error": "No reference data available"
}
```

**500 Server Error:**
```json
{
  "error": "Internal server error message"
}
```

---

### 3. Generate Detailed Report
```
POST /report
```
Generate granular analysis report with phrase highlighting and sentence-level risk scoring.

#### Request
```bash
curl -X POST http://localhost:5001/report \
  -H "Content-Type: application/json" \
  -d '{
  "text": "Original submitted text...",
  "analysis": {
    "combined_score": 0.75,
    "risk_level": "HIGH",
    "plagiarized_books": [...],
    "feature_names": [...],
    "essay_features": [...]
  }
}'
```

#### Response Structure
```json
{
  "summary": {
    "combined_score": 0.75,
    "matched_sources": 3,
    "total_sentences": 25,
    "total_words": 350,
    "unique_words": 165,
    "flesch_ease": 60.5,
    "fk_grade": 8.2
  },
  "ngram_frequency": {
    "bigrams": {
      "total_unique": 284,
      "total_occurrences": 318,
      "top_phrases": [
        {"phrase": "the quick", "count": 5, "density": 1.57},
        ...
      ]
    },
    "trigrams": {...},
    "quadgrams": {...}
  },
  "repeated_phrases": [
    {
      "phrase": "the quick brown fox",
      "count": 3,
      "word_length": 4,
      "score": 12,
      "token_positions": [0, 45, 120]
    }
  ],
  "highlighted_text_html": "<p>The <mark class='rep-0'>the quick brown fox</mark>...</p>",
  "paraphrased_segments": [
    {
      "sentence_index": 5,
      "sentence_text": "A swift auburn animal leaps across...",
      "suspicion_score": 0.65,
      "risk_level": "High",
      "signals": {
        "vocab_substitution": 0.4,
        "content_divergence": 0.6,
        "structural": 0.3,
        "ngram_novelty": 0.75
      },
      "reasons": [
        "3 rare/substituted words",
        "structural rewrite patterns",
        "diverges from surrounding text"
      ],
      "rare_words": ["auburn", "leaps", "climbs"]
    }
  ],
  "vocabulary_stats": {
    "total_tokens": 350,
    "unique_words": 165,
    "content_words": 140,
    "stop_words": 25,
    "type_token_ratio": 0.47,
    "hapax_legomena_count": 82,
    "avg_word_length": 4.8,
    "avg_sentence_length": 14.0,
    "total_sentences": 25,
    "punctuation_count": 45,
    "flesch_reading_ease": 60.5,
    "flesch_kincaid_grade": 8.2,
    "top_50_words": [
      {"word": "quick", "count": 5, "pct": 1.43},
      ...
    ]
  },
  "sentence_risk_map": [
    {
      "index": 0,
      "text": "The quick brown fox...",
      "repeated_phrases_found": ["the quick brown"],
      "risk_score": 0.8
    }
  ],
  "matched_books": [...],
  "feature_names": [...],
  "essay_features": [...]
}
```

#### Report Sections

1. **Summary:** Overall statistics and readability scores
2. **N-gram Frequency:** Bigrams, trigrams, and 4-grams with counts
3. **Repeated Phrases:** Multi-word phrases found multiple times
4. **Highlighted Text:** HTML with color-coded phrase highlighting
5. **Paraphrased Segments:** Suspicious sentences with 4-part analysis:
   - Vocabulary substitution (rare words)
   - Content divergence (different vocabulary)
   - Structural patterns (passive voice, nominalization)
   - N-gram novelty (differs from neighbors)
6. **Vocabulary Statistics:** TTR, readability, word frequency
7. **Sentence Risk Map:** Per-sentence risk heatmap
8. **Matched Books:** Source matching details

---

### 4. Report Viewer Page
```
GET /report-viewer
```
Returns the interactive report viewer HTML page.

**Response:** HTML page

---

### 5. Health Check
```
GET /health
```
Check if the application is running and database is loaded.

#### Response (200 OK)
```json
{
  "status": "ok",
  "books_loaded": 33
}
```

#### Response (503 Service Unavailable)
```json
{
  "status": "error",
  "message": "No reference data available"
}
```

---

## Request/Response Examples

### Example 1: Simple Text Analysis

**Request:**
```bash
curl -X POST http://localhost:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{
  "essay_text": "The quick brown fox jumps over the lazy dog."
}'
```

**Response:**
```json
{
  "combined_score": 0.15,
  "risk_level": "SAFE",
  "top_source": "No significant database match found",
  "plagiarized_books": [],
  "feature_names": ["Avg Word Len", ...],
  "essay_features": [3.8, 1.0, 0.7, 0.33, 0.11, 0.7, 1.1],
  "extracted_text": null
}
```

### Example 2: PDF Upload

**Request:**
```bash
curl -X POST http://localhost:5001/analyze \
  -F "file=@novel.pdf"
```

**Response:** (Same as above, but `extracted_text` will contain PDF content)

### Example 3: High-Risk Match

**Request:**
```bash
curl -X POST http://localhost:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{
  "essay_text": "It was the best of times, it was the worst of times..."
}'
```

**Response:**
```json
{
  "combined_score": 0.92,
  "risk_level": "CRITICAL",
  "top_source": "A Tale of Two Cities by Charles Dickens",
  "plagiarized_books": [
    {
      "book_title": "A Tale of Two Cities",
      "book_author": "Charles Dickens",
      "combined_score": 0.92,
      "risk_level": "CRITICAL",
      "ngram_score": 0.95,
      "semantic_score": 0.98,
      "stylometric_score": 0.78
    }
  ],
  ...
}
```

---

## Rate Limiting

Currently **no rate limits** are enforced. In production:
- Consider limiting requests per IP
- Set timeout on long-running analysis
- Implement request queuing for high volume

## Error Handling

All endpoints return standard HTTP status codes:

- **200 OK:** Successful request
- **400 Bad Request:** Invalid input (too short, wrong format)
- **404 Not Found:** Endpoint doesn't exist
- **500 Internal Server Error:** Server error

---

## CORS Headers

The API supports CORS from all origins (for development):
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

For production, update `src/main.py` to restrict origins.
