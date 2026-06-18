"""
Hybrid Data Storage for Copy Detection System
- SQLite: Text content and metadata storage
- FAISS: Vector embeddings for semantic search
"""

import os
import json
import sqlite3
import hashlib
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Vector DB
import faiss

# Embeddings
from sentence_transformers import SentenceTransformer

# Optional progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# --- CONFIGURATION FROM .env ---
DATA_FOLDER = os.getenv('DATA_FOLDER', 'test_data/Excel_Dataset')
DB_FOLDER = os.getenv('DATABASE_FOLDER', os.getenv('DB_FOLDER', 'database'))
SQLITE_DB_NAME = os.getenv('SQLITE_DB_NAME', 'documents.db')
FAISS_INDEX_NAME = os.getenv('FAISS_INDEX_NAME', 'faiss_index.bin')
FAISS_IDS_NAME = os.getenv('FAISS_IDS_NAME', 'faiss_ids.pkl')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
SKIP_EXISTING = os.getenv('SKIP_EXISTING', 'true').lower() == 'true'


class HybridDataStore:
    """
    Hybrid storage system combining SQLite (text/metadata) with FAISS (vectors).
    
    SQLite stores:
        - Document metadata (title, author, page count, etc.)
        - Full cleaned text content
        - Processing timestamps
    
    FAISS stores:
        - Text chunk embeddings for semantic search
        - Chunk IDs mapped back to SQLite
    """
    
    def __init__(
        self,
        data_folder: str = "Excel_Dataset",
        db_folder: str = "database",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.data_folder = Path(data_folder)
        self.db_folder = Path(db_folder)
        self.db_folder.mkdir(parents=True, exist_ok=True)
        
        # SQLite database path
        self.sqlite_path = self.db_folder / SQLITE_DB_NAME
        
        # FAISS index path
        self.faiss_index_path = self.db_folder / FAISS_INDEX_NAME
        self.faiss_ids_path = self.db_folder / FAISS_IDS_NAME
        
        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize databases
        self._init_sqlite()
        self._init_faiss()
        
        print(f"HybridDataStore initialized:")
        print(f"  - SQLite: {self.sqlite_path}")
        print(f"  - FAISS: {self.faiss_index_path}")
        print(f"  - Embedding dim: {self.embedding_dim}")
        print(f"  - Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    
    def _init_sqlite(self):
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Documents table - stores metadata and full text
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                title TEXT,
                author TEXT,
                subject TEXT,
                keywords TEXT,
                creator TEXT,
                producer TEXT,
                creation_date TEXT,
                modification_date TEXT,
                page_count INTEGER,
                file_size_bytes INTEGER,
                extraction_timestamp TEXT,
                source_type TEXT DEFAULT 'book',
                license_status TEXT DEFAULT 'unknown',
                copyright_status TEXT DEFAULT 'unknown',
                clean_text TEXT,
                text_hash TEXT,
                char_count INTEGER,
                word_count INTEGER,
                chunk_count INTEGER,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Chunks table - stores individual text chunks for reference
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                start_char INTEGER,
                end_char INTEGER,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON documents(filename)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_author ON documents(author)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_type ON documents(source_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_copyright_status ON documents(copyright_status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunks(chunk_id)')

        self._ensure_document_columns(cursor)
        self._normalize_document_metadata(cursor)
        
        conn.commit()
        conn.close()

    def _ensure_document_columns(self, cursor) -> None:
        """Backfill newer document metadata columns for older SQLite files."""
        cursor.execute("PRAGMA table_info(documents)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        required_columns = {
            "source_type": "TEXT DEFAULT 'book'",
            "license_status": "TEXT DEFAULT 'unknown'",
            "copyright_status": "TEXT DEFAULT 'unknown'",
        }

        for column_name, definition in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(f"ALTER TABLE documents ADD COLUMN {column_name} {definition}")

    def _normalize_document_metadata(self, cursor) -> None:
        """Fill newly introduced metadata fields with stable defaults."""
        cursor.execute(
            """
            UPDATE documents
            SET
                source_type = COALESCE(NULLIF(TRIM(source_type), ''), 'book'),
                license_status = COALESCE(NULLIF(TRIM(license_status), ''), 'unknown'),
                copyright_status = COALESCE(NULLIF(TRIM(copyright_status), ''), 'unknown')
            """
        )
    
    def _init_faiss(self):
        """Initialize or load FAISS index."""
        if self.faiss_index_path.exists() and self.faiss_ids_path.exists():
            # Load existing index
            self.faiss_index = faiss.read_index(str(self.faiss_index_path))
            with open(self.faiss_ids_path, 'rb') as f:
                self.chunk_id_map = pickle.load(f)
            print(f"  Loaded existing FAISS index with {self.faiss_index.ntotal} vectors")
        else:
            # Create new index (Inner Product for cosine similarity with normalized vectors)
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.chunk_id_map = []  # Maps FAISS index position to chunk_id
    
    def _save_faiss(self):
        """Save FAISS index and ID mapping to disk."""
        faiss.write_index(self.faiss_index, str(self.faiss_index_path))
        with open(self.faiss_ids_path, 'wb') as f:
            pickle.dump(self.chunk_id_map, f)
    
    def _generate_doc_id(self, filename: str) -> str:
        """Generate a unique document ID from filename."""
        return hashlib.md5(filename.encode()).hexdigest()[:12]
    
    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate a unique chunk ID."""
        return f"{doc_id}_chunk_{chunk_index:04d}"
    
    def _chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into overlapping chunks.
        
        Returns:
            List of tuples: (chunk_text, start_char, end_char)
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # Try to break at sentence boundary if not at end
            if end < text_len:
                # Look for sentence ending in last 100 chars
                search_start = max(end - 100, start)
                last_period = text.rfind('. ', search_start, end)
                last_newline = text.rfind('\n', search_start, end)
                
                break_point = max(last_period, last_newline)
                if break_point > start:
                    end = break_point + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append((chunk_text, start, end))
            
            # Move start with overlap
            start = end - self.chunk_overlap if end < text_len else text_len
        
        return chunks
    
    def _compute_text_hash(self, text: str) -> str:
        """Compute hash of text for deduplication."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if document already exists in database."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM documents WHERE doc_id = ?', (doc_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    
    def add_document(
        self,
        clean_text_path: str,
        metadata_path: str,
        skip_existing: bool = True
    ) -> Optional[str]:
        """
        Add a single document to the hybrid store.
        
        Args:
            clean_text_path: Path to *_clean.txt file
            metadata_path: Path to *_metadata.json file
            skip_existing: Skip if document already indexed
            
        Returns:
            doc_id if added, None if skipped or failed
        """
        clean_text_path = Path(clean_text_path)
        metadata_path = Path(metadata_path)
        
        # Load metadata
        if not metadata_path.exists():
            print(f"⚠️ Metadata not found: {metadata_path}")
            return None
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        filename = metadata.get('filename', clean_text_path.stem)
        doc_id = self._generate_doc_id(filename)
        
        # Check if already exists
        if skip_existing and self.document_exists(doc_id):
            return None
        
        # Load clean text
        if not clean_text_path.exists():
            print(f"⚠️ Clean text not found: {clean_text_path}")
            return None
        
        with open(clean_text_path, 'r', encoding='utf-8') as f:
            clean_text = f.read()
        
        if not clean_text.strip():
            print(f"⚠️ Empty text: {clean_text_path}")
            return None
        
        # Compute stats
        text_hash = self._compute_text_hash(clean_text)
        char_count = len(clean_text)
        word_count = len(clean_text.split())
        
        # Chunk the text
        chunks = self._chunk_text(clean_text)
        chunk_count = len(chunks)
        
        # Store in SQLite
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        try:
            # Insert document
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (doc_id, filename, title, author, subject, keywords, creator, producer,
                 creation_date, modification_date, page_count, file_size_bytes, source_type,
                 license_status, copyright_status,
                 extraction_timestamp, clean_text, text_hash, char_count, word_count, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc_id,
                filename,
                metadata.get('title'),
                metadata.get('author'),
                metadata.get('subject'),
                metadata.get('keywords'),
                metadata.get('creator'),
                metadata.get('producer'),
                metadata.get('creation_date'),
                metadata.get('modification_date'),
                metadata.get('page_count'),
                metadata.get('file_size_bytes'),
                metadata.get('source_type', 'book'),
                metadata.get('license_status', 'unknown'),
                metadata.get('copyright_status', 'unknown'),
                metadata.get('extraction_timestamp'),
                clean_text,
                text_hash,
                char_count,
                word_count,
                chunk_count
            ))
            
            # Insert chunks
            for i, (chunk_text, start, end) in enumerate(chunks):
                chunk_id = self._generate_chunk_id(doc_id, i)
                cursor.execute('''
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, doc_id, chunk_index, chunk_text, start_char, end_char)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (chunk_id, doc_id, i, chunk_text, start, end))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            print(f"❌ SQLite error for {filename}: {e}")
            return None
        finally:
            conn.close()
        
        # Store embeddings in FAISS
        try:
            chunk_texts = [chunk[0] for chunk in chunks]
            chunk_ids = [self._generate_chunk_id(doc_id, i) for i in range(len(chunks))]
            
            if chunk_texts:
                # Generate embeddings in batches and normalize for cosine similarity
                batch_size = 32
                all_embeddings = []
                
                for i in range(0, len(chunk_texts), batch_size):
                    batch = chunk_texts[i:i+batch_size]
                    batch_emb = self.embedding_model.encode(batch, show_progress_bar=False)
                    all_embeddings.append(batch_emb)
                
                embeddings = np.vstack(all_embeddings)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                # Add to FAISS index
                self.faiss_index.add(embeddings.astype('float32'))
                self.chunk_id_map.extend(chunk_ids)
        
        except Exception as e:
            print(f"❌ FAISS error for {filename}: {e}")
            return None
        
        return doc_id
    
    def ingest_all(self, skip_existing: bool = True) -> Dict[str, int]:
        """
        Ingest all documents from the data folder.
        
        Args:
            skip_existing: Skip documents already in database
            
        Returns:
            Stats dict with counts
        """
        stats = {'added': 0, 'skipped': 0, 'failed': 0, 'total': 0}
        
        # Find all clean text files
        clean_files = list(self.data_folder.glob("*_clean.txt"))
        stats['total'] = len(clean_files)
        
        print(f"\n{'='*60}")
        print(f"INGESTING DOCUMENTS TO HYBRID STORE")
        print(f"{'='*60}")
        print(f"Source folder: {self.data_folder}")
        print(f"Found {len(clean_files)} clean text files")
        print(f"{'='*60}\n")
        
        if not clean_files:
            print("❌ No clean text files found!")
            return stats
        
        iterator = tqdm(clean_files, desc="Indexing") if TQDM_AVAILABLE else clean_files
        
        for clean_path in iterator:
            # Derive metadata path
            base_name = clean_path.stem.replace('_clean', '')
            metadata_path = clean_path.parent / f"{base_name}_metadata.json"
            
            result = self.add_document(
                clean_text_path=str(clean_path),
                metadata_path=str(metadata_path),
                skip_existing=skip_existing
            )
            
            if result is None:
                if skip_existing and self.document_exists(self._generate_doc_id(base_name + '.pdf')):
                    stats['skipped'] += 1
                else:
                    stats['failed'] += 1
            else:
                stats['added'] += 1
        
        # Save FAISS index to disk
        self._save_faiss()
        
        print(f"\n{'='*60}")
        print(f"INGESTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total files:     {stats['total']}")
        print(f"Added:           {stats['added']}")
        print(f"Skipped:         {stats['skipped']}")
        print(f"Failed:          {stats['failed']}")
        print(f"{'='*60}")
        
        return stats
    
    # ==================== QUERY METHODS ====================
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Retrieve document metadata and text by doc_id."""
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM documents WHERE doc_id = ?', (doc_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def get_document_by_filename(self, filename: str) -> Optional[Dict]:
        """Retrieve document by original filename."""
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM documents WHERE filename = ?', (filename,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve chunk by chunk_id."""
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM chunks WHERE chunk_id = ?', (chunk_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def search_by_author(self, author: str) -> List[Dict]:
        """Search documents by author (partial match)."""
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT doc_id, filename, title, author, page_count FROM documents WHERE author LIKE ?',
            (f'%{author}%',)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def search_by_title(self, title: str) -> List[Dict]:
        """Search documents by title (partial match)."""
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT doc_id, filename, title, author, page_count FROM documents WHERE title LIKE ?',
            (f'%{title}%',)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
        exclude_doc_ids: List[str] = None
    ) -> List[Dict]:
        """
        Perform semantic search across all document chunks.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            exclude_doc_ids: Document IDs to exclude from results
            
        Returns:
            List of matching chunks with metadata and similarity scores
        """
        if self.faiss_index.ntotal == 0:
            return []
        
        # Generate query embedding and normalize
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search more results if we need to filter
        search_k = n_results * 3 if exclude_doc_ids else n_results
        
        # Query FAISS
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), min(search_k, self.faiss_index.ntotal))
        
        # Format results
        results = []
        exclude_doc_ids = set(exclude_doc_ids or [])
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.chunk_id_map):
                continue
            
            chunk_id = self.chunk_id_map[idx]
            chunk_data = self.get_chunk_by_id(chunk_id)
            
            if not chunk_data:
                continue
            
            doc_id = chunk_data['doc_id']
            
            # Skip excluded docs
            if doc_id in exclude_doc_ids:
                continue
            
            # Get document metadata
            doc = self.get_document_by_id(doc_id)
            
            results.append({
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'filename': doc['filename'] if doc else '',
                'title': doc['title'] if doc else '',
                'author': doc['author'] if doc else '',
                'source_type': doc['source_type'] if doc else 'unknown',
                'license_status': doc['license_status'] if doc else 'unknown',
                'copyright_status': doc['copyright_status'] if doc else 'unknown',
                'chunk_text': chunk_data['chunk_text'],
                'chunk_index': chunk_data['chunk_index'],
                'similarity': float(dist)  # Already cosine similarity due to normalization
            })
            
            if len(results) >= n_results:
                break
        
        return results
    
    def find_similar_documents(
        self,
        doc_id: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Find documents similar to a given document.
        
        Args:
            doc_id: Document ID to find similar documents for
            n_results: Number of similar documents to return
            
        Returns:
            List of similar documents with similarity scores
        """
        # Get document text
        doc = self.get_document_by_id(doc_id)
        if not doc or not doc.get('clean_text'):
            return []
        
        # Use first chunk as query (representative sample)
        text_sample = doc['clean_text'][:self.chunk_size]
        
        # Search excluding the source document
        results = self.semantic_search(
            query=text_sample,
            n_results=n_results * 3,  # Get more to deduplicate by doc
            exclude_doc_ids=[doc_id]
        )
        
        # Aggregate by document (take best chunk per doc)
        doc_scores = {}
        for r in results:
            other_doc_id = r['doc_id']
            if other_doc_id not in doc_scores or r['similarity'] > doc_scores[other_doc_id]['similarity']:
                doc_scores[other_doc_id] = r
        
        # Sort by similarity and return top n
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x['similarity'], reverse=True)
        return sorted_docs[:n_results]
    
    def get_all_documents(self, limit: int = None) -> List[Dict]:
        """Get all documents (metadata only, no full text)."""
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = '''
            SELECT doc_id, filename, title, author, source_type, license_status,
                   copyright_status, page_count, char_count, word_count, chunk_count, indexed_at
            FROM documents
            ORDER BY indexed_at DESC
        '''
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM documents')
        doc_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM chunks')
        chunk_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT SUM(char_count), SUM(word_count) FROM documents')
        totals = cursor.fetchone()
        
        conn.close()
        
        return {
            'documents': doc_count,
            'chunks_sqlite': chunk_count,
            'chunks_faiss': self.faiss_index.ntotal,
            'total_chars': totals[0] or 0,
            'total_words': totals[1] or 0
        }


