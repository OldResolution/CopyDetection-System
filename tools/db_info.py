"""
Database Info Script - Display SQLite and FAISS details
"""

import sqlite3
import pickle
from pathlib import Path

DB_FOLDER = Path(__file__).parent.parent / "database"
SQLITE_DB = DB_FOLDER / "documents.db"
FAISS_INDEX = DB_FOLDER / "faiss_index.bin"
FAISS_IDS = DB_FOLDER / "faiss_ids.pkl"

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def show_sqlite_info():
    print_header("SQLITE DATABASE")
    
    if not SQLITE_DB.exists():
        print("  Database not found!")
        return
    
    conn = sqlite3.connect(str(SQLITE_DB))
    cursor = conn.cursor()
    
    # Database file size
    db_size = SQLITE_DB.stat().st_size / (1024 * 1024)
    print(f"\n  File: {SQLITE_DB}")
    print(f"  Size: {db_size:.2f} MB")
    
    # Tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\n  Tables: {', '.join(tables)}")
    
    # Documents table
    print_header("DOCUMENTS TABLE")
    doc_count = cursor.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    print(f"\n  Total Documents: {doc_count}")
    
    # Aggregate stats
    stats = cursor.execute("""
        SELECT 
            SUM(page_count) as total_pages,
            SUM(char_count) as total_chars,
            SUM(word_count) as total_words,
            SUM(chunk_count) as total_chunks,
            AVG(word_count) as avg_words,
            MIN(word_count) as min_words,
            MAX(word_count) as max_words
        FROM documents
    """).fetchone()
    
    print(f"  Total Pages: {stats[0]:,}")
    print(f"  Total Characters: {stats[1]:,}")
    print(f"  Total Words: {stats[2]:,}")
    print(f"  Total Chunks: {stats[3]:,}")
    print(f"\n  Avg Words/Doc: {stats[4]:,.0f}")
    print(f"  Min Words: {stats[5]:,}")
    print(f"  Max Words: {stats[6]:,}")
    
    # Document list
    print("\n  Documents:")
    print("  " + "-" * 56)
    docs = cursor.execute("""
        SELECT doc_id, title, author, word_count, chunk_count 
        FROM documents 
        ORDER BY word_count DESC
    """).fetchall()
    
    for i, (doc_id, title, author, words, chunks) in enumerate(docs, 1):
        title_display = (title[:35] + "...") if title and len(title) > 38 else (title or "Untitled")
        author_display = author[:15] if author else "Unknown"
        print(f"  {i:2}. [{doc_id[:8]}] {title_display}")
        print(f"      Author: {author_display} | Words: {words:,} | Chunks: {chunks:,}")
    
    # Chunks table
    print_header("CHUNKS TABLE")
    chunk_count = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"\n  Total Chunks: {chunk_count:,}")
    
    chunk_stats = cursor.execute("""
        SELECT 
            AVG(LENGTH(chunk_text)) as avg_chars,
            MIN(LENGTH(chunk_text)) as min_chars,
            MAX(LENGTH(chunk_text)) as max_chars
        FROM chunks
    """).fetchone()
    
    print(f"  Avg Chars/Chunk: {chunk_stats[0]:,.0f}")
    print(f"  Min Chars: {chunk_stats[1]:,}")
    print(f"  Max Chars: {chunk_stats[2]:,}")
    
    conn.close()

def show_faiss_info():
    print_header("FAISS VECTOR INDEX")
    
    if not FAISS_INDEX.exists():
        print("  FAISS index not found!")
        return
    
    # File size
    index_size = FAISS_INDEX.stat().st_size / (1024 * 1024)
    print(f"\n  Index File: {FAISS_INDEX}")
    print(f"  Index Size: {index_size:.2f} MB")
    
    # Load FAISS to get details
    try:
        import faiss
        index = faiss.read_index(str(FAISS_INDEX))
        print(f"\n  Total Vectors: {index.ntotal:,}")
        print(f"  Dimension: {index.d}")
        print(f"  Index Type: IndexFlatIP (Inner Product / Cosine)")
        print(f"  Trained: {index.is_trained}")
    except ImportError:
        print("  (faiss-cpu not installed - limited info)")
    except Exception as e:
        print(f"  Error reading index: {e}")
    
    # Chunk ID mapping
    if FAISS_IDS.exists():
        ids_size = FAISS_IDS.stat().st_size / (1024 * 1024)
        print(f"\n  IDs File: {FAISS_IDS}")
        print(f"  IDs Size: {ids_size:.2f} MB")
        
        with open(FAISS_IDS, 'rb') as f:
            chunk_ids = pickle.load(f)
        
        print(f"  Mapped IDs: {len(chunk_ids):,}")
        
        # Sample IDs
        print("\n  Sample Chunk IDs:")
        for cid in chunk_ids[:5]:
            print(f"    - {cid}")
        if len(chunk_ids) > 5:
            print(f"    ... ({len(chunk_ids) - 5:,} more)")

def show_sync_status():
    print_header("SYNC STATUS")
    
    # Get SQLite chunk count
    conn = sqlite3.connect(str(SQLITE_DB))
    sqlite_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    
    # Get FAISS vector count
    with open(FAISS_IDS, 'rb') as f:
        faiss_count = len(pickle.load(f))
    
    print(f"\n  SQLite Chunks: {sqlite_chunks:,}")
    print(f"  FAISS Vectors: {faiss_count:,}")
    
    if sqlite_chunks == faiss_count:
        print(f"\n  Status: ✓ SYNCED")
    else:
        diff = abs(sqlite_chunks - faiss_count)
        print(f"\n  Status: ✗ OUT OF SYNC (difference: {diff:,})")

def main():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " DATABASE INFORMATION ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    show_sqlite_info()
    show_faiss_info()
    show_sync_status()
    
    print("\n")

if __name__ == "__main__":
    main()
