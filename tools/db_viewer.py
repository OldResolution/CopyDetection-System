
import streamlit as st
import sqlite3
import pickle
import os
from pathlib import Path

# Configuration
DB_FOLDER = Path(__file__).parent.parent / "database"
SQLITE_DB = DB_FOLDER / "documents.db"
FAISS_IDS = DB_FOLDER / "faiss_ids.pkl"

st.set_page_config(
    page_title="Document Database Viewer",
    page_icon="📚",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("📚 Database Viewer")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Documents", "Chunks Explorer", "FAISS Vectors", "Semantic Search"]
)

# Database connection helper
@st.cache_resource
def get_db_connection():
    return sqlite3.connect(str(SQLITE_DB), check_same_thread=False)

@st.cache_data
def load_faiss_ids():
    if FAISS_IDS.exists():
        with open(FAISS_IDS, 'rb') as f:
            return pickle.load(f)
    return []

# ============================================================
# PAGE: Overview
# ============================================================
if page == "Overview":
    st.title("📊 Database Overview")
    
    conn = get_db_connection()
    
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    
    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    total_words = conn.execute("SELECT SUM(word_count) FROM documents").fetchone()[0] or 0
    total_pages = conn.execute("SELECT SUM(page_count) FROM documents").fetchone()[0] or 0
    
    col1.metric("📄 Documents", f"{doc_count:,}")
    col2.metric("🧩 Chunks", f"{chunk_count:,}")
    col3.metric("📝 Total Words", f"{total_words:,}")
    col4.metric("📑 Total Pages", f"{total_pages:,}")
    
    # FAISS sync status
    faiss_ids = load_faiss_ids()
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💾 Storage Status")
        sync_status = "✅ Synced" if len(faiss_ids) == chunk_count else "⚠️ Out of Sync"
        st.write(f"**SQLite Chunks:** {chunk_count:,}")
        st.write(f"**FAISS Vectors:** {len(faiss_ids):,}")
        st.write(f"**Status:** {sync_status}")
    
    with col2:
        st.subheader("📈 Top Documents by Size")
        top_docs = conn.execute("""
            SELECT title, word_count, chunk_count 
            FROM documents 
            ORDER BY word_count DESC 
            LIMIT 5
        """).fetchall()
        for title, words, chunks in top_docs:
            st.write(f"**{title[:40]}...** - {words:,} words, {chunks:,} chunks")
    
    # Chart: Documents by word count
    st.divider()
    st.subheader("📊 Document Size Distribution")
    
    import pandas as pd
    df = pd.read_sql_query("""
        SELECT title, word_count, chunk_count, page_count 
        FROM documents 
        ORDER BY word_count DESC
    """, conn)
    
    st.bar_chart(df.set_index('title')['word_count'])

# ============================================================
# PAGE: Documents
# ============================================================
elif page == "Documents":
    st.title("📄 Documents")
    
    conn = get_db_connection()
    
    # Search filter
    search = st.text_input("🔍 Search documents", placeholder="Filter by title, author, or filename...")
    
    query = """
        SELECT doc_id, filename, title, author, page_count, word_count, chunk_count, indexed_at
        FROM documents
    """
    if search:
        query += f" WHERE title LIKE '%{search}%' OR author LIKE '%{search}%' OR filename LIKE '%{search}%'"
    query += " ORDER BY indexed_at DESC"
    
    import pandas as pd
    df = pd.read_sql_query(query, conn)
    
    st.write(f"**Showing {len(df)} documents**")
    
    # Display as table
    st.dataframe(
        df,
        column_config={
            "doc_id": st.column_config.TextColumn("ID", width="small"),
            "filename": st.column_config.TextColumn("Filename", width="medium"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "author": st.column_config.TextColumn("Author", width="medium"),
            "page_count": st.column_config.NumberColumn("Pages", format="%d"),
            "word_count": st.column_config.NumberColumn("Words", format="%,d"),
            "chunk_count": st.column_config.NumberColumn("Chunks", format="%,d"),
            "indexed_at": st.column_config.TextColumn("Indexed", width="medium"),
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Document detail view
    st.divider()
    st.subheader("📖 Document Details")
    
    doc_ids = df['doc_id'].tolist()
    doc_titles = df['title'].tolist()
    options = [f"{did[:8]} - {title[:50]}" for did, title in zip(doc_ids, doc_titles)]
    
    if options:
        selected = st.selectbox("Select a document", options)
        if selected:
            selected_id = selected.split(" - ")[0]
            # Find full doc_id
            full_id = [d for d in doc_ids if d.startswith(selected_id)][0]
            
            doc = conn.execute("""
                SELECT * FROM documents WHERE doc_id = ?
            """, (full_id,)).fetchone()
            
            if doc:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Title:** {doc[3]}")
                    st.write(f"**Author:** {doc[4] or 'N/A'}")
                    st.write(f"**Filename:** {doc[2]}")
                    st.write(f"**File Hash:** {doc[1][:16]}...")
                with col2:
                    st.write(f"**Pages:** {doc[5]}")
                    st.write(f"**Characters:** {doc[6]:,}")
                    st.write(f"**Words:** {doc[7]:,}")
                    st.write(f"**Chunks:** {doc[8]:,}")

# ============================================================
# PAGE: Chunks Explorer
# ============================================================
elif page == "Chunks Explorer":
    st.title("🧩 Chunks Explorer")
    
    conn = get_db_connection()
    
    # Select document
    docs = conn.execute("SELECT doc_id, title FROM documents ORDER BY title").fetchall()
    doc_options = {f"{d[0][:8]} - {d[1][:50]}": d[0] for d in docs}
    
    selected_doc = st.selectbox("Select Document", list(doc_options.keys()))
    
    if selected_doc:
        doc_id = doc_options[selected_doc]
        
        # Get chunks for this document
        chunks = conn.execute("""
            SELECT chunk_id, chunk_index, char_count, word_count, content
            FROM chunks
            WHERE doc_id = ?
            ORDER BY chunk_index
        """, (doc_id,)).fetchall()
        
        st.write(f"**Total Chunks:** {len(chunks)}")
        
        # Pagination
        chunks_per_page = 10
        total_pages = (len(chunks) + chunks_per_page - 1) // chunks_per_page
        
        page_num = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)
        
        start_idx = (page_num - 1) * chunks_per_page
        end_idx = min(start_idx + chunks_per_page, len(chunks))
        
        st.write(f"Showing chunks {start_idx + 1} to {end_idx} of {len(chunks)}")
        
        for chunk in chunks[start_idx:end_idx]:
            chunk_id, chunk_idx, chars, words, content = chunk
            with st.expander(f"Chunk {chunk_idx} | {words} words | ID: {chunk_id[:16]}..."):
                st.text_area("Content", content, height=200, disabled=True, key=chunk_id)

# ============================================================
# PAGE: FAISS Vectors
# ============================================================
elif page == "FAISS Vectors":
    st.title("🔢 FAISS Vector Index")
    
    faiss_ids = load_faiss_ids()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Vectors", f"{len(faiss_ids):,}")
    with col2:
        st.metric("Embedding Dimension", "384 (MiniLM)")
    
    st.divider()
    st.subheader("Vector ID Mapping")
    
    # Pagination for FAISS IDs
    items_per_page = 50
    total_pages = (len(faiss_ids) + items_per_page - 1) // items_per_page
    
    page_num = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)
    
    start_idx = (page_num - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(faiss_ids))
    
    st.write(f"Showing vectors {start_idx + 1} to {end_idx} of {len(faiss_ids)}")
    
    import pandas as pd
    df = pd.DataFrame({
        "FAISS Index": range(start_idx, end_idx),
        "Chunk ID": faiss_ids[start_idx:end_idx]
    })
    st.dataframe(df, hide_index=True, use_container_width=True)

# ============================================================
# PAGE: Semantic Search
# ============================================================
elif page == "Semantic Search":
    st.title("🔍 Semantic Search")
    
    st.warning("⚠️ This requires loading the embedding model (may take a moment on first use)")
    
    query = st.text_input("Enter search query", placeholder="Search for similar content...")
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    if st.button("Search", type="primary") and query:
        with st.spinner("Loading model and searching..."):
            try:
                import faiss
                import numpy as np
                from sentence_transformers import SentenceTransformer
                
                # Load model
                model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Load FAISS index
                faiss_index_path = DB_FOLDER / "faiss_index.bin"
                index = faiss.read_index(str(faiss_index_path))
                
                # Load chunk IDs
                faiss_ids = load_faiss_ids()
                
                # Embed query
                query_vec = model.encode([query], normalize_embeddings=True)
                
                # Search
                scores, indices = index.search(query_vec.astype('float32'), top_k)
                
                # Get results
                conn = get_db_connection()
                
                st.subheader("Search Results")
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < len(faiss_ids):
                        chunk_id = faiss_ids[idx]
                        
                        # Get chunk content
                        chunk = conn.execute("""
                            SELECT c.content, c.chunk_index, d.title
                            FROM chunks c
                            JOIN documents d ON c.doc_id = d.doc_id
                            WHERE c.chunk_id = ?
                        """, (chunk_id,)).fetchone()
                        
                        if chunk:
                            content, chunk_idx, title = chunk
                            with st.expander(f"#{i+1} | Score: {score:.4f} | {title[:40]}... (Chunk {chunk_idx})"):
                                st.write(content[:1000] + "..." if len(content) > 1000 else content)
                
            except ImportError as e:
                st.error(f"Missing dependency: {e}. Install with: pip install faiss-cpu sentence-transformers")
            except Exception as e:
                st.error(f"Search error: {e}")

# Footer
st.sidebar.divider()
st.sidebar.caption("Database: SQLite + FAISS Hybrid")
st.sidebar.caption(f"Path: {DB_FOLDER}")
