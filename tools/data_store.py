"""
CLI wrapper for rebuilding the canonical hybrid store in ../database.

The runtime HybridDataStore implementation lives in src.storage.hybrid_store.
Keep this module as an admin entry point only:

    python -m tools.data_store
"""

from src.storage.hybrid_store import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_FOLDER,
    DB_FOLDER,
    EMBEDDING_MODEL,
    FAISS_IDS_NAME,
    FAISS_INDEX_NAME,
    SQLITE_DB_NAME,
    SKIP_EXISTING,
    HybridDataStore,
)


def main() -> None:
    print("\n" + "=" * 60)
    print("HYBRID DATA STORE - Configuration")
    print("=" * 60)
    print(f"Data folder:      {DATA_FOLDER}")
    print(f"Database folder:  {DB_FOLDER}")
    print(f"SQLite DB:        {SQLITE_DB_NAME}")
    print(f"FAISS index:      {FAISS_INDEX_NAME}")
    print(f"Embedding model:  {EMBEDDING_MODEL}")
    print(f"Chunk size:       {CHUNK_SIZE}")
    print(f"Chunk overlap:    {CHUNK_OVERLAP}")
    print(f"Skip existing:    {SKIP_EXISTING}")
    print("=" * 60 + "\n")

    store = HybridDataStore(
        data_folder=DATA_FOLDER,
        db_folder=DB_FOLDER,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    store.ingest_all(skip_existing=SKIP_EXISTING)

    print("\nDatabase Stats:")
    db_stats = store.get_stats()
    for key, value in db_stats.items():
        print(f"  {key}: {value:,}")


if __name__ == "__main__":
    main()
