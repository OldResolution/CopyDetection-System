"""
Database-backed plagiarism detector using HybridDataStore (SQLite + FAISS).
"""
import sys
import os
import traceback
import numpy as np
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
from data_store import HybridDataStore

from src.config import MODEL_NAME
from src.common.text_processor import preprocess_text
from src.common.metrics import (
    calculate_ngram_similarity, 
    extract_stylometric_features, 
    calculate_stylometric_similarity
)


class DatabasePlagiarismDetector:
    """
    Plagiarism detector using SQLite + FAISS vector database.
    
    This uses pre-computed embeddings in FAISS for fast semantic search,
    then performs detailed analysis on top candidates.
    """
    
    def __init__(self, db_folder: str = "database"):
        print(f"\n[INFO] Initializing Database Detector...")
        self.db_folder = db_folder
        
        # Initialize hybrid data store
        try:
            self.store = HybridDataStore(
                data_folder="test_data/Excel_Dataset",
                db_folder=db_folder,
                embedding_model=MODEL_NAME,
                chunk_size=500,
                chunk_overlap=50
            )
            
            # Get stats
            stats = self.store.get_stats()
            print(f"[OK] Database loaded: {stats['documents']} documents, {stats['chunks_faiss']} chunks")
            
            if stats['documents'] == 0:
                print("[WARNING] Database is empty! Please run tools/data_store.py to populate it.")
                
        except Exception as e:
            print(f"[ERROR] Failed to initialize database: {e}")
            raise
    
    def analyze_text(self, essay: str) -> Dict:
        """
        Analyze text for plagiarism using database search.
        
        Args:
            essay: Text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        feature_names = ["Avg Word Len", "Avg Sent Len (Log)", "Type-Token",
                         "Stopwords", "Punctuation", "Unique Words", "Sent Count (Log)"]
        
        # 1. Extract essay features
        essay_feats = extract_stylometric_features(essay)
        essay_raw = preprocess_text(essay, False)
        essay_clean = preprocess_text(essay, True)

        response_base = {
            'feature_names': feature_names, 
            'essay_features': essay_feats.tolist(),
            'top_source': "No significant database match found", 
            'combined_score': 0.0, 
            'risk_level': "SAFE",
            'plagiarized_books': []
        }

        try:
            # PHASE 1: Semantic Search with FAISS (Fast!)
            print("[INFO] Searching database with FAISS semantic search...")
            search_results = self.store.semantic_search(
                query=essay[:5000],  # Use first 5000 chars for query
                n_results=100,  # Get top 100 chunks
                exclude_doc_ids=None
            )
            
            if not search_results:
                print("[INFO] No semantic matches found")
                return response_base
            
            print(f"[INFO] Found {len(search_results)} semantic matches")
            
            # PHASE 2: Group by document and get full texts
            doc_chunks = {}
            for result in search_results:
                doc_id = result['doc_id']
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = {
                        'title': result.get('title', 'Unknown'),
                        'author': result.get('author', 'Unknown'),
                        'chunks': [],
                        'max_similarity': 0.0
                    }
                doc_chunks[doc_id]['chunks'].append(result['chunk_text'])
                doc_chunks[doc_id]['max_similarity'] = max(
                    doc_chunks[doc_id]['max_similarity'],
                    result['similarity']
                )
            
            print(f"[INFO] Analyzing {len(doc_chunks)} unique documents...")
            
            # PHASE 3: Detailed Analysis on matched documents
            results = []
            seen_titles = set()
            
            for doc_id, doc_data in doc_chunks.items():
                try:
                    title = str(doc_data['title']).strip().lower()
                    
                    # Skip duplicates
                    if title in seen_titles:
                        continue
                    
                    # Get full document for detailed analysis
                    doc = self.store.get_document_by_id(doc_id)
                    if not doc or not doc.get('clean_text'):
                        continue
                    
                    book_text = doc['clean_text']
                    
                    # A. Semantic Score (from FAISS search)
                    # Use max similarity from any chunk
                    sem_s = float(doc_data['max_similarity'])
                    
                    # Hard filter: skip if semantic similarity too low
                    if sem_s < 0.3:
                        continue
                    
                    # B. N-gram similarity (on full text)
                    book_tokens = preprocess_text(book_text, False)
                    ngram_s = float(max(0.0, calculate_ngram_similarity(essay_raw, book_tokens, 3)))
                    
                    # C. Stylometric similarity
                    book_feats = extract_stylometric_features(book_text)
                    stylo_s = float(max(0.0, calculate_stylometric_similarity(essay_feats, book_feats)))
                    
                    # D. Weighted Combined Score
                    if sem_s > 0.7:
                        combined = (sem_s * 0.6) + (ngram_s * 0.2) + (stylo_s * 0.2)
                    elif ngram_s > 0.3:
                        combined = (ngram_s * 0.5) + (sem_s * 0.3) + (stylo_s * 0.2)
                    else:
                        combined = (sem_s * 0.5) + (ngram_s * 0.4) + (stylo_s * 0.1)
                    
                    # Penalty for weak matches
                    if sem_s < 0.65 and ngram_s < 0.05:
                        combined *= 0.4
                    
                    # Skip if below threshold
                    if combined < 0.3:
                        continue
                    
                    # E. Determine risk level
                    if combined >= 0.75:
                        risk = "CRITICAL"
                    elif combined >= 0.5:
                        risk = "HIGH"
                    elif combined >= 0.3:
                        risk = "MODERATE"
                    else:
                        risk = "SAFE"
                    
                    results.append({
                        'book_title': doc_data['title'],
                        'book_author': doc_data['author'],
                        'combined_score': float(combined),
                        'risk_level': risk,
                        'ngram_score': ngram_s,
                        'semantic_score': sem_s,
                        'stylometric_score': stylo_s
                    })
                    
                    seen_titles.add(title)
                    
                except Exception as e:
                    print(f"[ERROR] Failed to analyze document {doc_id}: {e}")
                    continue
            
            # Sort by combined score
            results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # PHASE 4: Format response
            if results:
                top_match = results[0]
                significant_matches = [r for r in results if r['combined_score'] >= 0.3]
                
                source_text = f"{top_match['book_title']} by {top_match['book_author']}"
                if len(significant_matches) > 1:
                    source_text = "Multiple Sources Detected"
                
                response_base.update({
                    'top_source': source_text,
                    'combined_score': top_match['combined_score'],
                    'risk_level': top_match['risk_level'],
                    'plagiarized_books': results[:10]
                })
                
                print(f"[OK] Analysis complete: {top_match['risk_level']} risk ({top_match['combined_score']:.2%})")
            else:
                print("[OK] No significant matches found")
                response_base.update({
                    'top_source': "No significant database match found",
                    'combined_score': 0.0,
                    'risk_level': "SAFE",
                    'plagiarized_books': []
                })
            
            return response_base
            
        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            traceback.print_exc()
            return response_base
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return self.store.get_stats()
