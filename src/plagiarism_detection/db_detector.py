"""
Database-backed plagiarism detector using HybridDataStore (SQLite + FAISS).
"""
import traceback
import numpy as np
from typing import Dict, List

from src.config import DATA_FOLDER, MODEL_NAME
from src.common.text_processor import preprocess_text
from src.common.metrics import (
    calculate_ngram_similarity, 
    extract_stylometric_features, 
    calculate_stylometric_similarity
)
from src.storage import HybridDataStore


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
                data_folder=DATA_FOLDER,
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

    @staticmethod
    def _normalize_rights_field(value: str, default: str = "unknown") -> str:
        if value is None:
            return default
        text = str(value).strip().lower()
        return text or default

    def _build_rights_context(self, source_type: str, license_status: str, copyright_status: str) -> str:
        source_type = self._normalize_rights_field(source_type, "source work")
        license_status = self._normalize_rights_field(license_status)
        copyright_status = self._normalize_rights_field(copyright_status)

        return (
            f"Source type: {source_type}. "
            f"License status: {license_status}. "
            f"Copyright status: {copyright_status}."
        )

    def _classify_legal_risk(
        self,
        semantic_score: float,
        ngram_score: float,
        stylometric_score: float,
        combined_score: float,
        license_status: str,
        copyright_status: str,
    ) -> Dict[str, str]:
        license_status = self._normalize_rights_field(license_status)
        copyright_status = self._normalize_rights_field(copyright_status)
        protected_source = (
            copyright_status not in {"public domain", "unknown"}
            or license_status not in {"public domain", "unknown"}
        )

        if combined_score < 0.3:
            return {
                "risk_level": "no actionable similarity detected",
                "legal_risk_code": "NO_ACTIONABLE_SIMILARITY",
                "legal_rationale": "The submission does not show enough overlap for a copyright-focused concern.",
            }

        if ngram_score >= 0.45 and semantic_score >= 0.7:
            return {
                "risk_level": "near-verbatim reproduction",
                "legal_risk_code": "NEAR_VERBATIM_REPRODUCTION",
                "legal_rationale": (
                    "Dense phrase overlap with strong semantic alignment suggests memorization or direct reuse of protected expression."
                    if protected_source
                    else "Dense phrase overlap suggests direct reuse, although the source rights status may reduce infringement exposure."
                ),
            }

        if semantic_score >= 0.6 and ngram_score < 0.45:
            return {
                "risk_level": "substantial similarity in protectable expression",
                "legal_risk_code": "SUBSTANTIAL_SIMILARITY",
                "legal_rationale": (
                    "Meaning-level overlap is strong without the same amount of verbatim reuse, which is more consistent with derivative rewriting."
                    if protected_source
                    else "Meaning-level overlap is strong, but the source rights status may permit some reuse."
                ),
            }

        if semantic_score >= 0.35 or stylometric_score >= 0.5:
            return {
                "risk_level": "thematic/stylistic influence only",
                "legal_risk_code": "THEMATIC_OR_STYLISTIC_INFLUENCE",
                "legal_rationale": "The overlap is more about themes, structure, or style than clear copying of protected expression.",
            }

        return {
            "risk_level": "no actionable similarity detected",
            "legal_risk_code": "NO_ACTIONABLE_SIMILARITY",
            "legal_rationale": "The available signals stay below a meaningful legal similarity threshold.",
        }
    
    def analyze_text(self, submission_text: str) -> Dict:
        """
        Analyze submitted text for copyright-oriented similarity.
        
        Args:
            submission_text: Text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        feature_names = ["Avg Word Len", "Avg Sent Len (Log)", "Type-Token",
                         "Stopwords", "Punctuation", "Unique Words", "Sent Count (Log)"]
        
        # 1. Extract submission features
        submission_feats = extract_stylometric_features(submission_text)
        submission_raw = preprocess_text(submission_text, False)
        submission_clean = preprocess_text(submission_text, True)

        response_base = {
            'feature_names': feature_names,
            'submission_features': submission_feats.tolist(),
            'top_source_work': "No significant source-work match found",
            'combined_score': 0.0,
            'risk_level': "no actionable similarity detected",
            'legal_risk_code': "NO_ACTIONABLE_SIMILARITY",
            'legal_rationale': "The submission does not show enough overlap for a copyright-focused concern.",
            'matched_sources': [],
        }

        try:
            # PHASE 1: Semantic Search with FAISS (Fast!)
            print("[INFO] Searching database with FAISS semantic search...")
            search_results = self.store.semantic_search(
                query=submission_text[:5000],  # Use first 5000 chars for query
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
                    
                    source_text = doc['clean_text']
                    source_type = doc.get('source_type', 'book')
                    license_status = doc.get('license_status', 'unknown')
                    copyright_status = doc.get('copyright_status', 'unknown')
                    
                    # A. Semantic Score (from FAISS search)
                    # Use max similarity from any chunk
                    sem_s = float(doc_data['max_similarity'])
                    
                    # Hard filter: skip if semantic similarity too low
                    if sem_s < 0.3:
                        continue
                    
                    # B. N-gram similarity (on full text)
                    source_tokens = preprocess_text(source_text, False)
                    ngram_s = float(max(0.0, calculate_ngram_similarity(submission_raw, source_tokens, 3)))
                    
                    # C. Stylometric similarity
                    source_feats = extract_stylometric_features(source_text)
                    stylo_s = float(max(0.0, calculate_stylometric_similarity(submission_feats, source_feats)))
                    
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
                    
                    legal_classification = self._classify_legal_risk(
                        semantic_score=sem_s,
                        ngram_score=ngram_s,
                        stylometric_score=stylo_s,
                        combined_score=combined,
                        license_status=license_status,
                        copyright_status=copyright_status,
                    )
                    
                    results.append({
                        'source_title': doc_data['title'],
                        'source_author': doc_data['author'],
                        'source_type': source_type,
                        'license_status': license_status,
                        'copyright_status': copyright_status,
                        'rights_context': self._build_rights_context(
                            source_type=source_type,
                            license_status=license_status,
                            copyright_status=copyright_status,
                        ),
                        'combined_score': float(combined),
                        'risk_level': legal_classification['risk_level'],
                        'legal_risk_code': legal_classification['legal_risk_code'],
                        'legal_rationale': legal_classification['legal_rationale'],
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
                
                source_text = f"{top_match['source_title']} by {top_match['source_author']}"
                if len(significant_matches) > 1:
                    source_text = "Multiple Sources Detected"
                
                response_base.update({
                    'top_source_work': source_text,
                    'combined_score': top_match['combined_score'],
                    'risk_level': top_match['risk_level'],
                    'legal_risk_code': top_match['legal_risk_code'],
                    'legal_rationale': top_match['legal_rationale'],
                    'matched_sources': results[:10]
                })
                
                print(f"[OK] Analysis complete: {top_match['risk_level']} ({top_match['combined_score']:.2%})")
            else:
                print("[OK] No significant matches found")
                response_base.update({
                    'top_source_work': "No significant source-work match found",
                    'combined_score': 0.0,
                    'risk_level': "no actionable similarity detected",
                    'legal_risk_code': "NO_ACTIONABLE_SIMILARITY",
                    'legal_rationale': "The submission does not show enough overlap for a copyright-focused concern.",
                    'matched_sources': []
                })
            
            return response_base
            
        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            traceback.print_exc()
            return response_base
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return self.store.get_stats()
