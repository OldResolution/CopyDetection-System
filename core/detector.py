import sys
import os
import pandas as pd
import numpy as np
import traceback

# Path Fix
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from sentence_transformers import SentenceTransformer, util
from config import MODEL_NAME

try:
    from utils.text_processor import preprocess_text
    from utils.metrics import (
        calculate_ngram_similarity, 
        calculate_jaccard_similarity, 
        extract_stylometric_features, 
        calculate_stylometric_similarity
    )
except ImportError as e:
    raise e

class AdvancedPlagiarismDetector:
    def __init__(self, reference_file_path):
        print(f"\n[INFO] Initializing Detector...")
        self.reference_file_path = reference_file_path
        self.books_df = self._load_reference_data()
        
        try:
            print("[INFO] Loading Semantic Model...")
            self.semantic_model = SentenceTransformer(MODEL_NAME)
            print("✓ Semantic model loaded")
        except:
            self.semantic_model = None

        if not self.books_df.empty:
            self._precompute_simple_features()

    def _load_reference_data(self):
        if not os.path.exists(self.reference_file_path):
            print(f"⚠ ERROR: File not found: {self.reference_file_path}")
            return pd.DataFrame()
        try:
            df = pd.read_excel(self.reference_file_path)
            if "text_content" not in df.columns: return pd.DataFrame()
            
            # Force string and clean
            df['text_content'] = df['text_content'].fillna('').astype(str)
            df = df[df['text_content'].str.len() > 50]
            
            cols = df.columns
            title_col = next((c for c in cols if 'title' in c.lower()), 'Title')
            author_col = next((c for c in cols if 'author' in c.lower()), 'Author')
            df['book_title'] = df[title_col] if title_col in cols else "Unknown Title"
            df['book_author'] = df[author_col] if author_col in cols else "Unknown Author"
            
            print(f"✓ Loaded {len(df)} valid books.")
            return df
        except Exception as e:
            print(f"✗ Data load error: {e}")
            return pd.DataFrame()

    def _precompute_simple_features(self):
        print("[INFO] Pre-processing tokens...")
        tokens_raw_list = []
        tokens_clean_list = []
        for text in self.books_df['text_content']:
            try:
                safe_text = str(text)
                tokens_raw_list.append(preprocess_text(safe_text, False))
                tokens_clean_list.append(preprocess_text(safe_text, True))
            except:
                tokens_raw_list.append([])
                tokens_clean_list.append([])
        self.books_df['tokens_raw'] = tokens_raw_list
        self.books_df['tokens_clean'] = tokens_clean_list
        print("✓ Tokens ready.")

    def calculate_chunked_semantic_similarity(self, essay, book_text):
        if not self.semantic_model: return 0.0
        try:
            essay_emb = self.semantic_model.encode(essay[:3000], convert_to_tensor=True)
            book_len = len(book_text)
            chunks = []
            offsets = [0, 0.25, 0.5, 0.75, 0.9]
            window = 3000 
            for off in offsets:
                start = int(book_len * off)
                chunk = book_text[start:start+window]
                if len(chunk) > 100: chunks.append(chunk)
            if not chunks: chunks = [book_text[:3000]] 

            max_sim = 0.0
            for chunk in chunks:
                chunk_emb = self.semantic_model.encode(chunk, convert_to_tensor=True)
                sim = float(util.pytorch_cos_sim(essay_emb, chunk_emb).item())
                if sim > max_sim: max_sim = sim
            return max_sim
        except: return 0.0

    def analyze_text(self, essay):
        feature_names = ["Avg Word Len", "Avg Sent Len (Log)", "Type-Token",
                         "Stopwords", "Punctuation", "Unique Words", "Sent Count (Log)"]
        
        # 1. Essay Features
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

        if self.books_df.empty: return response_base

        try:
            # PHASE 1: Quick Filter
            candidates = []
            for idx, row in self.books_df.iterrows():
                try:
                    jaccard = calculate_jaccard_similarity(essay_clean, row['tokens_clean'])
                    ngram = calculate_ngram_similarity(essay_raw, row['tokens_raw'], 2)
                    pre_score = jaccard + (ngram * 0.5)
                    candidates.append((idx, pre_score))
                except: continue
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, score in candidates[:50]]
            top_rows = self.books_df.loc[top_indices]

            # PHASE 2: Detailed Analysis
            results = []
            seen_titles = set()  # --- DEDUPLICATION SET ---

            for i, (_, row) in enumerate(top_rows.iterrows()):
                try:
                    title = str(row['book_title']).strip().lower()
                    
                    # --- CHECK DUPLICATES ---
                    if title in seen_titles:
                        continue
                    # ------------------------

                    book_text = str(row['text_content'])
                    
                    # A. Semantic Score (Hard Filter < 30%)
                    sem_s = float(max(0.0, self.calculate_chunked_semantic_similarity(essay, book_text)))
                    if sem_s < 0.3: continue
                    
                    # B. Other Metrics
                    ngram_s = float(max(0.0, calculate_ngram_similarity(essay_raw, row['tokens_raw'], 3)))
                    book_feats = extract_stylometric_features(book_text)
                    stylo_s = float(max(0.0, calculate_stylometric_similarity(essay_feats, book_feats)))

                    # Weighted Score
                    if sem_s > 0.7:
                        combined = (sem_s * 0.6) + (ngram_s * 0.2) + (stylo_s * 0.2)
                    elif ngram_s > 0.3:
                        combined = (ngram_s * 0.5) + (sem_s * 0.3) + (stylo_s * 0.2)
                    else:
                        # Heavy penalty if there is no exact phrasing match and semantic match is weak.
                        # Do not let stylometric writing style alone push the score > 0.3
                        combined = (sem_s * 0.5) + (ngram_s * 0.4) + (stylo_s * 0.1)

                    # Further penalty for strict false-positive prevention on texts loosely matching semantics
                    if sem_s < 0.65 and ngram_s < 0.05:
                        combined *= 0.4 

                    if combined < 0.3: continue 

                    if combined >= 0.75: risk = "CRITICAL"
                    elif combined >= 0.5: risk = "HIGH"
                    elif combined >= 0.3: risk = "MODERATE"
                    else: risk = "SAFE"

                    results.append({
                        'book_title': str(row['book_title']),
                        'book_author': str(row['book_author']),
                        'combined_score': float(combined),
                        'risk_level': risk,
                        'ngram_score': ngram_s,
                        'semantic_score': sem_s,
                        'stylometric_score': stylo_s
                    })
                    
                    # Mark title as seen so we don't add it again
                    seen_titles.add(title)

                except Exception:
                    continue

            results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # --- MULTIPLE SOURCE LOGIC ---
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
            else:
                response_base.update({
                    'top_source': "No significant database match found",
                    'combined_score': 0.0,
                    'risk_level': "SAFE",
                    'plagiarized_books': []
                })
            
            return response_base

        except Exception as e:
            traceback.print_exc()
            return response_base