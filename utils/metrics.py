import numpy as np
import nltk
from nltk.util import ngrams
import string
from utils.text_processor import STOP_WORDS

def calculate_ngram_similarity(essay_tokens, reference_tokens, n=3):
    if not essay_tokens or not reference_tokens or len(essay_tokens) < n:
        return 0.0
    try:
        essay_ngrams = set(ngrams(essay_tokens, n))
        ref_ngrams = set(ngrams(reference_tokens, n))
        if not essay_ngrams: return 0.0
        return len(essay_ngrams.intersection(ref_ngrams)) / len(essay_ngrams)
    except:
        return 0.0

def calculate_jaccard_similarity(list1, list2):
    s1, s2 = set(list1), set(list2)
    if not s1 or not s2: return 0.0
    union_len = len(s1.union(s2))
    if union_len == 0: return 0.0
    return len(s1.intersection(s2)) / union_len

def extract_stylometric_features(text):
    try:
        text = str(text)
        if not text.strip(): return np.zeros(7)

        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text.lower())
        words_alpha = [w for w in words if w.isalpha()]
        
        if not words_alpha: return np.zeros(7)
        
        num_words = len(words_alpha)
        num_total = len(words)
        
        def safe_div(a, b): return a / b if b > 0 else 0.0

        # 1. Avg Word Length
        f1 = np.mean([len(w) for w in words_alpha])
        
        # 2. Avg Sentence Length (Log scaled to prevent dominance)
        avg_sent_len = np.mean([len(nltk.word_tokenize(s)) for s in sentences]) if sentences else 0
        f2 = np.log1p(avg_sent_len) 
        
        # 3. Type-Token Ratio
        f3 = safe_div(len(set(words_alpha)), num_words)
        
        # 4. Stopword Ratio
        f4 = safe_div(len([w for w in words if w in STOP_WORDS]), num_total)
        
        # 5. Punctuation Ratio
        f5 = safe_div(len([w for w in words if w in string.punctuation]), num_total)
        
        # 6. Unique Words Ratio
        f6 = safe_div(len(set(words_alpha)), num_words)
        
        # 7. Sentence Count (Log scaled to prevent dominance)
        f7 = np.log1p(float(len(sentences)))
        
        # Return features as float array, ensuring no NaNs
        return np.nan_to_num(np.array([f1, f2, f3, f4, f5, f6, f7], dtype=float))
    except:
        return np.zeros(7)

def calculate_stylometric_similarity(essay_features, ref_features):
    try:
        v1 = np.nan_to_num(np.array(essay_features, dtype=float))
        v2 = np.nan_to_num(np.array(ref_features, dtype=float))
        
        # Check for empty vectors
        if np.sum(v1) == 0 or np.sum(v2) == 0:
            return 0.0
            
        dot = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
            
        similarity = dot / (norm_v1 * norm_v2)
        
        # Safety clip
        return float(max(0.0, min(1.0, similarity)))
    except:
        return 0.0