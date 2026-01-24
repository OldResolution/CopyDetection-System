import pandas as pd
import nltk
import numpy as np
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import string
from datetime import datetime

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# -------------------------------
# TEXT PREPROCESSING
# -------------------------------
def preprocess_text(text):
    """Preprocess text for analysis"""
    tokens = nltk.word_tokenize(str(text).lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

# -------------------------------
# N-GRAM SIMILARITY ANALYSIS
# -------------------------------
def calculate_ngram_similarity(essay, reference_text, n=3):
    """Calculate n-gram overlap similarity"""
    essay_tokens = preprocess_text(essay)
    ref_tokens = preprocess_text(reference_text)
    
    if not essay_tokens:
        return 0.0
    
    essay_ngrams = set(ngrams(essay_tokens, n))
    ref_ngrams = set(ngrams(ref_tokens, n))
    
    if not essay_ngrams:
        return 0.0
    
    overlap = essay_ngrams.intersection(ref_ngrams)
    similarity = len(overlap) / len(essay_ngrams)
    return similarity

def analyze_multiple_ngrams(essay, reference_text, ngram_sizes=[2, 3, 4, 5]):
    """Analyze multiple n-gram sizes"""
    ngram_results = {}
    
    print("📊 N-GRAM SIMILARITY ANALYSIS")
    print("-" * 50)
    
    for n in ngram_sizes:
        score = calculate_ngram_similarity(essay, reference_text, n=n)
        ngram_results[n] = score
        print(f"   {n}-gram similarity: {score:.4f} ({score*100:.2f}%)")
    
    avg_score = np.mean(list(ngram_results.values()))
    print(f"   📈 Average N-gram Score: {avg_score:.4f} ({avg_score*100:.2f}%)")
    
    return ngram_results, avg_score

# -------------------------------
# STYLOMETRIC ANALYSIS
# -------------------------------
def extract_stylometric_features(text):
    """Extract stylometric features from text"""
    sentences = nltk.sent_tokenize(str(text))
    words = nltk.word_tokenize(str(text).lower())
    stop_words = set(stopwords.words('english'))
    words_alpha = [w for w in words if w.isalpha()]
    
    if not words_alpha or not sentences:
        return np.zeros(7)
    
    # Calculate features
    avg_word_length = np.mean([len(w) for w in words_alpha])
    avg_sentence_length = np.mean([len(nltk.word_tokenize(s)) for s in sentences])
    type_token_ratio = len(set(words_alpha)) / (len(words_alpha) + 1)
    stopword_ratio = len([w for w in words if w in stop_words]) / (len(words) + 1)
    punctuation_ratio = len([w for w in words if w in string.punctuation]) / (len(words) + 1)
    unique_words_ratio = len(set(words_alpha)) / (len(words_alpha) + 1)
    sentence_count = len(sentences)
    
    return np.array([
        avg_word_length, avg_sentence_length, type_token_ratio, 
        stopword_ratio, punctuation_ratio, unique_words_ratio, sentence_count
    ])

def calculate_stylometric_similarity(essay, reference_text):
    """Calculate stylometric similarity using cosine similarity"""
    essay_features = extract_stylometric_features(essay).reshape(1, -1)
    ref_features = extract_stylometric_features(reference_text).reshape(1, -1)
    
    # Handle edge cases
    if np.any(np.isnan(essay_features)) or np.any(np.isnan(ref_features)):
        return 0.0, essay_features.flatten(), ref_features.flatten()
    
    similarity = cosine_similarity(essay_features, ref_features)[0][0]
    return max(0.0, float(similarity)), essay_features.flatten(), ref_features.flatten()

def analyze_stylometric_features(essay, reference_text):
    """Perform detailed stylometric analysis"""
    print("\n✍️  STYLOMETRIC SIMILARITY ANALYSIS")
    print("-" * 50)
    
    similarity, essay_features, ref_features = calculate_stylometric_similarity(essay, reference_text)
    
    print(f"   Stylometric similarity: {similarity:.4f} ({similarity*100:.2f}%)")
    
    feature_names = [
        "Avg Word Length", "Avg Sentence Length", "Type-Token Ratio",
        "Stopword Ratio", "Punctuation Ratio", "Unique Words Ratio", "Sentence Count"
    ]
    
    print("\n   📋 Detailed Feature Comparison:")
    print("   " + "="*60)
    print(f"   {'Feature':<20} {'Essay':<12} {'Reference':<12} {'Difference':<10}")
    print("   " + "-"*60)
    
    for i, name in enumerate(feature_names):
        if i < len(essay_features) and i < len(ref_features):
            diff = abs(essay_features[i] - ref_features[i])
            print(f"   {name:<20} {essay_features[i]:<12.3f} {ref_features[i]:<12.3f} {diff:<10.3f}")
    
    return similarity, essay_features, ref_features

# -------------------------------
# COMPREHENSIVE ANALYSIS FUNCTION
# -------------------------------
def comprehensive_analysis(essay, reference_text):
    """Perform complete analysis combining both methods"""
    
    # Header
    print("=" * 80)
    print("🔍 COMPREHENSIVE PLAGIARISM DETECTION ANALYSIS")
    print("=" * 80)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📄 Essay Statistics:")
    print(f"   • Length: {len(essay)} characters")
    print(f"   • Words: {len(essay.split())} words")
    print(f"   • Sentences: {len(nltk.sent_tokenize(essay))} sentences")
    print(f"📚 Reference Text: {len(reference_text):,} characters")
    print()
    
    # N-gram Analysis
    ngram_results, avg_ngram_score = analyze_multiple_ngrams(essay, reference_text)
    
    # Stylometric Analysis  
    stylometric_score, essay_features, ref_features = analyze_stylometric_features(essay, reference_text)
    
    # Combined Analysis
    print("\n🎯 COMBINED ANALYSIS RESULTS")
    print("-" * 50)
    
    # Weighted combination (adjustable weights)
    ngram_weight = 0.6
    stylometric_weight = 0.4
    combined_score = (avg_ngram_score * ngram_weight) + (stylometric_score * stylometric_weight)
    
    print(f"   📊 N-gram Component:      {avg_ngram_score:.4f} ({ngram_weight*100:.0f}% weight)")
    print(f"   ✍️  Stylometric Component: {stylometric_score:.4f} ({stylometric_weight*100:.0f}% weight)")
    print(f"   🎯 Final Combined Score:  {combined_score:.4f} ({combined_score*100:.2f}%)")
    
    
    # Detailed Score Breakdown
    print(f"\n📈 SCORE BREAKDOWN")
    print("-" * 50)
    score_bar_length = 50
    filled_length = int(score_bar_length * combined_score)
    bar = '█' * filled_length + '░' * (score_bar_length - filled_length)
    print(f"   Combined Score: |{bar}| {combined_score:.1%}")
    
    # Individual component bars
    ngram_filled = int(score_bar_length * avg_ngram_score)
    ngram_bar = '█' * ngram_filled + '░' * (score_bar_length - ngram_filled)
    print(f"   N-gram Score:   |{ngram_bar}| {avg_ngram_score:.1%}")
    
    stylo_filled = int(score_bar_length * stylometric_score)
    stylo_bar = '█' * stylo_filled + '░' * (score_bar_length - stylo_filled)
    print(f"   Stylometric:    |{stylo_bar}| {stylometric_score:.1%}")
    

    
    print("\n" + "=" * 80)
    print("📊 ANALYSIS COMPLETE")
    print("=" * 80)
    
    return {
        'ngram_scores': ngram_results,
        'avg_ngram_score': avg_ngram_score,
        'stylometric_score': stylometric_score,
        'combined_score': combined_score,
    }

# -------------------------------
# LOAD REFERENCE DATA
# -------------------------------
file_path = r"D:\CopyDetection-System-main\Excel_Dataset\processed_books_dataset-1.xlsx"

print("Loading reference data...")
try:
    df = pd.read_excel(file_path)
    reference_text = " ".join(df["text_content"].dropna().astype(str))
    print(f"✅ Reference data loaded successfully: {len(reference_text):,} characters")
except Exception as e:
    print(f"❌ Error loading reference data: {e}")
    reference_text = ""

# -------------------------------
# SAMPLE ESSAY FOR TESTING
# -------------------------------
student_essay = """
In a peaceful region surrounded by gentle hills, there lived a simple people who valued comfort and joy.  
Their lives were marked by festivals, good food, and stories shared by the hearth.  
However, far away in the dark lands, an old evil began to rise once more.  
Legends spoke of a golden ring with immense power, capable of destroying all who opposed it.  
When that ring came into the hands of an ordinary wanderer, his adventure turned into a mission that could decide the fate of every land.
The journey was long and perilous, filled with unexpected allies and dangerous enemies.
Through mountains and forests, across rivers and plains, the fellowship traveled with determination.
Each step brought them closer to their destiny, but also deeper into peril.
Magic and wisdom guided their path, while darkness threatened to consume everything they held dear.
"""

# -------------------------------
# RUN ANALYSIS
# -------------------------------
if reference_text:  # Only run if reference data loaded successfully
    results = comprehensive_analysis(student_essay, reference_text)
else:
    print("Cannot perform analysis without reference data.")