import pandas as pd
import nltk
import numpy as np
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import string
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

class ComprehensivePlagiarismDetector:
    def __init__(self, reference_file_path, model_name="all-MiniLM-L6-v2"):
        """Initialize the detector with reference data and semantic model"""
        self.reference_file_path = reference_file_path
        self.reference_text = self._load_reference_data()
        self.stop_words = set(stopwords.words('english'))
        
        # Load semantic similarity model
        print(" Loading Sentence Transformer model...")
        self.semantic_model = SentenceTransformer(model_name)
        print(" Semantic model loaded successfully.")
        
    def _load_reference_data(self):
        """Load and combine reference text from Excel file"""
        try:
            df = pd.read_excel(self.reference_file_path)
            reference_text = " ".join(df["text_content"].dropna().astype(str))
            print(f" Reference data loaded: {len(reference_text):,} characters")
            return reference_text
        except Exception as e:
            print(f" Error loading reference data: {e}")
            return ""
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        tokens = nltk.word_tokenize(str(text).lower())
        tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
        return tokens
    
    def calculate_ngram_similarity(self, essay, n=3):
        """Calculate n-gram overlap similarity"""
        essay_tokens = self.preprocess_text(essay)
        ref_tokens = self.preprocess_text(self.reference_text)
        
        if not essay_tokens:
            return 0.0
        
        essay_ngrams = set(ngrams(essay_tokens, n))
        ref_ngrams = set(ngrams(ref_tokens, n))
        
        if not essay_ngrams:
            return 0.0
        
        overlap = essay_ngrams.intersection(ref_ngrams)
        similarity = len(overlap) / len(essay_ngrams)
        return similarity
    
    def extract_stylometric_features(self, text):
        """Extract stylometric features from text"""
        sentences = nltk.sent_tokenize(str(text))
        words = nltk.word_tokenize(str(text).lower())
        words_alpha = [w for w in words if w.isalpha()]
        
        if not words_alpha or not sentences:
            return np.zeros(7)
        
        # Core stylometric features
        avg_word_length = np.mean([len(w) for w in words_alpha])
        avg_sentence_length = np.mean([len(nltk.word_tokenize(s)) for s in sentences])
        type_token_ratio = len(set(words_alpha)) / (len(words_alpha) + 1)
        stopword_ratio = len([w for w in words if w in self.stop_words]) / (len(words) + 1)
        punctuation_ratio = len([w for w in words if w in string.punctuation]) / (len(words) + 1)
        
        # Additional features
        unique_words_ratio = len(set(words_alpha)) / (len(words_alpha) + 1)
        sentence_count = len(sentences)
        
        return np.array([
            avg_word_length, avg_sentence_length, type_token_ratio, 
            stopword_ratio, punctuation_ratio, unique_words_ratio, sentence_count
        ])
    
    def calculate_stylometric_similarity(self, essay):
        """Calculate stylometric similarity using cosine similarity"""
        essay_features = self.extract_stylometric_features(essay).reshape(1, -1)
        ref_features = self.extract_stylometric_features(self.reference_text).reshape(1, -1)
        
        # Handle edge cases
        if np.any(np.isnan(essay_features)) or np.any(np.isnan(ref_features)):
            return 0.0
        
        similarity = cosine_similarity(essay_features, ref_features)[0][0]
        return max(0.0, float(similarity))  # Ensure non-negative
    
    def calculate_semantic_similarity(self, essay):
        """Calculate semantic similarity using sentence transformers"""
        try:
            # Encode both texts
            essay_embedding = self.semantic_model.encode(essay, convert_to_tensor=True)
            reference_embedding = self.semantic_model.encode(self.reference_text, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = util.pytorch_cos_sim(essay_embedding, reference_embedding).item()
            return max(0.0, float(similarity))  # Ensure non-negative
        except Exception as e:
            print(f"⚠ Semantic analysis error: {e}")
            return 0.0
    
    def analyze_text(self, essay, ngram_sizes=[2, 3, 4, 5]):
        """Perform comprehensive analysis combining all three methods"""
        print("=" * 80)
        print(" COMPREHENSIVE PLAGIARISM DETECTION ANALYSIS")
        print("=" * 80)
        print(f" Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Essay Length: {len(essay)} characters, {len(essay.split())} words")
        print()
        
        # N-gram Analysis
        print(" N-GRAM SIMILARITY ANALYSIS")
        print("-" * 50)
        ngram_scores = {}
        
        for n in ngram_sizes:
            score = self.calculate_ngram_similarity(essay, n=n)
            ngram_scores[n] = score
            print(f"   {n}-gram similarity: {score:.4f} ({score*100:.2f}%)")
        
        avg_ngram_score = np.mean(list(ngram_scores.values()))
        print(f"    Average N-gram Score: {avg_ngram_score:.4f} ({avg_ngram_score*100:.2f}%)")
        print()
        
        # Stylometric Analysis
        print("STYLOMETRIC SIMILARITY ANALYSIS")
        print("-" * 50)
        stylometric_score = self.calculate_stylometric_similarity(essay)
        print(f"   Stylometric similarity: {stylometric_score:.4f} ({stylometric_score*100:.2f}%)")
        
        # Feature comparison
        essay_features = self.extract_stylometric_features(essay)
        ref_features = self.extract_stylometric_features(self.reference_text)
        
        feature_names = [
            "Avg Word Length", "Avg Sentence Length", "Type-Token Ratio",
            "Stopword Ratio", "Punctuation Ratio", "Unique Words Ratio", "Sentence Count"
        ]
        
        print("\n    Detailed Feature Comparison:")
        for i, name in enumerate(feature_names):
            if i < len(essay_features) and i < len(ref_features):
                print(f"      {name:18}: Essay={essay_features[i]:.3f}, Reference={ref_features[i]:.3f}")
        print()
        
        # Semantic Analysis
        print(" SEMANTIC SIMILARITY ANALYSIS")
        print("-" * 50)
        semantic_score = self.calculate_semantic_similarity(essay)
        print(f"   Semantic similarity: {semantic_score:.4f} ({semantic_score*100:.2f}%)")
        print("    Note: Measures meaning and context similarity using AI embeddings")
        print()
        
        # Combined Analysis
        print(" COMBINED ANALYSIS RESULTS")
        print("-" * 50)
        
        # Weighted combination (you can adjust these weights)
        combined_score = (avg_ngram_score * 0.4) + (stylometric_score * 0.3) + (semantic_score * 0.3)
        
        print(f"   N-gram Component:      {avg_ngram_score:.4f} (40% weight)")
        print(f"  Stylometric Component: {stylometric_score:.4f} (30% weight)")
        print(f"   Semantic Component:    {semantic_score:.4f} (30% weight)")
        print(f"   Combined Score:        {combined_score:.4f} ({combined_score*100:.2f}%)")
        print()
        
        # Method comparison and insights
        print("🔬 METHOD-SPECIFIC INSIGHTS")
        print("-" * 50)
        scores = [avg_ngram_score, stylometric_score, semantic_score]
        methods = ["N-gram", "Stylometric", "Semantic"]
        
        max_method = methods[np.argmax(scores)]
        max_score = max(scores)
        
        print(f"   Highest similarity detected by: {max_method} ({max_score:.4f})")
        
        if semantic_score > avg_ngram_score and semantic_score > stylometric_score:
            print("    Semantic analysis shows high conceptual similarity")
            print("    May indicate paraphrasing or idea borrowing")
        elif avg_ngram_score > semantic_score and avg_ngram_score > stylometric_score:
            print("    N-gram analysis shows high textual overlap")
            print("    May indicate direct copying or close paraphrasing")
        elif stylometric_score > semantic_score and stylometric_score > avg_ngram_score:
            print("   Stylometric analysis shows similar writing patterns")
            print("    May indicate same author or learned writing style")
        print()
        
        # Recommendations
        print(" RECOMMENDATIONS")
        print("-" * 50)
        if combined_score >= 0.7:
            print("   ⚠ HIGH plagiarism risk detected!")
            print("   • Review the text thoroughly for potential copying")
            print("   • Check for proper citations and references")
            print("   • Consider manual verification of suspicious sections")
            if semantic_score > 0.7:
                print("   • High semantic similarity suggests conceptual borrowing")
            if avg_ngram_score > 0.7:
                print("   • High n-gram overlap suggests direct text copying")
        elif combined_score >= 0.4:
            print("   ⚠ MEDIUM plagiarism risk detected")
            print("   • Some similarities found that warrant investigation")
            print("   • Verify originality of key passages")
            print("   • Ensure proper attribution where needed")
            if semantic_score > stylometric_score and semantic_score > avg_ngram_score:
                print("   • Focus on semantic similarity - check for paraphrasing")
        elif combined_score >= 0.2:
            print("   ℹ LOW plagiarism risk")
            print("   • Minor similarities detected, likely coincidental")
            print("   • Standard review process recommended")
        else:
            print("    MINIMAL plagiarism risk")
            print("   • Text appears to be original")
            print("   • No significant similarities detected")
        
        print("=" * 80)
        
        # Return results for further processing if needed
        return {
            'ngram_scores': ngram_scores,
            'avg_ngram_score': avg_ngram_score,
            'stylometric_score': stylometric_score,
            'semantic_score': semantic_score,
            'combined_score': combined_score,
            'essay_features': essay_features,
            'reference_features': ref_features,
            'feature_names': feature_names
        }
    
    def create_text_visualization(self, results):
        """Create ASCII-based visualization of analysis results"""
        print("\n" + "=" * 80)
        print(" DETAILED ANALYSIS VISUALIZATION")
        print("=" * 80)
        
        # 1. N-gram scores bar chart (ASCII)
        print("\n N-GRAM SIMILARITY SCORES:")
        print("-" * 50)
        for n, score in results['ngram_scores'].items():
            bar_length = int(score * 40)  # Scale to 40 chars max
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"   {n}-gram: |{bar}| {score:.4f} ({score*100:.1f}%)")
        
        avg_bar_length = int(results['avg_ngram_score'] * 40)
        avg_bar = "█" * avg_bar_length + "░" * (40 - avg_bar_length)
        print(f"   Average: |{avg_bar}| {results['avg_ngram_score']:.4f} ({results['avg_ngram_score']*100:.1f}%)")
        
        # 2. Method comparison
        print("\n METHOD COMPARISON:")
        print("-" * 50)
        methods = [
            ("N-gram Analysis", results['avg_ngram_score'], "40% weight"),
            ("Stylometric Analysis", results['stylometric_score'], "30% weight"),
            ("Semantic Analysis", results['semantic_score'], "30% weight"),
            ("Combined Score", results['combined_score'], "Final Result")
        ]
        
        for method, score, weight in methods:
            bar_length = int(score * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"   {method:20}: |{bar}| {score:.4f} ({weight})")
        
        # 3. Feature comparison table
        print("\n STYLOMETRIC FEATURES COMPARISON:")
        print("-" * 80)
        print(f"{'Feature':<25} {'Essay Value':<15} {'Reference Value':<15} {'Difference':<15}")
        print("-" * 80)
        
        essay_features = results['essay_features']
        ref_features = results['reference_features']
        feature_names = results['feature_names']
        
        for i, name in enumerate(feature_names):
            if i < len(essay_features) and i < len(ref_features):
                diff = abs(essay_features[i] - ref_features[i])
                print(f"{name:<25} {essay_features[i]:<15.3f} {ref_features[i]:<15.3f} {diff:<15.3f}")
        
        # 4. Statistical summary
        print("\n STATISTICAL SUMMARY:")
        print("-" * 50)
        stats = [
            ("Highest N-gram Score", max(results['ngram_scores'].values())),
            ("Lowest N-gram Score", min(results['ngram_scores'].values())),
            ("N-gram Score Range", max(results['ngram_scores'].values()) - min(results['ngram_scores'].values())),
            ("Stylometric Similarity", results['stylometric_score']),
            ("Semantic Similarity", results['semantic_score']),
            ("Combined Final Score", results['combined_score']),
        ]
        
        for stat_name, stat_value in stats:
            if isinstance(stat_value, str):
                print(f"   {stat_name:25}: {stat_value}")
            else:
                print(f"   {stat_name:25}: {stat_value:.4f}")
        
        print("=" * 80)


# HARDCODED REFERENCE FILE PATH - Update this path to your reference Excel file
REFERENCE_FILE_PATH = r"D:\CopyDetection-System-main\Excel_Dataset\processed_books_dataset-1.xlsx"


def get_multiline_input():
    """Get multi-line input from user with improved instructions"""
    print("\n Enter or paste your essay text below.")
    print("   You can paste multiple paragraphs.")
    print("   When finished, type 'END' on a new line and press Enter.")
    print("-" * 80)
    
    lines = []
    try:
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)
    except EOFError:
        pass
    
    return '\n'.join(lines)


def get_input_method():
    """Ask user how they want to provide the essay text"""
    print("\n How would you like to provide the essay text?")
    print("   1. Type/paste text directly")
    print("   2. Load from a text file")
    
    while True:
        choice = input("\n   Enter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return choice
        print("    Invalid choice. Please enter 1 or 2.")
        
        # Remove quotes if user wrapped the path in quotes
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        elif file_path.startswith("'") and file_path.endswith("'"):
            file_path = file_path[1:-1]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f" File loaded: {len(text)} characters, {len(text.split())} words")
            return text
        except FileNotFoundError:
            print(f" Error: File not found at '{file_path}'")
            retry = input("   Would you like to try again? (yes/no): ").strip().lower()
            if retry not in ['yes', 'y']:
                return None
        except Exception as e:
            print(f" Error reading file: {e}")
            retry = input("   Would you like to try again? (yes/no): ").strip().lower()
            if retry not in ['yes', 'y']:
                return None


def get_reference_file_path():
    """Get the reference file path from user or use default"""
    print("\n Reference File Configuration")
    print("-" * 50)
    print(f"   Default path: {REFERENCE_FILE_PATH}")
    
    use_default = input("\n   Use default reference file path? (yes/no): ").strip().lower()
    
    if use_default in ['yes', 'y', '']:
        return REFERENCE_FILE_PATH
    
    while True:
        print("\n Enter the path to your reference Excel file:")
        user_path = input("   Path: ").strip()
        
        # Remove quotes if user wrapped the path in quotes
        if user_path.startswith('"') and user_path.endswith('"'):
            user_path = user_path[1:-1]
        elif user_path.startswith("'") and user_path.endswith("'"):
            user_path = user_path[1:-1]
        
        if os.path.exists(user_path):
            return user_path
        else:
            print(f" Error: File not found at '{user_path}'")
            retry = input("   Would you like to try again? (yes/no): ").strip().lower()
            if retry not in ['yes', 'y']:
                return None


def main():
    """Main function to run the plagiarism detector"""
    print("=" * 80)
    print(" COMPREHENSIVE PLAGIARISM DETECTION SYSTEM")
    print("=" * 80)
    print("\nWelcome! This tool analyzes text for potential plagiarism using:")
    print("  • N-gram analysis (text overlap)")
    print("  • Stylometric analysis (writing style)")
    print("  • Semantic analysis (meaning similarity)")
    
    # Get reference file path
    reference_path = get_reference_file_path()
    if not reference_path:
        print("\n Exiting program.")
        return
    
    try:
        # Initialize the detector
        detector = ComprehensivePlagiarismDetector(reference_path)
        
        while True:
            # Get input method
            input_method = get_input_method()
            
            # Get essay text based on user's choice
            if input_method == '1':
                student_essay = get_multiline_input()
                if not student_essay:
                    continue
            
            if not student_essay.strip():
                print("\n⚠ No text entered. Please try again.")
                continue
            
            # Perform comprehensive analysis
            results = detector.analyze_text(student_essay)
            
            # Create and display text-based visualization
            detector.create_text_visualization(results)
            
            print("\n ANALYSIS COMPLETE!")
            
            # Ask if user wants to analyze another essay
            print("\n" + "=" * 80)
            another = input("Would you like to analyze another essay? (yes/no): ").strip().lower()
            if another not in ['yes', 'y']:
                print("\n Thank you for using the Plagiarism Detection System!")
                break
            
    except Exception as e:
        print(f"\n An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()