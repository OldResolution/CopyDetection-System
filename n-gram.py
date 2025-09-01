import pandas as pd
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def preprocess(text):
    tokens = nltk.word_tokenize(str(text).lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens
# -------------------------------
# Preprocess text
# -------------------------------
def preprocess(text):
    tokens = nltk.word_tokenize(str(text).lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

# -------------------------------
# N-gram Overlap
# -------------------------------
def ngram_overlap(essay, reference_text, n=5):
    essay_tokens = preprocess(essay)
    ref_tokens = preprocess(reference_text)

    essay_ngrams = set(ngrams(essay_tokens, n))
    ref_ngrams = set(ngrams(ref_tokens, n))

    if not essay_ngrams:
        return 0

    overlap = essay_ngrams.intersection(ref_ngrams)
    similarity = len(overlap) / len(essay_ngrams)
    return similarity

# -------------------------------
# Load Reference Text from Excel
# -------------------------------
file_path = r"D:\CopyDetection-System-main\Excel_Dataset\processed_books_dataset-1.xlsx"

df = pd.read_excel(file_path)

# Combine all rows of 'text_content' column into one reference text
reference_text = " ".join(df["text_content"].dropna().astype(str))

# -------------------------------
# Example Essay to Compare
# -------------------------------
student_essay = """
In a peaceful region surrounded by gentle hills, there lived a simple people who valued comfort and joy.  
Their lives were marked by festivals, good food, and stories shared by the hearth.  
However, far away in the dark lands, an old evil began to rise once more.  
Legends spoke of a golden ring with immense power, capable of destroying all who opposed it.  
When that ring came into the hands of an ordinary wanderer, his adventure turned into a mission that could decide the fate of every land.
"""

score = ngram_overlap(student_essay, reference_text, n=3)  # using 4-grams
print(f"N-gram Similarity Score: {score:.2f}")