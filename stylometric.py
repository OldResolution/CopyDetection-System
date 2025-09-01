import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import string

# Download NLTK resources (only first time)
nltk.download('punkt')
nltk.download('punkt_tab')   # required in NLTK>=3.9
nltk.download('stopwords')

# -------------------------------
# Stylometric Feature Extraction
# -------------------------------
def stylometric_features(text):
    sentences = nltk.sent_tokenize(str(text))
    words = nltk.word_tokenize(str(text).lower())
    stop_words = set(stopwords.words("english"))
    words_alpha = [w for w in words if w.isalpha()]  # keep only words

    if not words_alpha or not sentences:
        return np.zeros(5)  # avoid crash on empty input

    avg_word_length = np.mean([len(w) for w in words_alpha])
    avg_sentence_length = np.mean([len(nltk.word_tokenize(s)) for s in sentences])
    type_token_ratio = len(set(words_alpha)) / (len(words_alpha)+1)
    stopword_ratio = len([w for w in words if w in stop_words]) / (len(words)+1)
    punctuation_ratio = len([w for w in words if w in string.punctuation]) / (len(words)+1)

    return np.array([avg_word_length, avg_sentence_length, type_token_ratio, stopword_ratio, punctuation_ratio])

# -------------------------------
# Stylometry Comparison
# -------------------------------
def stylometry_similarity(text1, text2):
    f1 = stylometric_features(text1).reshape(1, -1)
    f2 = stylometric_features(text2).reshape(1, -1)
    return float(cosine_similarity(f1, f2)[0][0])  # similarity between 0–1

# -------------------------------
# Load Reference from Excel
# -------------------------------
file_path = r"D:\CopyDetection-System-main\Excel_Dataset\processed_books_dataset-1.xlsx"

df = pd.read_excel(file_path)

print("✅ Excel loaded with columns:", df.columns)

# Combine all rows in 'text_content' into one big string
reference_text = " ".join(df["text_content"].dropna().astype(str))

print("✅ Length of reference text:", len(reference_text))

# -------------------------------
# Example Student Essay
# -------------------------------
student_essay = """
In a peaceful region surrounded by gentle hills, there lived a simple people who valued comfort and joy.
Their lives were marked by festivals, good food, and stories shared by the hearth.
However, far away in the dark lands, an old evil began to rise once more.
"""

# -------------------------------
# Run Stylometric Similarity
# -------------------------------
score = stylometry_similarity(student_essay, reference_text)

print(f"🎯 Stylometric Similarity Score: {score:.2f}")