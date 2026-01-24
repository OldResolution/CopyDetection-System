import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# ---------------------------
# Load spaCy model (for stylometry)
# ---------------------------
nlp = spacy.load("en_core_web_sm")

# ---------------------------
# Stylometry Functions
# ---------------------------
def extract_stylometry(text):
    """
    Extract stylometric features from text.
    """
    doc = nlp(text)
    sentences = list(doc.sents)
    words = [token.text for token in doc if not token.is_space and not token.is_punct]

    return {
        "sentence_count": len(sentences),
        "word_count": len(words),
        "avg_sentence_length": round(np.mean([len(sent.text.split()) for sent in sentences]), 2) if sentences else 0,
        "avg_word_length": round(np.mean([len(word) for word in words]), 2) if words else 0,
        "type_token_ratio": round(len(set(words)) / len(words), 3) if words else 0
    }

def compare_stylometry(ai_text, ref_text):
    """
    Compare stylometric features of AI-generated text vs reference paragraph.
    """
    ai_features = extract_stylometry(ai_text)
    ref_features = extract_stylometry(ref_text)
    return ai_features, ref_features

# ---------------------------
# Copyright Detection Functions
# ---------------------------
def detect_copyright(ai_text, df, vectorizer, tfidf_matrix, top_n=3):
    """
    Compare AI-generated text against reference dataset
    using TF-IDF + cosine similarity.
    """
    # Convert AI text into TF-IDF vector
    ai_vec = vectorizer.transform([ai_text])

    # Compute cosine similarity (AI vs all paragraphs)
    sim_scores = cosine_similarity(ai_vec, tfidf_matrix)[0]

    # Get indexes of top N most similar paragraphs
    top_matches = sim_scores.argsort()[-top_n:][::-1]

    # Collect results
    results = []
    for idx in top_matches:
        results.append({
            "similarity_score": round(sim_scores[idx], 3),
            "author": df['author'].iloc[idx],
            "book_title": df['book_title'].iloc[idx],
            "reference_text": df['text_content'].iloc[idx][:300] + "..."
        })
    return results

# ---------------------------
# Main Program
# ---------------------------
if __name__ == "__main__":
    # Step 1: Load dataset
    dataset_path = "excel_dataset/processed_books_dataset-1.xlsx"
    print("📂 Loading dataset from:", dataset_path)
    df = pd.read_excel(dataset_path)
    print("✅ Dataset loaded:", df.shape, "rows")

    # Step 2: Build TF-IDF matrix
    print("\n⚙️ Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['text_content'])
    print("✅ TF-IDF matrix built. Shape:", tfidf_matrix.shape)

    # Step 3: Example AI-generated text
    ai_output = """
    The spaceship hovered silently above the city, 
    casting shadows that stretched for miles across the desert.
    Its arrival marked the beginning of a new era for humankind.
    """

    # Step 4: Run copyright detection
    matches = detect_copyright(ai_output, df, vectorizer, tfidf_matrix, top_n=3)

    print("\n🔎 Top Matches (TF-IDF Similarity):")
    for m in matches:
        print(f"Score: {m['similarity_score']} | Author: {m['author']} | Book: {m['book_title']}")
        print(f"Ref: {m['reference_text']}\n")

    # Step 5: Stylometry comparison with best match
    best_ref_text = matches[0]["reference_text"]
    ai_stylo, ref_stylo = compare_stylometry(ai_output, best_ref_text)

    print("✍️ Stylometric Comparison (AI vs Best Match):")
    print("AI Output Features:", ai_stylo)
    print("Reference Features:", ref_stylo)