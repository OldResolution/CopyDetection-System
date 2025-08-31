from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == "__main__":
    # Load dataset
    df = pd.read_excel("Excel_Dataset/processed_books_dataset-1.xlsx")

    # Step 2: Build TF-IDF matrix
    print("\n⚙️ Building TF-IDF matrix...")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['text_content'])

    print("✅ TF-IDF matrix built.")
    print("Matrix shape:", tfidf_matrix.shape)


def detect_copyright(ai_text, top_n=3):
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
            "reference_text": df['text_content'].iloc[idx][:300] + "..."  # first 300 chars
        })

    return results


if __name__ == "__main__":
    # Step 1: Load dataset
    df = pd.read_excel("excel_dataset/processed_books_dataset-1.xlsx")

    # Step 2: Build TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['text_content'])

    # Step 3: Example AI-generated text
    ai_output = """
    The ring had a mysterious power, whispered about in dark corners of the Shire.
    Those who touched it felt a strange burden, as if their fate was no longer their own.
    """

    # Run detection
    matches = detect_copyright(ai_output, top_n=3)

    print("\n🔎 Top Matches:")
    for m in matches:
        print(f"Score: {m['similarity_score']} | Ref: {m['reference_text']}")

