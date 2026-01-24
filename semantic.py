import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# Load Pretrained Embedding Model
# -------------------------------
print("⚙ Loading Sentence Transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded.")

# -------------------------------
# Semantic Similarity Function
# -------------------------------
def semantic_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return similarity

# -------------------------------
# Load Reference Text from Excel
# -------------------------------
file_path = r"D:\CopyDetection-System-main\Excel_Dataset\processed_books_dataset-1.xlsx"

print(f"📂 Loading dataset from {file_path} ...")
df = pd.read_excel(file_path)
print("✅ Dataset loaded with shape:", df.shape)

# Use all rows in text_content column
if "text_content" not in df.columns:
    raise ValueError("❌ Column 'text_content' not found in Excel file.")

reference_text = " ".join(df["text_content"].dropna().astype(str))
print("✅ Reference text length:", len(reference_text))

# -------------------------------
# Example Student Essay
# -------------------------------
student_essay = """
The small folk lived peacefully in their hills, enjoying simple joys and celebrations,
but a great darkness was rising in a faraway land, threatening to change their fate forever.
"""

print("📝 Student essay ready. Length:", len(student_essay))

# -------------------------------
# Run Semantic Similarity
# -------------------------------
score = semantic_similarity(student_essay, reference_text)
print(f"\n🔎 Semantic Similarity Score: {score:.2f}")