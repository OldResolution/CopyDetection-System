import nltk
import re

# Safe NLTK Download
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")

try:
    STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
except Exception:
    STOP_WORDS = set()

def preprocess_text(text, remove_stopwords=True):
    try:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = nltk.word_tokenize(text)
        
        if remove_stopwords:
            return [t for t in tokens if t.isalpha() and len(t) > 2 and t not in STOP_WORDS]
        return [t for t in tokens if t.isalpha()]
    except:
        return []
