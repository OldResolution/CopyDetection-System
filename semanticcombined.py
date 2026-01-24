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
        print("⚙️ Loading Sentence Transformer model...")
        self.semantic_model = SentenceTransformer(model_name)
        print("✅ Semantic model loaded successfully.")
        
    def _load_reference_data(self):
        """Load and combine reference text from Excel file"""
        try:
            df = pd.read_excel(self.reference_file_path)
            reference_text = " ".join(df["text_content"].dropna().astype(str))
            print(f"✅ Reference data loaded: {len(reference_text):,} characters")
            return reference_text
        except Exception as e:
            print(f"❌ Error loading reference data: {e}")
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
            print(f"⚠️ Semantic analysis error: {e}")
            return 0.0
    
    def analyze_text(self, essay, ngram_sizes=[2, 3, 4, 5]):
        """Perform comprehensive analysis combining all three methods"""
        print("=" * 80)
        print("🔍 COMPREHENSIVE PLAGIARISM DETECTION ANALYSIS")
        print("=" * 80)
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📄 Essay Length: {len(essay)} characters, {len(essay.split())} words")
        print()
        
        # N-gram Analysis
        print("📊 N-GRAM SIMILARITY ANALYSIS")
        print("-" * 50)
        ngram_scores = {}
        
        for n in ngram_sizes:
            score = self.calculate_ngram_similarity(essay, n=n)
            ngram_scores[n] = score
            print(f"   {n}-gram similarity: {score:.4f} ({score*100:.2f}%)")
        
        avg_ngram_score = np.mean(list(ngram_scores.values()))
        print(f"   📈 Average N-gram Score: {avg_ngram_score:.4f} ({avg_ngram_score*100:.2f}%)")
        print()
        
        # Stylometric Analysis
        print("✍️ STYLOMETRIC SIMILARITY ANALYSIS")
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
        
        print("\n   📋 Detailed Feature Comparison:")
        for i, name in enumerate(feature_names):
            if i < len(essay_features) and i < len(ref_features):
                print(f"      {name:18}: Essay={essay_features[i]:.3f}, Reference={ref_features[i]:.3f}")
        print()
        
        # Semantic Analysis
        print("🧠 SEMANTIC SIMILARITY ANALYSIS")
        print("-" * 50)
        semantic_score = self.calculate_semantic_similarity(essay)
        print(f"   Semantic similarity: {semantic_score:.4f} ({semantic_score*100:.2f}%)")
        print("   📝 Note: Measures meaning and context similarity using AI embeddings")
        print()
        
        # Combined Analysis
        print("🎯 COMBINED ANALYSIS RESULTS")
        print("-" * 50)
        
        # Weighted combination (you can adjust these weights)
        # Updated to include semantic analysis: 40% n-gram + 30% stylometric + 30% semantic
        combined_score = (avg_ngram_score * 0.4) + (stylometric_score * 0.3) + (semantic_score * 0.3)
        
        print(f"   📊 N-gram Component:      {avg_ngram_score:.4f} (40% weight)")
        print(f"   ✍️ Stylometric Component: {stylometric_score:.4f} (30% weight)")
        print(f"   🧠 Semantic Component:    {semantic_score:.4f} (30% weight)")
        print(f"   🎯 Combined Score:        {combined_score:.4f} ({combined_score*100:.2f}%)")
        print()
        
        # Method comparison and insights
        print("🔬 METHOD-SPECIFIC INSIGHTS")
        print("-" * 50)
        scores = [avg_ngram_score, stylometric_score, semantic_score]
        methods = ["N-gram", "Stylometric", "Semantic"]
        
        max_method = methods[np.argmax(scores)]
        max_score = max(scores)
        
        print(f"   🥇 Highest similarity detected by: {max_method} ({max_score:.4f})")
        
        if semantic_score > avg_ngram_score and semantic_score > stylometric_score:
            print("   🧠 Semantic analysis shows high conceptual similarity")
            print("   💡 May indicate paraphrasing or idea borrowing")
        elif avg_ngram_score > semantic_score and avg_ngram_score > stylometric_score:
            print("   📊 N-gram analysis shows high textual overlap")
            print("   💡 May indicate direct copying or close paraphrasing")
        elif stylometric_score > semantic_score and stylometric_score > avg_ngram_score:
            print("   ✍️ Stylometric analysis shows similar writing patterns")
            print("   💡 May indicate same author or learned writing style")
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS")
        print("-" * 50)
        if combined_score >= 0.7:
            print("   ⚠️ HIGH plagiarism risk detected!")
            print("   • Review the text thoroughly for potential copying")
            print("   • Check for proper citations and references")
            print("   • Consider manual verification of suspicious sections")
            if semantic_score > 0.7:
                print("   • High semantic similarity suggests conceptual borrowing")
            if avg_ngram_score > 0.7:
                print("   • High n-gram overlap suggests direct text copying")
        elif combined_score >= 0.4:
            print("   ⚠️ MEDIUM plagiarism risk detected")
            print("   • Some similarities found that warrant investigation")
            print("   • Verify originality of key passages")
            print("   • Ensure proper attribution where needed")
            if semantic_score > stylometric_score and semantic_score > avg_ngram_score:
                print("   • Focus on semantic similarity - check for paraphrasing")
        elif combined_score >= 0.2:
            print("   ℹ️ LOW plagiarism risk")
            print("   • Minor similarities detected, likely coincidental")
            print("   • Standard review process recommended")
        else:
            print("   ✅ MINIMAL plagiarism risk")
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
        print("📈 DETAILED ANALYSIS VISUALIZATION")
        print("=" * 80)
        
        # 1. N-gram scores bar chart (ASCII)
        print("\n📊 N-GRAM SIMILARITY SCORES:")
        print("-" * 50)
        for n, score in results['ngram_scores'].items():
            bar_length = int(score * 40)  # Scale to 40 chars max
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"   {n}-gram: |{bar}| {score:.4f} ({score*100:.1f}%)")
        
        avg_bar_length = int(results['avg_ngram_score'] * 40)
        avg_bar = "█" * avg_bar_length + "░" * (40 - avg_bar_length)
        print(f"   Average: |{avg_bar}| {results['avg_ngram_score']:.4f} ({results['avg_ngram_score']*100:.1f}%)")
        
        # 2. Method comparison
        print("\n🎯 METHOD COMPARISON:")
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
        print("\n📋 STYLOMETRIC FEATURES COMPARISON:")
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
        
        # 4. Risk assessment gauge (ASCII)
        print("\n🚨 RISK ASSESSMENT GAUGE:")
        print("-" * 50)
        risk_score = results['combined_score']
        gauge_length = 60
        
        # 5. Statistical summary
        print("\n📊 STATISTICAL SUMMARY:")
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

if __name__ == "__main__":
    # Initialize the detector
    file_path = r"D:\CopyDetection-System-main\Excel_Dataset\processed_books_dataset-1.xlsx"
    detector = ComprehensivePlagiarismDetector(file_path)
    
    # Sample essay for testing
    student_essay = """
The Hobbits, a remarkable and often underestimated race, are distinguished not by power, wealth, or magical ability, but by a set of unique qualities that allowed them to thrive quietly alongside larger and more noticeable peoples. One of their most impressive abilities, noted from the earliest times, was their gift of disappearing quickly and silently when confronted with the approach of “big folk.” To outsiders—particularly Men—this ability appeared almost magical, as though the Hobbits had mastered spells of invisibility. Yet in truth, no magic was ever involved. The Hobbits never studied sorcery or even attempted to practice the secret arts pursued by Elves, Wizards, or other learned peoples. Their gift for elusiveness was the result of tradition, heredity, and careful practice. Generations of living close to the earth, developing instincts, and perfecting the art of moving quietly gave Hobbits a natural skill unmatched by any other race. It was a professional art, born of necessity and culture, that became second nature to them.Physically, Hobbits were small in stature, a fact that set them apart from nearly every other people of Middle-earth. Even compared to the Dwarves, who were known for their short and sturdy frames, Hobbits were smaller, though less stocky in build. They were not as muscular or rugged as Dwarves, nor did they possess the towering height of Men or the elegance of Elves. Typically, their height ranged between two and four feet. In modern times, however, they seldom reached even three feet in height. According to their own traditions and the stories handed down through the ages, their ancestors were taller, though gradual change had reduced their size over time. This decline in stature was accepted without complaint, for Hobbits valued comfort and practicality far more than physical grandeur. Their smallness allowed them to live unnoticed, hidden from the grand conflicts and struggles that often consumed larger races.The daily lives of Hobbits reflected the simplicity of their nature. They preferred the countryside to cities, enjoyed farming and gardening, and had a natural bond with the soil. Their hands were skilled at growing food, and their tables were always rich with simple but plentiful meals. Music, storytelling, and fellowship were central to their culture, and they valued traditions that brought families and neighbors together. Their homes were humble, often built into hillsides or nestled in quiet valleys, further blending them into the natural world. This way of life reinforced their tendency to avoid notice, for they did not seek attention or power. To many, their contentment with such a modest existence seemed peculiar, but it was precisely this quality that gave Hobbits their quiet strength.What outsiders often mistook for weakness was, in fact, a different kind of resilience. Hobbits did not need to wield swords or study enchantments to survive; instead, they relied on wit, discretion, and their deep understanding of the rhythms of nature. Their ability to vanish from sight, while practical, also symbolized their cultural philosophy: to avoid unnecessary conflict, to preserve peace, and to value the hidden joys of daily life. Unlike Men, who were often restless and ambitious, or Elves, who were burdened with ancient memories, Hobbits carried with them a lightness of spirit. They sought happiness in small things—good food, laughter, family, and song—finding in these simple treasures a satisfaction that other peoples rarely achieved.In conclusion, the Hobbits were more than just a small folk who lived in quiet corners of Middle-earth. Their ability to disappear swiftly and silently was a skill carefully developed over centuries, one that outsiders often misunderstood as magic. Their physical stature, though smaller than Dwarves and far less imposing than Men, reflected a history of adaptation and change, not weakness. And their cultural values—centered on peace, modesty, and fellowship—were strengths in themselves. To judge Hobbits by size alone is to miss the deeper truth: their resilience came not from force or sorcery, but from harmony with the earth and a spirit of endurance. In a world dominated by great wars and mighty rulers, the Hobbits proved that true strength often lies in simplicity, humility, and the ability to emain unseen until the moment truly matter.Beyond their physical traits and habits of disappearing, the Hobbits built a society that reflected their deepest values of peace, comfort, and fellowship. Their communities were typically small and tightly knit, with families that traced their roots back through many generations. Unlike Men, who often sought wealth, conquest, or prestige, Hobbits found meaning in cultivating the land and maintaining traditions. Their fields, gardens, and orchards were central to daily life, and meals were considered important social events. The act of eating together was more than nourishment—it was a celebration of kinship and shared joy. Six meals a day were not uncommon, and a Hobbit’s fondness for food became a symbol of their unhurried, contented approach to existence.
Homes, often called hobbit-holes, were another defining feature of their culture. Dug into the sides of hills, these dwellings were warm, round, and cozy, designed for comfort rather than grandeur. The round doors and windows reflected a love for harmony and balance, echoing their closeness to nature. Everything inside a Hobbit home was practical and crafted with care, from polished wooden furniture to shelves lined with jars of preserves and loaves of freshly baked bread. Visitors often remarked on the welcoming atmosphere, for Hobbits took pride in their hospitality. Guests were offered food, drink, and warmth, and conversations often stretched late into the night, filled with laughter and stories.Education among Hobbits was informal but rich in oral tradition. While few were scholars in the way that Elves or Men might be, Hobbits excelled in storytelling, music, and the preservation of local history. Songs and tales were passed down to children, ensuring that even the youngest Hobbits understood their heritage and values. These traditions reinforced their sense of identity, binding the community together through shared memory. Although they seldom wrote elaborate chronicles, their oral histories captured the essence of who they were—cheerful, resilient, and deeply connected to the land they loved.Despite their love of peace, Hobbits were not entirely ignorant of the dangers of the wider world. They knew of wars, dark powers, and the struggles of other races, though they deliberately chose to avoid such matters. This avoidance was not cowardice but wisdom, born of an understanding that conflict brought little good to those who sought only to live quietly. Their ability to vanish from sight when trouble approached was as much a cultural defense as it was a physical skill. They avoided unnecessary entanglements, preferring to preserve their safety through discretion. Yet, when pressed by extraordinary circumstances, Hobbits could demonstrate surprising courage. Their small stature and gentle nature hid an inner strength that emerged when tested, proving that bravery is not confined to those who are tall or strong.Another important aspect of Hobbit life was their appreciation for the simple beauties of the world. A well-tilled garden, a pipe of good tobacco, the smell of fresh bread, or a bright festival day brought them more happiness than treasures or power could. This perspective, which many outsiders might view as narrow or provincial, was in fact a profound philosophy of contentment. Hobbits found fulfillment not in chasing after greatness but in appreciating the ordinary. Their happiness was self-sustaining, built on the recognition that life’s truest joys lie in small, consistent pleasures rather than in distant ambitions. In this sense, Hobbits embodied a kind of wisdom that other peoples often overlooked.The strength of Hobbit society lay in its unity. Families were large, and kinship ties extended into broader communities where everyone knew each other’s names, histories, and habits. Celebrations brought villages together, whether for weddings, harvest festivals, or birthdays, which were occasions marked by feasting, games, and music. These gatherings reinforced bonds of friendship and loyalty, ensuring that even in times of hardship no one was left to struggle alone. Generosity was a common virtue; to share what one had, however little, was considered not merely polite but necessary for harmony. In this way, Hobbits created a culture of cooperation and trust, which gave them stability in a world often shaken by chaos and strife.In conclusion, Hobbits were more than a curious race of small folk with unusual abilities. Their lives offered a model of resilience rooted in peace, community, and simplicity. Their ability to disappear from sight reflected not only a practical skill but also a larger philosophy: that strength does not always come from force, but often from restraint, patience, and the choice to remain unseen. Their homes, traditions, and values highlighted an enduring love for the earth and the everyday joys of life. Though often overlooked by larger peoples, the Hobbits carried within them a quiet greatness, one that taught the world that even the smallest can hold wisdom and strength beyond measure.
Hobbits possessed from the very beginning the art of disappearing swiftly and silently, and this skill became one of their most defining traits. When larger folk came blundering by, especially those whom they did not wish to meet, Hobbits could remove themselves so quickly that it often seemed magical. To Men, their vanishing act appeared uncanny, as though they had studied spells and charms to hide themselves. Yet, Hobbits have never studied magic of any kind. Their talent for elusiveness came instead from heredity and practice, reinforced by a close friendship with the earth itself. Over generations, they developed this ability into a professional skill that no other race could rival. Bigger and clumsier peoples might stumble noisily through the woods, but the Hobbits, light of step and deeply attuned to their surroundings, could pass unseen.
For they are a little people, smaller even than Dwarves, though less stout and stocky in their frames. Their stature ranged between two and four feet of our measure, with most seldom reaching three feet in modern times. Ancient stories tell of Hobbits taller in days long past, but they have dwindled, as they themselves say, to a more modest height. This dwindling did not weaken them; rather, it shaped their character. In their smallness, they found safety, for few would suspect that such a little people could possess courage or resilience. This underestimation by others gave the Hobbits space to grow in their own quiet ways.Even among themselves, Hobbits often reflected on their unusual gift of disappearing swiftly and silently. It was not merely a trick but a way of life, a habit so ingrained that it became part of their cultural identity. When strangers entered their lands, Hobbits could vanish almost without trace, blending into the earth and hedgerows with a natural ease. To them, the ability to remain unnoticed was as important as farming or storytelling. While Men and Elves wrote histories of kings and battles, Hobbits preserved the art of silence, of slipping away from dangers they did not wish to meet.It is important to understand that their elusiveness was never the result of studying magic of any kind. They never opened grimoires, never practiced enchantments, and never invoked mysterious powers. Instead, their disappearing act came from heredity and practice, strengthened by their closeness to the land. They were people of the soil, gardeners, farmers, and keepers of simple homes, and it was this bond with the earth that made them inimitable. Bigger and clumsier races might mock their size, but those same races could never hope to move with such quiet grace. It was a professional skill, taught from parent to child, woven into their way of living.Their size and their silence defined them. For they are a little people, smaller than Dwarves, lighter in build, and rarely taller than three feet. Yet within their small bodies lived a spirit of remarkable endurance. Their dwindled height over the ages was balanced by a growing depth of character, one that valued peace, comfort, and community. The Hobbits’ strength was not in towering over others but in thriving below notice, finding harmony with one another and with the land they loved.    """
    
    # Perform comprehensive analysis
    results = detector.analyze_text(student_essay)
    
    # Create and display text-based visualization
    detector.create_text_visualization(results)
    
    # Additional summary
    print("\n🎯 ANALYSIS COMPLETE!")

    