
import re
import math
import string
from collections import Counter

import nltk
import numpy as np

# ── NLTK bootstrap ────────────────────────────────────────────────────────────
for pkg in ("punkt", "stopwords", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

try:
    STOP_WORDS = set(nltk.corpus.stopwords.words("english"))
except Exception:
    STOP_WORDS = set()


def _tokenize(text: str) -> list[str]:
    """Lowercase alpha tokens."""
    try:
        return [t for t in nltk.word_tokenize(text.lower()) if t.isalpha()]
    except Exception:
        return re.findall(r"[a-z]+", text.lower())


def _sentences(text: str) -> list[str]:
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        return re.split(r"(?<=[.!?])\s+", text.strip())


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_frequency_analysis(text: str, ns: tuple[int, ...] = (2, 3, 4)) -> dict:
    """
    Returns frequency tables for bigrams, trigrams, and 4-grams,
    filtering out stop-word-only n-grams.
    """
    tokens = _tokenize(text)
    content_tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    result = {}
    for n in ns:
        label = {2: "bigrams", 3: "trigrams", 4: "quadgrams"}.get(n, f"{n}-grams")
        raw_grams = _ngrams(tokens, n)
        freq = Counter(" ".join(g) for g in raw_grams)

        # Also build content-only version for highlighting
        content_grams = _ngrams(content_tokens, n)
        content_freq = Counter(" ".join(g) for g in content_grams)

        top_all = [
            {"phrase": ph, "count": cnt, "density": round(cnt / max(len(raw_grams), 1) * 100, 2)}
            for ph, cnt in freq.most_common(20)
        ]
        top_content = [
            {"phrase": ph, "count": cnt, "density": round(cnt / max(len(content_grams), 1) * 100, 2)}
            for ph, cnt in content_freq.most_common(20)
        ]

        result[label] = {
            "total_unique": len(freq),
            "total_occurrences": len(raw_grams),
            "top_phrases": top_all,
            "top_content_phrases": top_content,
        }
    return result


def find_repeated_phrases(text: str, min_n: int = 3, max_n: int = 6, min_count: int = 2) -> list[dict]:
    """
    Finds repeated multi-word phrases by splitting on whitespace so that
    the phrases can be matched back into the original text reliably.
    """
    raw_words = text.split()
    lower_words = [w.lower() for w in raw_words]

    seen: dict[str, list[int]] = {}
    for n in range(min_n, max_n + 1):
        for i in range(len(lower_words) - n + 1):
            phrase = " ".join(lower_words[i : i + n])
            seen.setdefault(phrase, []).append(i)

    repeated = [
        {
            "phrase": phrase,
            "count": len(positions),
            "word_length": len(phrase.split()),
            "score": len(positions) * len(phrase.split()),
            "token_positions": positions,
        }
        for phrase, positions in seen.items()
        if len(positions) >= min_count
    ]
    repeated.sort(key=lambda x: x["score"], reverse=True)

    # De-duplicate: drop shorter phrases already covered by a longer one
    final = []
    covered = set()
    for item in repeated:
        if item["phrase"] not in covered:
            final.append(item)
            words = item["phrase"].split()
            for sub_n in range(len(words) - 1, 1, -1):
                for start in range(len(words) - sub_n + 1):
                    covered.add(" ".join(words[start : start + sub_n]))
        if len(final) >= 30:
            break

    return final


def highlight_text_with_repeated_phrases(text: str, repeated_phrases: list[dict]) -> str:
    """
    Returns HTML with <mark> tags around every occurrence of each repeated
    phrase. Uses whitespace-flexible regex so it works on the raw original text.
    """
    if not repeated_phrases:
        return re.sub(r'\n', '<br>', text)

    max_score = repeated_phrases[0]["score"] if repeated_phrases else 1

    # Map lowercased phrase -> tier
    tier_map = {}
    for item in repeated_phrases:
        ratio = item["score"] / max_score
        tier = 0 if ratio >= 0.75 else 1 if ratio >= 0.5 else 2
        tier_map[item["phrase"].lower()] = tier

    # Sort longest-first so longer matches win over sub-phrases
    phrases_sorted = sorted(tier_map.keys(), key=len, reverse=True)

    def phrase_to_pattern(phrase: str) -> str:
        return r"\s+".join(re.escape(w) for w in phrase.split())

    combined = re.compile(
        r"(?i)(?<![\w])(" + "|".join(phrase_to_pattern(p) for p in phrases_sorted) + r")(?![\w])"
    )

    def replacer(m: re.Match) -> str:
        matched = m.group(0)
        key = re.sub(r"\s+", " ", matched).lower().strip()
        tier = tier_map.get(key, 2)
        return f"<mark class='rep-{tier}'>{matched}</mark>"

    highlighted = combined.sub(replacer, text)
    highlighted = re.sub(r'\n', '<br>', highlighted)
    return highlighted



def detect_paraphrased_segments(
    text: str, matched_sources: list[dict], threshold: float = 0.18
) -> list[dict]:
    """
    Detects potentially paraphrased sentences using four independent signals:

    1. Content-word overlap (Jaccard) between the sentence and the full
       submitted text — high overlap with common content words signals
       repetitive/paraphrased language.
    2. Synonym/substitution signal — measures how many words in the sentence
       are rare in the overall text (indicating swapped vocabulary).
    3. Structural suspicion — passive voice, nominalisations, unusual
       sentence length patterns typical of paraphrasing tools.
    4. Contextual n-gram novelty — sentence shares few n-grams with its
       neighbours, suggesting it was rewritten differently.
    """
    sentences = _sentences(text)
    if not sentences:
        return []

    # Global content-word frequency across the whole text
    all_tokens = _tokenize(text)
    global_freq = Counter(all_tokens)
    global_content = {w for w in all_tokens if w not in STOP_WORDS and len(w) > 3}
    total_words = max(len(all_tokens), 1)

    # Book semantic score as a global booster
    best_sem = max((b.get("semantic_score", 0) for b in matched_sources), default=0) if matched_sources else 0
    best_combined = max((b.get("combined_score", 0) for b in matched_sources), default=0) if matched_sources else 0

    # Pre-tokenize all sentences for n-gram context comparison
    sent_tokens = [_tokenize(s) for s in sentences]

    flagged = []

    for i, sent in enumerate(sentences):
        words = sent.split()
        if len(words) < 5:
            continue

        tokens = sent_tokens[i]
        content_tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 3]
        if not content_tokens:
            continue

       
        # Words that appear very rarely in the overall text but are used here
        # are candidates for synonym substitution
        rare_words = [w for w in content_tokens if global_freq.get(w, 0) == 1]
        vocab_sub_score = len(rare_words) / max(len(content_tokens), 1)

       
        # Low overlap means this sentence uses different vocabulary = rewrite
        sent_content = set(content_tokens)
        doc_content_sample = set(list(global_content)[:300])
        if doc_content_sample:
            overlap = len(sent_content & doc_content_sample)
            union = len(sent_content | doc_content_sample)
            content_divergence = 1.0 - (overlap / union if union else 0)
        else:
            content_divergence = 0.0

       
        sent_lower = sent.lower()
        structural_score = 0.0

        # Passive voice markers
        passive_markers = ["was ", "were ", "been ", "being ", "is said", "are said",
                          "was found", "were found", "has been", "have been"]
        if any(m in sent_lower for m in passive_markers):
            structural_score += 0.25

        # Nominalisation patterns (verb→noun conversion common in paraphrasing)
        nominal_endings = ["tion", "sion", "ment", "ness", "ity", "ance", "ence"]
        nominal_count = sum(1 for w in tokens if any(w.endswith(e) for e in nominal_endings))
        structural_score += min(nominal_count / max(len(tokens), 1) * 2, 0.3)

        # Unusually long sentence (paraphrase tools often expand)
        avg_sent_len = total_words / max(len(sentences), 1)
        if len(words) > avg_sent_len * 1.8:
            structural_score += 0.2

        # Transition/connector words common in paraphrased rewrites
        connectors = ["furthermore", "moreover", "however", "consequently",
                     "therefore", "subsequently", "additionally", "nevertheless",
                     "in addition", "as a result", "in contrast", "on the other hand"]
        if any(c in sent_lower for c in connectors):
            structural_score += 0.15

        structural_score = min(structural_score, 1.0)

        # ── Signal 4: N-gram novelty vs neighbours ────────────────────────
        # If this sentence shares few trigrams with its neighbours it may
        # have been rewritten while surrounding text was copied
        neighbour_tokens = []
        if i > 0: neighbour_tokens.extend(sent_tokens[i-1])
        if i < len(sentences)-1: neighbour_tokens.extend(sent_tokens[i+1])

        if neighbour_tokens and len(tokens) >= 3:
            sent_trigrams = set(_ngrams(tokens, 3))
            neigh_trigrams = set(_ngrams(neighbour_tokens, 3))
            if sent_trigrams and neigh_trigrams:
                trigram_overlap = len(sent_trigrams & neigh_trigrams) / len(sent_trigrams)
                novelty_score = 1.0 - trigram_overlap
            else:
                novelty_score = 0.5
        else:
            novelty_score = 0.3

       
        base_score = (
            vocab_sub_score    * 0.30 +
            content_divergence * 0.25 +
            structural_score   * 0.30 +
            novelty_score      * 0.15
        )

        # Boost if the overall analysis already found a strong match
        boost = 1.0 + (best_combined * 0.4) + (best_sem * 0.2)
        final_score = min(base_score * boost, 1.0)

        if final_score >= threshold:
            # Describe why it was flagged
            reasons = []
            if vocab_sub_score > 0.4:
                reasons.append(f"{len(rare_words)} rare/substituted words")
            if structural_score > 0.3:
                reasons.append("structural rewrite patterns")
            if novelty_score > 0.6:
                reasons.append("diverges from surrounding text")
            if not reasons:
                reasons.append("vocabulary and structure divergence")

            flagged.append({
                "sentence_index": i,
                "sentence_text": sent,
                "suspicion_score": round(final_score, 3),
                "risk_level": "High" if final_score >= 0.55 else "Medium" if final_score >= 0.35 else "Low",
                "signals": {
                    "vocab_substitution": round(vocab_sub_score, 3),
                    "content_divergence": round(content_divergence, 3),
                    "structural":         round(structural_score, 3),
                    "ngram_novelty":      round(novelty_score, 3),
                },
                "reasons": reasons,
                "rare_words": rare_words[:8],
            })

    flagged.sort(key=lambda x: x["suspicion_score"], reverse=True)
    return flagged[:20]


def vocabulary_stats(text: str) -> dict:
    tokens = _tokenize(text)
    sentences = _sentences(text)
    words_alpha = [t for t in tokens if len(t) > 1]
    content_words = [t for t in words_alpha if t not in STOP_WORDS]

    if not words_alpha:
        return {}

    # Flesch–Kincaid readability (approximate syllable count)
    def syllables(word: str) -> int:
        word = word.lower()
        count = len(re.findall(r"[aeiou]+", word))
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    total_syllables = sum(syllables(w) for w in words_alpha)
    total_words = len(words_alpha)
    total_sents = max(len(sentences), 1)

    fk_reading_ease = (
        206.835
        - 1.015 * (total_words / total_sents)
        - 84.6 * (total_syllables / total_words)
    )
    fk_grade = (
        0.39 * (total_words / total_sents)
        + 11.8 * (total_syllables / total_words)
        - 15.59
    )

    freq = Counter(words_alpha)
    hapax = [w for w, c in freq.items() if c == 1]
    punct_count = sum(1 for ch in text if ch in string.punctuation)

    return {
        "total_tokens": len(tokens),
        "unique_words": len(set(words_alpha)),
        "content_words": len(content_words),
        "stop_words": len(words_alpha) - len(content_words),
        "type_token_ratio": round(len(set(words_alpha)) / total_words, 3),
        "hapax_legomena_count": len(hapax),
        "hapax_examples": hapax[:10],
        "avg_word_length": round(np.mean([len(w) for w in words_alpha]), 2),
        "avg_sentence_length": round(total_words / total_sents, 1),
        "total_sentences": total_sents,
        "punctuation_count": punct_count,
        "flesch_reading_ease": round(fk_reading_ease, 1),
        "flesch_kincaid_grade": round(fk_grade, 1),
        "top_50_words": [
            {"word": w, "count": c, "pct": round(c / total_words * 100, 2)}
            for w, c in freq.most_common(50)
        ],
    }


def sentence_risk_map(text: str, repeated_phrases: list[dict]) -> list[dict]:
    """
    Assigns a repetition density score to each sentence.
    Matches using flexible whitespace regex so punctuation-attached words
    do not prevent matching.
    """
    sentences = _sentences(text)
    if not repeated_phrases:
        return [{"index": i, "text": s, "repeated_phrases_found": [], "risk_score": 0.0}
                for i, s in enumerate(sentences)]

    # Pre-compile one pattern per phrase using \s+ between words
    phrase_patterns = []
    for item in repeated_phrases:
        words = item["phrase"].split()
        pat = re.compile(
            r"(?i)" + r"\s+".join(re.escape(w) for w in words)
        )
        phrase_patterns.append((item["phrase"], item["count"], pat))

    # Normalise counts to get a 0-1 score
    max_count = max(item["count"] for item in repeated_phrases) if repeated_phrases else 1

    risk_map = []
    for idx, sent in enumerate(sentences):
        hits = []
        total_score = 0.0
        for phrase, count, pat in phrase_patterns:
            if pat.search(sent):
                hits.append(phrase)
                total_score += count

        # Scale: a sentence containing the most-repeated phrase scores ~1.0
        risk_score = round(min(total_score / (max_count * 2), 1.0), 3)
        risk_map.append(
            {
                "index": idx,
                "text": sent,
                "repeated_phrases_found": hits,
                "risk_score": risk_score,
            }
        )
    return risk_map

def generate_detailed_report(text: str, analysis: dict) -> dict:
    """
    Main entry point.
    
    Parameters
    ----------
    text     : The raw submitted text.
    analysis : The JSON dict previously returned by /analyze endpoint.
    
    Returns
    -------
    A dict ready to be jsonify()'d and consumed by the frontend report viewer.
    """
    matched_sources = analysis.get("matched_sources", [])

    # Core computations
    ngram_freq      = ngram_frequency_analysis(text)
    repeated        = find_repeated_phrases(text)
    highlighted_html= highlight_text_with_repeated_phrases(text, repeated)
    paraphrased     = detect_paraphrased_segments(text, matched_sources)
    vocab           = vocabulary_stats(text)
    sent_risk       = sentence_risk_map(text, repeated)

    return {
        "summary": {
            "combined_score":   analysis.get("combined_score", 0),
            "risk_level":       analysis.get("risk_level", "no actionable similarity detected"),
            "legal_risk_code":  analysis.get("legal_risk_code", "NO_ACTIONABLE_SIMILARITY"),
            "legal_rationale":  analysis.get("legal_rationale", ""),
            "top_source_work":  analysis.get("top_source_work", "No significant source-work match found"),
            "matched_sources":  len([b for b in matched_sources if b.get("combined_score", 0) >= 0.3]),
            "total_sentences":  vocab.get("total_sentences", 0),
            "total_words":      vocab.get("total_tokens", 0),
            "unique_words":     vocab.get("unique_words", 0),
            "flesch_ease":      vocab.get("flesch_reading_ease", 0),
            "fk_grade":         vocab.get("flesch_kincaid_grade", 0),
        },
        "ngram_frequency":       ngram_freq,
        "repeated_phrases":      repeated,
        "highlighted_text_html": highlighted_html,
        "paraphrased_segments":  paraphrased,
        "vocabulary_stats":      vocab,
        "sentence_risk_map":     sent_risk,
        "matched_sources":       matched_sources,
        "feature_names":         analysis.get("feature_names", []),
        "submission_features":   analysis.get("submission_features", []),
    }
