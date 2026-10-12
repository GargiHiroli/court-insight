"""Explanatory text analysis, not legal advice or litigation prediction."""
import re
from collections import Counter
from functools import lru_cache
from typing import Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LABELS = ['Good', 'Better', 'Worst']
# These categories describe textual signals, not court outcomes.
PHRASES = {
    'Good': [
        'the court granted the application', 'the claim is allowed',
        'the appeal is allowed', 'the petitioner is entitled to relief',
        'the evidence supports the claim', 'the order is in favor of the applicant'
    ],
    'Better': [
        'the case is adjourned for further hearing', 'the evidence requires consideration',
        'the court directed parties to file submissions', 'the matter remains pending',
        'the application is partly allowed', 'the court requested additional documents'
    ],
    'Worst': [
        'the petition is dismissed', 'the appeal is rejected',
        'the claim is denied', 'the court imposed a penalty',
        'the party failed to establish the case', 'the application is refused'
    ]
}
POS = {'granted', 'allowed', 'approved', 'relief', 'acquitted', 'succeeded', 'valid', 'accepted', 'favorable', 'upheld'}
NEG = {'dismissed', 'rejected', 'denied', 'penalty', 'convicted', 'failed', 'invalid', 'refused', 'violation', 'liable'}
NEUT = {'pending', 'adjourned', 'hearing', 'consideration', 'submission', 'review', 'interim', 'partly'}
STOP = {'this', 'that', 'with', 'from', 'were', 'have', 'their', 'there', 'would', 'shall', 'into', 'such', 'been', 'under', 'court', 'case', 'which', 'where', 'upon', 'they', 'them', 'said', 'also', 'after', 'before', 'between'}
TRAINED_MODEL: Optional[Pipeline] = None


def sentences(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if len(s.strip()) >= 15]


def tfidf_keywords(text: str, limit: int = 12):
    chunks = sentences(text)
    if not chunks:
        return []
    try:
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=3000)
        matrix = vec.fit_transform(chunks)
        scores = np.asarray(matrix.mean(axis=0)).flatten()
        ranking = np.argsort(scores)[::-1]
        terms = vec.get_feature_names_out()
        return [{'term': str(terms[i]), 'weight': round(float(scores[i]), 4)} for i in ranking[:limit] if scores[i] > 0]
    except ValueError:
        return []


def lexical_distribution(text: str):
    words = re.findall(r"\b[a-z]+\b", text.lower())
    counts = Counter(words)
    pos, neg, mid = sum(counts[w] for w in POS), sum(counts[w] for w in NEG), sum(counts[w] for w in NEUT)
    # Smoothed lexical baseline; highlights why this is not a legal outcome prediction.
    raw = np.array([pos + 0.5, mid + 0.5, neg + 0.5], dtype=float)
    raw = raw / raw.sum()
    return raw


@lru_cache(maxsize=1)
def semantic_components():
    # Downloads pretrained model on first request; requires internet on first run.
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    anchors = [p for label in LABELS for p in PHRASES[label]]
    anchor_labels = [label for label in LABELS for _ in PHRASES[label]]
    embeddings = model.encode(anchors, normalize_embeddings=True)
    return model, embeddings, anchor_labels


def semantic_distribution(text: str):
    model, anchors, anchor_labels = semantic_components()
    chunks = sentences(text)[:60] or [text[:1500]]
    embeddings = model.encode(chunks, normalize_embeddings=True)
    sims = np.asarray(embeddings) @ np.asarray(anchors).T
    scores = [float(np.mean(np.max(sims[:, [i for i, lbl in enumerate(anchor_labels) if lbl == label]], axis=1))) for label in LABELS]
    # Avoid negative values and use softmax for a relative *similarity* distribution.
    shifted = np.exp((np.array(scores) - max(scores)) * 5)
    return shifted / shifted.sum()


def train_model(rows):
    global TRAINED_MODEL
    docs = [r['text'] for r in rows]
    labels = [r['label'] for r in rows]
    if len(rows) < 9 or len(set(labels)) != 3 or min(Counter(labels).values()) < 3:
        raise ValueError('Provide at least 9 labeled examples, including at least 3 of each label.')
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=15000)),
        ('classifier', LogisticRegression(max_iter=1200, class_weight='balanced')),
    ])
    pipe.fit(docs, labels)
    TRAINED_MODEL = pipe
    return len(rows)


def analyze(text: str, use_embeddings: bool = False):
    clean = re.sub(r'\s+', ' ', text).strip()
    if len(clean) < 40:
        raise ValueError('Provide at least 40 characters of extractable text.')
    lex = lexical_distribution(clean)
    methods = ['lexical baseline']
    # Supervised TF-IDF model is available only after explicit training with user data.
    ml = None
    if TRAINED_MODEL is not None:
        model_probs = TRAINED_MODEL.predict_proba([clean])[0]
        ml = np.array([model_probs[list(TRAINED_MODEL.classes_).index(l)] for l in LABELS])
        methods.append('trained TF-IDF + logistic regression')
    emb = None
    warning = None
    if use_embeddings:
        try:
            emb = semantic_distribution(clean)
            methods.append('MiniLM sentence embeddings')
        except Exception as exc:
            warning = f'Embedding model unavailable ({type(exc).__name__}). Used other available methods.'
    weights, values = [0.25], [lex]
    if ml is not None:
        weights.append(0.55); values.append(ml)
    if emb is not None:
        weights.append(0.40 if ml is not None else 0.75); values.append(emb)
    probabilities = np.average(np.stack(values), axis=0, weights=weights)
    probabilities = probabilities / probabilities.sum()
    ranking = np.argsort(probabilities)[::-1]
    label = LABELS[int(ranking[0])]
    gap = float(probabilities[ranking[0]] - probabilities[ranking[1]])
    tokens = re.findall(r'\b\w+\b', clean)
    findings = []
    for s in sentences(text)[:120]:
        lower = s.lower()
        hits = [w for w in POS | NEG | NEUT if re.search(r'\b' + re.escape(w) + r'\b', lower)]
        if hits:
            findings.append({'excerpt': s[:360], 'signals': sorted(hits)[:8]})
    findings = sorted(findings, key=lambda x: len(x['signals']), reverse=True)[:5]
    return {
        'label': label,
        'ranges': [{'label': name, 'score': round(float(probabilities[i]) * 100, 1)} for i, name in enumerate(LABELS)],
        'relative_score': round(float(probabilities[ranking[0]]) * 100, 1),
        'uncertainty': 'high' if gap < 0.15 else ('moderate' if gap < 0.3 else 'lower'),
        'words': len(tokens), 'characters': len(clean),
        'keywords': tfidf_keywords(text), 'evidence': findings,
        'methods': methods, 'warning': warning,
        'disclaimer': 'Experimental linguistic classification only. Labels are relative textual signals, not legal merits, accuracy, probability of winning, legal advice, or a substitute for lawyer review.'
    }
