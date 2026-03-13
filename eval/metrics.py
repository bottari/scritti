import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import spacy
except Exception:  # pragma: no cover - optional dependency
    spacy = None

try:
    from nltk.corpus import wordnet as wn
except Exception:  # pragma: no cover - optional dependency
    wn = None


IMAGERY_LEXICON = {
    "mist",
    "moon",
    "sun",
    "river",
    "forest",
    "leaf",
    "leaves",
    "stone",
    "wind",
    "rain",
    "storm",
    "shadow",
    "shadows",
    "light",
    "glow",
    "flame",
    "ash",
    "snow",
    "ice",
    "ocean",
    "sea",
    "wave",
    "waves",
    "sky",
    "cloud",
    "clouds",
    "flower",
    "flowers",
    "petal",
    "petals",
    "dust",
    "smoke",
    "ember",
    "root",
    "roots",
    "tree",
    "trees",
    "bird",
    "birds",
    "wolf",
    "wolves",
    "deer",
    "feather",
    "feathers",
    "scent",
    "fragrance",
    "taste",
    "bitter",
    "sweet",
    "sour",
    "salt",
    "salty",
    "silence",
    "whisper",
    "whispers",
    "echo",
    "echoes",
    "breath",
    "breathing",
    "night",
    "dawn",
    "twilight",
    "dusk",
    "horizon",
    "stars",
    "star",
    "thunder",
    "fog",
    "moss",
    "soil",
    "earth",
    "garden",
    "garden",
}

DIALOGUE_VERBS = {"say", "tell", "ask", "reply", "whisper", "shout", "cry"}
MOTION_PERCEPTION_VERBS = {
    "walk",
    "look",
    "see",
    "stand",
    "move",
    "turn",
    "stare",
    "watch",
    "glance",
    "hear",
    "listen",
    "run",
    "step",
    "wander",
}

WORD_RE = re.compile(r"[a-zA-Z']+")
SPECIAL_TOKEN_RE = re.compile(
    r"(<\|im_start\|>|<\|im_end\|>|<\|eot_id\|>|<\|endoftext\|>|<\|assistant\|>|<\|user\|>|<\|system\|>)"
)


def clean_generation_text(text: str) -> str:
    if not text:
        return ""
    cleaned = SPECIAL_TOKEN_RE.sub(" ", text)
    return " ".join(cleaned.split())


@dataclass
class SpacyResources:
    nlp: Optional["spacy.language.Language"]


def load_spacy(model: str = "en_core_web_sm") -> SpacyResources:
    if spacy is None:
        return SpacyResources(nlp=None)
    try:
        nlp = spacy.load(model)
    except Exception:
        nlp = None
    return SpacyResources(nlp=nlp)


def tokenize_words(text: str) -> List[str]:
    cleaned = clean_generation_text(text)
    return [m.group(0).lower() for m in WORD_RE.finditer(cleaned)]


def _expand_imagery_lexicon(seed_lexicon: set) -> set:
    if wn is None:
        return set(seed_lexicon)
    expanded = set(seed_lexicon)
    try:
        for word in seed_lexicon:
            for syn in wn.synsets(word):
                for lemma in syn.lemma_names():
                    term = lemma.lower().replace("_", " ")
                    if " " in term:
                        continue
                    if term.isalpha():
                        expanded.add(term)
    except LookupError:
        # WordNet data not downloaded
        return set(seed_lexicon)
    return expanded


_EXPANDED_IMAGERY_LEXICON = None


def get_imagery_lexicon() -> set:
    global _EXPANDED_IMAGERY_LEXICON
    if _EXPANDED_IMAGERY_LEXICON is None:
        _EXPANDED_IMAGERY_LEXICON = _expand_imagery_lexicon(IMAGERY_LEXICON)
    return _EXPANDED_IMAGERY_LEXICON


def imagery_density(text: str, imagery_lexicon: Optional[set] = None) -> float:
    tokens = tokenize_words(text)
    if not tokens:
        return 0.0
    lex = imagery_lexicon or get_imagery_lexicon()
    hits = sum(1 for t in tokens if t in lex)
    return hits / len(tokens)


def lexical_novelty(text: str, corpus_token_set: set) -> float:
    tokens = tokenize_words(text)
    if not tokens:
        return 0.0
    novel = [t for t in tokens if t not in corpus_token_set]
    return len(novel) / len(tokens)


def metaphor_density(text: str, nlp: Optional["spacy.language.Language"]) -> float:
    text = clean_generation_text(text)
    if not text.strip():
        return 0.0
    if nlp is None:
        # Fallback regex for simple patterns
        patterns = [
            re.compile(r"\b\w+\s+is\s+\w+\b", re.IGNORECASE),
            re.compile(r"\b\w+\s+becomes\s+\w+\b", re.IGNORECASE),
            re.compile(r"\b\w+\s+like\s+\w+\b", re.IGNORECASE),
            re.compile(r"\b\w+\s+of\s+\w+\b", re.IGNORECASE),
        ]
        hits = sum(len(p.findall(text)) for p in patterns)
        tokens = tokenize_words(text)
        return hits / max(1, len(tokens))

    doc = nlp(text)
    hits = 0
    noun_like = {"NOUN", "PROPN"}

    # Pattern 1: X is Y / X becomes Y (copular)
    for token in doc:
        if token.lemma_.lower() in {"be", "become"}:
            head = token.head if token.dep_ == "cop" else token
            subj = [c for c in head.children if c.dep_ in {"nsubj", "nsubjpass"}]
            comp = [c for c in head.children if c.dep_ in {"attr", "acomp", "oprd"}]
            if not comp:
                comp = [head] if head.pos_ in noun_like else []
            if subj and comp:
                if any(s.pos_ in noun_like for s in subj) and any(
                    c.pos_ in noun_like or c.pos_ == "ADJ" for c in comp
                ):
                    hits += 1

    # Pattern 2: X like Y (comparative)
    for token in doc:
        if token.lemma_.lower() == "like" and token.pos_ in {"ADP", "SCONJ"}:
            pobj = [c for c in token.children if c.dep_ == "pobj"]
            head = token.head
            subj = [c for c in head.children if c.dep_ in {"nsubj", "nsubjpass"}]
            if pobj and subj:
                if any(s.pos_ in noun_like for s in subj) and any(p.pos_ in noun_like for p in pobj):
                    hits += 1

    # Pattern 3: X of Y (noun-noun linking)
    for token in doc:
        if token.lemma_.lower() == "of" and token.dep_ == "prep":
            pobj = [c for c in token.children if c.dep_ == "pobj"]
            head = token.head
            if pobj and head.pos_ in noun_like and any(p.pos_ in noun_like for p in pobj):
                hits += 1

    return hits / max(1, len(doc))


def compound_imagery_density(text: str, nlp: Optional["spacy.language.Language"]) -> float:
    text = clean_generation_text(text)
    tokens = tokenize_words(text)
    if not tokens:
        return 0.0
    if nlp is None:
        patterns = [
            re.compile(r"\b\w+\s+of\s+\w+\b", re.IGNORECASE),
            re.compile(r"\b\w+\s+\w+\b", re.IGNORECASE),
        ]
        hits = sum(len(p.findall(text)) for p in patterns)
        return (hits / len(tokens)) * 100.0

    doc = nlp(text)
    hits = 0
    noun_like = {"NOUN", "PROPN"}

    # noun chunks: adjective+noun and noun noun
    for chunk in doc.noun_chunks:
        nouns = [t for t in chunk if t.pos_ in noun_like]
        adjs = [t for t in chunk if t.pos_ == "ADJ"]
        if len(nouns) >= 2:
            hits += 1
        elif len(nouns) == 1 and len(adjs) >= 1:
            hits += 1

    # explicit X of Y patterns
    for token in doc:
        if token.lemma_.lower() == "of" and token.dep_ == "prep":
            pobj = [c for c in token.children if c.dep_ == "pobj"]
            head = token.head
            if pobj and head.pos_ in noun_like and any(p.pos_ in noun_like for p in pobj):
                hits += 1

    return (hits / len(tokens)) * 100.0


def narrative_density(text: str, nlp: Optional["spacy.language.Language"]) -> float:
    text = clean_generation_text(text)
    tokens = tokenize_words(text)
    if not tokens:
        return 0.0
    if nlp is None:
        verb_ratio = len(re.findall(r"\b\w+(ed|ing)\b", text, flags=re.IGNORECASE)) / max(1, len(tokens))
        dialogue_ratio = (
            len(re.findall(r"\b(he|she)\s+said\b", text, flags=re.IGNORECASE))
            + len(re.findall(r"\"|“|”", text))
        ) / max(1, len(tokens))
        motion_ratio = (
            len(re.findall(r"\b(walk|look|see|stand|move|turn|stare|watch|glance|hear|listen|run|step|wander)\b", text, flags=re.IGNORECASE))
            / max(1, len(tokens))
        )
        return (verb_ratio + dialogue_ratio + motion_ratio) / 3.0

    doc = nlp(text)
    verb_count = sum(1 for t in doc if t.pos_ == "VERB")
    verb_ratio = verb_count / max(1, len(doc))

    dialogue_hits = 0
    for token in doc:
        if token.lemma_.lower() in DIALOGUE_VERBS and token.pos_ == "VERB":
            subj = [c for c in token.children if c.dep_ in {"nsubj", "nsubjpass"}]
            if any(s.lemma_.lower() in {"he", "she", "they", "i", "we", "you"} for s in subj):
                dialogue_hits += 1
    quote_hits = text.count('"') + text.count("“") + text.count("”")
    dialogue_ratio = (dialogue_hits + quote_hits) / max(1, len(doc))

    motion_hits = sum(
        1
        for t in doc
        if t.pos_ == "VERB" and t.lemma_.lower() in MOTION_PERCEPTION_VERBS
    )
    motion_ratio = motion_hits / max(1, len(doc))

    return (verb_ratio + dialogue_ratio + motion_ratio) / 3.0


def _noun_pair_distance(
    nouns: List[str],
    embedder,
    cache: Optional[Dict[str, np.ndarray]] = None,
    batch_size: int = 64,
) -> float:
    if len(nouns) < 2:
        return 0.0
    cache = cache if cache is not None else {}
    missing = [n for n in nouns if n not in cache]
    if missing:
        embeddings = embedder.encode(missing, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
        for noun, emb in zip(missing, embeddings):
            cache[noun] = emb
    distances = []
    for i in range(len(nouns) - 1):
        sim = float(np.dot(cache[nouns[i]], cache[nouns[i + 1]]))
        distances.append(1.0 - sim)
    return float(np.mean(distances)) if distances else 0.0


def conceptual_distance_score(
    text: str,
    nlp: Optional["spacy.language.Language"],
    embedder,
    cache: Optional[Dict[str, np.ndarray]] = None,
    max_nouns: int = 128,
    batch_size: int = 64,
) -> float:
    text = clean_generation_text(text)
    if nlp is None or not text.strip():
        return 0.0
    doc = nlp(text)
    nouns = [t.text for t in doc if t.pos_ in {"NOUN", "PROPN"}]
    if len(nouns) > max_nouns:
        nouns = nouns[:max_nouns]
    return _noun_pair_distance(nouns, embedder, cache=cache, batch_size=batch_size)


def surreal_imagery_score(
    text: str,
    nlp: Optional["spacy.language.Language"],
    embedder,
    cache: Optional[Dict[str, np.ndarray]] = None,
    max_nouns: int = 128,
    batch_size: int = 64,
) -> float:
    text = clean_generation_text(text)
    if nlp is None or not text.strip():
        return 0.0
    doc = nlp(text)
    nouns = [t.text for t in doc if t.pos_ in {"NOUN", "PROPN"}]
    if len(nouns) > max_nouns:
        nouns = nouns[:max_nouns]
    return _noun_pair_distance(nouns, embedder, cache=cache, batch_size=batch_size)


def semantic_drift(
    base_texts: Sequence[str],
    lora_texts: Sequence[str],
    embedder,
    batch_size: int = 64,
) -> List[float]:
    if len(base_texts) != len(lora_texts):
        raise ValueError("base_texts and lora_texts must be the same length.")
    base_emb = embedder.encode(list(base_texts), normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    lora_emb = embedder.encode(list(lora_texts), normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    drifts = []
    for b, l in zip(base_emb, lora_emb):
        sim = float(np.dot(b, l))
        drifts.append(1.0 - sim)
    return drifts


def build_corpus_token_set(corpus_texts: Iterable[str]) -> set:
    token_set = set()
    for text in corpus_texts:
        token_set.update(tokenize_words(text))
    return token_set


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = p + eps
    q = q + eps
    return float(np.sum(p * np.log(p / q)))


def token_distribution(texts: Iterable[str], tokenizer) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        for idx in ids:
            counts[idx] = counts.get(idx, 0) + 1
    return counts


def token_distribution_shift(base_texts: Iterable[str], lora_texts: Iterable[str], tokenizer) -> float:
    base_counts = token_distribution(base_texts, tokenizer)
    lora_counts = token_distribution(lora_texts, tokenizer)
    vocab_ids = sorted(set(base_counts.keys()) | set(lora_counts.keys()))
    if not vocab_ids:
        return 0.0
    base_vec = np.array([base_counts.get(i, 0) for i in vocab_ids], dtype=np.float64)
    lora_vec = np.array([lora_counts.get(i, 0) for i in vocab_ids], dtype=np.float64)
    base_prob = base_vec / base_vec.sum() if base_vec.sum() > 0 else base_vec
    lora_prob = lora_vec / lora_vec.sum() if lora_vec.sum() > 0 else lora_vec
    return kl_divergence(lora_prob, base_prob)

