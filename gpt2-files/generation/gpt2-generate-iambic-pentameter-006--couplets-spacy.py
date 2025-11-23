import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
import re
from nltk.corpus import cmudict
import numpy as np
from functools import lru_cache
import spacy

# =========================================================
#                     CONFIGURATION
# =========================================================
MODEL_NAME = "gpt2"
ADAPTER_PATH = r"C:\Users\micha\Desktop\projects\mercury\gpt2-finetuned-poetry-mercury-04\final_model"
PROMPT = "Awake, my"
MAX_GENERATION_ATTEMPTS = 25
TOKENS_TO_GENERATE = 400  # Increased for more candidates
MAX_RECONSTRUCTION_ATTEMPTS = 15
ALLOW_WORD_REPETITION_THRESHOLD = 0.2  # Allow some word repetition

device = "cuda" if torch.cuda.is_available() else "cpu"
bf16_available = torch.cuda.is_bf16_supported()

# =========================================================
#                 LOAD DEPENDENCIES
# =========================================================
try:
    cmu = cmudict.dict()
except LookupError:
    nltk.download("cmudict")
    cmu = cmudict.dict()

print("Loading SpaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading SpaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# =========================================================
#                 METER AND RHYME ANALYSIS
# =========================================================

@lru_cache(maxsize=10000)
def get_stress_and_syllables(word):
    """Returns stress pattern, syllable count, and phonemes."""
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    if word and word in cmu:
        phonemes = cmu[word][0]
        stress = ''.join([p[-1] for p in phonemes if p[-1].isdigit()])
        return stress, len(stress), tuple(phonemes)
    return None, 0, None

@lru_cache(maxsize=5000)
def get_rhyming_phonemes(word):
    """Returns phonemes from the primary stressed vowel onwards."""
    stress_markers = ['1', '2']
    word_clean = clean_word(word)
    
    if not word_clean:
        return None
    
    _, _, phonemes = get_stress_and_syllables(word_clean)
    if not phonemes:
        return None

    vowels = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
    rhyme_start_index = -1
    
    for i in range(len(phonemes) - 1, -1, -1):
        if len(phonemes[i]) > 0 and phonemes[i][-1] in stress_markers and phonemes[i][:2] in vowels:
            rhyme_start_index = i
            break
            
    if rhyme_start_index == -1:
        return None

    return tuple(phonemes[rhyme_start_index:])

@lru_cache(maxsize=5000)
def clean_word(word):
    """Cleans punctuation and converts to lowercase."""
    return re.sub(r"[^a-zA-Z]", "", word).lower()

def check_rhyme(word1, word2):
    """Checks if two words rhyme and are not the same word."""
    if not word1 or not word2:
        return False
    
    clean_w1 = clean_word(word1)
    clean_w2 = clean_word(word2)
    
    if clean_w1 == clean_w2 and clean_w1 != "":
        return False

    p1 = get_rhyming_phonemes(word1)
    p2 = get_rhyming_phonemes(word2)
    
    if p1 is None or p2 is None:
        return False
        
    return p1 == p2

def get_phrase_stress_pattern(words):
    """Get the complete stress pattern for a phrase."""
    full_stress = ""
    total_syllables = 0
    
    for word in words:
        stress, syllables, _ = get_stress_and_syllables(word)
        if stress is None:
            return None, 0
        full_stress += stress
        total_syllables += syllables
    
    return full_stress, total_syllables

def check_iambic_pentameter(word_sequence, allow_trochaic_substitution=True):
    """
    Check if a sequence forms valid iambic pentameter.
    Allows trochaic substitution (10 instead of 01) at specific positions.
    Returns (is_valid, stress_pattern, has_substitution, substitution_positions)
    """
    full_stress, total_syllables = get_phrase_stress_pattern(word_sequence)
    
    if full_stress is None or total_syllables != 10:
        return False, full_stress if full_stress else "", False, []
    
    # Normalize secondary stress
    full_stress = full_stress.replace('2', '1')
    
    # Check strict iambic pattern: 0101010101
    is_strict_iambic = True
    for i in range(0, 10, 2):
        if full_stress[i] != '0' or full_stress[i+1] != '1':
            is_strict_iambic = False
            break
    
    if is_strict_iambic:
        return True, full_stress, False, []
    
    # If not strict iambic, check for trochaic substitution
    if allow_trochaic_substitution:
        substitution_positions = []
        is_valid_with_sub = True
        
        for i in range(0, 10, 2):
            foot_pattern = full_stress[i:i+2]
            
            if foot_pattern == '01':  # Normal iamb
                continue
            elif foot_pattern == '10':  # Trochaic substitution
                foot_number = (i // 2) + 1
                # Trochaic substitution most common in positions 1, 3, 4
                # Less common in position 5 (usually avoided)
                # Position 2 is rare but allowed
                if foot_number == 5:
                    # Allow but note it's unusual
                    substitution_positions.append(foot_number)
                else:
                    substitution_positions.append(foot_number)
            else:
                # Invalid foot pattern (00, 11, etc.)
                is_valid_with_sub = False
                break
        
        if is_valid_with_sub and len(substitution_positions) > 0:
            # Limit to reasonable number of substitutions (typically 1-2)
            if len(substitution_positions) <= 2:
                return True, full_stress, True, substitution_positions
    
    return False, full_stress, False, []

# =========================================================
#         SENTENCE AND PHRASE EXTRACTION
# =========================================================

def generate_free_text(model, tokenizer, prompt, num_tokens=300):
    """Generate poetry."""
    
    # Add poetry-like context to encourage verse generation
    poetry_prompt = f"{prompt}"
    
    prompt_ids = tokenizer(poetry_prompt, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=num_tokens,
            do_sample=True,
            top_p=0.85,  # Slightly higher for more diversity
            top_k=50,    # Add top_k for better quality
            temperature=0.9,  # Slightly higher for creativity
            repetition_penalty=1.25,  # Increased to reduce repetition
            no_repeat_ngram_size=2,  # Prevent repeating 3-grams
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = generated_text.replace(poetry_prompt, "", 1).strip()
    
    return generated_text

def extract_coherent_phrases(text):
    """
    Extract grammatically coherent phrases from text that could form
    iambic pentameter lines with minimal modification.
    """
    doc = nlp(text)
    phrases = []
    
    # Extract complete sentences
    for sent in doc.sents:
        sent_text = sent.text.strip()
        words = [clean_word(token.text) for token in sent if clean_word(token.text)]
        
        if len(words) > 0:
            stress, syllables = get_phrase_stress_pattern(words)
            
            phrases.append({
                'text': sent_text,
                'words': words,
                'tokens': [token for token in sent],
                'syllables': syllables,
                'stress': stress,
                'type': 'sentence',
                'grammatical': True
            })
    
    # Extract clauses (more granular than sentences)
    for sent in doc.sents:
        # Find verb phrases as clause boundaries
        root_verb = None
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                root_verb = token
                break
        
        if root_verb:
            # Extract subject + verb + object/complement
            clause_tokens = [root_verb]
            
            # Get subjects
            for child in root_verb.children:
                if child.dep_ in ('nsubj', 'nsubjpass'):
                    clause_tokens.extend([child] + list(child.subtree))
            
            # Get objects/complements
            for child in root_verb.children:
                if child.dep_ in ('dobj', 'attr', 'prep', 'advmod'):
                    clause_tokens.extend([child] + list(child.subtree))
            
            # Sort by position
            clause_tokens = sorted(set(clause_tokens), key=lambda t: t.i)
            
            if len(clause_tokens) >= 3:
                clause_text = ' '.join([t.text for t in clause_tokens])
                words = [clean_word(t.text) for t in clause_tokens if clean_word(t.text)]
                
                if len(words) > 0:
                    stress, syllables = get_phrase_stress_pattern(words)
                    
                    phrases.append({
                        'text': clause_text,
                        'words': words,
                        'tokens': clause_tokens,
                        'syllables': syllables,
                        'stress': stress,
                        'type': 'clause',
                        'grammatical': True
                    })
    
    # Extract noun phrases
    for chunk in doc.noun_chunks:
        words = [clean_word(token.text) for token in chunk if clean_word(token.text)]
        
        if len(words) > 0:
            stress, syllables = get_phrase_stress_pattern(words)
            
            phrases.append({
                'text': chunk.text,
                'words': words,
                'tokens': [token for token in chunk],
                'syllables': syllables,
                'stress': stress,
                'type': 'noun_phrase',
                'grammatical': True
            })
    
    # Extract verb phrases
    for token in doc:
        if token.pos_ == "VERB":
            vp_tokens = [token]
            for child in token.children:
                if child.dep_ in ('aux', 'auxpass', 'neg', 'advmod', 'dobj', 'prep'):
                    vp_tokens.extend([child] + list(child.subtree))
            
            vp_tokens = sorted(set(vp_tokens), key=lambda t: t.i)
            
            if len(vp_tokens) >= 2:
                vp_text = ' '.join([t.text for t in vp_tokens])
                words = [clean_word(t.text) for t in vp_tokens if clean_word(t.text)]
                
                if len(words) > 0:
                    stress, syllables = get_phrase_stress_pattern(words)
                    
                    phrases.append({
                        'text': vp_text,
                        'words': words,
                        'tokens': vp_tokens,
                        'syllables': syllables,
                        'stress': stress,
                        'type': 'verb_phrase',
                        'grammatical': True
                    })
    
    return phrases

def adjust_phrase_to_iambic(phrase, target_syllables=10):
    """
    Try to adjust a phrase to fit iambic pentameter by:
    1. Adding or removing words at edges
    2. Finding sub-phrases that fit
    3. Trying minor syllable adjustments
    Returns list of adjusted versions
    """
    adjusted = []
    words = phrase['words']
    tokens = phrase['tokens']
    
    # Try the phrase as-is
    if phrase['syllables'] == target_syllables:
        is_iambic, pattern, has_sub, sub_pos = check_iambic_pentameter(words)
        if is_iambic:
            adjusted.append({
                'text': phrase['text'],
                'words': words,
                'syllables': phrase['syllables'],
                'stress': phrase['stress'],
                'iambic': True,
                'modification': 'none',
                'trochaic_substitution': has_sub,
                'substitution_positions': sub_pos
            })
    
    # Try phrases within ±2 syllables and adjust
    syllable_tolerance = 2
    if abs(phrase['syllables'] - target_syllables) <= syllable_tolerance:
        # Try removing words from the start
        for i in range(1, min(3, len(words))):  # Only remove up to 2 words
            sub_words = words[i:]
            stress, syllables = get_phrase_stress_pattern(sub_words)
            
            if syllables == target_syllables:
                is_iambic, pattern, has_sub, sub_pos = check_iambic_pentameter(sub_words)
                if is_iambic:
                    sub_text = ' '.join([t.text for t in tokens[i:]])
                    adjusted.append({
                        'text': sub_text,
                        'words': sub_words,
                        'syllables': syllables,
                        'stress': stress,
                        'iambic': True,
                        'modification': f'removed_start_{i}',
                        'trochaic_substitution': has_sub,
                        'substitution_positions': sub_pos
                    })
        
        # Try removing words from the end
        for i in range(max(3, len(words) - 2), len(words)):  # Only remove up to 2 words
            sub_words = words[:i]
            stress, syllables = get_phrase_stress_pattern(sub_words)
            
            if syllables == target_syllables:
                is_iambic, pattern, has_sub, sub_pos = check_iambic_pentameter(sub_words)
                if is_iambic:
                    sub_text = ' '.join([t.text for t in tokens[:i]])
                    adjusted.append({
                        'text': sub_text,
                        'words': sub_words,
                        'syllables': syllables,
                        'stress': stress,
                        'iambic': True,
                        'modification': f'removed_end_{len(words)-i}',
                        'trochaic_substitution': has_sub,
                        'substitution_positions': sub_pos
                    })
    
    # Only try trimming both ends for longer phrases
    if len(words) > 6:
        for i in range(1, min(3, len(words))):
            for j in range(max(len(words) - 2, i + 3), len(words)):
                sub_words = words[i:j]
                if len(sub_words) < 3:
                    continue
                    
                stress, syllables = get_phrase_stress_pattern(sub_words)
                
                if syllables == target_syllables:
                    is_iambic, pattern, has_sub, sub_pos = check_iambic_pentameter(sub_words)
                    if is_iambic:
                        sub_text = ' '.join([t.text for t in tokens[i:j]])
                        adjusted.append({
                            'text': sub_text,
                            'words': sub_words,
                            'syllables': syllables,
                            'stress': stress,
                            'iambic': True,
                            'modification': f'trimmed_{i}_{len(words)-j}',
                            'trochaic_substitution': has_sub,
                            'substitution_positions': sub_pos
                        })
    
    return adjusted

def find_combinable_phrases(phrases, target_syllables=10):
    """
    Find pairs or groups of phrases that can combine to form iambic pentameter.
    """
    combinations = []
    
    for i, phrase1 in enumerate(phrases):
        if phrase1['syllables'] >= target_syllables:
            continue
        
        needed = target_syllables - phrase1['syllables']
        
        # Look for a second phrase that fills the gap
        for j, phrase2 in enumerate(phrases[i+1:], start=i+1):
            if phrase2['syllables'] == needed:
                # Try combining
                combined_words = phrase1['words'] + phrase2['words']
                is_iambic, pattern, has_sub, sub_pos = check_iambic_pentameter(combined_words)  # FIX: unpack 4 values
                
                if is_iambic:
                    combined_text = phrase1['text'] + ' ' + phrase2['text']
                    combinations.append({
                        'text': combined_text,
                        'words': combined_words,
                        'syllables': target_syllables,
                        'stress': pattern,
                        'iambic': True,
                        'components': [phrase1, phrase2],
                        'type': 'combined',
                        'trochaic_substitution': has_sub,
                        'substitution_positions': sub_pos
                    })
            
            # Also try if phrase2 is smaller than needed
            elif phrase2['syllables'] < needed:
                remaining = needed - phrase2['syllables']
                
                # Look for a third phrase
                for k, phrase3 in enumerate(phrases[j+1:], start=j+1):
                    if phrase3['syllables'] == remaining:
                        combined_words = phrase1['words'] + phrase2['words'] + phrase3['words']
                        is_iambic, pattern, has_sub, sub_pos = check_iambic_pentameter(combined_words)  # FIX: unpack 4 values
                        
                        if is_iambic:
                            combined_text = phrase1['text'] + ' ' + phrase2['text'] + ' ' + phrase3['text']
                            combinations.append({
                                'text': combined_text,
                                'words': combined_words,
                                'syllables': target_syllables,
                                'stress': pattern,
                                'iambic': True,
                                'components': [phrase1, phrase2, phrase3],
                                'type': 'combined',
                                'trochaic_substitution': has_sub,
                                'substitution_positions': sub_pos
                            })
    
    return combinations

# =========================================================
#         Avoid Repetitive Words
# =========================================================

def has_repeated_words(words):
    """
    Check if a line has repeated words (case-insensitive).
    Ignores common articles/conjunctions that can reasonably repeat.
    Returns True if problematic repetition found.
    """
    # Words that are OK to repeat
    allowed_repeats = {'the', 'a', 'an', 'and', 'or', 'but', 'to', 'of', 'in', 'on', 'at', 'for', 'with', 'is', 'are', 'was', 'were', 'it', 'its', "it's", 'that', 'this', 'as', 'by', 'from', 'be', 'not','he', 'she', 'or', 'they', 'we', 'you', 'his', 'hers', 'me', 'my', 'your', 'our', 'so', 'if', 'then'}
    
    # Count each word (lowercase, excluding allowed repeats)
    word_counts = {}
    for word in words:
        word_lower = word.lower()
        if word_lower not in allowed_repeats:
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
    
    # Check if any content word appears more than once
    for word, count in word_counts.items():
        if count > 1:
            return True, word
    
    return False, None


# =========================================================
#         COUPLET RECONSTRUCTION
# =========================================================

def find_iambic_pentameter_lines(text):
    """
    Extract all possible iambic pentameter lines from text.
    Returns list of candidate lines with metadata.
    Now filters out lines with repeated words.
    """
    print("\n[Analysis] Extracting coherent phrases from generated text...")
    phrases = extract_coherent_phrases(text)
    print(f"[Analysis] Found {len(phrases)} grammatical phrases")
    
    # Count by type
    phrase_types = {}
    for p in phrases:
        phrase_types[p['type']] = phrase_types.get(p['type'], 0) + 1
    print(f"[Analysis] Phrase breakdown: {phrase_types}")
    
    candidates = []
    
    # Try to adjust existing phrases to iambic pentameter
    print("\n[Analysis] Adjusting phrases to fit iambic pentameter...")
    for phrase in phrases:
        adjusted = adjust_phrase_to_iambic(phrase)
        candidates.extend(adjusted)
    
    print(f"[Analysis] Found {len(candidates)} phrases that fit IP through adjustment")
    
    # Try combining phrases
    print("\n[Analysis] Looking for combinable phrases...")
    combinations = find_combinable_phrases(phrases)
    candidates.extend(combinations)
    
    print(f"[Analysis] Found {len(combinations)} valid combinations")
    
    # FILTER: Remove candidates with repeated words
    before_filter = len(candidates)
    filtered_candidates = []
    
    for candidate in candidates:
        has_repeat, repeated_word = has_repeated_words(candidate['words'])
        if has_repeat:
            continue  # Skip this candidate
        filtered_candidates.append(candidate)
    
    candidates = filtered_candidates
    print(f"[Analysis] Filtered out {before_filter - len(candidates)} lines with repeated words")
    print(f"[Analysis] Total candidate IP lines: {len(candidates)}")
    
    # Score each candidate for naturalness
    for candidate in candidates:
        doc = nlp(candidate['text'])
        
        # Grammaticality score (now more stringent - max 100 points)
        grammatical_score = 0
        
        has_verb = any(token.pos_ == "VERB" for token in doc)
        has_subject = any(token.dep_ in ('nsubj', 'nsubjpass') for token in doc)
        has_object = any(token.dep_ in ('dobj', 'pobj', 'attr') for token in doc)
        has_proper_structure = has_verb and has_subject
        
        # More demanding scoring system
        if has_verb:
            grammatical_score += 30
        if has_subject:
            grammatical_score += 35
        if has_object:
            grammatical_score += 15
        if has_proper_structure:  # Bonus for having both subject and verb
            grammatical_score += 20
        
        # Check for complete sentence structure
        num_sentences = len(list(doc.sents))
        if num_sentences == 1:  # Single complete sentence
            # Check if it has proper sentence structure
            root = [token for token in doc if token.dep_ == "ROOT"]
            if root and root[0].pos_ == "VERB":
                grammatical_score += 20  # Bonus for proper sentence
        
        # Penalize fragments or run-ons
        if num_sentences == 0:
            grammatical_score -= 20
        elif num_sentences > 2:
            grammatical_score -= 10
        
        # Penalize if it starts with non-sentence starters
        if len(doc) > 0:
            first_token = doc[0]
            # Should start with DET, NOUN, PRON, ADJ, or ADV
            if first_token.pos_ not in ('DET', 'NOUN', 'PRON', 'ADJ', 'ADV', 'VERB'):
                grammatical_score -= 15
        
        # Ensure score is between 0-100
        grammatical_score = max(0, min(100, grammatical_score))
        
        candidate['grammatical_score'] = grammatical_score
        candidate['semantic_vector'] = doc.vector if doc.has_vector else None
    
    # Sort by grammatical score
    candidates.sort(key=lambda x: x['grammatical_score'], reverse=True)
    
    return candidates

def extract_ngrams(words, n):
    """Extract all n-grams from a list of words."""
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngrams.add(ngram)
    return ngrams

def calculate_word_repetition(words1, words2):
    """
    Calculate what percentage of words appear in both lines.
    Returns percentage of repeated words.
    """
    set1 = set(w.lower() for w in words1)
    set2 = set(w.lower() for w in words2)
    
    # Common words that are OK to repeat
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'for', 'with', 'these', 'those', 'is', 'are', 'was', 'were', 'it', 'its', "it's", 'that', 'this', 'as', 'by', 'from', 'be', 'not', 'me', 'you', 'us', 'her', 'she', 'he'}
    
    # Remove stopwords from consideration
    content1 = set1 - stopwords
    content2 = set2 - stopwords
    
    if not content1 or not content2:
        return 0.0
    
    # Calculate overlap of content words
    overlap = len(content1 & content2)
    total_unique = len(content1 | content2)
    
    return overlap / total_unique if total_unique > 0 else 0.0

def calculate_ngram_overlap(words1, words2, max_n=4):
    """
    Calculate the proportion of overlapping n-grams between two lines.
    Returns dict with overlap percentages for different n-gram sizes.
    """
    overlap_scores = {}
    
    for n in range(2, max_n + 1):
        ngrams1 = extract_ngrams(words1, n)
        ngrams2 = extract_ngrams(words2, n)
        
        if len(ngrams1) == 0 or len(ngrams2) == 0:
            overlap_scores[n] = 0.0
            continue
        
        # Count overlapping n-grams
        overlap = len(ngrams1 & ngrams2)
        total = len(ngrams1) + len(ngrams2)
        
        # Calculate overlap percentage
        overlap_pct = (2 * overlap) / total if total > 0 else 0
        overlap_scores[n] = overlap_pct
    
    return overlap_scores

def lines_too_similar(line1_words, line2_words, threshold=0.5):
    """
    Check if two lines are too similar based on n-gram overlap.
    Returns True if they share too many n-grams (indicating repetition).
    """
    overlap = calculate_ngram_overlap(line1_words, line2_words, max_n=4)
    
    # Reject if ANY n-gram size has >50% overlap
    # Bigrams (n=2) are especially important
    if overlap.get(2, 0) > threshold:
        return True, overlap
    if overlap.get(3, 0) > 0.3:  # Stricter for trigrams
        return True, overlap
    if overlap.get(4, 0) > 0.2:  # Even stricter for 4-grams
        return True, overlap
    
    return False, overlap

def find_rhyming_couplet(candidates, require_rhyme=True, min_grammatical_score=80):
    """
    Find a pair of iambic pentameter lines that rhyme.
    Ensures lines don't repeat the same n-grams.
    Only accepts lines with high grammatical scores.
    Ensures rhyming words are different words (not identical).
    Now checks ALL possible pairs and has stricter similarity requirements.
    """
    if len(candidates) < 2:
        return None
    
    best_couplet = None
    best_score = -1
    
    # Filter candidates first to only high-quality ones
    quality_candidates = [c for c in candidates if c['grammatical_score'] >= min_grammatical_score]
    
    print(f"[Pairing] Checking {len(quality_candidates)} quality candidates...")
    
    # Try to find rhyming pairs - check ALL combinations
    for i, line1 in enumerate(quality_candidates):
        line1_end = line1['words'][-1]
        line1_end_clean = clean_word(line1_end)
        
        # Check against ALL other candidates (not just those after i)
        for j, line2 in enumerate(quality_candidates):
            if i == j:  # Skip comparing a line with itself
                continue
                
            line2_end = line2['words'][-1]
            line2_end_clean = clean_word(line2_end)
            
            # FILTER: Reject if end words are identical
            if line1_end_clean == line2_end_clean:
                continue
            
            rhymes = check_rhyme(line1_end, line2_end)
            
            if require_rhyme and not rhymes:
                continue
            
            # CHECK: Word repetition (excluding common stopwords) - STRICTER THRESHOLD
            word_repetition = calculate_word_repetition(line1['words'], line2['words'])
            if word_repetition > 0.15:  # Reduced from 0.3 to 0.15 (15%)
                continue
            
            # CHECK: Ensure lines aren't too similar (no repeated n-grams)
            too_similar, overlap = lines_too_similar(line1['words'], line2['words'])
            
            if too_similar:
                continue
            
            # Calculate semantic similarity
            if line1.get('semantic_vector') is not None and line2.get('semantic_vector') is not None:
                similarity = np.dot(line1['semantic_vector'], line2['semantic_vector']) / (
                    np.linalg.norm(line1['semantic_vector']) * np.linalg.norm(line2['semantic_vector']) + 1e-10
                )
            else:
                similarity = 0.5
            
            # Penalize high semantic similarity (likely means repetitive content)
            # Sweet spot is moderate similarity (0.3-0.7)
            if similarity > 0.9:
                similarity_score = 50  # Penalize extreme similarity
            elif similarity > 0.7:
                similarity_score = 70
            else:
                similarity_score = similarity * 100
            
            # Calculate combined score
            score = (
                line1['grammatical_score'] * 0.3 +
                line2['grammatical_score'] * 0.3 +
                similarity_score * 0.2 +
                (100 if rhymes else 0) * 0.2
            )
            
            # Penalty for n-gram overlap (even if below threshold)
            avg_overlap = sum(overlap.values()) / len(overlap) if overlap else 0
            score -= avg_overlap * 50  # Reduce score based on overlap
            
            # Additional penalty for word repetition
            score -= word_repetition * 100  # More penalty for repeated words
            
            if score > best_score:
                best_score = score
                best_couplet = {
                    'line1': line1,
                    'line2': line2,
                    'rhymes': rhymes,
                    'similarity': similarity,
                    'ngram_overlap': overlap,
                    'word_repetition': word_repetition,
                    'score': score
                }
    
    if best_couplet:
        print(f"[Pairing] ✓ Found couplet with score {best_score:.1f}")
    else:
        print(f"[Pairing] ✗ No valid couplet found in this pool")
    
    return best_couplet

# =========================================================
#                MAIN GENERATION FUNCTION
# =========================================================

def generate_coherent_couplet(model, tokenizer, prompt, model_name="Model", max_attempts=20):
    """
    Generate coherent iambic pentameter couplet by preserving sentence structure.
    Accumulates all candidate lines across attempts for better pairing.
    """
    all_candidates = []  # Accumulate candidates across all attempts
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*60}")
        print(f"[{model_name}] ATTEMPT {attempt}/{max_attempts}")
        print(f"{'='*60}")
        
        # Generate free text
        print(f"[{model_name}] Generating {TOKENS_TO_GENERATE} tokens...")
        generated_text = generate_free_text(model, tokenizer, prompt, TOKENS_TO_GENERATE)
        print(f"[{model_name}] Generated text:\n{generated_text}...\n")
        
        # Find iambic pentameter lines
        new_candidates = find_iambic_pentameter_lines(generated_text)
        
        if len(new_candidates) == 0:
            print(f"[{model_name}] No valid IP lines found this attempt")
        else:
            print(f"[{model_name}] Found {len(new_candidates)} new candidate lines")
            
            # Add new candidates to accumulated pool
            all_candidates.extend(new_candidates)
            
            print(f"\n[{model_name}] Top 5 candidates from this attempt:")
            for i, cand in enumerate(new_candidates[:5]):
                print(f"  {i+1}. [{cand['grammatical_score']:.0f}] {cand['text']}")
        
        print(f"\n[{model_name}] Total accumulated candidates: {len(all_candidates)}")
        
        if len(all_candidates) == 0:
            print(f"[{model_name}] No candidates yet, continuing...")
            continue
        
        # Sort all accumulated candidates by quality
        all_candidates.sort(key=lambda x: x['grammatical_score'], reverse=True)
        
        # Try to find rhyming couplet from ALL accumulated candidates
        require_rhyme = (attempt <= MAX_RECONSTRUCTION_ATTEMPTS)
        
        print(f"\n[{model_name}] Searching for {'rhyming' if require_rhyme else 'any'} couplet in {len(all_candidates)} total candidates...")
        print(f"[{model_name}] Requiring grammatical score >= 80/100")
        
        # Count how many candidates meet the grammatical threshold
        high_quality_candidates = [c for c in all_candidates if c['grammatical_score'] >= 80]
        print(f"[{model_name}] Candidates meeting threshold: {len(high_quality_candidates)}/{len(all_candidates)}")
        
        result = find_rhyming_couplet(all_candidates, require_rhyme=require_rhyme, min_grammatical_score=80)
        
        if result:
            print(f"\n[{model_name}] ✓✓✓ COUPLET FOUND! ✓✓✓\n")
            
            line1_text = result['line1']['text']
            line2_text = result['line2']['text']
            
            couplet = line1_text + "\n" + line2_text
            
            return couplet, attempt, generated_text, result, all_candidates
        else:
            print(f"[{model_name}] No suitable couplet found yet, generating more candidates...")
    
    print(f"\n[{model_name}] ✗ Could not find a couplet after {max_attempts} attempts")
    print(f"[{model_name}] Total candidates accumulated: {len(all_candidates)}")
    return None, max_attempts, None, None, all_candidates

# =========================================================
#                     LOAD MODELS
# =========================================================
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"\n{'='*60}")
print("LOADING FINE-TUNED MODEL")
print(f"{'='*60}")
print(f"Loading model from {ADAPTER_PATH}...")

try:
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        ADAPTER_PATH,
        torch_dtype=torch.bfloat16 if bf16_available else torch.float16,
    )
    finetuned_model.to(device)
    finetuned_model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# =========================================================
#             GENERATE COUPLET
# =========================================================
print(f"\n{'='*60}")
print(f"GENERATING FROM PROMPT: '{PROMPT}'")
print(f"{'='*60}")

print(f"\n{'#'*60}")
print("COHERENT SENTENCE-BASED METHOD")
print("GENERATE → EXTRACT SENTENCES → ADJUST TO IP → FIND RHYMING PAIR")
print(f"{'#'*60}")

couplet, attempts, generated_text, result, candidates = generate_coherent_couplet(
    finetuned_model, tokenizer, PROMPT, model_name="Fine-tuned", max_attempts=MAX_GENERATION_ATTEMPTS
)

# =========================================================
#                     RESULTS
# =========================================================
print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}\n")

if couplet:
    print(f"Prompt: '{PROMPT}'\n")
    print("COHERENT IAMBIC PENTAMETER COUPLET:")
    print("-" * 40)
    print(couplet)
    print("-" * 40)
    print(f"\nAttempts needed: {attempts}")
    print(f"Lines rhyme: {result['rhymes']}")
    print(f"Semantic similarity: {result['similarity']:.3f}")
    print(f"Word repetition (content): {result['word_repetition']:.1%}")
    print(f"N-gram overlap: {result['ngram_overlap']}")
    print(f"Overall score: {result['score']:.1f}/100")
    
    print(f"\nLine 1 Analysis:")
    print(f"  Text: {result['line1']['text']}")
    print(f"  End word: '{result['line1']['words'][-1]}'")
    print(f"  Type: {result['line1'].get('type', 'combined')}")
    print(f"  Modification: {result['line1'].get('modification', 'none')}")
    print(f"  Grammatical score: {result['line1']['grammatical_score']:.0f}/100")
    print(f"  Stress pattern: {result['line1']['stress']}")
    if result['line1'].get('trochaic_substitution', False):
        sub_pos = result['line1'].get('substitution_positions', [])
        print(f"  ⚠ Trochaic substitution in foot/feet: {sub_pos}")
    
    print(f"\nLine 2 Analysis:")
    print(f"  Text: {result['line2']['text']}")
    print(f"  End word: '{result['line2']['words'][-1]}'")
    print(f"  Type: {result['line2'].get('type', 'combined')}")
    print(f"  Modification: {result['line2'].get('modification', 'none')}")
    print(f"  Grammatical score: {result['line2']['grammatical_score']:.0f}/100")
    print(f"  Stress pattern: {result['line2']['stress']}")
    if result['line2'].get('trochaic_substitution', False):
        sub_pos = result['line2'].get('substitution_positions', [])
        print(f"  ⚠ Trochaic substitution in foot/feet: {sub_pos}")
    
    print(f"\nOriginal generated text:")
    print("-" * 40)
    print(generated_text)
    print("-" * 40)
    
    print(f"\nTotal candidates found: {len(candidates) if candidates else 0}")
else:
    print(f"✗ Could not generate a couplet after {attempts} attempts")

print(f"\n{'='*60}")
print("COHERENCE METHOD SUMMARY")
print(f"{'='*60}")
print("✓ Preserves actual sentences and phrases from generated text")
print("✓ Uses SpaCy to identify grammatical structures")
print("✓ Adjusts phrases minimally to fit iambic pentameter")
print("✓ Combines shorter phrases when needed")
print("✓ Scores lines by grammaticality and semantic coherence")
print("✓ Only accepts lines with high grammatical scores (≥80/100)")
print("✓ Rejects pairs with identical end words")
print("✓ Allows trochaic substitution (up to 2 feet)")
print("✓ Much more coherent than word-pool reconstruction!")
print(f"{'='*60}")