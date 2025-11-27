import re
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as tr_logging
import nltk
from nltk.corpus import cmudict
import spacy
from datetime import datetime

# --- Quiet down extraneous logs (transformers prints) ---
tr_logging.set_verbosity_error()
warnings.filterwarnings("ignore", ".*symlinks.*")

# --- CONFIG ---
MODEL_NAME = "gpt2-large"
ADAPTER_PATH = r"D:/models/gpt2-large-poetry-mercury-12-unfrozen-top-layers-010-50epochs-separate-txt-files/final_model_unfrozen"
PROMPT = """ 
fleece lid of eternity
near silvery dunes
"""

GENERATION_KWARGS = {
    "max_length": 350,
    "min_length": 75,
    "num_return_sequences": 1,
    "do_sample": True,
    "top_k": 0,
    "top_p": 0.5,
    # "temperature": 1.5,
    "repetition_penalty": 1.25,
    # "no_repeat_ngram_size": 3,
    "pad_token_id": None  # Will be set below    
}

device = "cuda" if torch.cuda.is_available() else "cpu"
bf16_available = torch.cuda.is_bf16_supported()

# --- Load spaCy model ---
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# --- CMU DICT (syllable counting) ---
try:
    CMU = cmudict.dict()
except LookupError:
    nltk.download("cmudict")
    CMU = cmudict.dict()


def count_syllables(word: str) -> int:
    w = word.lower()
    if w in CMU:
        return min(len([ph for ph in pron if ph[-1].isdigit()]) for pron in CMU[w])
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_v = False
    for ch in w:
        if ch in vowels:
            if not prev_v:
                count += 1
            prev_v = True
        else:
            prev_v = False
    if w.endswith("e"):
        count = max(1, count - 1)
    return max(1, count)


def count_syllables_line(line: str) -> int:
    return sum(count_syllables(w) for w in line.split())


def clean_text(t: str) -> str:
    t = t.replace("<|endoftext|>", "")
    t = re.sub(r"[\r\n]+", " ", t)
    return t.strip()


# --- Load tokenizer & model (and set pad token consistently) ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Ensure pad token exists and is same as eos (GPT-2 has no pad by default)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
tokenizer.pad_token_id = tokenizer.pad_token_id

print("Loading model (fine-tuned checkpoint)...")
model = AutoModelForCausalLM.from_pretrained(
    ADAPTER_PATH,
    dtype=torch.bfloat16 if bf16_available else torch.float16,
)
model.to(device)
model.eval()

# Make sure model config knows pad_token_id too
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.eos_token_id


def generate_poetic_text(model, tokenizer, prompt):
    """Generate a chunk of poetic text to extract phrases from."""
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    
    # Set pad_token_id in generation kwargs
    gen_kwargs = GENERATION_KWARGS.copy()
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs
    )
    
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    text = clean_text(text)
    
    # Remove the original prompt
    continuation = text.replace(prompt, "").strip()
    
    return continuation


def get_span_with_modifiers(token):
    """Get a token along with its modifiers (adjectives, determiners, etc)."""
    span = [token]
    
    # Add modifiers that come before
    for child in token.children:
        if child.dep_ in ("det", "amod", "compound", "poss") and child.i < token.i:
            span.insert(0, child)
    
    return sorted(span, key=lambda t: t.i)


def clean_phrase(phrase):
    """Clean the phrase: remove extra whitespace and non-letter characters."""
    # Keep letters, apostrophes, and spaces
    cleaned = re.sub(r"[^a-zA-Z' ]+", " ", phrase)
    # Remove extra whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def is_valid_phrase(phrase):
    """Check if a phrase meets minimum quality criteria."""
    if not phrase:
        return False
    
    words = phrase.split()
    
    # At least 2 words
    if len(words) < 2:
        return False
    
    # Not too long (for haiku generation, shorter is better)
    if len(words) > 7:
        return False
    
    # Filter out phrases that are just articles and common words
    stop_words = {"the", "a", "an", "this", "that", "these", "those"}
    content_words = [w for w in words if w.lower() not in stop_words]
    
    if len(content_words) < 1:
        return False
    
    return True


def extract_phrases(text):
    """
    Extract meaningful phrases using spaCy's linguistic analysis.
    Focuses on subject-verb constructions and noun chunks for semantic coherence.
    """
    doc = nlp(text)
    phrases = []
    
    # Strategy 1: Extract subject-verb-object patterns
    for token in doc:
        if token.pos_ == "VERB":
            # Get the subject(s)
            subjects = [child for child in token.children 
                       if child.dep_ in ("nsubj", "nsubjpass")]
            
            # Get direct objects or complements
            objects = [child for child in token.children 
                      if child.dep_ in ("dobj", "attr", "pobj", "dative")]
            
            # Build phrases with subject + verb
            for subj in subjects:
                # Get the full subject span (including modifiers)
                subj_span = get_span_with_modifiers(subj)
                verb_span = [token]
                
                # Add auxiliary verbs
                aux_verbs = [child for child in token.children 
                           if child.dep_ in ("aux", "auxpass")]
                verb_span = sorted(aux_verbs + verb_span, key=lambda t: t.i)
                
                phrase_text = " ".join([t.text for t in subj_span + verb_span])
                phrase_text = clean_phrase(phrase_text)
                if is_valid_phrase(phrase_text):
                    phrases.append(phrase_text)
                
                # Also create subject + verb + object phrases
                for obj in objects:
                    obj_span = get_span_with_modifiers(obj)
                    full_phrase = " ".join([t.text for t in subj_span + verb_span + obj_span])
                    full_phrase = clean_phrase(full_phrase)
                    if is_valid_phrase(full_phrase):
                        phrases.append(full_phrase)
    
    # Strategy 2: Extract meaningful noun chunks
    for chunk in doc.noun_chunks:
        cleaned = clean_phrase(chunk.text)
        if is_valid_phrase(cleaned):
            phrases.append(cleaned)
    
    # Strategy 3: Extract prepositional phrases with their objects
    for token in doc:
        if token.pos_ == "ADP":  # Preposition
            prep_phrase = []
            prep_phrase.append(token)
            
            # Get the object of preposition
            pobj = [child for child in token.children if child.dep_ == "pobj"]
            for obj in pobj:
                obj_span = get_span_with_modifiers(obj)
                phrase_text = " ".join([t.text for t in prep_phrase + obj_span])
                phrase_text = clean_phrase(phrase_text)
                if is_valid_phrase(phrase_text):
                    phrases.append(phrase_text)
    
    # Strategy 4: Also fall back to simple comma/punctuation splits for poetic phrases
    # This helps capture phrases that might not fit grammatical patterns
    chunks = re.split(r'[.!?\n]+', text)
    for chunk in chunks:
        sub_phrases = re.split(r'[,;]+', chunk)
        for phrase in sub_phrases:
            cleaned = re.sub(r"[^a-zA-Z' ]+", "", phrase).strip()
            if cleaned and len(cleaned.split()) >= 2 and len(cleaned.split()) <= 7:
                phrases.append(cleaned)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_phrases = []
    for phrase in phrases:
        phrase_lower = phrase.lower()
        if phrase_lower not in seen:
            seen.add(phrase_lower)
            unique_phrases.append(phrase)
    
    return unique_phrases


def find_all_haiku_combinations(phrases, max_results=50):
    """
    Find ALL valid combinations of 3 phrases that form a 5-7-5 haiku.
    Returns a list of tuples [(line1, line2, line3), ...] or empty list.
    """
    # Score each phrase by syllable count
    phrase_data = []
    for phrase in phrases:
        syl = count_syllables_line(phrase)
        if 3 <= syl <= 9:  # Only consider reasonable syllable counts
            phrase_data.append((phrase, syl))
    
    if len(phrase_data) < 3:
        return []
    
    # Organize phrases by syllable count for efficiency
    five_syl = [(p, i) for i, (p, s) in enumerate(phrase_data) if s == 5]
    seven_syl = [(p, i) for i, (p, s) in enumerate(phrase_data) if s == 7]
    
    haikus = []
    
    # Find all exact 5-7-5 combinations
    for p1, i1 in five_syl:
        for p2, i2 in seven_syl:
            if i1 == i2:  # Skip if same phrase
                continue
            for p3, i3 in five_syl:
                if i3 == i1 or i3 == i2:  # Skip if same phrase
                    continue
                
                # Valid haiku found!
                haikus.append((p1, p2, p3))
                
                if len(haikus) >= max_results:
                    return haikus
    
    return haikus


def generate_haiku_from_sample(model, tokenizer, prompt, attempt_num):
    """Generate text, extract phrases, and assemble into ALL possible haikus."""
    print(f"\n{'='*60}")
    print(f"Attempt #{attempt_num}")
    print(f"{'='*60}")
    
    # Step 1: Generate poetic text
    print("\n[1] Generating poetic text...")
    generated_text = generate_poetic_text(model, tokenizer, prompt)
    print(f"\nGenerated text (full):\n{generated_text}\n")
    
    # Step 2: Extract phrases using spaCy
    print("[2] Extracting phrases from generated text (using spaCy)...")
    phrases = extract_phrases(generated_text)
    print(f"\nFound {len(phrases)} phrases:")
    for i, phrase in enumerate(phrases[:20], 1):  # Show first 20
        syl = count_syllables_line(phrase)
        print(f"  {i}. ({syl} syl) {phrase}")
    if len(phrases) > 20:
        print(f"  ... and {len(phrases) - 20} more")
    
    # Step 3: Find ALL haiku combinations
    print("\n[3] Searching for ALL 5-7-5 combinations...")
    haikus = find_all_haiku_combinations(phrases, max_results=50)
    
    if haikus:
        print(f"\n✓ Found {len(haikus)} valid haiku(s)!")
        print("\n" + "="*60)
        print("🌿 ALL GENERATED HAIKUS:")
        print("="*60)
        
        for idx, (l1, l2, l3) in enumerate(haikus, 1):
            s1, s2, s3 = count_syllables_line(l1), count_syllables_line(l2), count_syllables_line(l3)
            print(f"\nHaiku #{idx} ({s1}-{s2}-{s3}):")
            print(f"  {l1}")
            print(f"  {l2}")
            print(f"  {l3}")
        
        print("="*60)
        return haikus
    else:
        print("\n✗ Could not find valid 5-7-5 combination from these phrases")
        return None


# --- Run generation attempts ---
print("\n" + "="*60)
print("HAIKU GENERATOR - Sample First, Assemble ALL Possible (with spaCy)")
print("="*60)
print(f"\nPrompt: \"{PROMPT}\"")

all_haikus_found = []

for attempt in range(5):  # Reduced attempts since we're finding multiple per generation
    haikus = generate_haiku_from_sample(model, tokenizer, PROMPT, attempt + 1)
    if haikus:
        all_haikus_found.extend(haikus)

if all_haikus_found:
    print("\n" + "="*60)
    print(f"🌿 SUMMARY: Found {len(all_haikus_found)} total haiku(s) across all attempts")
    print("="*60)
    
    # Show top 10 unique haikus
    seen = set()
    unique_haikus = []
    for haiku in all_haikus_found:
        haiku_str = str(haiku)
        if haiku_str not in seen:
            seen.add(haiku_str)
            unique_haikus.append(haiku)
    
    print(f"\nShowing {min(10, len(unique_haikus))} unique haiku(s):\n")
    for idx, (l1, l2, l3) in enumerate(unique_haikus[:10], 1):
        print(f"#{idx}:")
        print(f"  {l1}")
        print(f"  {l2}")
        print(f"  {l3}")
        print()
    
    # Save all unique haikus to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"haiku-output-{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(f"HAIKU GENERATION OUTPUT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Prompt: {PROMPT.strip()}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total haikus found: {len(unique_haikus)}\n")
        f.write("="*60 + "\n\n")
        
        for idx, (l1, l2, l3) in enumerate(unique_haikus, 1):
            s1, s2, s3 = count_syllables_line(l1), count_syllables_line(l2), count_syllables_line(l3)
            f.write(f"Haiku #{idx} ({s1}-{s2}-{s3}):\n")
            f.write(f"  {l1}\n")
            f.write(f"  {l2}\n")
            f.write(f"  {l3}\n")
            f.write("\n")
    
    print(f"\n✓ Saved {len(unique_haikus)} unique haikus to: {filename}")
    
else:
    print("\n" + "="*60)
    print("❌ Could not generate valid haikus after all attempts.")
    print("Try running again or adjusting the prompt.")
    print("="*60)