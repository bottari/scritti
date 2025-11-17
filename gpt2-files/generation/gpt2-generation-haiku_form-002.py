import re
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as tr_logging
import nltk
from nltk.corpus import cmudict
from itertools import combinations

# --- Quiet down extraneous logs (transformers prints) ---
tr_logging.set_verbosity_error()
warnings.filterwarnings("ignore", ".*symlinks.*")

# --- CONFIG ---
MODEL_NAME = "gpt2"
ADAPTER_PATH = r"C:\Users\micha\Desktop\projects\mercury\gpt2-finetuned-poetry-mercury-04--copy-attempt\final_model"  # r"C:\Users\micha\Desktop\projects\mercury\gpt2-finetuned-poetry-mercury-04\final_model"
PROMPT = """ 
fleece lid of eternity
near silvery dunes
"""

GENERATION_KWARGS = {
    "max_length": 500,
    "min_length": 50,
    "num_return_sequences": 1,
    "do_sample": True,
    "top_k": 0,
    "top_p": 0.75,
    # "temperature": 1.5,
    "repetition_penalty": 1.25,
    # "no_repeat_ngram_size": 3,
    "pad_token_id": None  # Will be set below    
}

device = "cuda" if torch.cuda.is_available() else "cpu"
bf16_available = torch.cuda.is_bf16_supported()

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


def extract_phrases(text):
    """Break text into phrases based on natural breaks (commas, line breaks, etc)."""
    # First, split on sentence-ending punctuation and newlines
    chunks = re.split(r'[.!?\n]+', text)
    
    phrases = []
    for chunk in chunks:
        # Further split on commas and semicolons
        sub_phrases = re.split(r'[,;]+', chunk)
        for phrase in sub_phrases:
            # Clean: keep only letters, apostrophes, and spaces
            cleaned = re.sub(r"[^a-zA-Z' ]+", "", phrase).strip()
            if cleaned and len(cleaned.split()) >= 2:  # At least 2 words
                phrases.append(cleaned)
    
    return phrases


def find_haiku_combination(phrases, max_search=1000):
    """
    Try to find a combination of 3 phrases that form a valid 5-7-5 haiku.
    Returns (line1, line2, line3) or None.
    """
    # Score each phrase by syllable count
    phrase_data = []
    for phrase in phrases:
        syl = count_syllables_line(phrase)
        if 3 <= syl <= 9:  # Only consider reasonable syllable counts
            phrase_data.append((phrase, syl))
    
    if len(phrase_data) < 3:
        return None
    
    # Try to find combinations that match 5-7-5
    # First, try to find exact matches
    five_syl = [p for p, s in phrase_data if s == 5]
    seven_syl = [p for p, s in phrase_data if s == 7]
    
    if len(five_syl) >= 2 and len(seven_syl) >= 1:
        # Perfect! We have exact matches
        return (five_syl[0], seven_syl[0], five_syl[1] if len(five_syl) > 1 else five_syl[0])
    
    # If no exact matches, try close combinations (within 1 syllable)
    attempts = 0
    for i, (p1, s1) in enumerate(phrase_data):
        if attempts >= max_search:
            break
        if abs(s1 - 5) > 1:  # Line 1 should be close to 5
            continue
        for j, (p2, s2) in enumerate(phrase_data):
            if i == j or abs(s2 - 7) > 1:  # Line 2 should be close to 7
                continue
            for k, (p3, s3) in enumerate(phrase_data):
                if k == i or k == j or abs(s3 - 5) > 1:  # Line 3 should be close to 5
                    continue
                
                attempts += 1
                # Check if this is a valid haiku
                if s1 == 5 and s2 == 7 and s3 == 5:
                    return (p1, p2, p3)
    
    return None


def generate_haiku_from_sample(model, tokenizer, prompt, attempt_num):
    """Generate text, extract phrases, and assemble into haiku."""
    print(f"\n{'='*60}")
    print(f"Attempt #{attempt_num}")
    print(f"{'='*60}")
    
    # Step 1: Generate poetic text
    print("\n[1] Generating poetic text...")
    generated_text = generate_poetic_text(model, tokenizer, prompt)
    print(f"\nGenerated text (full):\n{generated_text}\n")
    
    # Step 2: Extract phrases
    print("[2] Extracting phrases from generated text...")
    phrases = extract_phrases(generated_text)
    print(f"\nFound {len(phrases)} phrases:")
    for i, phrase in enumerate(phrases[:15], 1):  # Show first 15
        syl = count_syllables_line(phrase)
        print(f"  {i}. ({syl} syl) {phrase}")
    if len(phrases) > 15:
        print(f"  ... and {len(phrases) - 15} more")
    
    # Step 3: Find haiku combination
    print("\n[3] Searching for 5-7-5 combination...")
    haiku = find_haiku_combination(phrases)
    
    if haiku:
        l1, l2, l3 = haiku
        s1, s2, s3 = count_syllables_line(l1), count_syllables_line(l2), count_syllables_line(l3)
        print(f"\n✓ Found valid haiku ({s1}-{s2}-{s3}):")
        print(f"  Line 1: {l1}")
        print(f"  Line 2: {l2}")
        print(f"  Line 3: {l3}")
        return haiku
    else:
        print("\n✗ Could not find valid 5-7-5 combination from these phrases")
        return None


# --- Run generation attempts ---
print("\n" + "="*60)
print("HAIKU GENERATOR - Sample First, Assemble Second")
print("="*60)
print(f"\nPrompt: \"{PROMPT}\"")

for attempt in range(15):  # Reduced attempts since this is more efficient
    haiku = generate_haiku_from_sample(model, tokenizer, PROMPT, attempt + 1)
    if haiku:
        print("\n" + "="*60)
        print("🌿 FINAL HAIKU (valid 5-7-5):")
        print("="*60)
        for line in haiku:
            print(line)
        print("="*60)
        break
else:
    print("\n" + "="*60)
    print("❌ Could not generate valid haiku after 15 attempts.")
    print("Try running again or adjusting the prompt.")
    print("="*60)