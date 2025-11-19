import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
import re
from nltk.corpus import cmudict
import numpy as np

# =========================================================
#                     CONFIGURATION
# =========================================================
MODEL_NAME = "gpt2"
ADAPTER_PATH = r"C:\Users\micha\Desktop\projects\mercury\gpt2-finetuned-poetry-mercury-04\final_model"
PROMPT = "Running, reaching, rather"
MAX_GENERATION_ATTEMPTS = 100 
MAX_TOKENS_PER_LINE = 50 # Safety limit for tokens to prevent infinite loops

device = "cuda" if torch.cuda.is_available() else "cpu"
bf16_available = torch.cuda.is_bf16_supported()

# =========================================================
#                 LOAD WORD PHONEMES
# =========================================================
try:
    cmu = cmudict.dict()
except LookupError:
    nltk.download("cmudict")
    cmu = cmudict.dict()
    
# =========================================================
#                 STRESS PATTERN ANALYSIS
# =========================================================

def get_stress_and_syllables(word):
    """
    Returns the cleaned stress pattern string and syllable count.
    0 = unstressed, 1 = primary stress, 2 = secondary stress
    """
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    if word in cmu:
        # Get the first pronunciation
        phonemes = cmu[word][0]
        stress = ''.join([p[-1] for p in phonemes if p[-1].isdigit()])
        return stress, len(stress)
    return None, 0 # Word not found

def clean_line_for_check(line):
    """Removes punctuation for dictionary lookup but keeps spacing."""
    return re.sub(r"[^a-zA-Z' ]", "", line).strip()

def is_iambic_pentameter_viable(line):
    """
    Checks if a partial line is still viable for iambic pentameter generation.
    Returns (is_viable, total_syllables, current_stress_pattern)
    """
    clean_line = clean_line_for_check(line)
    words = clean_line.split()
    if not words:
        return True, 0, ""

    full_stress = ""
    total_syllables = 0
    for word in words:
        stress, syllables = get_stress_and_syllables(word)
        if stress is None or syllables == 0:
            # If any word is not in dictionary, we can't verify meter
            return False, 0, ""
        full_stress += stress
        total_syllables += syllables
    
    # Check if we've gone over the syllable limit
    if total_syllables > 10:
        return False, total_syllables, full_stress

    # Convert secondary stress (2) to primary (1) for simplified analysis
    full_stress = full_stress.replace('2', '1')

    # Check for iambic consistency: 010101...
    # i is the syllable index (0-indexed)
    for i, stress in enumerate(full_stress):
        expected_stress = '1' if i % 2 != 0 else '0' # Expect 0 at even index, 1 at odd index
        
        if stress != expected_stress:
            # We strictly reject any immediate substitution (0 where 1 is expected, or 1 where 0 is expected)
            return False, total_syllables, full_stress # Violated iambic pattern

    return True, total_syllables, full_stress

def check_iambic_pentameter_line(line):
    """
    Check if a line is true iambic pentameter.
    Returns (is_valid, feet, stress_pattern)
    """
    clean_line = clean_line_for_check(line)
    words = clean_line.split()
    if not words:
        return False, 0, ""
    
    full_stress = ""
    for word in words:
        stress, _ = get_stress_and_syllables(word)
        if stress is None:
            return False, 0, ""
        full_stress += stress
    
    if len(full_stress) != 10:
        return False, len(full_stress) / 2, full_stress
    
    full_stress = full_stress.replace('2', '1')
    
    # Check for perfect iambic pattern: 0101010101
    feet = 0
    is_valid = True
    
    for i in range(0, len(full_stress), 2):
        if i + 1 < len(full_stress):
            # Check for iamb (0-1)
            if full_stress[i] == '0' and full_stress[i+1] == '1':
                feet += 1
            else:
                is_valid = False
                break
    
    if is_valid and feet == 5:
        return True, feet, full_stress
    
    return False, feet, full_stress

# =========================================================
#                 CONSTRAINED GENERATION LOOP
# =========================================================

def generate_iambic_pentameter_line(model, tokenizer, prompt, model_name="Model", max_attempts=100):
    """
    Generates a single line of iambic pentameter by iteratively checking viability.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n[{model_name}] === LINE ATTEMPT {attempt}/{max_attempts} ===")
        current_line_tokens = prompt_ids.clone()
        
        # We start the line with the prompt and continue generating 
        # The prompt itself is NOT checked for iambic meter.

        for token_count in range(MAX_TOKENS_PER_LINE):
            
            # 1. Get the logits for the next token
            with torch.no_grad():
                outputs = model(current_line_tokens)
            next_token_logits = outputs.logits[:, -1, :]
            
            # 2. Apply Top-P and Top-K filtering for sampling (replicate huggingface settings)
            # You'd typically use a more sophisticated sampler like the Hugging Face Pipeline
            # but here we replicate basic sampling logic for demonstration.
            top_p = 0.75
            
            # --- Apply Top-P Sampling ---
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[:, indices_to_remove] = -float('Inf') # Mask probabilities

            # 3. Create a set of viable next tokens
            viable_next_token = None
            
            # Sample from the filtered logits until we find a viable token, or we give up.
            # We sample many times to try and find one that works.
            for _ in range(50): # Try up to 50 samples
                
                # Resample based on new masked logits
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                
                # Check for numerical stability/empty distribution
                if torch.isinf(next_token_logits).all():
                    print(f"[{model_name}] ✗ Distribution is empty (all tokens masked). Aborting line attempt.")
                    break

                next_token = torch.multinomial(probs, num_samples=1)
                
                # Tentatively add the new token
                tentative_ids = torch.cat([current_line_tokens, next_token], dim=-1)
                tentative_text = tokenizer.decode(tentative_ids[0], skip_special_tokens=True)
                
                # CRITICAL: Separate the prompt and the generated text
                generated_segment = tentative_text.replace(prompt, "", 1).strip()
                
                # Check the line viability (only the part following the prompt)
                is_viable, syllables, pattern = is_iambic_pentameter_viable(generated_segment)

                if is_viable:
                    viable_next_token = next_token
                    break # Found a viable token, stop sampling
                else:
                    # Mask the non-viable token to avoid sampling it again in this iteration
                    next_token_logits[:, next_token.item()] = -float('Inf')
            
            
            if viable_next_token is None:
                print(f"[{model_name}] ✗ No viable token found for this step. Restarting line attempt.")
                break # Break inner loop, restart full attempt

            # 4. Token is viable, update the current line
            current_line_tokens = torch.cat([current_line_tokens, viable_next_token], dim=-1)
            
            # 5. Check if a line is completed (10 syllables)
            if syllables == 10:
                final_line_text = generated_segment # This is the full line generated after the prompt
                
                # Now, check the completed 10-syllable line with the strict IP function
                is_valid, feet, final_pattern = check_iambic_pentameter_line(final_line_text)
                
                if is_valid:
                    print(f"\n[{model_name}] ✓✓✓ FOUND IAMBIC PENTAMETER! ✓✓✓")
                    print(f"[{model_name}] Line: {final_line_text}")
                    print(f"[{model_name}] Feet: {feet}")
                    print(f"[{model_name}] Stress pattern: {final_pattern}")
                    return final_line_text, feet, final_pattern, attempt
                else:
                    print(f"[{model_name}] ✗ 10 syllables, but not perfect iambic: {final_pattern}. Restarting line attempt.")
                    break # Break inner loop, restart full attempt
            
            # If we reach the maximum token count without 10 syllables, restart
            if token_count == MAX_TOKENS_PER_LINE - 1:
                 print(f"[{model_name}] ✗ Reached max tokens without completing a line. Restarting line attempt.")
                 break

    # If all attempts fail
    print(f"\n[{model_name}] ❌ Could not find iambic pentameter after {max_attempts} attempts")
    return None, 0, "", max_attempts

# =========================================================
#                     LOAD MODELS
# =========================================================
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"\n{'='*60}")
print("LOADING FINE-TUNED MODEL")
print(f"{'='*60}")
print(f"Loading fine-tuned model from {ADAPTER_PATH}...")
finetuned_model = AutoModelForCausalLM.from_pretrained(
    ADAPTER_PATH,
    torch_dtype=torch.bfloat16 if bf16_available else torch.float16,
)
finetuned_model.to(device)
finetuned_model.eval()
print("Fine-tuned model loaded successfully!")

# =========================================================
#         GENERATE ONLY IAMBIC PENTAMETER IS FOUND
# =========================================================
print(f"\n{'='*60}")
print(f"GENERATING FROM PROMPT: '{PROMPT}'")
print(f"{'='*60}\n")

# Generate from fine-tuned model
print(f"\n{'#'*60}")
print("FINE-TUNED MODEL")
print(f"{'#'*60}\n")

# NOTE: The new function attempts to generate ONE full line of iambic pentameter
finetuned_line, finetuned_feet, finetuned_pattern, finetuned_attempts = generate_iambic_pentameter_line(
    finetuned_model, tokenizer, PROMPT, model_name="Fine-tuned", max_attempts=MAX_GENERATION_ATTEMPTS
)

# =========================================================
#                     RESULTS
# =========================================================
print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}\n")

print(f"Prompt: '{PROMPT}' (EXCLUDED from iambic pentameter checking)\n")

print("\nFINE-TUNED MODEL:")
if finetuned_line:
    print(f"   Line found: {finetuned_line}")
    print(f"   Feet: {finetuned_feet}")
    print(f"   Stress pattern: {finetuned_pattern}")
    print(f"   Attempts needed: {finetuned_attempts}")
else:
    print(f"   ❌ Could not find iambic pentameter in {finetuned_attempts} attempts")

print(f"\n{'='*60}")