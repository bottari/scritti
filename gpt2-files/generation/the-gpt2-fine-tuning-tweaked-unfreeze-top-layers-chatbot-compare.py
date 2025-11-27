import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# Suppress some common warnings
warnings.filterwarnings("ignore", "huggingface_hub cache-system uses symlinks")
warnings.filterwarnings("ignore", "You are using a model of type gpt2")

# --- CONFIGURATION ---
MODEL_NAME = "gpt2-large"
ADAPTER_PATH = "D:/models/gpt2-large-poetry-mercury-12-unfrozen-top-layers-010-50epochs-separate-txt-files/final_model_unfrozen"
prompt = """material worth, wrists shaken"""

# Generation parameters (kept consistent for fair comparison)
GENERATION_KWARGS = {
    "max_length": 750,
    "min_length": 150,
    "num_return_sequences": 1,
    "do_sample": True,
    "top_k": 0,
    "top_p": 0.5,
    "temperature": 1.5,
    "repetition_penalty": 1.025,
    "no_repeat_ngram_size": 3,
    "pad_token_id": None # Will be set below    
}

# Setup device and precision
device = "cuda" if torch.cuda.is_available() else "cpu"
bf16_available = torch.cuda.is_bf16_supported()

# --- STEP 1: LOAD BASE MODEL & TOKENIZER ---
print(f"Loading base model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if bf16_available else torch.float16,
)
model.to(device)
model.eval()

# Finalize generation settings
GENERATION_KWARGS["pad_token_id"] = tokenizer.eos_token_id
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
streamer = TextStreamer(tokenizer, skip_prompt=False) # Set up the streamer

# --- STEP 2: GENERATE WITH BASE MODEL (STREAMING) ---
print("\n" + "="*80)
print(f"--- BASE MODEL (GPT-2 Large) STREAMING: '{prompt}' ---")
print("="*80)

with torch.no_grad():
    model.generate(
        input_ids,
        streamer=streamer, # Pass the streamer object
        **GENERATION_KWARGS
    )
print("\n" + "-"*80) # Separator after streaming is complete


# --- STEP 3: LOAD FINE-TUNED CHECKPOINT ---
print(f"Loading fine-tuned checkpoint from: {ADAPTER_PATH}")

try:
    # Overwrite the 'model' variable with the fine-tuned checkpoint
    model = AutoModelForCausalLM.from_pretrained(
        ADAPTER_PATH,
        torch_dtype=torch.bfloat16 if bf16_available else torch.float16,
    )
    # Move the new model to the device
    model.to(device)
    model.eval()
    print("✅ Fine-tuned model loaded successfully.")

except Exception as e:
    print(f"\n❌ ERROR: Failed to load fine-tuned model from {ADAPTER_PATH}")
    print(f"Details: {e}")
    exit()

# --- STEP 4: GENERATE WITH FINE-TUNED MODEL (STREAMING) ---
print("\n" + "="*80)
print(f"--- FINE-TUNED MODEL (POSEIA DI MICHELE BOTTARI) STREAMING: '{prompt}' ---")
print("="*80)

with torch.no_grad():
    model.generate(
        input_ids,
        streamer=streamer, # Pass the streamer object
        **GENERATION_KWARGS
    )

print("\n" + "="*80)