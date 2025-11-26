import torch
import warnings
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# --- CONFIGURATION ---
BASE_MODEL_NAME = "gpt2-large"
FINE_TUNED_PATH = r"D:\models\gpt2-large-finetuned-poetry-mercury-25-fixed-003\final_model"
PROMPT = "repeating in wind"

# Generation parameters (Optimized for poetry)
GENERATION_KWARGS = {
    "max_new_tokens": 250,
    "min_length": 50,
    "num_return_sequences": 1,
    "do_sample": True,
    "top_k": 0,
    "top_p": 0.75,
    # "temperature": 0.5,
    "repetition_penalty": 1.25,
    "bad_words_ids": None
}

# Suppress warnings
warnings.filterwarnings("ignore", "huggingface_hub cache-system uses symlinks")
warnings.filterwarnings("ignore", "You are using a model of type gpt2")

def main():
    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16_available = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_available else torch.float16
    print(f"🚀 Running on: {device.upper()} (Precision: {dtype})")

    # ==========================================
    # PART 1: BASE MODEL GENERATION
    # ==========================================
    print(f"\n1️⃣  Loading Base Model: {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=dtype)
    model.to(device)
    model.eval()

    # Setup inputs for Base Model
    GENERATION_KWARGS["pad_token_id"] = tokenizer.eos_token_id
    GENERATION_KWARGS["eos_token_id"] = tokenizer.eos_token_id
    
    # Prepare prompt with attention mask to silence warnings
    full_prompt = tokenizer.bos_token + PROMPT if tokenizer.bos_token else PROMPT
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    
    # Streamer fix: skip_special_tokens=True hides <|endoftext|>
    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)

    print("\n" + "="*60)
    print(f"--- BASE MODEL (GPT-2 Large) STREAMING: '{PROMPT}' ---")
    print("="*60)

    with torch.no_grad():
        model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask, 
            streamer=streamer, 
            **GENERATION_KWARGS
        )

    print("\n" + "-"*60)
    print("Base model generation complete. Switching models...")
    
    # CLEANUP: Free VRAM before loading the next model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ==========================================
    # PART 2: FINE-TUNED MODEL GENERATION
    # ==========================================
    print(f"\n2️⃣  Loading Fine-Tuned Model: {FINE_TUNED_PATH}...")
    
    try:
        # Reload tokenizer from fine-tuned path (crucial for custom tokens/settings)
        tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_PATH)
        model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_PATH, torch_dtype=dtype)
        model.to(device)
        model.eval()
        print("✅ Fine-tuned model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading fine-tuned model: {e}")
        return

    # Diagnostic: Check Newline ID for the fine-tuned tokenizer
    newline_id = tokenizer.encode('\n', add_special_tokens=False)[0]
    print(f"🔍 Diagnostic: Newline token ID is [{newline_id}]")

    # Setup inputs for Fine-Tuned Model
    GENERATION_KWARGS["pad_token_id"] = tokenizer.eos_token_id
    GENERATION_KWARGS["eos_token_id"] = tokenizer.eos_token_id
    
    full_prompt = tokenizer.bos_token + PROMPT if tokenizer.bos_token else PROMPT
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    
    # Streamer fix: skip_special_tokens=True hides <|endoftext|>
    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)

    print("\n" + "="*60)
    print(f"--- FINE-TUNED MODEL STREAMING: '{PROMPT}' ---")
    print("="*60)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            streamer=streamer, 
            **GENERATION_KWARGS
        )

    # ==========================================
    # PART 3: DIAGNOSTICS (Fine-Tuned Only)
    # ==========================================
    print("\n" + "="*60)
    print("📊 FINE-TUNED DIAGNOSTICS")
    print("="*60)

    # Decode without special tokens for analysis
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Isolate generated content
    if generated_text.startswith(PROMPT):
        generated_only = generated_text[len(PROMPT):]
    else:
        generated_only = generated_text

    visual_newlines = generated_only.count('\n')
    token_newlines = (outputs[0] == newline_id).sum().item()

    print(f"📝 Raw Output Text:\n{generated_only!r}")
    print("-" * 60)
    print(f"✅ Visual lines generated: {visual_newlines + 1}")
    print(f"✅ Newline characters (\\n) found: {visual_newlines}")
    print(f"✅ Newline token IDs found in tensor: {token_newlines}")

    if visual_newlines > 0:
        print("\n🎉 SUCCESS: The model is correctly generating line breaks!")
    elif token_newlines > 0:
        print("\n⚠️  WARNING: Newline tokens exist in data but aren't decoding visually.")
    else:
        print("\n❌ FAILURE: No line breaks generated.")

if __name__ == "__main__":
    main()