import os
import torch
import warnings
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer
)
from peft import PeftModel
from huggingface_hub import login 
from peft.tuners.lora.layer import LoraLayer 
import copy 

# --- CONFIGURATION ---
# 1. TOKEN: Paste your Hugging Face token here (Must have Read access)
HF_TOKEN = "" 

# 2. MODEL PATHS: Must match the paths used in train_llama_poetry.py
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B"
ADAPTER_PATH = r"D:\models\llama3-8b-poetry-mercury-25-qlora-012\final_modelv2"

# 3. GENERATION SETTINGS
MAX_NEW_TOKENS = 256
TEMPERATURE = 1.8
TOP_P = 0.45
TOP_K = 150
REPETITION_PENALTY = 1.2 
NUM_RETURN_SEQUENCES = 1

PROMPT = "Coming upon the lake at night"

# Suppress warnings
warnings.filterwarnings("ignore")

# --- UTILITIES ---

def setup_auth():
    """Handles authentication explicitly if token is provided."""
    if HF_TOKEN:
        print(f"🔐 Logging in with explicit token...")
        login(token=HF_TOKEN)
    else:
        print("ℹ️  No explicit token in script. Relying on 'huggingface-cli login'...")

def print_diagnostics():
    """Check and print system and GPU info."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {gpu_memory:.1f} GB")
    else:
        print("❌ NO GPU DETECTED. Generation requires a GPU if the model was trained with QLoRA.")
        exit()

def get_base_model_and_tokenizer(bnb_config):
    """Loads the pure base model (without any adapter attachment)."""
    print(f"\n🔄 Loading base model ({BASE_MODEL_NAME}) and tokenizer...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            token=HF_TOKEN if HF_TOKEN else True
        )
    except OSError as e:
        print(f"\n❌ BASE MODEL ACCESS ERROR: {e}")
        print(f"1. Go to: https://huggingface.co/{BASE_MODEL_NAME}")
        print("2. Click 'Accept License' (Wait 1-2 mins if you just did it)")
        print("3. Ensure your token in the script has 'Read' permissions.")
        exit()
        
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN if HF_TOKEN else True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer

def get_fine_tuned_model(base_model, ADAPTER_PATH):
    """Attaches the LoRA adapter to a base model instance."""
    print(f"🔗 Attaching LoRA adapter from {ADAPTER_PATH}...")
    if not os.path.exists(ADAPTER_PATH):
        print(f"❌ ADAPTER NOT FOUND: Please ensure {ADAPTER_PATH} exists and contains your saved LoRA files.")
        exit()
        
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    peft_model.eval() 
    return peft_model
        
def check_lora_activation(model, label):
    """
    DEBUG FUNCTION: Checks if the model instance contains any LoRA layers 
    (PeftModel/LoraLayer types) to confirm its state.
    """
    lora_found = False
    
    if isinstance(model, PeftModel):
        print(f"🔍 DEBUG CHECK ({label}): Model is wrapped by PeftModel (Adapter is active).")
        lora_found = True
    else:
        print(f"🔍 DEBUG CHECK ({label}): Model is a standard AutoModelForCausalLM (Expected for base).")

    lora_layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            lora_layer_count += 1
            lora_found = True
            
    if lora_layer_count > 0:
        print(f"✅ LoRA Layers Found: {lora_layer_count} instances of LoraLayer.")
    elif isinstance(model, PeftModel):
         pass
    else:
        print(f"❌ No LoraLayer instances found in model modules.")
        
    return lora_found

def stream_generation(model, tokenizer, prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty, label):
    """
    Generates text token-by-token and streams the output 
    directly to the console using TextStreamer.
    """
    print(f"\n==================== {label} =====================")
    print(f"PROMPT: {prompt}")
    print(f"Settings: Temp={temperature}, Top_P={top_p}, Top_K={top_k}, Repetition Penalty={repetition_penalty}")
    print("\n--- Continuation (Streaming Output) ---")

    # Prepare input
    input_text = tokenizer.bos_token + prompt 
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Define stopping criteria (Llama 3 EOT token ID)
    llama_3_eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    stopping_ids = [tokenizer.eos_token_id, llama_3_eot_id]
    
    # Initialize the TextStreamer
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    with torch.no_grad():
        model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id, 
            eos_token_id=stopping_ids,
            repetition_penalty=repetition_penalty,
            streamer=streamer
        )
    
    print("----------------------------------------------------------")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    setup_auth()
    print_diagnostics()

    # QUANTIZATION CONFIG
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("\n✨ Starting Poetry Generation Comparison...")
    
    # --- RUN GENERATION on Base Model (PURE) ---
    pure_base_model, tokenizer = get_base_model_and_tokenizer(bnb_config)
    
    check_lora_activation(pure_base_model, "Pure Base Model")

    stream_generation(  # Changed from non_streaming_generation
        model=pure_base_model, 
        tokenizer=tokenizer,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        label="Base Model (Llama-3.1-8B, PURE INSTANCE)"
    )
    
    # Free up memory
    del pure_base_model
    torch.cuda.empty_cache()
    
    # --- RUN GENERATION on Fine-Tuned Model (STREAMING) ---
    print("\n🔄 Reloading base model for adapter attachment...")
    peft_base_model, tokenizer = get_base_model_and_tokenizer(bnb_config)
    
    peft_model = get_fine_tuned_model(peft_base_model, ADAPTER_PATH)
    
    check_lora_activation(peft_model, "Fine-Tuned Model")

    stream_generation(
        model=peft_model, 
        tokenizer=tokenizer,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        label="Fine-Tuned Model (Llama-3.1-8B + QLoRA, Adapter ACTIVE)"
    )

    print("\n✨ Comparison Complete!")