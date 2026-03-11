import os
import torch
import warnings
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer,
)
from peft import PeftModel
from huggingface_hub import login
from peft.tuners.lora.layer import LoraLayer

# --- CONFIGURATION ---
HF_TOKEN = ""

BASE_MODEL_NAME = "Qwen/Qwen3.5-0.8B"
TRUST_REMOTE_CODE = True
ADAPTER_PATH = r"D:\models\qwen3-5-0-8b-poetry-mercury-qlora-8bit-003"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.5
TOP_P = 0.45
TOP_K = 15
REPETITION_PENALTY = 1.2
NUM_RETURN_SEQUENCES = 1

PROMPT = "Tell me what the forest thought"

warnings.filterwarnings("ignore")


# --- UTILITIES ---
def setup_auth():
    """Handles authentication explicitly if token is provided."""
    if HF_TOKEN:
        print("Logging in with explicit token...")
        login(token=HF_TOKEN)
    else:
        print("No explicit token in script. Relying on 'huggingface-cli login'...")


def print_diagnostics():
    """Check and print system and GPU info."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {gpu_memory:.1f} GB")
    else:
        print("NO GPU DETECTED. Generation requires a GPU if the model was trained with QLoRA.")
        raise SystemExit(1)


def get_base_model_and_tokenizer(bnb_config):
    """Loads the pure base model (without any adapter attachment)."""
    print(f"\nLoading base model ({BASE_MODEL_NAME}) and tokenizer...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=TRUST_REMOTE_CODE,
            token=HF_TOKEN if HF_TOKEN else True,
        )
    except OSError as e:
        print(f"\nBASE MODEL ACCESS ERROR: {e}")
        print(f"1. Go to: https://huggingface.co/{BASE_MODEL_NAME}")
        print("2. Ensure access is accepted and token has read permissions.")
        raise SystemExit(1)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        token=HF_TOKEN if HF_TOKEN else True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def get_fine_tuned_model(base_model, adapter_path):
    """Attaches the LoRA adapter to a base model instance."""
    print(f"Attaching LoRA adapter from {adapter_path}...")
    if not os.path.exists(adapter_path):
        print(f"ADAPTER NOT FOUND: Ensure {adapter_path} exists and contains saved LoRA files.")
        raise SystemExit(1)

    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    peft_model.eval()
    return peft_model


def check_lora_activation(model, label):
    """Checks whether LoRA is active on the model instance."""
    lora_found = False

    if isinstance(model, PeftModel):
        print(f"DEBUG CHECK ({label}): Model is wrapped by PeftModel (Adapter is active).")
        lora_found = True
    else:
        print(f"DEBUG CHECK ({label}): Model is a standard AutoModelForCausalLM (Expected for base).")

    lora_layer_count = 0
    for _, module in model.named_modules():
        if isinstance(module, LoraLayer):
            lora_layer_count += 1
            lora_found = True

    if lora_layer_count > 0:
        print(f"LoRA Layers Found: {lora_layer_count} instances of LoraLayer.")
    elif not isinstance(model, PeftModel):
        print("No LoraLayer instances found in model modules.")

    return lora_found


def stream_generation(model, tokenizer, prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty, label):
    """Generates text token-by-token and streams output to console."""
    print(f"\n==================== {label} ====================")
    print(f"PROMPT: {prompt}")
    print(f"Settings: Temp={temperature}, Top_P={top_p}, Top_K={top_k}, Repetition Penalty={repetition_penalty}")
    print("\n--- Continuation (Streaming Output) ---")

    bos = tokenizer.bos_token or ""
    input_text = bos + prompt
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    stopping_ids = []
    if tokenizer.eos_token_id is not None:
        stopping_ids.append(tokenizer.eos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0:
        stopping_ids.append(eot_id)
    eos_token_id = stopping_ids if stopping_ids else None

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    with torch.no_grad():
        model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=NUM_RETURN_SEQUENCES,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
        )

    print("----------------------------------------------------------")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    setup_auth()
    print_diagnostics()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("\nStarting Poetry Generation Comparison...")

    pure_base_model, tokenizer = get_base_model_and_tokenizer(bnb_config)
    check_lora_activation(pure_base_model, "Pure Base Model")

    stream_generation(
        model=pure_base_model,
        tokenizer=tokenizer,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        label="Base Model (Qwen3.5-0.8B, PURE INSTANCE)",
    )

    del pure_base_model
    torch.cuda.empty_cache()

    print("\nReloading base model for adapter attachment...")
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
        label="Fine-Tuned Model (Qwen3.5-0.8B + QLoRA, Adapter ACTIVE)",
    )

    print("\nComparison Complete!")
