import os
import torch
import warnings
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from huggingface_hub import login # Added for explicit login

# --- CRITICAL ENVIRONMENT CONFIGURATION FOR WINDOWS ---
os.environ["DS_BUILD_EXTENSIONS"] = "0"
# Enable memory optimizations
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# -----------------------------------------------------


# --- CONFIGURATION ---
# 1. GET ACCESS: Go to https://huggingface.co/meta-llama/Llama-3.1-8B and click "Accept License"
# 2. PASTE TOKEN: Paste your token here (Settings -> Access Tokens -> Read) to bypass CLI issues
HF_TOKEN = "" 

MODEL_NAME = "meta-llama/Llama-3.1-8B" 
DATASET_PATH = r"C:\Users\micha\Desktop\projects\mercury\poetry_txt"
OUTPUT_DIR = "D:/models/llama3-8b-poetry-mercury-25-qlora-014"

# Training Parameters - Note 16GB VRAM (RTX 5070 Ti) Limitation
MAX_LENGTH = 256 
PER_DEVICE_TRAIN_BATCH_SIZE = 1  # Keep at 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 10
LEARNING_RATE = 2e-4 

# LoRA Config - REDUCED for stability
LORA_R = 64
LORA_ALPHA = 256
LORA_DROPOUT = 0.05

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
        print(f"⚠️  Monitor GPU temp during training. If it exceeds 80°C, stop and improve cooling.")
    else:
        print("❌ NO GPU DETECTED. QLoRA requires a GPU.")
        exit()

def tokenize_function(examples, tokenizer, max_length):
    """Tokenizes text ensuring newlines are preserved."""
    processed_texts = []
    for text in examples["text"]:
        processed_text = text.replace('\r\n', '\n')
        processed_text = tokenizer.bos_token + processed_text + tokenizer.eos_token
        processed_texts.append(processed_text)
    
    tokenized = tokenizer(
        processed_texts, 
        truncation=True, 
        max_length=max_length,
        add_special_tokens=False, 
        padding=False, 
    )
    return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}

def load_text_corpus_from_folder(data_path):
    """Load text files preserving formatting."""
    print(f"\n📖 Loading corpus from folder: {data_path}...")
    all_documents = []
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Path {data_path} does not exist.")
        exit()

    for filename in sorted(os.listdir(data_path)):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        all_documents.append(content)
            except Exception as e:
                print(f" ⚠️ Error reading {filename}: {e}")
                    
    raw_dataset = Dataset.from_dict({'text': all_documents})
    print(f"✅ Loaded {len(raw_dataset)} documents.")
    return raw_dataset

def verify_newline_tokenization(tokenizer):
    """Checks if newlines are being tokenized as distinct tokens."""
    print("\n🔍 Verifying Newline Tokenization...")
    test_str = "Line1\nLine2"
    tokens = tokenizer.encode(test_str, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]
    
    if '\n' in decoded or 'Ċ' in str(decoded):
        print("✅ Newlines are distinct.")
    else:
        print("⚠️ Newlines might be merged.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    setup_auth()
    print_diagnostics()
    
    # 1. QUANTIZATION CONFIG
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 2. LOAD MODEL & TOKENIZER
    print(f"🔄 Loading {MODEL_NAME} in 4-bit...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            use_cache=False,
            token=HF_TOKEN if HF_TOKEN else True
        )
    except OSError as e:
        print(f"\n❌ ACCESS ERROR: {e}")
        print("\n👉 SOLUTION:")
        print(f"1. Go to: https://huggingface.co/{MODEL_NAME}")
        print("2. Click 'Accept License' (Wait 1-2 mins if you just did it)")
        print("3. Ensure your token in the script has 'Read' permissions.")
        exit()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN if HF_TOKEN else True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    verify_newline_tokenization(tokenizer)

    # 3. PREPARE FOR LORA
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head", "embed_tokens"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. LOAD & PROCESS DATA
    raw_dataset = load_text_corpus_from_folder(DATASET_PATH)
    
    lm_datasets = raw_dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=["text"]
    )

    # 5. TRAINING ARGUMENTS - OPTIMIZED FOR STABILITY
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=True, 
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=5,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        # NEW: Memory optimization settings
        torch_empty_cache_steps=10,  # Clear cache every 10 steps
        dataloader_num_workers=0,    # Disable multiprocessing (can cause memory spikes)
        dataloader_pin_memory=False, # Reduce memory overhead
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=lm_datasets,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 6. TRAIN
    print("\n🚀 Starting QLoRA Training...")
    trainer.train()

    # 7. SAVE
    print(f"\n💾 Saving adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n✨ Done!")