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
from huggingface_hub import login

# --- CRITICAL ENVIRONMENT CONFIGURATION FOR WINDOWS ---
os.environ["DS_BUILD_EXTENSIONS"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# -----------------------------------------------------


# --- CONFIGURATION ---
HF_TOKEN = "" 

MODEL_NAME = "meta-llama/Llama-3.1-8B" 
DATASET_PATH = r"C:\Users\micha\Desktop\projects\mercury\poetry_txt"
# Use absolute path and ensure it exists
OUTPUT_DIR = r"D:\models\llama3-8b-poetry-mercury-26-qlora-8bit-019"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training Parameters - Anti-Overfitting Configuration
MAX_LENGTH = 128
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 8
LEARNING_RATE = 5e-5

# LoRA Config - Regularization to prevent overfitting
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.1

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
    
    # 1. QUANTIZATION CONFIG - 8-BIT
    print("🔧 Using 8-bit quantization")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    # 2. LOAD MODEL & TOKENIZER
    print(f"🔄 Loading {MODEL_NAME} in 8-bit...")
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
        bias="all",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. LOAD & PROCESS DATA WITH TRAIN/VAL SPLIT
    raw_dataset = load_text_corpus_from_folder(DATASET_PATH)
    
    print(f"📊 Splitting data: 90% train, 10% validation...")
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    
    train_dataset = split_dataset['train'].map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=["text"]
    )
    
    eval_dataset = split_dataset['test'].map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=["text"]
    )

    # 5. TRAINING ARGUMENTS - AGGRESSIVE MEMORY OPTIMIZATION + REGULARIZATION
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=True, 
        logging_steps=10,
        save_strategy="no",  # ⚠️ CRITICAL: Disable automatic checkpoint saving (causing disk freeze)
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        torch_empty_cache_steps=5,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        max_grad_norm=0.3,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=False,  # Disable this since we're not saving checkpoints
        metric_for_best_model="loss",
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 6. TRAIN
    print("\n🚀 Starting QLoRA Training with 8-bit quantization + anti-overfitting...")
    print(f"📊 LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}, dropout: {LORA_DROPOUT}")
    print(f"🎯 Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
    print(f"⚠️  WATCH validation loss - stop early if it stops decreasing!")
    trainer.train()

    # 7. SAVE - MINIMAL AND DIRECT
    print(f"\n💾 Saving adapter to {OUTPUT_DIR}...")
    print("⏳ Writing to disk (do NOT interrupt)...")
    
    try:
        # Force everything off GPU first
        trainer.model = trainer.model.cpu()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Give the system a moment
        import time
        time.sleep(2)
        
        # Simple, direct save
        trainer.model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print("✅ Adapter saved successfully!")
        print(f"✅ Location: {OUTPUT_DIR}")
        print("\n✨ Training complete!")
        
    except Exception as e:
        print(f"❌ Save failed: {e}")
        import traceback
        traceback.print_exc()
        raise