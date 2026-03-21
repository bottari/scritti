import os
import random
import inspect
import time
import traceback
import warnings

import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# --- CRITICAL ENVIRONMENT CONFIGURATION FOR WINDOWS ---
os.environ["DS_BUILD_EXTENSIONS"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# -----------------------------------------------------


# --- CONFIGURATION ---
HF_TOKEN = ""

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
TRUST_REMOTE_CODE = True
DATASET_PATH = r"C:\Users\micha\Desktop\projects\mercury\poetry_txt_whitman"
OUTPUT_DIR = r"D:\models\qwen3-5-0-8b-poetry-mercury-qlora-8bit-March21-002"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training Parameters - regularized baseline
MAX_LENGTH = 256
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

# LoRA Config - regularization to reduce overfit
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.2

# Data augmentation for poetic texture drift (train set only)
AUGMENT_TRAIN_TEXT = True
AUGMENT_PROB = 0.35
TOKEN_DROP_PROB = 0.05
LINE_SHUFFLE_PROB = 0.2

# Corpus chunking — controls how the single .txt is split into training examples
MIN_CHUNK_LINES = 1    # discard stanzas shorter than this many lines
CHUNK_GROUP_SIZE = 2   # group this many stanzas into one training example

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
        print("Monitor GPU temp during training. If it exceeds 80C, stop and improve cooling.")
    else:
        print("NO GPU DETECTED. QLoRA requires a GPU.")
        raise SystemExit(1)


def lightly_augment_poetic_text(text):
    """Apply mild corruption to reduce literal coherence while preserving style."""
    if random.random() > AUGMENT_PROB:
        return text

    if random.random() < LINE_SHUFFLE_PROB:
        stanzas = text.split("\n\n")
        shuffled_stanzas = []
        for stanza in stanzas:
            lines = stanza.split("\n")
            if len(lines) > 2:
                random.shuffle(lines)
            shuffled_stanzas.append("\n".join(lines))
        text = "\n\n".join(shuffled_stanzas)

    processed_lines = []
    for line in text.split("\n"):
        words = line.split(" ")
        if len(words) > 4:
            kept = [w for w in words if random.random() > TOKEN_DROP_PROB]
            if len(kept) < 2:
                kept = words[:2]
            processed_lines.append(" ".join(kept))
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)


def tokenize_function(examples, tokenizer, max_length, augment=False):
    """Tokenizes text ensuring newlines are preserved."""
    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""
    processed_texts = []

    for text in examples["text"]:
        if augment:
            text = lightly_augment_poetic_text(text)
        processed_text = text.replace("\r\n", "\n")
        processed_text = bos + processed_text + eos
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
    """
    Load text files and split into stanza-grouped chunks.

    Because a single large .txt file produces only 1 dataset row (making
    train/test splitting impossible), we break each file on double-newlines
    into stanzas, filter out very short ones, then regroup them in windows
    of CHUNK_GROUP_SIZE stanzas so each training example has enough context.
    """
    print(f"\nLoading corpus from folder: {data_path}...")
    all_chunks = []

    if not os.path.exists(data_path):
        print(f"Error: Path {data_path} does not exist.")
        raise SystemExit(1)

    for filename in sorted(os.listdir(data_path)):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_path, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if not content:
                    continue

                # Split on blank lines → individual stanzas / sections
                raw_stanzas = content.split("\n\n")
                stanzas = [
                    s.strip() for s in raw_stanzas
                    if len(s.strip().splitlines()) >= MIN_CHUNK_LINES
                ]

                # Group stanzas into windows of CHUNK_GROUP_SIZE
                for i in range(0, len(stanzas), CHUNK_GROUP_SIZE):
                    group = stanzas[i : i + CHUNK_GROUP_SIZE]
                    if group:
                        all_chunks.append("\n\n".join(group))

                print(f"  {filename}: {len(stanzas)} stanzas → ~{len(all_chunks)} chunks")

            except Exception as e:
                print(f"Warning: Error reading {filename}: {e}")

    if len(all_chunks) < 10:
        print(
            f"\nWARNING: Only {len(all_chunks)} chunks produced. "
            "Consider lowering CHUNK_GROUP_SIZE or MIN_CHUNK_LINES, "
            "or adding more source files."
        )

    raw_dataset = Dataset.from_dict({"text": all_chunks})
    print(f"Total training chunks: {len(raw_dataset)}")
    return raw_dataset


def verify_newline_tokenization(tokenizer):
    """Checks if newlines are being tokenized as distinct tokens."""
    print("\nVerifying newline tokenization...")
    test_str = "Line1\nLine2"
    tokens = tokenizer.encode(test_str, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]

    if "\n" in decoded or "Ċ" in str(decoded):
        print("Newlines are distinct.")
    else:
        print("Warning: Newlines might be merged.")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    set_seed(42)
    setup_auth()
    print_diagnostics()

    print("Using 8-bit quantization")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    print(f"Loading {MODEL_NAME} in 8-bit...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=TRUST_REMOTE_CODE,
            token=HF_TOKEN if HF_TOKEN else True,
        )
    except ValueError as e:
        if "qwen3_5" in str(e):
            print("\nMODEL ARCHITECTURE NOT SUPPORTED BY CURRENT TRANSFORMERS VERSION.")
            print("Upgrade inside your active env and retry:")
            print("python -m pip install --upgrade transformers accelerate peft")
            print("If still unsupported, install latest main branch:")
            print("python -m pip install --upgrade git+https://github.com/huggingface/transformers.git")
            raise SystemExit(1)
        raise
    except OSError as e:
        print(f"\nACCESS ERROR: {e}")
        print("\nSOLUTION:")
        print(f"1. Go to: https://huggingface.co/{MODEL_NAME}")
        print("2. Ensure access is accepted and token has read permissions.")
        raise

    # Some model classes do not accept `use_cache` in constructor kwargs.
    # Disable cache here to keep training memory behavior consistent.
    if hasattr(model, "config"):
        model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        token=HF_TOKEN if HF_TOKEN else True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    verify_newline_tokenization(tokenizer)

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    raw_dataset = load_text_corpus_from_folder(DATASET_PATH)

    print("Splitting data: 90% train, 10% validation...")
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = split_dataset["train"].map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH, augment=AUGMENT_TRAIN_TEXT),
        batched=True,
        remove_columns=["text"],
    )

    eval_dataset = split_dataset["test"].map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH, augment=False),
        batched=True,
        remove_columns=["text"],
    )

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "evaluation_strategy" if "evaluation_strategy" in ta_params else "eval_strategy"

    training_kwargs = {
        "output_dir": OUTPUT_DIR,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "fp16": False,
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "torch_empty_cache_steps": 5,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "remove_unused_columns": False,
        "max_grad_norm": 0.3,
        "eval_steps": 100,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    training_kwargs[strategy_key] = "steps"

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\nStarting QLoRA training with 8-bit quantization + anti-overfitting...")
    print(f"LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}, dropout: {LORA_DROPOUT}")
    print(f"Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
    print("Watch validation loss. Early stopping will trigger if loss plateaus.")
    trainer.train()

    print(f"\nSaving adapter to {OUTPUT_DIR}...")
    print("Writing to disk (do not interrupt)...")

    try:
        trainer.model = trainer.model.cpu()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)

        trainer.model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
        tokenizer.save_pretrained(OUTPUT_DIR)

        print("Adapter saved successfully.")
        print(f"Location: {OUTPUT_DIR}")
        print("Training complete.")

    except Exception as e:
        print(f"Save failed: {e}")
        traceback.print_exc()
        raise