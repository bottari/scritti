import os
import torch
import warnings
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Suppress the Hugging Face symlink warning (common on Windows)
warnings.filterwarnings("ignore", "huggingface_hub cache-system uses symlinks")
warnings.filterwarnings("ignore", "resume_download is deprecated")

# --- CONFIGURATION ---
MODEL_NAME = "meta-llama/Llama-3.1-8B"
DATASET_PATH = "concatenated_poetry_corpus-clean-001.txt"
OUTPUT_DIR = r"D:\models\llama-3-8b-q4-finetuned-poetry-mercury-11-test-001-r64-alpha128-3e4-lngth1024-50epochs"

MAX_LENGTH = 1024
GRADIENT_ACCUMULATION_STEPS = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 1
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4


# --- DIAGNOSTICS ---
def print_diagnostics():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        compute_capability = ".".join(map(str, torch.cuda.get_device_capability(0)))
        torch_version = torch.__version__
        print("--------------------------------------------------")
        print(f"✅ GPU DETECTED: {device_name}")
        print(f"✅ TORCH VERSION: {torch_version}")
        print(f"✅ COMPUTE CAPABILITY: {compute_capability}")
        print("--------------------------------------------------")
    else:
        print("--------------------------------------------------")
        print("⚠️ NO GPU DETECTED. Training will be extremely slow on CPU.")
        print("--------------------------------------------------")


def clean_text(example):
    text = example["text"]

    # 🛑 CRITICAL FIX: Explicitly remove all known Llama-3/general special tokens
    # These often appear as literal strings in aggregated training corpora.
    text = text.replace("<|begin_of_text|>", "") 
    text = text.replace("<|end_of_text|>", "")
    text = text.replace("<|endoftext|>", "\n\n") # Your original replacement
    text = text.replace("<|startoftext|>", "")
    text = text.replace("<|eot_id|>", "")
    
    # 💡 Llama 3.1 also uses specific header tokens you should remove if they exist
    text = text.replace("<|start_header_id|>", "")
    text = text.replace("<|end_header_id|>", "")

    text = text.strip()

    return {"text": text}


# --- TOKENIZATION ---
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="do_not_pad")


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // MAX_LENGTH) * MAX_LENGTH
    result = {
        k: [t[i: i + MAX_LENGTH] for i in range(0, total_length, MAX_LENGTH)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# ==============================================================================
if __name__ == "__main__":

    print_diagnostics()

    print(f"🆕 Starting from base model: {MODEL_NAME}")
    global tokenizer

    # Disable automatic BOS token injection (Option A)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_bos_token=False)

    # Fix tokenizer padding / special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Tokenizer Special Tokens:")
    print("  BOS:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("  EOS:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("  PAD:", tokenizer.pad_token, tokenizer.pad_token_id)

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="all",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- DATASET ---
    try:
        raw_dataset = load_dataset("text", data_files={"train": DATASET_PATH})["train"]
    except Exception as e:
        print(f"\n❌ ERROR: Could not load dataset from {DATASET_PATH}")
        print(f"Details: {e}")
        exit()

    # Clean special tokens (Option B)
    raw_dataset = raw_dataset.map(clean_text, num_proc=1)

    # Sanity check a sample
    print("Sample cleaned text:\n", raw_dataset[0]["text"][:400], "\n---\n")

    # Tokenize
    print("Tokenizing and chunking dataset...")
    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=["text"]
    )

    # Chunk into blocks
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=1)

    num_samples = len(lm_datasets)
    print(f"✅ Loaded {num_samples} processed samples")

    # --- TRAINING ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_dir='./logs',
        logging_steps=20,
        bf16=True,
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
    )

    trainer.train()

    # --- SAVE ---
    final_output_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_output_dir, exist_ok=True)

    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"\n🎉 Training complete! Final model saved to: {final_output_dir}")
    print("You can now use this model for inference.")