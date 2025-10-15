import os
import torch
import warnings
# Import load_dataset from datasets
from datasets import load_dataset 
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

# --- CONFIGURATION ---
MODEL_NAME = "gpt2-large"
# 🚩 Path to your folder containing the individual .txt files
DATASET_PATH = r"C:\Users\micha\Desktop\projects\mercury\poetry_txt" 
OUTPUT_DIR = "D:/models/gpt2-large-poetry-mercury-12-unfrozen-top-layers-010-50epochs-separate-txt-files"

# Training Parameters
MAX_LENGTH = 512
PER_DEVICE_TRAIN_BATCH_SIZE = 4            
GRADIENT_ACCUMULATION_STEPS = 2            
NUM_EPOCHS = 50
LEARNING_RATE = 5e-5

# Unfreezing Parameter (Set how many transformer blocks to unfreeze)
NUM_LAYERS_TO_UNFREEZE = 36 

# Suppress common warnings
warnings.filterwarnings("ignore", "huggingface_hub cache-system uses symlinks")
warnings.filterwarnings("ignore", "resume_download is deprecated")
warnings.filterwarnings("ignore", "You are using a model of type gpt2")


# --- UTILITIES ---

def print_diagnostics():
    """Check and print system and GPU info."""
    try:
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
    except Exception as e:
        print(f"Error during diagnostics: {e}")
        
# --- MODIFIED: Tokenize and handle length in a single step ---
def tokenize_function(examples, tokenizer, max_length):
    """
    Tokenizes each text entry (document), appends the EOS token, and then 
    applies padding/truncation to MAX_LENGTH.
    """
    # CRITICAL: Append the EOS token to explicitly mark the end of a training example.
    texts_with_eos = [
        text + tokenizer.eos_token 
        for text in examples["text"]
    ]
    
    return tokenizer(
        texts_with_eos, 
        truncation=True, 
        padding="max_length", # Pad/truncate to MAX_LENGTH
        max_length=max_length 
    )

# --- NEW: Function to display a sample of the processed data ---
def inspect_processed_sample(lm_datasets, tokenizer, num_tokens=100):
    """Decode and print the first sample from the processed dataset."""
    if not lm_datasets:
        print("⚠️ No processed data to inspect.")
        return

    print("\n--- 🔎 INSPECTING PROCESSED TRAINING SAMPLE ---")
    # Get the first sample's input IDs
    sample_ids = lm_datasets[0]["input_ids"]
    
    # Slice to the requested number of tokens
    sample_tokens = sample_ids[:num_tokens]
    
    # Decode the tokens back into human-readable text
    decoded_text = tokenizer.decode(sample_tokens, skip_special_tokens=False)

    print(f"Dataset length: {len(lm_datasets)}")
    print(f"First sample token length: {len(sample_ids)}")
    print(f"Decoded Sample (First {num_tokens} tokens):\n")
    print("--------------------------------------------------")
    print(decoded_text)
    print("--------------------------------------------------")
    print("NOTE: The presence of '<|endoftext|>' at the end of the content means")
    print("the tokenizer correctly added the EOS token to mark the end of the document.")


def unfreeze_top_layers(model, num_layers):
    """
    Freezes all layers and then unfreezes the top N transformer blocks, 
    LayerNorms, and the Language Modeling head.
    """
    # 1. Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # GPT-2 transformer blocks are named 'h'.
    total_layers = len(model.transformer.h)
    unfreeze_start_index = total_layers - num_layers

    print("\n--- Layer Unfreezing Logic ---")
    print(f"Total Transformer Blocks: {total_layers}")
    print(f"🥶 Freezing all layers. Unfreezing the top {num_layers} blocks:")

    # 2. Unfreeze the top N layers (transformer blocks)
    for i in range(unfreeze_start_index, total_layers):
        layer = model.transformer.h[i]
        for param in layer.parameters():
            param.requires_grad = True
        print(f"   ✅ Unfrozen layer: model.transformer.h[{i}]")

    # 3. Unfreeze the final Language Modeling head
    for param in model.lm_head.parameters():
        param.requires_grad = True
    print("   ✅ Unfrozen the Language Modeling Head (lm_head)")

    # 4. Unfreeze all normalization layers (recommended for transfer learning)
    for name, module in model.named_modules():
        if 'ln' in name or 'LayerNorm' in name:
            for param in module.parameters():
                param.requires_grad = True
    print("   ✅ Unfrozen all LayerNorms.")

    # 5. Print diagnostics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / all_params

    print("\n--- Trainable Parameters Summary ---")
    print(f"trainable params: {trainable_params:,}")
    print(f"all params: {all_params:,}")
    print(f"trainable%: {trainable_percent:.4f}%")
    print("------------------------------------")

    return model

# 🚩 MODIFIED FUNCTION: Loads each file as a single document
def load_text_corpus_from_folder(data_path):
    """Loads all text files from a directory into a Hugging Face Dataset, treating each file as one document."""
    print(f"\n📖 Loading corpus from folder: {data_path}...")
    
    all_documents = []
    
    # Manually iterate over files to ensure each file is a single document object
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        all_documents.append(content)
            except Exception as e:
                print(f"⚠️ Could not read file {filename}: {e}")
                continue
                    
    if not all_documents:
        print("⚠️ No content found in the text files. Check the folder path and file contents.")
        exit()

    # Create the dataset from the list of complete documents
    raw_dataset = load_dataset('text', data_files={'train': os.path.join(data_path, '*.txt')}, split='train')
    raw_dataset = raw_dataset.from_dict({'text': all_documents})

    print(f"✅ Loaded {len(raw_dataset)} individual documents from the folder.")
    return raw_dataset

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print_diagnostics()
    print(f"🆕 Starting from base model: {MODEL_NAME}")

    # --- MODEL & TOKENIZER ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Ensure EOS token is used for padding/end of document
    if tokenizer.pad_token is None:
        # GPT-2's EOS token is '<|endoftext|>'
        tokenizer.pad_token = tokenizer.eos_token 

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        use_cache=False
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- UNFREEZE LAYERS ---
    model = unfreeze_top_layers(model, NUM_LAYERS_TO_UNFREEZE)

    # --- DATASET LOADING ---
    raw_dataset = load_text_corpus_from_folder(DATASET_PATH)
    
    # --- TOKENIZATION & DOCUMENT-LEVEL CHUNKING ---
    print("Tokenizing and padding/truncating dataset...")
    
    tokenized_datasets = raw_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, MAX_LENGTH), 
        batched=True,
        num_proc=1,
        remove_columns=["text"],
    )
    
    # Add the labels
    def add_labels(examples):
        # For Causal Language Modeling, labels are simply the input_ids
        examples["labels"] = examples["input_ids"].copy()
        return examples
        
    lm_datasets = tokenized_datasets.map(add_labels, batched=True)
    
    # ----------------------------------------------------------------------
    # --- DATA VERIFICATION STEP ---
    # ----------------------------------------------------------------------
    # Inspect 250 tokens from the first document
    inspect_processed_sample(lm_datasets, tokenizer, num_tokens=250) 
    
    print(f"✅ Final dataset size (number of training samples): {len(lm_datasets)}")
    
    # --- TRAINING ARGS ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        # Using 4 (batch size) * 4 (gradient steps) = 16 effective batch size
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_dir='./logs',
        logging_strategy="epoch",
        logging_steps=100,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True, 
        save_strategy="steps",
        save_steps=85,
        load_best_model_at_end=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
    )

    # --- TRAINING ---
    print("\n🚀 Starting fine-tuning...")
    trainer.train()

    # --- SAVE FINE-TUNED MODEL ---
    print("\n💾 Saving fine-tuned model...")
    final_output_dir = os.path.join(OUTPUT_DIR, "final_model_unfrozen")
    os.makedirs(final_output_dir, exist_ok=True)

    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print(f"\n🎉 Training complete!")
    print(f"✅ Final model saved to: {final_output_dir}")