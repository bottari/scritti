import os
import torch
import warnings
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# --- CONFIGURATION ---
MODEL_NAME = "gpt2"
DATASET_PATH = r"C:\Users\micha\Desktop\projects\mercury\poetry_txt"
OUTPUT_DIR = "D:/models/gpt2-finetuned-poetry-mercury-concatenated-008"

# Training Parameters - AGGRESSIVE SETTINGS
BLOCK_SIZE = 128  # Smaller chunks = more training examples = more line break exposure
PER_DEVICE_TRAIN_BATCH_SIZE = 8  # Larger batch
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 100  # MANY epochs to override GPT-2's prose bias
LEARNING_RATE = 5e-5  # Higher LR to learn faster

NUM_LAYERS_TO_UNFREEZE = 12

warnings.filterwarnings("ignore")


def print_diagnostics():
    if torch.cuda.is_available():
        print("="*60)
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print("="*60)
    else:
        print("⚠️ NO GPU")


def load_and_concatenate_corpus(data_path):
    """
    Load all poems and concatenate them into ONE LONG TEXT.
    This is a different strategy: instead of treating each poem as separate,
    we create one continuous stream of poetry with line breaks throughout.
    """
    print(f"\n📖 Loading corpus from: {data_path}")
    print(f"   Strategy: Concatenate all poems into continuous text\n")
    
    all_poems = []
    
    for filename in sorted(os.listdir(data_path)):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', newline='') as f:
                    content = f.read().strip()
                    if content:
                        newlines = content.count('\n')
                        all_poems.append(content)
                        print(f"   ✅ {filename}: {newlines} newlines")
            except Exception as e:
                print(f"   ⚠️ Error reading {filename}: {e}")
    
    # Concatenate all poems with double EOS token as separator
    # This creates: poem1<|endoftext|><|endoftext|>poem2<|endoftext|><|endoftext|>...
    concatenated = "<|endoftext|><|endoftext|>".join(all_poems)
    
    total_newlines = concatenated.count('\n')
    total_chars = len(concatenated)
    
    print(f"\n📊 Concatenated Corpus Statistics:")
    print(f"   Total poems: {len(all_poems)}")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Total newlines: {total_newlines:,}")
    print(f"   Newline density: {(total_newlines/total_chars)*100:.2f}%")
    
    return concatenated


def create_chunked_dataset(text, tokenizer, block_size):
    """
    Tokenize the entire corpus and split into overlapping chunks.
    This maximizes the number of training examples with line breaks.
    """
    print(f"\n🔪 Chunking text into blocks of {block_size} tokens...")
    
    # Tokenize the entire concatenated text
    tokenized = tokenizer(text, add_special_tokens=False)
    input_ids = tokenized["input_ids"]
    
    print(f"   Total tokens: {len(input_ids):,}")
    
    # Split into chunks with stride (overlapping windows)
    # Stride = block_size // 2 means 50% overlap
    stride = block_size // 2
    
    chunks = []
    for i in range(0, len(input_ids) - block_size + 1, stride):
        chunk = input_ids[i:i + block_size]
        chunks.append(chunk)
    
    # Count newlines in chunks (token 198 is '\n' in GPT-2)
    chunks_with_newlines = sum(1 for chunk in chunks if 198 in chunk)
    
    print(f"   Created {len(chunks):,} chunks")
    print(f"   Chunks containing newlines: {chunks_with_newlines:,} ({100*chunks_with_newlines/len(chunks):.1f}%)")
    print(f"   Stride: {stride} tokens (50% overlap)")
    
    # Verify newline presence
    total_newline_tokens = sum(chunk.count(198) for chunk in chunks)
    avg_newlines_per_chunk = total_newline_tokens / len(chunks)
    print(f"   Average newlines per chunk: {avg_newlines_per_chunk:.2f}")
    
    # Create dataset
    dataset = Dataset.from_dict({
        "input_ids": chunks,
        "labels": chunks.copy()  # For causal LM, labels = input_ids
    })
    
    return dataset


def inspect_chunks(dataset, tokenizer, num_samples=3):
    """Inspect some chunks to verify newlines are present."""
    print("\n" + "="*70)
    print("🔎 CHUNK INSPECTION")
    print("="*70)
    
    for i in range(min(num_samples, len(dataset))):
        chunk_ids = dataset[i]["input_ids"]
        decoded = tokenizer.decode(chunk_ids, skip_special_tokens=False)
        
        newline_count = chunk_ids.count(198)
        
        print(f"\nChunk {i+1}:")
        print(f"Newlines: {newline_count}")
        print("-" * 70)
        print(decoded)
        print("-" * 70)
    
    print("="*70 + "\n")


def unfreeze_top_layers(model, num_layers):
    """Unfreeze layers for training."""
    for param in model.parameters():
        param.requires_grad = False

    total_layers = len(model.transformer.h)
    start_idx = total_layers - num_layers

    print("\n--- Unfreezing Layers ---")
    for i in range(start_idx, total_layers):
        for param in model.transformer.h[i].parameters():
            param.requires_grad = True
    print(f"   ✅ Unfrozen top {num_layers} transformer blocks")

    for param in model.lm_head.parameters():
        param.requires_grad = True
    print("   ✅ Unfrozen LM head")

    for name, module in model.named_modules():
        if 'ln' in name or 'LayerNorm' in name:
            for param in module.parameters():
                param.requires_grad = True
    print("   ✅ Unfrozen LayerNorms")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} ({100*trainable/total:.1f}%)\n")

    return model


if __name__ == "__main__":
    print_diagnostics()
    print(f"🆕 Base model: {MODEL_NAME}")
    print(f"🎯 Strategy: Concatenate corpus + small chunks + many epochs")
    print(f"   This maximizes exposure to line break patterns\n")

    # --- TOKENIZER & MODEL ---
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        use_cache=False
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    print("✅ Model loaded\n")

    # --- UNFREEZE LAYERS ---
    model = unfreeze_top_layers(model, NUM_LAYERS_TO_UNFREEZE)

    # --- LOAD AND PREPARE DATA ---
    concatenated_text = load_and_concatenate_corpus(DATASET_PATH)
    
    # --- CREATE CHUNKED DATASET ---
    dataset = create_chunked_dataset(concatenated_text, tokenizer, BLOCK_SIZE)
    
    # --- INSPECT ---
    inspect_chunks(dataset, tokenizer, num_samples=3)
    
    print(f"✅ Final dataset: {len(dataset):,} training chunks\n")
    
    # --- TRAINING ARGS ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # Data collator - no need for dynamic padding since all chunks are same size
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # --- TRAIN ---
    print("🚀 Starting training...")
    print(f"   100 epochs × {len(dataset):,} chunks = massive line break exposure")
    print(f"   This will take a while but should finally teach the model!\n")
    
    trainer.train()

    # --- SAVE ---
    print("\n💾 Saving model...")
    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\n🎉 Training complete!")
    print(f"✅ Saved to: {final_dir}")
    print("\n📝 Key changes in this approach:")
    print("   • Concatenated all poems into one continuous text")
    print("   • Created many small overlapping chunks (128 tokens)")
    print("   • Trained for 100 epochs for maximum exposure")
    print("   • Should finally override GPT-2's prose bias!")