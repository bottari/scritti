"""
Fine-tune Gemma on one local poetry .txt corpus using raw CLM text examples.

Dataset construction lives in preprocess_whitman_clm_jsonl.py. This script keeps
the training path only: load model/tokenizer, build or cache raw {"text": "..."}
examples, train, and save the adapter/merged model.

Usage:
    pip install unsloth trl transformers datasets accelerate bitsandbytes
    python gemma-tune-on-poetry-single-txt-file.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

# Avoid a Windows / RTX 50-series Torch Inductor cuBLAS crash in Unsloth's
# compiled fused-loss path. These are runtime stability flags, not tuning
# hyperparameters, and must be set before importing unsloth/torch.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import unsloth
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

from preprocess_whitman_clm_jsonl import (
    add_bos,
    build_units,
    chunk_units,
    clean_source_text,
    extract_reflective_fragments,
    maybe_add_reflective_prefix,
    print_quality_report,
    read_text,
    token_count,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

POETRY_FILE = r"C:\Users\micha\Desktop\projects\mercury\Leaves_of_Grass_1882.txt"

OUTPUT_DIR = r"D:\models\gemma4-poetry-finetune-whitmanv3"
DATASET_CACHE = r"D:\models\whitman_clm_training_data.jsonl"
USE_EXISTING_DATASET_CACHE = True

MODEL_NAME = "google/gemma-4-E2B-it"
MAX_SEQ_LENGTH = 2048

MIN_CHUNK_TOKENS = 200
TARGET_CHUNK_TOKENS = 600
MAX_CHUNK_TOKENS = 800
HARD_MAX_TOKENS = 2000
OVERLAP_RATIO = 0.15
REFLECTIVE_PREFIX_RATE = 0.15
INCLUDE_BOS = True
PREVIEW_COUNT = 3

LORA_RANK = 64
TRAIN_LAST_N_BASE_LAYERS = 2
TRAIN_EMBEDDINGS = False
TRAIN_LM_HEAD = True
SAVE_MERGED_MODEL = False

BATCH_SIZE = 1
GRAD_ACCUM = 4
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
RANDOM_SEED = 42
MAX_STEPS = None


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------

def build_clm_samples(tokenizer) -> list[dict[str, str]]:
    corpus_path = Path(POETRY_FILE)
    rng = random.Random(RANDOM_SEED)

    print(f"\nReading corpus: {corpus_path}")
    raw_text = read_text(corpus_path)
    cleaned_text = clean_source_text(raw_text)
    if not cleaned_text:
        raise ValueError("The cleaned corpus is empty. Check the source file and filters.")

    units = build_units(tokenizer, cleaned_text, HARD_MAX_TOKENS)
    chunks = chunk_units(
        tokenizer=tokenizer,
        units=units,
        min_tokens=MIN_CHUNK_TOKENS,
        target_tokens=TARGET_CHUNK_TOKENS,
        max_tokens=MAX_CHUNK_TOKENS,
        hard_max_tokens=HARD_MAX_TOKENS,
        overlap_ratio=OVERLAP_RATIO,
    )

    fragments = extract_reflective_fragments(cleaned_text)
    chunks = [
        maybe_add_reflective_prefix(rng, chunk, fragments, REFLECTIVE_PREFIX_RATE)
        for chunk in chunks
    ]

    samples = []
    for chunk in chunks:
        if token_count(tokenizer, chunk) > HARD_MAX_TOKENS:
            continue
        samples.append({"text": add_bos(tokenizer, chunk, INCLUDE_BOS)})

    if not samples:
        raise ValueError("No CLM samples were produced from the corpus.")

    write_jsonl(Path(DATASET_CACHE), samples)
    print(f"Cached CLM JSONL dataset -> {DATASET_CACHE}")
    print_quality_report(tokenizer, samples, PREVIEW_COUNT)
    return samples


def load_clm_samples_from_jsonl(tokenizer, jsonl_path: str) -> list[dict[str, str]]:
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"JSONL dataset not found: {path}\n"
            "Run preprocess_whitman_clm_jsonl.py first, or set "
            "USE_EXISTING_DATASET_CACHE = False to rebuild it here."
        )

    samples: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc

            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Line {line_number} of {path} has no non-empty text field.")
            samples.append({"text": text})

    if not samples:
        raise ValueError(f"No samples found in JSONL dataset: {path}")

    print(f"\nLoaded CLM JSONL dataset <- {path}")
    print_quality_report(tokenizer, samples, PREVIEW_COUNT)
    return samples


# ---------------------------------------------------------------------------
# 3. Partial base unfreezing helpers
# ---------------------------------------------------------------------------

def find_transformer_layers(model):
    matches = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
            continue

        first = module[0]
        child_names = {child_name for child_name, _ in first.named_children()}
        has_text_block_shape = (
            {"self_attn", "mlp"}.issubset(child_names)
            or {"attention", "feed_forward"}.issubset(child_names)
        )
        name_looks_right = "layer" in name.lower() or "block" in name.lower()

        if has_text_block_shape or name_looks_right:
            matches.append((len(module), name, module, has_text_block_shape))

    if matches:
        matches.sort(key=lambda item: (item[3], item[0]), reverse=True)
        _, name, module, _ = matches[0]
        return module, name

    seen = [
        f"{name} ({len(module)})"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.ModuleList)
    ]
    raise RuntimeError(
        "Could not find transformer layers to unfreeze. "
        f"ModuleList candidates seen: {seen[:20]}"
    )


def unfreeze_more_model(model) -> None:
    if TRAIN_LAST_N_BASE_LAYERS > 0:
        layers, path = find_transformer_layers(model)
        n_layers = min(TRAIN_LAST_N_BASE_LAYERS, len(layers))
        for layer in layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"Unfroze last {n_layers}/{len(layers)} transformer layer(s) at {path}.")

    if TRAIN_EMBEDDINGS:
        for name, module in model.named_modules():
            if name.endswith("embed_tokens"):
                for param in module.parameters():
                    param.requires_grad = True
                print(f"Unfroze embeddings at {name}.")

    if TRAIN_LM_HEAD:
        for name, module in model.named_modules():
            if name.endswith("lm_head"):
                for param in module.parameters():
                    param.requires_grad = True
                print(f"Unfroze language-model head at {name}.")


def save_merged_16bit_model(model, tokenizer, merged_dir: str) -> None:
    """Merge LoRA into the live model without using Unsloth's merged-save path."""
    print("\nSaving merged 16-bit model via PEFT merge_and_unload()...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)


# ---------------------------------------------------------------------------
# 4. Model, trainer, save
# ---------------------------------------------------------------------------

print("Loading model and tokenizer...")
LOAD_IN_4BIT = TRAIN_LAST_N_BASE_LAYERS <= 0
if not LOAD_IN_4BIT:
    print("Partial base-model tuning enabled: loading in fp16/bf16 instead of 4-bit.")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    dtype=None,
)

random.seed(RANDOM_SEED)
examples = (
    load_clm_samples_from_jsonl(tokenizer, DATASET_CACHE)
    if USE_EXISTING_DATASET_CACHE
    else build_clm_samples(tokenizer)
)
hf_dataset = Dataset.from_list(examples)

MODULES_TO_SAVE = []
if TRAIN_EMBEDDINGS:
    MODULES_TO_SAVE.append("embed_tokens")
if TRAIN_LM_HEAD:
    MODULES_TO_SAVE.append("lm_head")
MODULES_TO_SAVE = MODULES_TO_SAVE or None

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    modules_to_save=MODULES_TO_SAVE,
    use_gradient_checkpointing="unsloth",
    random_state=RANDOM_SEED,
)

unfreeze_more_model(model)
model.print_trainable_parameters()

use_bf16 = torch.cuda.is_bf16_supported()

training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS if MAX_STEPS is None else 1,
    max_steps=MAX_STEPS if MAX_STEPS is not None else -1,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    learning_rate=LEARNING_RATE,
    fp16=not use_bf16,
    bf16=use_bf16,
    optim="adamw_8bit",
    gradient_checkpointing=True,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    output_dir=OUTPUT_DIR,
    weight_decay=0.01,
    seed=RANDOM_SEED,
    report_to="none",
    dataloader_num_workers=0,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=hf_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=True,
    args=training_args,
)

print("\nStarting raw CLM fine-tuning...\n")
trainer_stats = trainer.train()

print("\nTraining complete.")
print(f"  Total steps  : {trainer_stats.global_step}")
print(f"  Training loss: {trainer_stats.training_loss:.4f}")

if TRAIN_LAST_N_BASE_LAYERS > 0:
    print(
        "\nNote: base layers were unfrozen. Use the merged_fp16 output for inference; "
        "the lightweight LoRA adapter does not contain all base-layer weight changes."
    )

adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"\nLoRA adapter saved -> {adapter_dir}")

if SAVE_MERGED_MODEL:
    merged_dir = os.path.join(OUTPUT_DIR, "merged_fp16")
    trainer.optimizer = None
    trainer.lr_scheduler = None
    save_merged_16bit_model(model, tokenizer, merged_dir)
    print(f"Merged fp16 model saved -> {merged_dir}")
else:
    print("\nSkipped merged fp16 export because SAVE_MERGED_MODEL = False.")

print("\nRaw CLM fine-tuning run complete.")
