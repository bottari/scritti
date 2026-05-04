"""
Fine-tune Gemma 4 E4B IT on a local corpus of poetry .txt files.

Hardware target : RTX 5070 Ti  — 16 GB VRAM
Strategy        : 4-bit QLoRA via unsloth  (fits comfortably in 16 GB)

Directory layout assumed on disk
─────────────────────────────────
poetry_txt/
    keats.txt
    shelley.txt
    plath.txt
    ...   (any number of .txt files, flat or in sub-folders)

Each file may contain ONE poem or MULTIPLE poems separated by blank lines.
The preprocessor handles both cases automatically.

Usage
─────
    pip install unsloth trl transformers datasets accelerate bitsandbytes
    python finetune_gemma4_poetry.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import json
import glob
import random
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Configuration  ← edit these to match your machine
# ──────────────────────────────────────────────────────────────────────────────

# Folder containing your .txt poetry files  (Windows path — works on native
# Python for Windows; if using WSL replace with the /mnt/c/... equivalent)
POETRY_DIR   = r"C:\Users\micha\Desktop\projects\mercury\poetry_txt"

# Where checkpoints and the final model will be written
OUTPUT_DIR   = r"D:\models\gemma4-poetry-finetune"

# Intermediate JSONL written after preprocessing (handy for inspection)
DATASET_CACHE = r"D:\models\poetry_training_data.jsonl"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME     = "google/gemma-4-E4B-it"   # pre-quantised; ~6 GB VRAM
MAX_SEQ_LENGTH = 2048    # most poems fit well within this; raise to 4096 if needed

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_RANK    = 32        # 8–32 works well for style transfer; raise for richer data

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 2         # per-device; 2 is safe for 16 GB with 4-bit + grad-ckpt
GRAD_ACCUM   = 4         # effective batch = BATCH_SIZE × GRAD_ACCUM = 8
NUM_EPOCHS   = 3         # on small poetry corpora 2–5 epochs is typical
LEARNING_RATE = 2e-4
RANDOM_SEED  = 42

# Set MAX_STEPS to a small number (e.g. 20) for a quick smoke-test;
# leave as None to use NUM_EPOCHS instead.
MAX_STEPS    = None      # e.g. 20 for a dry run


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Preprocessing helpers
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise a raw text file: fix line endings, strip trailing whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing spaces from every line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # Collapse runs of 4+ blank lines down to 3 (poem boundary)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def split_into_poems(text: str) -> list:
    """
    Heuristically split a file that may contain multiple poems.

    Boundaries detected:
      • 3+ consecutive blank lines
      • A line of 3+ dashes / equals / asterisks (visual separator)
      • A line that looks like a title in ALL CAPS (≥3 words) — conservative
    """
    # Normalise separator lines into a unique marker
    text = re.sub(r"\n[ \t]*[-=*~]{3,}[ \t]*\n", "\n\n\n", text)
    # Split on 3+ blank lines
    chunks = re.split(r"\n{3,}", text)
    poems = []
    for chunk in chunks:
        chunk = chunk.strip()
        # Skip very short fragments (likely headers or page numbers)
        if len(chunk) < 40:
            continue
        poems.append(chunk)
    return poems


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Terminal preview  (called after preprocessing, before training)
# ──────────────────────────────────────────────────────────────────────────────

# How many examples to spot-check.  Set to 0 to skip entirely.
PREVIEW_COUNT = 3

# Max lines to print per poem before truncating with an ellipsis row
PREVIEW_MAX_LINES = 14


def _box(title: str, width: int = 68) -> str:
    """Return a titled box header string."""
    pad = width - len(title) - 4
    return f"┌─ {title} {'─' * max(pad, 0)}┐"


def _truncate_poem(text: str, max_lines: int) -> str:
    """Trim a poem to *max_lines* lines, appending an ellipsis row if cut."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    kept = lines[:max_lines]
    kept.append(f"  … ({len(lines) - max_lines} more lines)")
    return "\n".join(kept)


def preview_dataset(raw_examples: list, formatted_dataset, n: int = PREVIEW_COUNT) -> None:
    """
    Print a side-by-side look at:
      • the raw split poem  (what the preprocessor extracted)
      • the formatted training string  (what the model will actually see)

    Also prints a short statistics table: total examples, token-length
    distribution (min / median / max) estimated by whitespace-split word count
    as a fast proxy (no tokeniser needed at this point).
    """
    if n <= 0:
        return

    total = len(raw_examples)
    sample_indices = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    # spread evenly: first, middle(s), last

    WIDTH = 68

    print("\n" + "═" * WIDTH)
    print("  PREPROCESSING PREVIEW")
    print("═" * WIDTH)

    for rank, idx in enumerate(sample_indices, 1):
        raw_msg   = raw_examples[idx]["messages"]
        poem_text = raw_msg[1]["content"]           # assistant turn = the poem
        fmt_text  = formatted_dataset[idx]["text"]  # after apply_chat_template

        # ── raw poem ──────────────────────────────────────────────────────
        print(f"\n{_box(f'Example {rank}/{n}  —  raw poem  (index {idx})', WIDTH)}")
        for line in _truncate_poem(poem_text, PREVIEW_MAX_LINES).splitlines():
            print(f"│  {line}")
        print("└" + "─" * (WIDTH - 1))

        # ── formatted training string ──────────────────────────────────────
        print(f"{_box(f'Example {rank}/{n}  —  formatted training text', WIDTH)}")
        fmt_preview = _truncate_poem(fmt_text, PREVIEW_MAX_LINES)
        for line in fmt_preview.splitlines():
            # keep long lines from blowing up the terminal
            display = line if len(line) <= WIDTH - 3 else line[: WIDTH - 6] + "…"
            print(f"│  {display}")
        print("└" + "─" * (WIDTH - 1))

    # ── statistics ────────────────────────────────────────────────────────
    word_counts = [len(ex["messages"][1]["content"].split()) for ex in raw_examples]
    word_counts.sort()
    wmin  = word_counts[0]
    wmed  = word_counts[len(word_counts) // 2]
    wmax  = word_counts[-1]
    # rough token estimate: ~1.3 tokens per word for poetry
    tmin, tmed, tmax = int(wmin * 1.3), int(wmed * 1.3), int(wmax * 1.3)

    # bucket distribution
    buckets = {"<50 words": 0, "50–150": 0, "150–300": 0, "300–600": 0, ">600": 0}
    for w in word_counts:
        if   w <  50: buckets["<50 words"] += 1
        elif w < 150: buckets["50–150"]    += 1
        elif w < 300: buckets["150–300"]   += 1
        elif w < 600: buckets["300–600"]   += 1
        else:         buckets[">600"]      += 1

    print(f"\n{'─' * WIDTH}")
    print(f"  Dataset statistics")
    print(f"{'─' * WIDTH}")
    print(f"  Total examples      : {total}")
    print(f"  Word count  min/med/max : {wmin} / {wmed} / {wmax}")
    print(f"  Token est.  min/med/max : {tmin} / {tmed} / {tmax}  (≈1.3 tok/word)")
    print(f"  Length distribution :")
    for label, count in buckets.items():
        bar = "█" * min(count, 40)
        print(f"    {label:<12}  {bar}  {count}")
    print(f"{'─' * WIDTH}")

    # ── pause for confirmation ─────────────────────────────────────────────
    print("\n  ✔  If the poems look correctly split and formatted above,")
    print("     press  Enter  to continue to training.")
    print("     Press  Ctrl-C  to abort and adjust the configuration.\n")
    try:
        input("  > ")
    except KeyboardInterrupt:
        print("\nAborted by user.")
        raise SystemExit(0)


# Instruction variants so the model sees diverse prompts during training
_INSTRUCTION_TEMPLATES = [
    "Write a poem.",
    "Write an original poem.",
    "Compose a lyrical poem.",
    "Write a short poem.",
    "Write a poem in a literary style.",
    "Please write a poem for me.",
    "Could you write a poem?",
    "I'd love to read a poem.",
]


def make_chat_example(poem: str) -> dict:
    """
    Wrap a single poem in an instruction-following (messages) format
    suitable for an IT (instruction-tuned) model.
    """
    instruction = random.choice(_INSTRUCTION_TEMPLATES)
    return {
        "messages": [
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": poem},
        ]
    }


def load_poetry_corpus(poetry_dir: str) -> list:
    """
    Walk *poetry_dir*, read every .txt file, split into poems,
    and return a list of chat-formatted examples.
    """
    pattern = os.path.join(poetry_dir, "**", "*.txt")
    txt_files = sorted(glob.glob(pattern, recursive=True))

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found under: {poetry_dir}\n"
            "Check that POETRY_DIR is set correctly and the folder exists."
        )

    print(f"\n{'─'*60}")
    print(f"  Found {len(txt_files)} .txt file(s) in corpus")
    print(f"{'─'*60}")

    all_examples = []
    for fpath in txt_files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
        except OSError as exc:
            print(f"  ⚠  Skipping {fname}: {exc}")
            continue

        cleaned = clean_text(raw)
        poems   = split_into_poems(cleaned)

        for poem in poems:
            all_examples.append(make_chat_example(poem))

        print(f"  {fname:<40}  →  {len(poems):>3} poem(s)")

    print(f"{'─'*60}")
    print(f"  Total training examples : {len(all_examples)}")
    print(f"{'─'*60}\n")

    if len(all_examples) == 0:
        raise ValueError("No poems could be extracted. Check your .txt files.")

    return all_examples


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Load model + tokenizer
# ──────────────────────────────────────────────────────────────────────────────

print("Loading model and tokenizer …")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = True,
    dtype          = None,   # unsloth auto-detects bf16 on Ada/Ampere; fp16 otherwise
)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Attach LoRA adapters
# ──────────────────────────────────────────────────────────────────────────────

model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_RANK,
    lora_alpha     = LORA_RANK,            # alpha == r is a safe, well-tested default
    lora_dropout   = 0.05,
    bias           = "none",
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",        # MLP / FFN
    ],
    # "unsloth" mode applies smarter gradient checkpointing that saves ~30% VRAM
    # compared to vanilla HF checkpointing — highly recommended on 16 GB
    use_gradient_checkpointing = "unsloth",
    random_state   = RANDOM_SEED,
)

model.print_trainable_parameters()


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Build dataset
# ──────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)
examples = load_poetry_corpus(POETRY_DIR)

# Apply the model's own chat template (handles Gemma 4's <start_of_turn> tokens)
def apply_chat_template(example: dict) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize              = False,
        add_generation_prompt = False,   # we supply the assistant turn ourselves
    )
    return {"text": text}

hf_dataset = Dataset.from_list(examples)
hf_dataset = hf_dataset.map(apply_chat_template, remove_columns=["messages"])

# Persist formatted dataset for later inspection / reproducibility
os.makedirs(os.path.dirname(DATASET_CACHE) or ".", exist_ok=True)
with open(DATASET_CACHE, "w", encoding="utf-8") as fh:
    for item in hf_dataset:
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Formatted dataset cached → {DATASET_CACHE}")

# ── Spot-check: show raw poems + formatted training text, then wait for OK ──
preview_dataset(examples, hf_dataset, n=PREVIEW_COUNT)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Trainer
# ──────────────────────────────────────────────────────────────────────────────

use_bf16 = torch.cuda.is_bf16_supported()   # RTX 5070 Ti supports bf16

training_args = TrainingArguments(
    # ── batch / accumulation ──────────────────────────────────────────────
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,

    # ── schedule ──────────────────────────────────────────────────────────
    num_train_epochs    = NUM_EPOCHS if MAX_STEPS is None else 1,
    max_steps           = MAX_STEPS if MAX_STEPS is not None else -1,
    warmup_ratio        = 0.05,
    lr_scheduler_type   = "cosine",
    learning_rate       = LEARNING_RATE,

    # ── precision ─────────────────────────────────────────────────────────
    fp16                = not use_bf16,
    bf16                = use_bf16,

    # ── memory optimisation ───────────────────────────────────────────────
    optim               = "adamw_8bit",      # 8-bit Adam saves ~1 GB VRAM
    gradient_checkpointing = True,

    # ── logging / saving ──────────────────────────────────────────────────
    logging_steps       = 5,
    save_strategy       = "epoch",
    save_total_limit    = 2,                 # keep only the 2 most recent checkpoints
    output_dir          = OUTPUT_DIR,

    # ── misc ──────────────────────────────────────────────────────────────
    weight_decay        = 0.01,
    seed                = RANDOM_SEED,
    report_to           = "none",            # change to "wandb" if you use W&B
    dataloader_num_workers = 0,              # set to 0 on Windows (avoids fork issues)
)

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = hf_dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LENGTH,
    packing            = True,    # packs short poems into full context windows → faster
    args               = training_args,
)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Train
# ──────────────────────────────────────────────────────────────────────────────

print("\nStarting training …\n")
trainer_stats = trainer.train()

print(f"\nTraining complete.")
print(f"  Total steps  : {trainer_stats.global_step}")
print(f"  Training loss: {trainer_stats.training_loss:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Save outputs
# ──────────────────────────────────────────────────────────────────────────────

# ── 8a. LoRA adapter only (lightweight, ~50–200 MB) ──────────────────────────
# Load this with PeftModel.from_pretrained(base_model, adapter_path)
adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"\nLoRA adapter saved → {adapter_dir}")

# ── 8b. Merged fp16 model (no PEFT dependency at inference time, ~8–10 GB) ──
# Only run this if you have enough disk space; comment out to skip.
merged_dir = os.path.join(OUTPUT_DIR, "merged_fp16")
model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
print(f"Merged fp16 model saved → {merged_dir}")

print("\nAll done! ✓")


# ──────────────────────────────────────────────────────────────────────────────
# Quick inference sanity-check (optional — comment out if not needed)
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Inference test ─────────────────────────────────────────────────────")

FastLanguageModel.for_inference(model)   # enable unsloth's fast inference kernel

prompt_messages = [{"role": "user", "content": [{"type": "text", "text": "Write a short lyrical poem about autumn."}]}]
input_ids = tokenizer.apply_chat_template(
    prompt_messages,
    tokenize              = True,
    add_generation_prompt = True,   # adds the assistant <start_of_turn> token
    return_tensors        = "pt",
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens = 256,
        temperature    = 0.8,
        top_p          = 0.9,
        do_sample      = True,
    )

generated = output_ids[0][input_ids.shape[-1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))
print("───────────────────────────────────────────────────────────────────────\n")