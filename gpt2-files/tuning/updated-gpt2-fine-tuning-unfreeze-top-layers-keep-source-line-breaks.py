import argparse
import warnings
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]


def print_diagnostics() -> None:
    if torch.cuda.is_available():
        print("=" * 60)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 60)
    else:
        print("No GPU detected, running on CPU.")


def load_text_corpus(data_path: Path) -> str:
    """Accept a directory of .txt files or a single .txt file."""
    source = data_path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Dataset path not found: {source}")

    if source.is_file():
        if source.suffix.lower() != ".txt":
            raise ValueError("Dataset file must be a .txt file.")
        content = source.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Dataset file is empty: {source}")
        return content

    txt_files = sorted(source.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found under: {source}")

    all_poems = []
    for txt_file in txt_files:
        content = txt_file.read_text(encoding="utf-8").strip()
        if content:
            all_poems.append(content)

    if not all_poems:
        raise ValueError(f"No non-empty .txt content found under: {source}")

    return "<|endoftext|><|endoftext|>".join(all_poems)


def create_chunked_dataset(text: str, tokenizer, block_size: int) -> Dataset:
    tokenized = tokenizer(text, add_special_tokens=False)
    input_ids = tokenized["input_ids"]

    if len(input_ids) < block_size:
        raise ValueError(
            f"Not enough tokens ({len(input_ids)}) for block size {block_size}. "
            "Use a smaller block size or larger dataset."
        )

    stride = max(1, block_size // 2)
    chunks = []
    for i in range(0, len(input_ids) - block_size + 1, stride):
        chunks.append(input_ids[i : i + block_size])

    return Dataset.from_dict({"input_ids": chunks, "labels": chunks.copy()})


def unfreeze_top_layers(model, num_layers: int):
    for param in model.parameters():
        param.requires_grad = False

    total_layers = len(model.transformer.h)
    num_layers = max(0, min(num_layers, total_layers))
    start_idx = total_layers - num_layers

    for i in range(start_idx, total_layers):
        for param in model.transformer.h[i].parameters():
            param.requires_grad = True

    for param in model.lm_head.parameters():
        param.requires_grad = True

    for name, module in model.named_modules():
        if "ln" in name or "LayerNorm" in name:
            for param in module.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-2 fine-tuning with line-break preserving chunks.")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--dataset-path", default=str(REPO_ROOT / "poetry_txt_whitman"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts" / "gpt2-whitman-quickstart"))
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-layers-to-unfreeze", type=int, default=4)
    parser.add_argument("--save-total-limit", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_diagnostics()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)

    print(f"Base model: {args.model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Output dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None,
        device_map="auto" if torch.cuda.is_available() else None,
        use_cache=False,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model = unfreeze_top_layers(model, args.num_layers_to_unfreeze)

    concatenated_text = load_text_corpus(dataset_path)
    dataset = create_chunked_dataset(concatenated_text, tokenizer, args.block_size)
    print(f"Training chunks: {len(dataset):,}")

    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=args.save_total_limit,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"Training complete. Saved to: {final_dir}")


if __name__ == "__main__":
    main()
