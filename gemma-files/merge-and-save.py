"""
merge_lora.py  –  Merge a LoRA checkpoint into the Gemma 4 base model and save.

The checkpoint only contains LoRA delta weights.  When transformers tries to
randomly initialise the remaining (base-model) keys into 4-bit quantised
(Byte) tensors it crashes with:
    NotImplementedError: "normal_kernel_cuda" not implemented for 'Byte'

Fix: monkeypatch _initialize_missing_keys to a no-op before loading.
This is safe — the LoRA adapter supplies the actual values for those layers.

Usage
-----
    python merge_lora.py
    python merge_lora.py --adapter D:\\models\\my-checkpoint --output D:\\models\\my-merged
    python merge_lora.py --save-method merged_4bit
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("unsloth").setLevel(logging.ERROR)

# ── defaults ────────────────────────────────────────────────────────────────
ADAPTER_PATH = r"D:\models\gemma4-poetry-finetune-whitmanv3\checkpoint-400"
OUTPUT_PATH  = r"D:\models\gemma4-poetry-merged"
MAX_SEQ_LEN  = 2048
# ────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Merge Unsloth LoRA adapter into base model.")
    p.add_argument("--adapter",        default=ADAPTER_PATH, help="Path to LoRA checkpoint folder")
    p.add_argument("--output",         default=OUTPUT_PATH,  help="Where to save the merged model")
    p.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LEN)
    p.add_argument(
        "--save-method",
        default="merged_16bit",
        choices=["merged_16bit", "merged_4bit", "lora"],
        help=(
            "merged_16bit : full precision, best quality (~10 GB)  [default]\n"
            "merged_4bit  : quantised, smaller (~5 GB)\n"
            "lora         : save adapter only (no merge)"
        ),
    )
    return p.parse_args()


def patch_transformers():
    """
    Prevent transformers from calling normal_() on 4-bit (Byte) tensors
    when it tries to initialise weights that are absent from the checkpoint.
    The LoRA adapter provides the real values; random init is not needed.
    """
    import transformers.modeling_utils as mu

    def _noop_initialize_missing_keys(self, *args, **kwargs):
        pass  # skip – LoRA adapter supplies these weights

    mu.PreTrainedModel._initialize_missing_keys = _noop_initialize_missing_keys
    print("  ✓ Patched transformers: missing-key init disabled")


def main():
    args = parse_args()

    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        answer = input(
            f"\n  Output folder '{output_dir}' already exists and is not empty.\n"
            "  Overwrite? [y/N] "
        ).strip().lower()
        if answer != "y":
            print("  Aborted.")
            sys.exit(0)

    print("\n── Step 1 / 2 : Loading model + adapter ────────────────────────────────")
    print(f"  adapter : {args.adapter}")

    # Must patch before unsloth / transformers are imported so the patched
    # method is in place when from_pretrained triggers _initialize_missing_keys.
    patch_transformers()

    from unsloth import FastLanguageModel  # noqa: PLC0415

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    print("  ✓ Model + adapter loaded")

    print("\n── Step 2 / 2 : Merging and saving ─────────────────────────────────────")
    print(f"  method  : {args.save_method}")
    print(f"  output  : {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained_merged(
        str(output_dir),
        tokenizer,
        save_method=args.save_method,
    )
    print("  ✓ Merged model saved\n")

    print("Done!  Point DEFAULT_MODEL in gemma-chat.py at:")
    print(f"  {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()