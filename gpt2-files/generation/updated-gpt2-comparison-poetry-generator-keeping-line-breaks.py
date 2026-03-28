import argparse
import gc
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

warnings.filterwarnings("ignore", "huggingface_hub cache-system uses symlinks")
warnings.filterwarnings("ignore", "You are using a model of type gpt2")

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base GPT-2 output to a fine-tuned checkpoint.")
    parser.add_argument("--base-model", default="gpt2")
    parser.add_argument(
        "--fine-tuned-path",
        default=str(REPO_ROOT / "artifacts" / "gpt2-whitman-quickstart" / "final_model"),
    )
    parser.add_argument("--prompt", default="repeating in wind")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--min-length", type=int, default=30)
    return parser.parse_args()


def generation_kwargs(args, tokenizer):
    return {
        "max_new_tokens": args.max_new_tokens,
        "min_length": args.min_length,
        "num_return_sequences": 1,
        "do_sample": True,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def run_generation(model_name_or_path: str, prompt: str, device: str, dtype, args) -> tuple[str, torch.Tensor, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
    model.to(device)
    model.eval()

    full_prompt = tokenizer.bos_token + prompt if tokenizer.bos_token else prompt
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            streamer=streamer,
            **generation_kwargs(args, tokenizer),
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return decoded, output_ids[0], tokenizer


def main() -> None:
    args = parse_args()

    fine_tuned_path = Path(args.fine_tuned_path).expanduser().resolve()
    if not fine_tuned_path.exists():
        raise FileNotFoundError(f"Fine-tuned path not found: {fine_tuned_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Running on: {device.upper()} ({dtype})")

    print("\n" + "=" * 60)
    print(f"BASE MODEL STREAMING ({args.base_model})")
    print("=" * 60)
    base_decoded, _, _ = run_generation(args.base_model, args.prompt, device, dtype, args)

    print("\n" + "=" * 60)
    print(f"FINE-TUNED MODEL STREAMING ({fine_tuned_path})")
    print("=" * 60)
    fine_decoded, fine_ids, fine_tokenizer = run_generation(str(fine_tuned_path), args.prompt, device, dtype, args)

    if args.prompt in fine_decoded:
        generated_only = fine_decoded.split(args.prompt, 1)[1]
    else:
        generated_only = fine_decoded

    newline_id = fine_tokenizer.encode("\n", add_special_tokens=False)[0]
    visual_newlines = generated_only.count("\n")
    token_newlines = (fine_ids == newline_id).sum().item()

    print("\n" + "=" * 60)
    print("FINE-TUNED DIAGNOSTICS")
    print("=" * 60)
    print(f"Prompt: {args.prompt!r}")
    print(f"Base chars: {len(base_decoded)}")
    print(f"Fine-tuned chars: {len(fine_decoded)}")
    print(f"Fine-tuned visual newline count: {visual_newlines}")
    print(f"Fine-tuned newline token count: {token_newlines}")


if __name__ == "__main__":
    main()
