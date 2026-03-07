import argparse
import json
import random
import warnings
from pathlib import Path

import torch
from huggingface_hub import login
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- CONFIGURATION ---
HF_TOKEN = ""

# Model A: Llama 3 + QLoRA adapter
MODEL_A_LABEL = "llama3-poetry-mercury-26"
MODEL_A_BASE = "meta-llama/Llama-3.1-8B"
MODEL_A_ADAPTER_PATH = r"D:\models\choice-models\llama3-8b-poetry-mercury-26-qlora-8bit-019\final_model"

# Model B: GPT-2 fine-tuned checkpoint
MODEL_B_LABEL = "gpt2-finetuned-poetry-mercury-04"
MODEL_B_BASE = "gpt2"
MODEL_B_FINETUNED_PATH = r"D:\models\choice-models\gpt2-finetuned-poetry-mercury-04\final_model"
MODEL_B_LINE_BREAK_TOKEN = "<|line|>"

# Llama generation settings
LLAMA_MAX_NEW_TOKENS = 250
LLAMA_TEMPERATURE = 0.5
LLAMA_TOP_P = 0.75
LLAMA_TOP_K = 10
LLAMA_REPETITION_PENALTY = 1.25
LLAMA_NUM_RETURN_SEQUENCES = 1

# GPT-2 generation settings
GPT2_MAX_LENGTH = 350
GPT2_MIN_LENGTH = 100
GPT2_NUM_RETURN_SEQUENCES = 1
GPT2_DO_SAMPLE = True
GPT2_TOP_K = 50
GPT2_TOP_P = 0.9
GPT2_TEMPERATURE = 0.85
GPT2_REPETITION_PENALTY = 1.2
GPT2_NO_REPEAT_NGRAM_SIZE = 2

warnings.filterwarnings("ignore")


def setup_auth(hf_token: str | None) -> None:
    if hf_token:
        print("Logging in with explicit Hugging Face token...")
        login(token=hf_token)
    else:
        print("No explicit token set in script/CLI. Using existing local HF login if available.")


def print_diagnostics() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {gpu_memory:.1f} GB")


def build_llama_quant_config() -> BitsAndBytesConfig:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )


def load_prompts(prompts_path: Path) -> list[str]:
    data = json.loads(prompts_path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        prompts = [item for item in data if isinstance(item, str) and item.strip()]
    elif isinstance(data, dict) and isinstance(data.get("prompts"), list):
        prompts = [item for item in data["prompts"] if isinstance(item, str) and item.strip()]
    else:
        raise ValueError("prompts.json must be either a list of strings or {'prompts': [...]}.")

    if not prompts:
        raise ValueError("No valid prompts found in prompts.json.")

    return prompts


def get_llama_model_and_tokenizer(base_model_name: str, adapter_path: str, hf_token: str | None):
    adapter = Path(adapter_path)
    if not adapter.exists():
        raise FileNotFoundError(f"Llama adapter not found: {adapter}")

    bnb_config = build_llama_quant_config()
    auth_token = hf_token if hf_token else True

    print(f"Loading Llama base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=auth_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=auth_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Attaching Llama adapter: {adapter}")
    model = PeftModel.from_pretrained(base_model, str(adapter))
    model.eval()

    lora_count = sum(1 for _, module in model.named_modules() if isinstance(module, LoraLayer))
    print(f"LoRA check [{MODEL_A_LABEL}] wrapped={isinstance(model, PeftModel)}, lora_layers={lora_count}")

    return tokenizer, model


def get_gpt2_model_and_tokenizer(finetuned_path: str):
    model_path = Path(finetuned_path)
    if not model_path.exists():
        raise FileNotFoundError(f"GPT-2 fine-tuned path not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading GPT-2 tokenizer/model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype)
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def generate_with_llama(model, tokenizer, prompt: str) -> str:
    input_text = tokenizer.bos_token + prompt
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    llama_eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    stop_ids = [tokenizer.eos_token_id]
    if isinstance(llama_eot_id, int) and llama_eot_id >= 0:
        stop_ids.append(llama_eot_id)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=LLAMA_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=LLAMA_TEMPERATURE,
            top_p=LLAMA_TOP_P,
            top_k=LLAMA_TOP_K,
            num_return_sequences=LLAMA_NUM_RETURN_SEQUENCES,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids,
            repetition_penalty=LLAMA_REPETITION_PENALTY,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated[len(input_text) :].strip() if generated.startswith(input_text) else generated.strip()


def generate_with_gpt2(model, tokenizer, prompt: str, line_break_token: str) -> str:
    device = model.device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=GPT2_MAX_LENGTH,
            min_length=GPT2_MIN_LENGTH,
            num_return_sequences=GPT2_NUM_RETURN_SEQUENCES,
            do_sample=GPT2_DO_SAMPLE,
            top_k=GPT2_TOP_K,
            top_p=GPT2_TOP_P,
            temperature=GPT2_TEMPERATURE,
            repetition_penalty=GPT2_REPETITION_PENALTY,
            no_repeat_ngram_size=GPT2_NO_REPEAT_NGRAM_SIZE,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=False)
    return text.replace(line_break_token, "\n")


def make_random_ids(count: int, min_id: int = 100000, max_id: int = 999999) -> list[int]:
    pool_size = max_id - min_id + 1
    if count > pool_size:
        raise ValueError("Not enough unique IDs available for requested prompt count.")
    return random.sample(range(min_id, max_id + 1), k=count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate side-by-side outputs from Llama3 QLoRA and GPT-2 fine-tuned models"
    )
    parser.add_argument("--prompts", default="prompts.json", help="Path to prompts JSON file")
    parser.add_argument("--output", default="poem_review_app/outputs.json", help="Side-by-side output file")
    parser.add_argument(
        "--flat-output",
        default="poem_review_app/outputs_flat.json",
        help="Flat output file for analysis/debugging",
    )

    parser.add_argument("--hf-token", default=HF_TOKEN, help="Optional Hugging Face token")

    parser.add_argument("--model-a-label", default=MODEL_A_LABEL)
    parser.add_argument("--model-a-base", default=MODEL_A_BASE)
    parser.add_argument("--model-a-adapter", default=MODEL_A_ADAPTER_PATH)

    parser.add_argument("--model-b-label", default=MODEL_B_LABEL)
    parser.add_argument("--model-b-base", default=MODEL_B_BASE)
    parser.add_argument("--model-b-finetuned", default=MODEL_B_FINETUNED_PATH)
    parser.add_argument("--model-b-line-break-token", default=MODEL_B_LINE_BREAK_TOKEN)

    args = parser.parse_args()

    setup_auth(args.hf_token or None)
    print_diagnostics()

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    prompts = load_prompts(prompts_path)
    ids = make_random_ids(len(prompts))

    llama_tokenizer, llama_model = get_llama_model_and_tokenizer(
        base_model_name=args.model_a_base,
        adapter_path=args.model_a_adapter,
        hf_token=args.hf_token or None,
    )

    gpt2_tokenizer, gpt2_model = get_gpt2_model_and_tokenizer(args.model_b_finetuned)

    side_by_side_rows = []
    flat_rows = []

    for idx, prompt in enumerate(prompts):
        poem_id = ids[idx]

        poem_a = generate_with_llama(llama_model, llama_tokenizer, prompt)
        poem_b = generate_with_gpt2(gpt2_model, gpt2_tokenizer, prompt, args.model_b_line_break_token)

        side_by_side_rows.append(
            {
                "id": poem_id,
                "prompt": prompt,
                "outputs": [
                    {"model": args.model_a_label, "base_model": args.model_a_base, "poem": poem_a},
                    {"model": args.model_b_label, "base_model": args.model_b_base, "poem": poem_b},
                ],
            }
        )

        flat_rows.append(
            {
                "id": poem_id,
                "model": args.model_a_label,
                "base_model": args.model_a_base,
                "prompt": prompt,
                "poem": poem_a,
            }
        )
        flat_rows.append(
            {
                "id": poem_id,
                "model": args.model_b_label,
                "base_model": args.model_b_base,
                "prompt": prompt,
                "poem": poem_b,
            }
        )

        print(f"[{idx + 1}/{len(prompts)}] Generated both model outputs for ID {poem_id}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(side_by_side_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    flat_output_path = Path(args.flat_output)
    flat_output_path.parent.mkdir(parents=True, exist_ok=True)
    flat_output_path.write_text(json.dumps(flat_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved side-by-side outputs to: {output_path.resolve()}")
    print(f"Saved flat outputs to: {flat_output_path.resolve()}")


if __name__ == "__main__":
    main()

