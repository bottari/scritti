import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - optional dependency
    PeftModel = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.05
    do_sample: bool = True
    seed: int = 42


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_model(
    model_name_or_path: str,
    device: str,
    load_in_8bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model.to(device)

    model.eval()
    return model, tokenizer


def load_lora_model(
    base_model_name_or_path: str,
    lora_path: str,
    device: str,
    load_in_8bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if PeftModel is None:
        raise RuntimeError("peft is not installed. Please install peft to load LoRA weights.")
    base_model, tokenizer = load_base_model(base_model_name_or_path, device, load_in_8bit=load_in_8bit)
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    if not load_in_8bit:
        lora_model.to(device)
    lora_model.eval()
    return lora_model, tokenizer


def _generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    cfg: GenerationConfig,
    device: str,
    num_return_sequences: int,
) -> List[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
            do_sample=cfg.do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return decoded


def generate_samples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Iterable[str],
    num_samples: int,
    cfg: Optional[GenerationConfig] = None,
    device: Optional[str] = None,
    batch_size: int = 8,
    num_return_sequences: int = 1,
    log_every: int = 0,
    label: str = "",
) -> List[Dict[str, str]]:
    cfg = cfg or GenerationConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(cfg.seed)

    prompts_list = list(prompts)
    if not prompts_list:
        raise ValueError("prompts list is empty.")

    samples: List[Dict[str, str]] = []
    prompt_idx = 0
    batch_idx = 0
    start = time.time()

    while len(samples) < num_samples:
        batch_prompts = []
        for _ in range(batch_size):
            batch_prompts.append(prompts_list[prompt_idx % len(prompts_list)])
            prompt_idx += 1
            if len(samples) + (len(batch_prompts) * num_return_sequences) >= num_samples:
                break

        outputs = _generate_batch(model, tokenizer, batch_prompts, cfg, device, num_return_sequences)

        # model.generate returns num_return_sequences per prompt in order
        for i, prompt in enumerate(batch_prompts):
            for j in range(num_return_sequences):
                idx = i * num_return_sequences + j
                if idx >= len(outputs):
                    break
                samples.append({"prompt": prompt, "text": outputs[idx]})
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                break

        batch_idx += 1
        if log_every and batch_idx % log_every == 0:
            elapsed = time.time() - start
            print(f"[gen]{' ' + label if label else ''} {len(samples)}/{num_samples} samples in {elapsed:.1f}s")

    return samples
