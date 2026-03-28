import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - optional dependency
    PeftModel = None


REPO_ROOT = Path(__file__).resolve().parent

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")


def expand_env_placeholders(value: str) -> str:
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def repl(match: re.Match[str]) -> str:
        var_name = match.group(1)
        env_val = os.getenv(var_name)
        if env_val is None or not env_val.strip():
            raise ValueError(
                f"Missing required environment variable: {var_name}. "
                f"Set it in your environment or .env file."
            )
        return env_val

    return pattern.sub(repl, value)


def _expand_config(value):
    if isinstance(value, dict):
        return {k: _expand_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_config(v) for v in value]
    if isinstance(value, str):
        return expand_env_placeholders(value)
    return value


def resolve_path(value: str | None, config_dir: Path) -> str | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((config_dir / candidate).resolve())


def load_yaml(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _expand_config(raw)
    cfg["_config_path"] = str(config_path)
    return cfg


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _normalize_prompt_entry(entry: Any, idx: int) -> Dict[str, str]:
    if isinstance(entry, str):
        return {"prompt_id": str(idx), "prompt": entry}
    if isinstance(entry, dict):
        prompt = entry.get("prompt") or entry.get("text") or entry.get("input")
        prompt_id = entry.get("prompt_id") or entry.get("id") or entry.get("uid") or str(idx)
        if prompt is None:
            raise ValueError("Prompt entry missing 'prompt' field.")
        return {"prompt_id": str(prompt_id), "prompt": str(prompt)}
    raise ValueError(f"Unsupported prompt entry type: {type(entry)}")


def _read_prompt_lines(path: str) -> List[Dict[str, str]]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{"prompt_id": str(idx), "prompt": line} for idx, line in enumerate(lines)]


def load_prompt_set(path: str) -> List[Dict[str, str]]:
    source = Path(path)
    if source.suffix.lower() == ".txt":
        return _read_prompt_lines(str(source))

    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("prompt_set JSON must be a list of prompts or prompt objects.")
    return [_normalize_prompt_entry(entry, idx) for idx, entry in enumerate(data)]


def prepare_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name_or_path: str, device: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = prepare_tokenizer(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def load_finetuned_model(
    base_model_name_or_path: str,
    finetuned_path: str,
    device: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = prepare_tokenizer(base_model_name_or_path)
    if PeftModel is None:
        raise RuntimeError("peft is not installed. Please install peft to load LoRA weights.")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, finetuned_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_texts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    temperature: float,
    top_p: float,
    max_tokens: int,
    device: str,
    batch_size: int,
) -> List[str]:
    outputs: List[str] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[start : start + batch_size])
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        batch_texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        outputs.extend(batch_texts)
    return outputs


def repetition_index(text: str, tokenizer: AutoTokenizer) -> float:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return 0.0
    unique = len(set(ids))
    return float((len(ids) - unique) / len(ids))


def perplexity(text: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str) -> float:
    ids = tokenizer.encode(text, return_tensors="pt")
    if ids.numel() <= 1:
        return float("nan")
    ids = ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids=ids, labels=ids)
        loss = outputs.loss
    if loss is None or math.isnan(float(loss)):
        return float("nan")
    return float(torch.exp(loss).item())


def resolve_meter_score(optional: bool = False) -> Optional[Any]:
    try:
        from eval.metrics import meter_score  # type: ignore

        return meter_score
    except Exception as exc:
        if optional:
            return None
        raise RuntimeError(
            "meter_score not found. Define meter_score in eval/metrics.py or adjust the import."
        ) from exc


def compute_semantic_similarity(
    embedder: SentenceTransformer,
    base_texts: Sequence[str],
    finetuned_texts: Sequence[str],
) -> List[float]:
    if len(base_texts) != len(finetuned_texts):
        raise ValueError("base_texts and finetuned_texts must be same length for similarity.")
    base_emb = embedder.encode(list(base_texts), normalize_embeddings=True, show_progress_bar=False)
    fine_emb = embedder.encode(list(finetuned_texts), normalize_embeddings=True, show_progress_bar=False)
    return [float(np.dot(b, f)) for b, f in zip(base_emb, fine_emb)]


def iter_sweep(values: Iterable[float]) -> List[float]:
    return list(values) if isinstance(values, list) else list(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven eval runner with prompt sweep.")
    parser.add_argument("--config", default="eval_config_sweep.yaml", help="Path to eval config file")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    config_dir = Path(cfg.get("_config_path", Path(args.config).resolve())).parent

    model_base = cfg.get("model_base")
    model_finetuned_raw = cfg.get("model_finetuned")
    prompt_set_raw = cfg.get("prompt_set")
    generation = cfg.get("generation", {})
    metrics = cfg.get("metrics", [])
    output_cfg = cfg.get("output", {})

    model_finetuned = resolve_path(model_finetuned_raw, config_dir)
    prompt_set = resolve_path(prompt_set_raw, config_dir)

    if not model_base or not model_finetuned or not prompt_set:
        raise ValueError("model_base, model_finetuned, and prompt_set are required in config.")

    temps = iter_sweep(generation.get("temperature", [0.7]))
    top_ps = iter_sweep(generation.get("top_p", [0.95]))
    max_tokens = int(generation.get("max_tokens", 120))
    batch_size = int(generation.get("batch_size", 4))

    log_dir_raw = output_cfg.get("log_dir", "eval_logs")
    log_dir = Path(resolve_path(log_dir_raw, config_dir) or (REPO_ROOT / "eval_logs"))
    run_name = output_cfg.get("run_name", datetime.now().strftime("eval_%Y%m%d_%H%M%S"))
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(log_dir)

    prompt_entries = load_prompt_set(prompt_set)
    prompts = [p["prompt"] for p in prompt_entries]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model, base_tokenizer = load_base_model(model_base, device)
    finetuned_model, finetuned_tokenizer = load_finetuned_model(model_base, model_finetuned, device)

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    meter_fn = None
    if "meter_score" in metrics:
        meter_fn = resolve_meter_score(optional=True)
        if meter_fn is None:
            print("Warning: meter_score requested but not available; filling NaN.")

    rows: List[Dict[str, Any]] = []

    for temperature in temps:
        for top_p in top_ps:
            base_texts = generate_texts(
                base_model,
                base_tokenizer,
                prompts,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=max_tokens,
                device=device,
                batch_size=batch_size,
            )
            finetuned_texts = generate_texts(
                finetuned_model,
                finetuned_tokenizer,
                prompts,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=max_tokens,
                device=device,
                batch_size=batch_size,
            )

            sims = compute_semantic_similarity(embedder, base_texts, finetuned_texts)

            for idx, prompt_entry in enumerate(prompt_entries):
                prompt_id = prompt_entry["prompt_id"]
                prompt = prompt_entry["prompt"]

                base_text = base_texts[idx]
                fine_text = finetuned_texts[idx]

                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "model_type": "base",
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "meter_score": float(meter_fn(base_text)) if "meter_score" in metrics and meter_fn is not None else float("nan"),
                        "semantic_similarity": float(sims[idx]) if "semantic_similarity" in metrics else float("nan"),
                        "repetition_index": float(repetition_index(base_text, base_tokenizer)) if "repetition_index" in metrics else float("nan"),
                        "perplexity": float(perplexity(base_text, base_model, base_tokenizer, device)) if "perplexity" in metrics else float("nan"),
                        "output_length": len(base_text),
                        "prompt": prompt,
                        "output": base_text,
                    }
                )

                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "model_type": "finetuned",
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "meter_score": float(meter_fn(fine_text)) if "meter_score" in metrics and meter_fn is not None else float("nan"),
                        "semantic_similarity": float(sims[idx]) if "semantic_similarity" in metrics else float("nan"),
                        "repetition_index": float(repetition_index(fine_text, finetuned_tokenizer)) if "repetition_index" in metrics else float("nan"),
                        "perplexity": float(perplexity(fine_text, finetuned_model, finetuned_tokenizer, device)) if "perplexity" in metrics else float("nan"),
                        "output_length": len(fine_text),
                        "prompt": prompt,
                        "output": fine_text,
                    }
                )

    df = pd.DataFrame(rows)
    csv_path = log_dir / f"{run_name}_{run_stamp}.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "mean_meter_score": float(df["meter_score"].mean()) if "meter_score" in df else float("nan"),
        "mean_semantic_similarity": float(df["semantic_similarity"].mean()) if "semantic_similarity" in df else float("nan"),
        "mean_perplexity": float(df["perplexity"].mean()) if "perplexity" in df else float("nan"),
        "mean_repetition_index": float(df["repetition_index"].mean()) if "repetition_index" in df else float("nan"),
    }

    summary_path = log_dir / f"{run_name}_{run_stamp}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
