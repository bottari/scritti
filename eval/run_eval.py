import argparse
import gc
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

from generation import GenerationConfig, generate_samples, load_base_model, load_lora_model
from latent_space import build_latent_space_groups, plot_latent_space_groups
from metrics import (
    build_corpus_token_set,
    compound_imagery_density,
    conceptual_distance_score,
    imagery_density,
    lexical_novelty,
    load_spacy,
    metaphor_density,
    narrative_density,
    semantic_drift,
    surreal_imagery_score,
    token_distribution_shift,
)
from plots import (
    plot_compound_imagery_distribution,
    plot_conceptual_distance_distribution,
    plot_histogram_compare,
    plot_imagery_distribution,
    plot_narrative_distribution,
    plot_novelty_distribution,
    plot_semantic_drift_scatter,
    plot_surrealism_index,
)
from reference_poetry_loader import load_human_poems


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = "gpt2"
DEFAULT_LORA_PATH = "${SCRITTI_EVAL_LORA_PATH}"
DEFAULT_HUMAN_CORPUS = REPO_ROOT / "poetry_txt_whitman"

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")


def is_qwen_model(name_or_path: str) -> bool:
    if not name_or_path:
        return False
    return "qwen" in name_or_path.lower()


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


def resolve_config_path(value: str | None, config_dir: Path) -> str | None:
    if not value:
        return value
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((config_dir / candidate).resolve())


def read_lines(path: str) -> List[str]:
    source = Path(path).expanduser()
    if source.is_dir():
        lines: List[str] = []
        for txt_file in sorted(source.glob("*.txt")):
            file_lines = [line.strip() for line in txt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            lines.extend(file_lines)
        return lines

    return [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def format_delta(base: float, other: float) -> str:
    if base == 0:
        return "n/a"
    return f"{((other - base) / base) * 100:.0f}%"


def load_config(path: str) -> Dict:
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _expand_config(raw)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def _sample_if_needed(items: List[str], max_items: int, seed: int = 42) -> List[str]:
    if max_items <= 0 or len(items) <= max_items:
        return items
    rng = random.Random(seed)
    return rng.sample(items, max_items)


def fix_tokenizer_padding(tokenizer):
    if tokenizer and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def release_model(model_obj) -> None:
    if model_obj is None:
        return
    del model_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base vs LoRA surreal poetry outputs.")
    parser.add_argument("--config", default=None)

    parser.add_argument("--base-model", default=None)
    parser.add_argument("--lora-path", default=None)

    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-lora", action="store_true")

    parser.add_argument("--base-8bit", action="store_true")
    parser.add_argument("--lora-8bit", action="store_true")
    parser.add_argument("--base-4bit", action="store_true")
    parser.add_argument("--lora-4bit", action="store_true")
    parser.add_argument("--base-8bit-cpu-offload", action="store_true")
    parser.add_argument("--lora-8bit-cpu-offload", action="store_true")

    parser.add_argument("--extra-base-model", default=None)
    parser.add_argument("--extra-lora-path", default=None)
    parser.add_argument("--extra-base-label", default=None)
    parser.add_argument("--extra-lora-label", default=None)
    parser.add_argument("--extra-8bit", action="store_true")
    parser.add_argument("--extra-4bit", action="store_true")
    parser.add_argument("--extra-8bit-cpu-offload", action="store_true")

    parser.add_argument("--prompts", default=None)
    parser.add_argument("--corpus", default=None)

    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)

    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embed-batch-size", type=int, default=None)
    parser.add_argument("--max-nouns", type=int, default=None)

    parser.add_argument("--human-corpus", default=None)
    parser.add_argument("--human-max-lines", type=int, default=None)

    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--umap-max-points", type=int, default=None)

    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    config_dir = Path(cfg.get("_config_path")).parent if cfg.get("_config_path") else Path.cwd()

    base_model = args.base_model or cfg.get("base_model") or DEFAULT_BASE_MODEL
    lora_path_raw = args.lora_path or cfg.get("lora_path") or DEFAULT_LORA_PATH
    lora_path = resolve_config_path(lora_path_raw, config_dir) if lora_path_raw else None

    extra_base_model = args.extra_base_model or cfg.get("extra_base_model")
    extra_lora_raw = args.extra_lora_path or cfg.get("extra_lora_path")
    extra_lora_path = resolve_config_path(extra_lora_raw, config_dir) if extra_lora_raw else None

    prompts_raw = args.prompts or cfg.get("prompts")
    corpus_raw = args.corpus or cfg.get("corpus")
    prompts_path = resolve_config_path(prompts_raw, config_dir) if prompts_raw else None
    corpus_path = resolve_config_path(corpus_raw, config_dir) if corpus_raw else None

    num_samples = args.num_samples or int(cfg.get("num_samples", 200))
    max_new_tokens = args.max_new_tokens or int(cfg.get("max_new_tokens", 128))

    temperature = args.temperature if args.temperature is not None else float(cfg.get("temperature", 0.9))
    top_p = args.top_p if args.top_p is not None else float(cfg.get("top_p", 0.95))
    top_k = args.top_k if args.top_k is not None else int(cfg.get("top_k", 50))

    repetition_penalty = args.repetition_penalty if args.repetition_penalty is not None else float(
        cfg.get("repetition_penalty", 1.05)
    )

    embed_batch_size = args.embed_batch_size if args.embed_batch_size is not None else int(cfg.get("embed_batch_size", 64))
    max_nouns = args.max_nouns if args.max_nouns is not None else int(cfg.get("max_nouns", 128))
    umap_max_points = args.umap_max_points if args.umap_max_points is not None else int(cfg.get("umap_max_points", 0))

    human_corpus_raw = args.human_corpus or cfg.get("human_corpus") or str(DEFAULT_HUMAN_CORPUS)
    human_corpus_path = resolve_config_path(human_corpus_raw, config_dir)
    human_lines = load_human_poems(human_corpus_path)
    if args.human_max_lines is not None and args.human_max_lines > 0:
        human_lines = _sample_if_needed(human_lines, args.human_max_lines)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[config] loaded")
    print(f"[config] device={device}")
    print(f"[config] base_model={base_model}")
    print(f"[config] lora_path={lora_path}")
    print(f"[config] prompts={prompts_path}")
    print(f"[config] corpus={corpus_path}")
    print(f"[config] human_corpus={human_corpus_path}")
    print(f"[config] extra_base_model={extra_base_model or '(disabled)'}")
    print(f"[config] extra_lora_path={extra_lora_path or '(disabled)'}")

    # Guard known-unstable combo seen on some Windows CUDA + bitsandbytes setups:
    # Llama LoRA + int8 + CPU offload can fail at runtime inside bnb kernels.
    if (
        not args.skip_lora
        and args.lora_8bit
        and args.lora_8bit_cpu_offload
        and "llama" in base_model.lower()
    ):
        print(
            "[warn] Detected Llama LoRA with --lora-8bit and --lora-8bit-cpu-offload. "
            "Auto-falling back to --lora-4bit for stability."
        )
        args.lora_8bit = False
        args.lora_8bit_cpu_offload = False
        args.lora_4bit = True

    if bool(extra_base_model) ^ bool(extra_lora_path):
        raise ValueError(
            "extra_base_model and extra_lora_path must both be set (or both empty). "
            "Set SCRITTI_EVAL_EXTRA_BASE_MODEL and SCRITTI_EVAL_EXTRA_LORA_PATH together."
        )

    base_model_obj = None
    lora_model = None
    base_tokenizer_for_shift = None
    lora_tokenizer_for_shift = None

    extra_base_model_obj = None
    extra_tokenizer = None
    extra_lora_model = None

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    if not prompts_path or not corpus_path:
        raise ValueError("prompts and corpus paths are required via config or CLI.")

    prompts = read_lines(prompts_path)
    corpus_lines = read_lines(corpus_path)
    corpus_tokens = build_corpus_token_set(corpus_lines)

    print("Starting generation...")

    base_samples = []
    lora_samples = []
    extra_base_samples = []
    extra_lora_samples = []
    human_samples = []

    if not args.skip_base:
        print("[load] loading base model...")
        base_model_obj, base_tokenizer = load_base_model(
            base_model,
            device,
            load_in_8bit=args.base_8bit,
            load_in_4bit=args.base_4bit,
            cpu_offload=args.base_8bit_cpu_offload,
            trust_remote_code=is_qwen_model(base_model),
        )
        fix_tokenizer_padding(base_tokenizer)

        base_samples = generate_samples(
            base_model_obj,
            base_tokenizer,
            prompts,
            num_samples=num_samples,
            cfg=gen_cfg,
            device=device,
            batch_size=args.batch_size,
            label="base",
        )
        base_tokenizer_for_shift = base_tokenizer
        release_model(base_model_obj)
        base_model_obj = None

    if not args.skip_lora:
        if not lora_path:
            raise ValueError("lora_path is required unless --skip-lora is set.")
        print("[load] loading lora model...")
        lora_model, lora_tokenizer = load_lora_model(
            base_model,
            lora_path,
            device,
            load_in_8bit=args.lora_8bit,
            load_in_4bit=args.lora_4bit,
            cpu_offload=args.lora_8bit_cpu_offload,
            trust_remote_code=is_qwen_model(base_model),
        )
        fix_tokenizer_padding(lora_tokenizer)

        lora_samples = generate_samples(
            lora_model,
            lora_tokenizer,
            prompts,
            num_samples=num_samples,
            cfg=gen_cfg,
            device=device,
            batch_size=args.batch_size,
            label="lora",
        )
        lora_tokenizer_for_shift = lora_tokenizer
        release_model(lora_model)
        lora_model = None

    if extra_base_model:
        print("[load] loading extra base model...")
        extra_base_model_obj, extra_tokenizer = load_base_model(
            extra_base_model,
            device,
            load_in_8bit=args.extra_8bit,
            load_in_4bit=args.extra_4bit,
            cpu_offload=args.extra_8bit_cpu_offload,
            trust_remote_code=is_qwen_model(extra_base_model),
        )
        fix_tokenizer_padding(extra_tokenizer)

    if extra_lora_path:
        print("[load] loading extra lora model...")
        extra_lora_model, extra_tokenizer = load_lora_model(
            extra_base_model,
            extra_lora_path,
            device,
            load_in_8bit=args.extra_8bit,
            load_in_4bit=args.extra_4bit,
            cpu_offload=args.extra_8bit_cpu_offload,
            trust_remote_code=is_qwen_model(extra_base_model),
        )
        fix_tokenizer_padding(extra_tokenizer)

    if extra_base_model_obj:
        extra_base_samples = generate_samples(
            extra_base_model_obj,
            extra_tokenizer,
            prompts,
            num_samples=num_samples,
            cfg=gen_cfg,
            device=device,
            batch_size=args.batch_size,
            label="extra_base",
        )

    if extra_lora_model:
        extra_lora_samples = generate_samples(
            extra_lora_model,
            extra_tokenizer,
            prompts,
            num_samples=num_samples,
            cfg=gen_cfg,
            device=device,
            batch_size=args.batch_size,
            label="extra_lora",
        )

    base_texts = [s["text"] for s in base_samples]
    lora_texts = [s["text"] for s in lora_samples]
    extra_base_texts = [s["text"] for s in extra_base_samples]
    extra_lora_texts = [s["text"] for s in extra_lora_samples]
    if human_lines:
        human_samples = [{"prompt": "", "text": line} for line in human_lines]

    embedder = SentenceTransformer(args.embedding_model)
    spacy_resources = load_spacy()

    drift_scores = []
    if base_texts and lora_texts and len(base_texts) == len(lora_texts):
        drift_scores = semantic_drift(base_texts, lora_texts, embedder, batch_size=embed_batch_size)

    noun_embed_cache: Dict[str, np.ndarray] = {}

    rows = []

    def enrich(samples, label, drift_values=None):
        for idx, sample in enumerate(samples):
            text = sample["text"]

            imagery = imagery_density(text)
            metaphor = metaphor_density(text, spacy_resources.nlp)

            surreal = surreal_imagery_score(
                text,
                spacy_resources.nlp,
                embedder,
                cache=noun_embed_cache,
                max_nouns=max_nouns,
                batch_size=embed_batch_size,
            )

            conceptual = conceptual_distance_score(
                text,
                spacy_resources.nlp,
                embedder,
                cache=noun_embed_cache,
                max_nouns=max_nouns,
                batch_size=embed_batch_size,
            )

            compound_imagery = compound_imagery_density(text, spacy_resources.nlp)
            narrative = narrative_density(text, spacy_resources.nlp)

            row = {
                "model": label,
                "prompt": sample["prompt"],
                "text": text,
                "imagery_density": imagery,
                "compound_imagery_density": compound_imagery,
                "narrative_density": narrative,
                "lexical_novelty": lexical_novelty(text, corpus_tokens),
                "metaphor_density": metaphor,
                "surreal_imagery_score": surreal,
                "conceptual_distance_score": conceptual,
                "surrealism_index": imagery + metaphor + surreal + conceptual,
            }

            if drift_values is not None and idx < len(drift_values):
                row["semantic_drift_from_base"] = drift_values[idx]

            rows.append(row)

    enrich(base_samples, "base")
    enrich(lora_samples, "lora", drift_values=drift_scores)
    enrich(extra_base_samples, "extra_base")
    enrich(extra_lora_samples, "extra_lora")
    enrich(human_samples, "human")

    df = pd.DataFrame(rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_raw = args.output_dir or cfg.get("output_dir")
    if output_dir_raw:
        configured_output = Path(resolve_config_path(output_dir_raw, config_dir))
        # Treat config/CLI output_dir as a parent directory unless it clearly points
        # to a file path. This preserves timestamped run folders by default.
        if configured_output.suffix:
            output_dir = configured_output.parent / timestamp
        else:
            output_dir = configured_output / timestamp
    else:
        output_dir = REPO_ROOT / "eval_outputs" / timestamp

    ensure_dir(output_dir)

    df_path = output_dir / "eval_samples.csv"
    df.to_csv(df_path, index=False)

    print("Saved:", df_path)

    if not df.empty:
        plot_imagery_distribution(df, str(output_dir / "imagery_density.png"))
        plot_novelty_distribution(df, str(output_dir / "lexical_novelty.png"))
        plot_compound_imagery_distribution(df, str(output_dir / "compound_imagery_density.png"))
        plot_narrative_distribution(df, str(output_dir / "narrative_density.png"))
        plot_conceptual_distance_distribution(df, str(output_dir / "conceptual_distance_score.png"))
        plot_surrealism_index(df, str(output_dir / "surrealism_index.png"))
        plot_histogram_compare(df, "metaphor_density", str(output_dir / "metaphor_density_hist.png"))
        plot_histogram_compare(df, "surreal_imagery_score", str(output_dir / "surreal_imagery_hist.png"))

        if "semantic_drift_from_base" in df.columns and df["semantic_drift_from_base"].notnull().any():
            plot_semantic_drift_scatter(df, str(output_dir / "semantic_drift_scatter.png"))

    if not args.skip_umap:
        groups = []
        labels = []
        colors = []

        def add_group(label: str, texts: List[str], color: str) -> None:
            if not texts:
                return
            sampled = _sample_if_needed(texts, umap_max_points) if umap_max_points else texts
            groups.append(sampled)
            labels.append(label)
            colors.append(color)

        add_group("base", base_texts, "#1f77b4")
        add_group("lora", lora_texts, "#ff7f0e")
        add_group("extra_base", extra_base_texts, "#2ca02c")
        add_group("extra_lora", extra_lora_texts, "#d62728")
        add_group("human", human_lines, "#9467bd")

        if groups:
            coords = build_latent_space_groups(groups, embedder)
            group_sizes = [len(g) for g in groups]
            plot_latent_space_groups(
                coords,
                group_sizes,
                labels,
                colors,
                str(output_dir / "latent_space_umap.png"),
            )

    summary_rows = []
    if not df.empty:
        metric_cols = [
            "imagery_density",
            "compound_imagery_density",
            "narrative_density",
            "lexical_novelty",
            "metaphor_density",
            "surreal_imagery_score",
            "conceptual_distance_score",
            "surrealism_index",
            "semantic_drift_from_base",
        ]
        for model in sorted(df["model"].unique()):
            group = df[df["model"] == model]
            row = {"model": model}
            for col in metric_cols:
                if col in group.columns:
                    values = group[col].dropna()
                    if not values.empty:
                        row[col] = float(values.mean())
            summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / "metrics_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print("Saved:", summary_path)

    report_lines = []
    shift_tokenizer = lora_tokenizer_for_shift or base_tokenizer_for_shift
    if base_texts and lora_texts and shift_tokenizer:
        shift = token_distribution_shift(base_texts, lora_texts, shift_tokenizer)
        report_lines.append(f"token_distribution_shift base->lora: {shift:.6f}")
    if extra_base_texts and extra_lora_texts and extra_tokenizer:
        shift = token_distribution_shift(extra_base_texts, extra_lora_texts, extra_tokenizer)
        report_lines.append(f"token_distribution_shift extra_base->extra_lora: {shift:.6f}")

    if report_lines:
        report_path = output_dir / "metrics_report.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print("Saved:", report_path)


if __name__ == "__main__":
    main()
