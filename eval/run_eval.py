import argparse
import os
import random
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

from generation import GenerationConfig, generate_samples, load_base_model, load_lora_model
from latent_space import build_latent_space, plot_latent_space
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

DEFAULT_BASE_MODEL = "gpt2"
DEFAULT_LORA_PATH = r"D:\models\choice-models\gpt2-finetuned-poetry-mercury-04"
DEFAULT_HUMAN_CORPUS = r"C:\Users\micha\Desktop\projects\scritti\data\human_poetry.txt"


def read_lines(path: str) -> List[str]:
    if os.path.isdir(path):
        lines: List[str] = []
        for name in sorted(os.listdir(path)):
            if not name.lower().endswith(".txt"):
                continue
            file_path = os.path.join(path, name)
            with open(file_path, "r", encoding="utf-8") as f:
                lines.extend([line.strip() for line in f if line.strip()])
        return lines
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_delta(base: float, other: float) -> str:
    if base == 0:
        return "n/a"
    return f"{((other - base) / base) * 100:.0f}%"


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _sample_if_needed(items: List[str], max_items: int, seed: int = 42) -> List[str]:
    if max_items <= 0 or len(items) <= max_items:
        return items
    rng = random.Random(seed)
    return rng.sample(items, max_items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base vs LoRA surreal poetry outputs.")
    parser.add_argument("--config", default=None, help="Path to eval_config.yaml")
    parser.add_argument("--base-model", default=None, help="Base model name or path")
    parser.add_argument("--lora-path", default=None, help="Path to LoRA adapter")

    parser.add_argument("--extra-model", default=None, help="Optional extra full model path")
    parser.add_argument("--extra-base-model", default=None, help="Base model for extra LoRA adapter")
    parser.add_argument("--extra-lora-path", default=None, help="LoRA adapter path for extra model")
    parser.add_argument("--extra-8bit", action="store_true", help="Load extra model in 8-bit")
    parser.add_argument("--extra-label", default="extra", help="Label for extra comparator")

    parser.add_argument("--prompts", default=None, help="Path to prompts txt file")
    parser.add_argument("--corpus", default=None, help="Path to training corpus text file")
    parser.add_argument("--num-samples", type=int, default=None, help="Samples per model")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--num-return-sequences", type=int, default=None)

    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embed-batch-size", type=int, default=None)
    parser.add_argument("--max-nouns", type=int, default=None)

    parser.add_argument("--human-corpus", default=None, help="Optional human poetry corpus txt file")
    parser.add_argument("--human-max-lines", type=int, default=None)

    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP latent plot")
    parser.add_argument("--umap-max-points", type=int, default=None)

    parser.add_argument("--gen-log-every", type=int, default=None)

    parser.add_argument("--color-by-imagery", action="store_true", help="Color latent points by imagery density")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}

    base_model = args.base_model or cfg.get("base_model") or DEFAULT_BASE_MODEL
    lora_path = args.lora_path or cfg.get("lora_path") or DEFAULT_LORA_PATH

    extra_model = args.extra_model or cfg.get("extra_model")
    extra_base_model = args.extra_base_model or cfg.get("extra_base_model")
    extra_lora_path = args.extra_lora_path or cfg.get("extra_lora_path")
    extra_label = args.extra_label or cfg.get("extra_label", "extra")
    extra_8bit = args.extra_8bit or bool(cfg.get("extra_8bit", False))

    prompts_path = args.prompts or cfg.get("prompts")
    corpus_path = args.corpus or cfg.get("corpus")
    num_samples = args.num_samples or int(cfg.get("num_samples", 200))
    max_new_tokens = args.max_new_tokens or int(cfg.get("max_new_tokens", 128))
    temperature = args.temperature if args.temperature is not None else float(cfg.get("temperature", 0.9))
    top_p = args.top_p if args.top_p is not None else float(cfg.get("top_p", 0.95))
    top_k = args.top_k if args.top_k is not None else int(cfg.get("top_k", 50))
    repetition_penalty = args.repetition_penalty if args.repetition_penalty is not None else float(
        cfg.get("repetition_penalty", 1.05)
    )
    num_return_sequences = args.num_return_sequences if args.num_return_sequences is not None else int(
        cfg.get("num_return_sequences", 1)
    )

    embed_batch_size = args.embed_batch_size if args.embed_batch_size is not None else int(cfg.get("embed_batch_size", 64))
    max_nouns = args.max_nouns if args.max_nouns is not None else int(cfg.get("max_nouns", 128))

    human_max_lines = args.human_max_lines if args.human_max_lines is not None else int(cfg.get("human_max_lines", 2000))
    skip_umap = args.skip_umap or bool(cfg.get("skip_umap", False))
    umap_max_points = args.umap_max_points if args.umap_max_points is not None else int(cfg.get("umap_max_points", 3000))

    gen_log_every = args.gen_log_every if args.gen_log_every is not None else int(cfg.get("gen_log_every", 5))

    missing = [
        name
        for name, val in [
            ("base_model", base_model),
            ("lora_path", lora_path),
            ("prompts", prompts_path),
            ("corpus", corpus_path),
        ]
        if not val
    ]
    if missing:
        raise ValueError(f"Missing required config values: {', '.join(missing)}")

    if extra_lora_path and not extra_base_model:
        raise ValueError("extra_base_model is required when extra_lora_path is provided")

    output_dir = args.output_dir or os.path.join(
        "eval_outputs", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    ensure_dir(output_dir)

    prompts = read_lines(prompts_path)
    corpus_lines = read_lines(corpus_path)
    corpus_tokens = build_corpus_token_set(corpus_lines)
    human_lines = load_human_poems(args.human_corpus or DEFAULT_HUMAN_CORPUS)
    human_lines = _sample_if_needed(human_lines, human_max_lines)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    base_model_obj, tokenizer = load_base_model(base_model, device)
    lora_model, _ = load_lora_model(base_model, lora_path, device)

    extra_model_obj = None
    if extra_lora_path:
        extra_model_obj, _ = load_lora_model(extra_base_model, extra_lora_path, device, load_in_8bit=extra_8bit)
    elif extra_model:
        extra_model_obj, _ = load_base_model(extra_model, device, load_in_8bit=extra_8bit)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    print("Starting generation...")

    base_samples = generate_samples(
        base_model_obj,
        tokenizer,
        prompts,
        num_samples=num_samples,
        cfg=gen_cfg,
        device=device,
        batch_size=args.batch_size,
        num_return_sequences=num_return_sequences,
        log_every=gen_log_every,
        label="base",
    )
    lora_samples = generate_samples(
        lora_model,
        tokenizer,
        prompts,
        num_samples=num_samples,
        cfg=gen_cfg,
        device=device,
        batch_size=args.batch_size,
        num_return_sequences=num_return_sequences,
        log_every=gen_log_every,
        label="lora",
    )

    extra_samples = []
    if extra_model_obj is not None:
        extra_samples = generate_samples(
            extra_model_obj,
            tokenizer,
            prompts,
            num_samples=num_samples,
            cfg=gen_cfg,
            device=device,
            batch_size=args.batch_size,
            num_return_sequences=num_return_sequences,
            log_every=gen_log_every,
            label=extra_label,
        )

    base_texts = [s["text"] for s in base_samples]
    lora_texts = [s["text"] for s in lora_samples]
    extra_texts = [s["text"] for s in extra_samples] if extra_samples else []

    embedder = SentenceTransformer(args.embedding_model)
    spacy_resources = load_spacy()

    drift_scores = semantic_drift(base_texts, lora_texts, embedder, batch_size=embed_batch_size)
    extra_drift_scores = semantic_drift(base_texts, extra_texts, embedder, batch_size=embed_batch_size) if extra_samples else []

    noun_embed_cache: Dict[str, np.ndarray] = {}

    def enrich(samples, model_name, drift_values=None):
        rows = []
        for i, sample in enumerate(samples):
            text = sample["text"]
            imagery = imagery_density(text)
            metaphor = metaphor_density(text, spacy_resources.nlp)
            surreal = surreal_imagery_score(
                text, spacy_resources.nlp, embedder, cache=noun_embed_cache, max_nouns=max_nouns, batch_size=embed_batch_size
            )
            conceptual = conceptual_distance_score(
                text, spacy_resources.nlp, embedder, cache=noun_embed_cache, max_nouns=max_nouns, batch_size=embed_batch_size
            )
            compound_imagery = compound_imagery_density(text, spacy_resources.nlp)
            narrative = narrative_density(text, spacy_resources.nlp)
            row = {
                "model": model_name,
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
            if drift_values is not None:
                row["semantic_drift_from_base"] = drift_values[i]
            else:
                row["semantic_drift_from_base"] = np.nan
            rows.append(row)
        return rows

    base_rows = enrich(base_samples, "base")
    lora_rows = enrich(lora_samples, "lora", drift_scores)
    extra_rows = enrich(extra_samples, extra_label, extra_drift_scores) if extra_samples else []

    human_rows = []
    if human_lines:
        for text in human_lines:
            imagery = imagery_density(text)
            metaphor = metaphor_density(text, spacy_resources.nlp)
            surreal = surreal_imagery_score(
                text, spacy_resources.nlp, embedder, cache=noun_embed_cache, max_nouns=max_nouns, batch_size=embed_batch_size
            )
            conceptual = conceptual_distance_score(
                text, spacy_resources.nlp, embedder, cache=noun_embed_cache, max_nouns=max_nouns, batch_size=embed_batch_size
            )
            compound_imagery = compound_imagery_density(text, spacy_resources.nlp)
            narrative = narrative_density(text, spacy_resources.nlp)
            human_rows.append(
                {
                    "model": "human",
                    "prompt": "reference",
                    "text": text,
                    "imagery_density": imagery,
                    "compound_imagery_density": compound_imagery,
                    "narrative_density": narrative,
                    "lexical_novelty": lexical_novelty(text, corpus_tokens),
                    "metaphor_density": metaphor,
                    "surreal_imagery_score": surreal,
                    "conceptual_distance_score": conceptual,
                    "surrealism_index": imagery + metaphor + surreal + conceptual,
                    "semantic_drift_from_base": np.nan,
                }
            )

    df = pd.DataFrame(base_rows + lora_rows + extra_rows + human_rows)

    token_shift = token_distribution_shift(base_texts, lora_texts, tokenizer)

    summary_metrics = [
        "imagery_density",
        "compound_imagery_density",
        "narrative_density",
        "lexical_novelty",
        "semantic_drift_from_base",
        "metaphor_density",
        "surreal_imagery_score",
        "conceptual_distance_score",
        "surrealism_index",
    ]

    summary_rows = []
    for metric in summary_metrics:
        base_avg = float(df[df["model"] == "base"][metric].mean())
        lora_avg = float(df[df["model"] == "lora"][metric].mean())
        extra_avg = float(df[df["model"] == extra_label][metric].mean()) if extra_rows else np.nan
        human_avg = float(df[df["model"] == "human"][metric].mean()) if human_rows else np.nan
        summary_row = {
            "metric": metric,
            "base": base_avg,
            "lora": lora_avg,
            "human": human_avg,
            "delta_lora_vs_base": format_delta(base_avg, lora_avg),
            "delta_human_vs_base": format_delta(base_avg, human_avg) if human_rows else "n/a",
        }
        if extra_rows:
            summary_row[extra_label] = extra_avg
            summary_row[f"delta_{extra_label}_vs_base"] = format_delta(base_avg, extra_avg)
        summary_rows.append(summary_row)

    summary_rows.append(
        {
            "metric": "token_distribution_shift",
            "base": 0.0,
            "lora": float(token_shift),
            "human": np.nan,
            "delta_lora_vs_base": "n/a",
            "delta_human_vs_base": "n/a",
        }
    )

    summary_df = pd.DataFrame(summary_rows)

    df_path = os.path.join(output_dir, "eval_samples.csv")
    summary_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(df_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_imagery_distribution(df, os.path.join(output_dir, "imagery_density_dist.png"))
    plot_compound_imagery_distribution(df, os.path.join(output_dir, "compound_imagery_dist.png"))
    plot_narrative_distribution(df, os.path.join(output_dir, "narrative_density_dist.png"))
    plot_conceptual_distance_distribution(df, os.path.join(output_dir, "conceptual_distance_dist.png"))
    plot_novelty_distribution(df, os.path.join(output_dir, "lexical_novelty_dist.png"))
    plot_semantic_drift_scatter(df, os.path.join(output_dir, "semantic_drift_scatter.png"))
    plot_histogram_compare(df, "imagery_density", os.path.join(output_dir, "imagery_density_hist.png"))
    plot_surrealism_index(df, os.path.join(output_dir, "surrealism_index_box.png"))

    if not skip_umap:
        # Downsample for UMAP to avoid OOM on large corpora
        base_umap = _sample_if_needed(base_texts, umap_max_points)
        lora_umap = _sample_if_needed(lora_texts, umap_max_points)
        extra_umap = _sample_if_needed(extra_texts, umap_max_points) if extra_texts else None
        human_umap = _sample_if_needed(human_lines, umap_max_points) if human_lines else None

        coords = build_latent_space(
            base_umap,
            lora_umap,
            embedder,
            human_umap,
            extra_umap,
        )
        imagery_values = None
        if args.color_by_imagery:
            imagery_values = np.array(
                [imagery_density(t) for t in base_umap]
                + [imagery_density(t) for t in lora_umap]
                + ([imagery_density(t) for t in extra_umap] if extra_umap else [])
                + ([imagery_density(t) for t in human_umap] if human_umap else []),
                dtype=np.float64,
            )
        plot_latent_space(
            coords,
            base_count=len(base_umap),
            lora_count=len(lora_umap),
            extra_count=len(extra_umap) if extra_umap else 0,
            extra_label=extra_label,
            output_path=os.path.join(output_dir, "latent_poetry_space.png"),
            imagery_values=imagery_values,
        )

    print("Saved:")
    print(df_path)
    print(summary_path)
    print("Token distribution shift (KL):", token_shift)


if __name__ == "__main__":
    main()
