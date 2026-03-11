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

    parser.add_argument("--skip-base", action="store_true", help="Skip base model generation")
    parser.add_argument("--skip-lora", action="store_true", help="Skip LoRA model generation")

    parser.add_argument("--base-8bit", action="store_true", help="Load base model in 8-bit")
    parser.add_argument("--lora-8bit", action="store_true", help="Load LoRA base in 8-bit")
    parser.add_argument("--base-4bit", action="store_true", help="Load base model in 4-bit")
    parser.add_argument("--lora-4bit", action="store_true", help="Load LoRA base in 4-bit")
    parser.add_argument("--base-8bit-cpu-offload", action="store_true")
    parser.add_argument("--lora-8bit-cpu-offload", action="store_true")

    parser.add_argument("--extra-base-model", default=None, help="Base model for extra LoRA adapter")
    parser.add_argument("--extra-lora-path", default=None, help="LoRA adapter path for extra model")
    parser.add_argument("--extra-base-label", default=None)
    parser.add_argument("--extra-lora-label", default=None)
    parser.add_argument("--extra-8bit", action="store_true", help="Load extra model in 8-bit")
    parser.add_argument("--extra-4bit", action="store_true", help="Load extra model in 4-bit")
    parser.add_argument("--extra-8bit-cpu-offload", action="store_true")

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

    parser.add_argument("--base-batch-size", type=int, default=None)
    parser.add_argument("--lora-batch-size", type=int, default=None)
    parser.add_argument("--extra-batch-size", type=int, default=None)

    parser.add_argument("--base-max-new-tokens", type=int, default=None)
    parser.add_argument("--lora-max-new-tokens", type=int, default=None)
    parser.add_argument("--extra-max-new-tokens", type=int, default=None)

    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embed-batch-size", type=int, default=None)
    parser.add_argument("--max-nouns", type=int, default=None)

    parser.add_argument("--human-corpus", default=None, help="Optional human poetry corpus txt file")
    parser.add_argument("--human-max-lines", type=int, default=None)

    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP latent plot")
    parser.add_argument("--umap-max-points", type=int, default=None)

    parser.add_argument("--gen-log-every", type=int, default=None)
    parser.add_argument("--heartbeat-seconds", type=int, default=None)

    parser.add_argument("--color-by-imagery", action="store_true", help="Color latent points by imagery density")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}

    base_model = args.base_model or cfg.get("base_model") or DEFAULT_BASE_MODEL
    lora_path = args.lora_path or cfg.get("lora_path") or DEFAULT_LORA_PATH

    extra_base_model = args.extra_base_model or cfg.get("extra_base_model")
    extra_lora_path = args.extra_lora_path or cfg.get("extra_lora_path")
    extra_base_label = args.extra_base_label or cfg.get("extra_base_label", "gpt2_base")
    extra_lora_label = args.extra_lora_label or cfg.get("extra_lora_label", "gpt2_lora")
    extra_8bit = args.extra_8bit or bool(cfg.get("extra_8bit", False))
    extra_4bit = args.extra_4bit or bool(cfg.get("extra_4bit", False))
    extra_8bit_cpu_offload = args.extra_8bit_cpu_offload or bool(cfg.get("extra_8bit_cpu_offload", False))

    skip_base = args.skip_base or bool(cfg.get("skip_base", False))
    skip_lora = args.skip_lora or bool(cfg.get("skip_lora", False))
    base_8bit = args.base_8bit or bool(cfg.get("base_8bit", False))
    lora_8bit = args.lora_8bit or bool(cfg.get("lora_8bit", False))
    base_4bit = args.base_4bit or bool(cfg.get("base_4bit", False))
    lora_4bit = args.lora_4bit or bool(cfg.get("lora_4bit", False))
    base_8bit_cpu_offload = args.base_8bit_cpu_offload or bool(cfg.get("base_8bit_cpu_offload", False))
    lora_8bit_cpu_offload = args.lora_8bit_cpu_offload or bool(cfg.get("lora_8bit_cpu_offload", False))

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

    base_batch_size = args.base_batch_size if args.base_batch_size is not None else int(cfg.get("base_batch_size", args.batch_size))
    lora_batch_size = args.lora_batch_size if args.lora_batch_size is not None else int(cfg.get("lora_batch_size", args.batch_size))
    extra_batch_size = args.extra_batch_size if args.extra_batch_size is not None else int(cfg.get("extra_batch_size", 8))

    base_max_new_tokens = args.base_max_new_tokens if args.base_max_new_tokens is not None else int(
        cfg.get("base_max_new_tokens", max_new_tokens)
    )
    lora_max_new_tokens = args.lora_max_new_tokens if args.lora_max_new_tokens is not None else int(
        cfg.get("lora_max_new_tokens", max_new_tokens)
    )
    extra_max_new_tokens = args.extra_max_new_tokens if args.extra_max_new_tokens is not None else int(
        cfg.get("extra_max_new_tokens", max_new_tokens)
    )

    embed_batch_size = args.embed_batch_size if args.embed_batch_size is not None else int(cfg.get("embed_batch_size", 64))
    max_nouns = args.max_nouns if args.max_nouns is not None else int(cfg.get("max_nouns", 128))

    human_max_lines = args.human_max_lines if args.human_max_lines is not None else int(cfg.get("human_max_lines", 2000))
    skip_umap = args.skip_umap or bool(cfg.get("skip_umap", False))
    umap_max_points = args.umap_max_points if args.umap_max_points is not None else int(cfg.get("umap_max_points", 3000))

    gen_log_every = args.gen_log_every if args.gen_log_every is not None else int(cfg.get("gen_log_every", 5))
    heartbeat_seconds = (
        args.heartbeat_seconds
        if args.heartbeat_seconds is not None
        else int(cfg.get("heartbeat_seconds", 60))
    )

    missing = []
    if not skip_base and not base_model:
        missing.append("base_model")
    if not skip_lora and not lora_path:
        missing.append("lora_path")
    if not prompts_path:
        missing.append("prompts")
    if not corpus_path:
        missing.append("corpus")
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

    base_model_obj = None
    lora_model = None
    tokenizer = None

    if not skip_base:
        base_model_obj, tokenizer = load_base_model(
            base_model,
            device,
            load_in_8bit=base_8bit,
            load_in_4bit=base_4bit,
            cpu_offload=base_8bit_cpu_offload,
        )
    if not skip_lora:
        lora_model, tokenizer = load_lora_model(
            base_model,
            lora_path,
            device,
            load_in_8bit=lora_8bit,
            load_in_4bit=lora_4bit,
            cpu_offload=lora_8bit_cpu_offload,
        )

    extra_base_model_obj = None
    extra_lora_model = None
    extra_tokenizer = None

    if extra_base_model:
        extra_base_model_obj, extra_tokenizer = load_base_model(
            extra_base_model,
            device,
            load_in_8bit=extra_8bit,
            load_in_4bit=extra_4bit,
            cpu_offload=extra_8bit_cpu_offload,
        )
    if extra_lora_path:
        extra_lora_model, extra_tokenizer = load_lora_model(
            extra_base_model,
            extra_lora_path,
            device,
            load_in_8bit=extra_8bit,
            load_in_4bit=extra_4bit,
            cpu_offload=extra_8bit_cpu_offload,
        )

    gen_cfg_base = GenerationConfig(
        max_new_tokens=base_max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    gen_cfg_lora = GenerationConfig(
        max_new_tokens=lora_max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    gen_cfg_extra = GenerationConfig(
        max_new_tokens=extra_max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    print("Starting generation...")

    base_samples = []
    lora_samples = []
    extra_base_samples = []
    extra_lora_samples = []

    if base_model_obj is not None:
        base_samples = generate_samples(
            base_model_obj,
            tokenizer,
            prompts,
            num_samples=num_samples,
            cfg=gen_cfg_base,
            device=device,
            batch_size=base_batch_size,
            num_return_sequences=num_return_sequences,
            log_every=gen_log_every,
            label="llama_base",
            heartbeat_seconds=heartbeat_seconds,
        )

    if lora_model is not None:
        lora_samples = generate_samples(
            lora_model,
            tokenizer,
            prompts,
            num_samples=num_samples,
            cfg=gen_cfg_lora,
            device=device,
            batch_size=lora_batch_size,
            num_return_sequences=num_return_sequences,
            log_every=gen_log_every,
            label="llama_lora",
            heartbeat_seconds=heartbeat_seconds,
        )

    if extra_base_model_obj is not None:
        extra_base_samples = generate_samples(
            extra_base_model_obj,
            extra_tokenizer,
            prompts,
            num_samples=num_samples,
            cfg=gen_cfg_extra,
            device=device,
            batch_size=extra_batch_size,
            num_return_sequences=num_return_sequences,
            log_every=gen_log_every,
            label=extra_base_label,
            heartbeat_seconds=heartbeat_seconds,
        )

    if extra_lora_model is not None:
        extra_lora_samples = generate_samples(
            extra_lora_model,
            extra_tokenizer,
            prompts,
            num_samples=num_samples,
            cfg=gen_cfg_extra,
            device=device,
            batch_size=extra_batch_size,
            num_return_sequences=num_return_sequences,
            log_every=gen_log_every,
            label=extra_lora_label,
            heartbeat_seconds=heartbeat_seconds,
        )

    base_texts = [s["text"] for s in base_samples]
    lora_texts = [s["text"] for s in lora_samples]
    extra_base_texts = [s["text"] for s in extra_base_samples]
    extra_lora_texts = [s["text"] for s in extra_lora_samples]

    embedder = SentenceTransformer(args.embedding_model)
    spacy_resources = load_spacy()

    drift_scores = semantic_drift(base_texts, lora_texts, embedder, batch_size=embed_batch_size) if base_texts and lora_texts else []
    extra_drift_scores = (
        semantic_drift(extra_base_texts, extra_lora_texts, embedder, batch_size=embed_batch_size)
        if extra_base_texts and extra_lora_texts
        else []
    )

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
            if drift_values is not None and i < len(drift_values):
                row["semantic_drift_from_base"] = drift_values[i]
            else:
                row["semantic_drift_from_base"] = np.nan
            rows.append(row)
        return rows

    base_rows = enrich(base_samples, "llama_base", drift_scores) if base_samples else []
    lora_rows = enrich(lora_samples, "llama_lora", drift_scores) if lora_samples else []
    extra_base_rows = enrich(extra_base_samples, extra_base_label, None) if extra_base_samples else []
    extra_lora_rows = enrich(extra_lora_samples, extra_lora_label, extra_drift_scores) if extra_lora_samples else []

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

    df = pd.DataFrame(base_rows + lora_rows + extra_base_rows + extra_lora_rows + human_rows)

    token_shift_primary = token_distribution_shift(base_texts, lora_texts, tokenizer) if base_texts and lora_texts else np.nan
    token_shift_extra = (
        token_distribution_shift(extra_base_texts, extra_lora_texts, extra_tokenizer)
        if extra_base_texts and extra_lora_texts
        else np.nan
    )

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

    model_labels = [label for label in ["llama_base", "llama_lora", extra_base_label, extra_lora_label, "human"] if label in df["model"].unique()]

    summary_rows = []
    for metric in summary_metrics:
        row = {"metric": metric}
        base_value = np.nan
        for label in model_labels:
            value = float(df[df["model"] == label][metric].mean()) if not df[df["model"] == label].empty else np.nan
            row[label] = value
            if label == "llama_base":
                base_value = value
        for label in model_labels:
            if label != "llama_base" and not np.isnan(base_value):
                row[f"delta_{label}_vs_llama_base"] = format_delta(base_value, row.get(label, np.nan))
        summary_rows.append(row)

    summary_rows.append(
        {
            "metric": "token_distribution_shift_llama",
            "llama_base": 0.0,
            "llama_lora": float(token_shift_primary) if not np.isnan(token_shift_primary) else np.nan,
        }
    )
    if not np.isnan(token_shift_extra):
        summary_rows.append(
            {
                "metric": f"token_distribution_shift_{extra_base_label}",
                extra_base_label: 0.0,
                extra_lora_label: float(token_shift_extra),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    df_path = os.path.join(output_dir, "eval_samples.csv")
    summary_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(df_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    if not df.empty:
        plot_imagery_distribution(df, os.path.join(output_dir, "imagery_density_dist.png"))
        plot_compound_imagery_distribution(df, os.path.join(output_dir, "compound_imagery_dist.png"))
        plot_narrative_distribution(df, os.path.join(output_dir, "narrative_density_dist.png"))
        plot_conceptual_distance_distribution(df, os.path.join(output_dir, "conceptual_distance_dist.png"))
        plot_novelty_distribution(df, os.path.join(output_dir, "lexical_novelty_dist.png"))
        plot_histogram_compare(df, "imagery_density", os.path.join(output_dir, "imagery_density_hist.png"))
        plot_surrealism_index(df, os.path.join(output_dir, "surrealism_index_box.png"), order=["llama_base", "llama_lora", "gpt2_base", "gpt2_lora", "human"])

    if not skip_umap and not df.empty:
        groups = []
        labels = []
        colors = []
        cmaps = []

        def add_group(texts, label, color, cmap):
            if texts:
                groups.append(texts)
                labels.append(label)
                colors.append(color)
                cmaps.append(cmap)

        add_group(_sample_if_needed(base_texts, umap_max_points), "llama_base", "#2a6fdb", "Blues")
        add_group(_sample_if_needed(lora_texts, umap_max_points), "llama_lora", "#d64541", "Reds")
        add_group(_sample_if_needed(extra_base_texts, umap_max_points), extra_base_label, "#f28e2b", "Oranges")
        add_group(_sample_if_needed(extra_lora_texts, umap_max_points), extra_lora_label, "#7b2cbf", "Purples")
        add_group(_sample_if_needed(human_lines, umap_max_points), "human", "#2ca25f", "Greens")

        coords = build_latent_space_groups(groups, embedder)

        imagery_values = None
        if args.color_by_imagery:
            imagery_values = np.array([imagery_density(t) for group in groups for t in group], dtype=np.float64)

        plot_latent_space_groups(
            coords,
            group_sizes=[len(g) for g in groups],
            labels=labels,
            colors=colors,
            cmaps=cmaps,
            output_path=os.path.join(output_dir, "latent_poetry_space.png"),
            imagery_values=imagery_values,
        )

    print("Saved:")
    print(df_path)
    print(summary_path)
    print("Token distribution shift (llama):", token_shift_primary)
    if not np.isnan(token_shift_extra):
        print("Token distribution shift (extra):", token_shift_extra)


if __name__ == "__main__":
    main()
