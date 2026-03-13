import argparse
import os
import random
from datetime import datetime
from typing import Dict, List

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


def is_qwen_model(name_or_path: str) -> bool:
    if not name_or_path:
        return False
    return "qwen" in name_or_path.lower()


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


def fix_tokenizer_padding(tokenizer):
    if tokenizer and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


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

    base_model = args.base_model or cfg.get("base_model") or DEFAULT_BASE_MODEL
    lora_path = args.lora_path or cfg.get("lora_path") or DEFAULT_LORA_PATH

    extra_base_model = args.extra_base_model or cfg.get("extra_base_model")
    extra_lora_path = args.extra_lora_path or cfg.get("extra_lora_path")

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

    embed_batch_size = args.embed_batch_size if args.embed_batch_size is not None else int(cfg.get("embed_batch_size", 64))
    max_nouns = args.max_nouns if args.max_nouns is not None else int(cfg.get("max_nouns", 128))
    umap_max_points = args.umap_max_points if args.umap_max_points is not None else int(cfg.get("umap_max_points", 0))

    human_lines = load_human_poems(args.human_corpus or DEFAULT_HUMAN_CORPUS)
    if args.human_max_lines is not None and args.human_max_lines > 0:
        human_lines = _sample_if_needed(human_lines, args.human_max_lines)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    base_model_obj = None
    tokenizer = None
    lora_model = None

    if not args.skip_base:
        base_model_obj, tokenizer = load_base_model(
            base_model,
            device,
            load_in_8bit=args.base_8bit,
            load_in_4bit=args.base_4bit,
            cpu_offload=args.base_8bit_cpu_offload,
            trust_remote_code=is_qwen_model(base_model),
        )
        fix_tokenizer_padding(tokenizer)

    if not args.skip_lora:
        lora_model, tokenizer = load_lora_model(
            base_model,
            lora_path,
            device,
            load_in_8bit=args.lora_8bit,
            load_in_4bit=args.lora_4bit,
            cpu_offload=args.lora_8bit_cpu_offload,
            trust_remote_code=is_qwen_model(base_model),
        )
        fix_tokenizer_padding(tokenizer)

    extra_base_model_obj = None
    extra_tokenizer = None
    extra_lora_model = None

    if extra_base_model:
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

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    prompts = read_lines(prompts_path)
    corpus_lines = read_lines(corpus_path)
    corpus_tokens = build_corpus_token_set(corpus_lines)

    print("Starting generation...")

    base_samples = []
    lora_samples = []
    extra_base_samples = []
    extra_lora_samples = []
    human_samples = []

    if base_model_obj:
        base_samples = generate_samples(
            base_model_obj,
            tokenizer,
            prompts,
            num_samples=num_samples,
            cfg=gen_cfg,
            device=device,
            batch_size=args.batch_size,
            label="base",
        )

    if lora_model:
        lora_samples = generate_samples(
            lora_model,
            tokenizer,
            prompts,
            num_samples=num_samples,
            cfg=gen_cfg,
            device=device,
            batch_size=args.batch_size,
            label="lora",
        )

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

    output_dir = args.output_dir or os.path.join(
        "eval_outputs", datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    ensure_dir(output_dir)

    df_path = os.path.join(output_dir, "eval_samples.csv")
    df.to_csv(df_path, index=False)

    print("Saved:", df_path)

    if not df.empty:
        plot_imagery_distribution(df, os.path.join(output_dir, "imagery_density.png"))
        plot_novelty_distribution(df, os.path.join(output_dir, "lexical_novelty.png"))
        plot_compound_imagery_distribution(df, os.path.join(output_dir, "compound_imagery_density.png"))
        plot_narrative_distribution(df, os.path.join(output_dir, "narrative_density.png"))
        plot_conceptual_distance_distribution(df, os.path.join(output_dir, "conceptual_distance_score.png"))
        plot_surrealism_index(df, os.path.join(output_dir, "surrealism_index.png"))
        plot_histogram_compare(df, "metaphor_density", os.path.join(output_dir, "metaphor_density_hist.png"))
        plot_histogram_compare(df, "surreal_imagery_score", os.path.join(output_dir, "surreal_imagery_hist.png"))

        if "semantic_drift_from_base" in df.columns and df["semantic_drift_from_base"].notnull().any():
            plot_semantic_drift_scatter(df, os.path.join(output_dir, "semantic_drift_scatter.png"))

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
                os.path.join(output_dir, "latent_space_umap.png"),
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
        summary_path = os.path.join(output_dir, "metrics_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print("Saved:", summary_path)

    report_lines = []
    if base_texts and lora_texts and tokenizer:
        shift = token_distribution_shift(base_texts, lora_texts, tokenizer)
        report_lines.append(f"token_distribution_shift base->lora: {shift:.6f}")
    if extra_base_texts and extra_lora_texts and extra_tokenizer:
        shift = token_distribution_shift(extra_base_texts, extra_lora_texts, extra_tokenizer)
        report_lines.append(f"token_distribution_shift extra_base->extra_lora: {shift:.6f}")

    if report_lines:
        report_path = os.path.join(output_dir, "metrics_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print("Saved:", report_path)


if __name__ == "__main__":
    main()

