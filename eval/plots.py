from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_imagery_distribution(df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(8, 4.5))
    sns.kdeplot(data=df, x="imagery_density", hue="model", fill=True, common_norm=False, alpha=0.4)
    plt.title("Imagery Density Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_novelty_distribution(df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(8, 4.5))
    sns.kdeplot(data=df, x="lexical_novelty", hue="model", fill=True, common_norm=False, alpha=0.4)
    plt.title("Lexical Novelty Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_semantic_drift_scatter(df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(6.5, 5))
    sns.scatterplot(
        data=df[df["model"] == "lora"],
        x="semantic_drift_from_base",
        y="imagery_density",
        alpha=0.7,
    )
    plt.title("Semantic Drift vs Imagery Density (LoRA)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_histogram_compare(df: pd.DataFrame, metric: str, output_path: str) -> None:
    plt.figure(figsize=(7, 4.5))
    sns.histplot(data=df, x=metric, hue="model", bins=30, stat="density", element="step")
    plt.title(f"Histogram Comparison: {metric}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_surrealism_index(df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df, x="model", y="surrealism_index", order=["base", "lora", "human"])
    plt.title("Surrealism Index by Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_compound_imagery_distribution(df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(8, 4.5))
    sns.kdeplot(data=df, x="compound_imagery_density", hue="model", fill=True, common_norm=False, alpha=0.4)
    plt.title("Compound Imagery Density Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_narrative_distribution(df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(8, 4.5))
    sns.kdeplot(data=df, x="narrative_density", hue="model", fill=True, common_norm=False, alpha=0.4)
    plt.title("Narrative Density Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_conceptual_distance_distribution(df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(8, 4.5))
    sns.kdeplot(data=df, x="conceptual_distance_score", hue="model", fill=True, common_norm=False, alpha=0.4)
    plt.title("Conceptual Distance Score Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
