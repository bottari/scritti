from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import umap
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("umap-learn is required for latent space plots. Install umap-learn.") from exc


def _embed_texts(texts: Iterable[str], embedder: SentenceTransformer) -> np.ndarray:
    embeddings = embedder.encode(list(texts), normalize_embeddings=True)
    return np.array(embeddings)


def build_latent_space(
    base_texts: Iterable[str],
    lora_texts: Iterable[str],
    embedder: SentenceTransformer,
    human_texts: Optional[Iterable[str]] = None,
    extra_texts: Optional[Iterable[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    base_emb = _embed_texts(base_texts, embedder)
    lora_emb = _embed_texts(lora_texts, embedder)
    extra_emb = _embed_texts(extra_texts, embedder) if extra_texts is not None else None
    human_emb = _embed_texts(human_texts, embedder) if human_texts is not None else None

    stacks = [base_emb, lora_emb]
    if extra_emb is not None:
        stacks.append(extra_emb)
    if human_emb is not None:
        stacks.append(human_emb)

    all_emb = np.vstack(stacks)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=random_state,
    )
    coords = reducer.fit_transform(all_emb)
    return coords


def plot_latent_space(
    coords: np.ndarray,
    base_count: int,
    lora_count: int,
    output_path: str,
    imagery_values: Optional[np.ndarray] = None,
    extra_count: int = 0,
    extra_label: str = "Extra",
) -> None:
    plt.figure(figsize=(7.5, 6))

    idx = 0
    base_slice = coords[idx : idx + base_count]
    idx += base_count
    lora_slice = coords[idx : idx + lora_count]
    idx += lora_count
    extra_slice = coords[idx : idx + extra_count] if extra_count > 0 else np.empty((0, 2))
    idx += extra_count
    human_slice = coords[idx:]

    if imagery_values is None:
        plt.scatter(base_slice[:, 0], base_slice[:, 1], s=14, c="#2a6fdb", alpha=0.75, label="Base")
        plt.scatter(lora_slice[:, 0], lora_slice[:, 1], s=14, c="#d64541", alpha=0.75, label="LoRA")
        if extra_count > 0:
            plt.scatter(extra_slice[:, 0], extra_slice[:, 1], s=14, c="#f28e2b", alpha=0.75, label=extra_label)
        if len(human_slice) > 0:
            plt.scatter(human_slice[:, 0], human_slice[:, 1], s=14, c="#2ca25f", alpha=0.75, label="Human")
    else:
        vmin = float(np.min(imagery_values))
        vmax = float(np.max(imagery_values))
        plt.scatter(
            base_slice[:, 0],
            base_slice[:, 1],
            s=14,
            c=imagery_values[:base_count],
            cmap="Blues",
            alpha=0.75,
            vmin=vmin,
            vmax=vmax,
            label="Base",
        )
        plt.scatter(
            lora_slice[:, 0],
            lora_slice[:, 1],
            s=14,
            c=imagery_values[base_count : base_count + lora_count],
            cmap="Reds",
            alpha=0.75,
            vmin=vmin,
            vmax=vmax,
            label="LoRA",
        )
        if extra_count > 0:
            start = base_count + lora_count
            end = start + extra_count
            plt.scatter(
                extra_slice[:, 0],
                extra_slice[:, 1],
                s=14,
                c=imagery_values[start:end],
                cmap="Oranges",
                alpha=0.75,
                vmin=vmin,
                vmax=vmax,
                label=extra_label,
            )
        if len(human_slice) > 0:
            start = base_count + lora_count + extra_count
            plt.scatter(
                human_slice[:, 0],
                human_slice[:, 1],
                s=14,
                c=imagery_values[start:],
                cmap="Greens",
                alpha=0.75,
                vmin=vmin,
                vmax=vmax,
                label="Human",
            )
        plt.colorbar(label="Imagery Density")

    plt.title("Latent Poetry Space (UMAP)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()
