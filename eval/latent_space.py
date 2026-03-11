from typing import Iterable, List, Optional, Sequence

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


def build_latent_space_groups(
    groups: List[Sequence[str]],
    embedder: SentenceTransformer,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    embeddings = []
    for texts in groups:
        if texts:
            embeddings.append(_embed_texts(texts, embedder))
    if not embeddings:
        return np.empty((0, 2))

    all_emb = np.vstack(embeddings)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=random_state,
    )
    coords = reducer.fit_transform(all_emb)
    return coords


def plot_latent_space_groups(
    coords: np.ndarray,
    group_sizes: List[int],
    labels: List[str],
    colors: List[str],
    output_path: str,
    imagery_values: Optional[np.ndarray] = None,
    cmaps: Optional[List[str]] = None,
) -> None:
    plt.figure(figsize=(7.5, 6))

    idx = 0
    for i, size in enumerate(group_sizes):
        if size <= 0:
            continue
        group_slice = coords[idx : idx + size]
        idx += size
        if imagery_values is None:
            plt.scatter(
                group_slice[:, 0],
                group_slice[:, 1],
                s=14,
                c=colors[i],
                alpha=0.75,
                label=labels[i],
            )
        else:
            group_vals = imagery_values[idx - size : idx]
            cmap = cmaps[i] if cmaps and i < len(cmaps) else "viridis"
            plt.scatter(
                group_slice[:, 0],
                group_slice[:, 1],
                s=14,
                c=group_vals,
                cmap=cmap,
                alpha=0.75,
                label=labels[i],
            )

    if imagery_values is not None:
        plt.colorbar(label="Imagery Density")

    plt.title("Latent Poetry Space (UMAP)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()
