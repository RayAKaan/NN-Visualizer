from __future__ import annotations

from typing import Dict, List
import numpy as np

from ..dataset_manager import dataset_manager
from ..graph_engine import NetworkGraph
from ..forward_engine import run_forward_full
from .pca_projector import pca_project
from .tsne_projector import tsne_project
from .umap_projector import umap_project


def compute_embeddings(graph: NetworkGraph, dataset_id: str, layer_index: int, n_samples: int = 200, method: str = "pca") -> Dict:
    method = (method or "pca").lower()
    dataset = dataset_manager.get(dataset_id)
    train = dataset.get("train", [])
    if isinstance(train, list):
        samples = train[:n_samples]
        xs = np.asarray([s["x"] for s in samples], dtype=np.float32)
        ys = np.asarray([s["y"] for s in samples], dtype=np.float32)
    else:
        xs = np.asarray(train.get("x", []), dtype=np.float32)[:n_samples]
        ys = np.asarray(train.get("y", []), dtype=np.float32)[:n_samples]

    activations = []
    labels = []
    preds = []
    for i in range(len(xs)):
        steps, output, layer_outputs = run_forward_full(graph, xs[i].tolist())
        layer_key = str(layer_index)
        act = layer_outputs.get(layer_key)
        if act is None:
            act = output
        activations.append(np.asarray(act, dtype=np.float32).reshape(-1))
        labels.append(int(np.argmax(ys[i])) if ys[i].ndim else int(ys[i]))
        preds.append(int(np.argmax(output)))


    if method not in {"pca", "tsne", "umap"}:
        method = "pca"
    X = np.asarray(activations, dtype=np.float32)
    if method == "tsne":
        proj = tsne_project(X, n_components=2, perplexity=min(30.0, max(5.0, X.shape[0] / 5)))
        variance = np.array([0.0, 0.0], dtype=np.float32)
    elif method == "umap":
        proj = umap_project(X, n_components=2, n_neighbors=min(15, max(5, X.shape[0] // 3)))
        variance = np.array([0.0, 0.0], dtype=np.float32)
    else:
        proj, variance = pca_project(X, n_components=2)
    projections = [
        {"index": i, "coords": proj[i].tolist(), "label": labels[i], "predicted": preds[i]} for i in range(len(proj))
    ]

    return {
        "projections": projections,
        "variance_explained": variance.tolist(),
        "cluster_metrics": {
            "silhouette_score": 0.0,
            "inter_class_distance": 0.0,
            "intra_class_distance": 0.0,
        },
        "method_params_used": {"method": method},
    }
