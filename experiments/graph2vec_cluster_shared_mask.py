"""
Graph2Vec-based per-layer head clustering and cluster-shared sparse mask utilities.

Used by run_graph2vec_cluster_shared_mask_experiment.py; does not modify the
single-cluster experiment in run_shared_layer_mask_experiment.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)

from experiments.run_shared_layer_mask_experiment import (
    AttentionForwardContext,
    build_causal_valid_mask,
    compute_directional_coverage_similarity,
    normalize_attention,
    query_abs_positions_from_last_q,
    select_representative_head,
)

try:
    import networkx as nx

    _HAS_NETWORKX = True
except ImportError:
    nx = None  # type: ignore
    _HAS_NETWORKX = False

try:
    from karateclub import Graph2Vec as KarateGraph2Vec

    _HAS_KARATECLUB = True
except ImportError:
    KarateGraph2Vec = None  # type: ignore
    _HAS_KARATECLUB = False


# ---------------------------------------------------------------------------
# Attention binarization (configurable)
# ---------------------------------------------------------------------------


def _normalize_attn_rows(attn: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = attn.sum(dim=-1, keepdim=True).clamp_min(eps)
    return attn / denom


def _causal_row_slice(
    attn: torch.Tensor,
    query_abs_positions: torch.Tensor,
) -> Tuple[torch.Tensor, List[int]]:
    """Return per-row causal slices and absolute query positions as Python ints."""
    last_q, seq_len = attn.shape
    rows: List[torch.Tensor] = []
    abs_positions: List[int] = []
    for row in range(last_q):
        abs_q = int(query_abs_positions[row].item())
        causal_len = min(abs_q + 1, seq_len)
        rows.append(attn[row, :causal_len])
        abs_positions.append(abs_q)
    return attn, abs_positions


def binarize_attention_top_p(
    attn: torch.Tensor,
    top_p: float,
    query_abs_positions: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    attn = _normalize_attn_rows(attn, eps=eps)
    last_q, seq_len = attn.shape
    out = torch.zeros((last_q, seq_len), dtype=torch.bool, device=attn.device)
    for row in range(last_q):
        abs_q = int(query_abs_positions[row].item())
        causal_len = abs_q + 1
        row_attn = attn[row, :causal_len]
        total = float(row_attn.sum().item())
        if total <= 0:
            out[row, :causal_len] = True
            continue
        target = top_p * total
        sorted_idx = torch.argsort(row_attn, descending=True)
        cum = 0.0
        for idx in sorted_idx.tolist():
            cum += float(row_attn[idx].item())
            out[row, idx] = True
            if cum >= target:
                break
    return out


def binarize_attention_top_k(
    attn: torch.Tensor,
    top_k: int,
    query_abs_positions: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    attn = _normalize_attn_rows(attn, eps=eps)
    last_q, seq_len = attn.shape
    out = torch.zeros((last_q, seq_len), dtype=torch.bool, device=attn.device)
    for row in range(last_q):
        abs_q = int(query_abs_positions[row].item())
        causal_len = abs_q + 1
        k = min(top_k, causal_len)
        if k <= 0:
            continue
        row_attn = attn[row, :causal_len]
        top_idx = torch.topk(row_attn, k=k, largest=True).indices
        out[row, top_idx] = True
    return out


def binarize_attention_threshold(
    attn: torch.Tensor,
    threshold: float,
    query_abs_positions: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    attn = _normalize_attn_rows(attn, eps=eps)
    last_q, seq_len = attn.shape
    out = torch.zeros((last_q, seq_len), dtype=torch.bool, device=attn.device)
    for row in range(last_q):
        abs_q = int(query_abs_positions[row].item())
        causal_len = abs_q + 1
        out[row, :causal_len] = attn[row, :causal_len] >= threshold
    return out


def binarize_attention_map(
    attn: torch.Tensor,
    method: str,
    query_abs_positions: torch.Tensor,
    *,
    top_p: float = 0.95,
    top_k: int = 128,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Binarize head attention [last_q, seq_len] under causal constraint."""
    if method == "top_p":
        return binarize_attention_top_p(attn, top_p, query_abs_positions)
    if method == "top_k":
        return binarize_attention_top_k(attn, top_k, query_abs_positions)
    if method == "threshold":
        return binarize_attention_threshold(attn, threshold, query_abs_positions)
    raise ValueError(f"Unknown binarize_method: {method}")


# ---------------------------------------------------------------------------
# Binary attention -> graph
# ---------------------------------------------------------------------------


def _key_node_label(
    key_pos: int,
    query_abs_pos: int,
    *,
    local_window: int,
    sink_tokens: int,
) -> str:
    """Position-aware key token label for Graph2Vec node features."""
    if key_pos < sink_tokens:
        return "key_sink"
    dist = query_abs_pos - key_pos
    if dist < 0:
        return "key_future"
    if dist <= local_window:
        return "key_local"
    if dist <= 4 * local_window:
        return "key_near"
    if key_pos < query_abs_pos // 2:
        return "key_far"
    return "key_middle"


def _graph_node_counts(g: "nx.Graph") -> Tuple[int, int]:
    return g.number_of_nodes(), g.number_of_edges()


def compute_graph_stats(
    graphs: List["nx.Graph"],
    backend: Optional[str],
) -> Dict[str, Any]:
    """Aggregate graph size statistics for one layer."""
    if not graphs:
        return {
            "num_graphs": 0,
            "avg_num_nodes": 0.0,
            "avg_num_edges": 0.0,
            "max_num_nodes": 0,
            "max_num_edges": 0,
            "min_num_nodes": 0,
            "min_num_edges": 0,
            "graph_embedding_backend": backend,
        }
    node_counts = [g.number_of_nodes() for g in graphs]
    edge_counts = [g.number_of_edges() for g in graphs]
    return {
        "num_graphs": len(graphs),
        "avg_num_nodes": float(np.mean(node_counts)),
        "avg_num_edges": float(np.mean(edge_counts)),
        "max_num_nodes": int(max(node_counts)),
        "max_num_edges": int(max(edge_counts)),
        "min_num_nodes": int(min(node_counts)),
        "min_num_edges": int(min(edge_counts)),
        "graph_embedding_backend": backend,
    }


def binary_attention_to_graph(
    binary_map: Union[torch.Tensor, np.ndarray],
    query_abs_positions: Union[torch.Tensor, np.ndarray],
    graph_type: str,
    *,
    local_window: int = 256,
    sink_tokens: int = 4,
) -> "nx.Graph":
    """
    Convert binary attention [last_q, seq_len] to a NetworkX graph for Graph2Vec.

    graph_type:
      - bipartite: query/key node types with WL-compatible string labels
      - directed_token: token-position nodes with directed edges (stored as DiGraph,
        converted to undirected if needed for Graph2Vec)
    """
    if not _HAS_NETWORKX:
        raise ImportError("networkx is required for graph conversion: pip install networkx")

    if isinstance(binary_map, torch.Tensor):
        binary = binary_map.cpu().numpy().astype(bool)
    else:
        binary = np.asarray(binary_map).astype(bool)

    if isinstance(query_abs_positions, torch.Tensor):
        q_abs = query_abs_positions.cpu().numpy().astype(int)
    else:
        q_abs = np.asarray(query_abs_positions).astype(int)

    last_q, seq_len = binary.shape
    ref_query_abs = int(q_abs[-1]) if len(q_abs) > 0 else seq_len - 1

    if graph_type == "bipartite":
        g = nx.Graph()
        for qi in range(last_q):
            g.add_node(f"q_{qi}", label="query_tail", feature="query_tail")
        for qi in range(last_q):
            abs_q = int(q_abs[qi])
            causal_len = min(abs_q + 1, seq_len)
            row = binary[qi, :causal_len]
            selected_keys = np.nonzero(row)[0]
            for ki in selected_keys.tolist():
                node_k = f"k_{ki}"
                if node_k not in g:
                    klbl = _key_node_label(
                        int(ki),
                        ref_query_abs,
                        local_window=local_window,
                        sink_tokens=sink_tokens,
                    )
                    g.add_node(node_k, label=klbl, feature=klbl)
                g.add_edge(f"q_{qi}", node_k)

        if g.number_of_edges() == 0 and last_q > 0:
            fallback_ki = max(0, ref_query_abs)
            node_k = f"k_{fallback_ki}"
            klbl = _key_node_label(
                fallback_ki,
                ref_query_abs,
                local_window=local_window,
                sink_tokens=sink_tokens,
            )
            g.add_node(node_k, label=klbl, feature=klbl)
            g.add_edge(f"q_{last_q - 1}", node_k)
        return g

    if graph_type == "directed_token":
        g = nx.DiGraph()
        for qi in range(last_q):
            abs_q = int(q_abs[qi])
            node_q = f"t_{abs_q}"
            qlbl = _key_node_label(
                abs_q, abs_q, local_window=local_window, sink_tokens=sink_tokens
            )
            g.add_node(node_q, label=f"query_{qlbl}", feature=f"query_{qlbl}")
            causal_len = min(abs_q + 1, seq_len)
            row = binary[qi, :causal_len]
            selected_keys = np.nonzero(row)[0]
            for ki in selected_keys.tolist():
                node_k = f"t_{int(ki)}"
                klbl = _key_node_label(
                    int(ki),
                    abs_q,
                    local_window=local_window,
                    sink_tokens=sink_tokens,
                )
                g.add_node(node_k, label=f"key_{klbl}", feature=f"key_{klbl}")
                g.add_edge(node_q, node_k)

        if g.number_of_edges() == 0 and last_q > 0:
            abs_q = int(q_abs[last_q - 1])
            node_q = f"t_{abs_q}"
            fallback_ki = max(0, abs_q)
            node_k = f"t_{fallback_ki}"
            g.add_node(node_q, label="query_tail", feature="query_tail")
            g.add_node(
                node_k,
                label=f"key_{_key_node_label(fallback_ki, abs_q, local_window=local_window, sink_tokens=sink_tokens)}",
                feature=f"key_{_key_node_label(fallback_ki, abs_q, local_window=local_window, sink_tokens=sink_tokens)}",
            )
            g.add_edge(node_q, node_k)
        return g

    raise ValueError(f"Unknown graph_type: {graph_type}")


def _graph_to_karateclub(g: "nx.Graph") -> "nx.Graph":
    """
    Convert arbitrary NetworkX graph to karateclub-compatible graph.

    Requirements:
    - Nodes relabeled to consecutive integers 0..n-1.
    - Preserve node labels as string attributes (label + feature).
    - Directed graphs converted to undirected.
    """
    if isinstance(g, nx.DiGraph):
        base = g.to_undirected()
    else:
        base = g

    mapping = {node: i for i, node in enumerate(base.nodes())}
    out = nx.Graph()
    for node, new_id in mapping.items():
        data = base.nodes[node]
        label = str(data.get("label", "0"))
        out.add_node(new_id, label=label, feature=label)
    for u, v in base.edges():
        out.add_edge(mapping[u], mapping[v])
    return out


# ---------------------------------------------------------------------------
# Graph2Vec embedding + clustering
# ---------------------------------------------------------------------------


class Graph2VecHeadEmbedder:
    """Embed head graphs via karateclub Graph2Vec with fallbacks."""

    def __init__(
        self,
        dimensions: int = 128,
        wl_iterations: int = 2,
        workers: int = 1,
        seed: int = 42,
    ) -> None:
        self.dimensions = dimensions
        self.wl_iterations = wl_iterations
        self.workers = workers
        self.seed = seed
        self.backend_used: Optional[str] = None

    def fit_transform(self, graphs: List["nx.Graph"]) -> np.ndarray:
        if not graphs:
            self.backend_used = "empty_graphs"
            return np.zeros((0, self.dimensions), dtype=np.float32)

        if _HAS_KARATECLUB:
            try:
                kgraphs = [_graph_to_karateclub(g) for g in graphs]
                # karateclub.Graph2Vec does not expose seed/random_state; may be non-deterministic.
                model_kwargs: Dict[str, Any] = {
                    "dimensions": self.dimensions,
                    "wl_iterations": self.wl_iterations,
                    "workers": self.workers,
                    "attributed": True,
                }
                try:
                    model = KarateGraph2Vec(**model_kwargs, seed=self.seed)
                except TypeError:
                    try:
                        model = KarateGraph2Vec(**model_kwargs)
                    except TypeError:
                        logger.debug(
                            "karateclub.Graph2Vec does not accept attributed; "
                            "falling back without node features"
                        )
                        model_kwargs.pop("attributed", None)
                        try:
                            model = KarateGraph2Vec(**model_kwargs, seed=self.seed)
                        except TypeError:
                            model = KarateGraph2Vec(**model_kwargs)
                model.fit(kgraphs)
                emb = model.get_embedding()
                if emb.shape[0] != len(graphs):
                    raise RuntimeError(
                        f"karateclub returned {emb.shape[0]} embeddings for {len(graphs)} graphs"
                    )
                self.backend_used = "karateclub.Graph2Vec"
                logger.info("Graph embedding backend used: %s", self.backend_used)
                return emb.astype(np.float32)
            except Exception as exc:
                logger.warning("karateclub Graph2Vec failed (%s); trying fallback", exc)
                self.backend_used = "fallback_wl_doc2vec_after_karateclub_failure"
                logger.info("Graph embedding backend used: %s", self.backend_used)
                return self._fallback_wl_doc2vec(graphs)

        logger.warning(
            "karateclub is not installed; using WL+Doc2Vec fallback. "
            "This is not strict karateclub.Graph2Vec. Install: pip install karateclub"
        )
        self.backend_used = "fallback_wl_doc2vec_no_karateclub"
        logger.info("Graph embedding backend used: %s", self.backend_used)
        return self._fallback_wl_doc2vec(graphs)

    def _fallback_wl_doc2vec(self, graphs: List["nx.Graph"]) -> np.ndarray:
        try:
            from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        except ImportError as exc:
            raise ImportError(
                "Neither karateclub nor gensim available for Graph2Vec fallback. "
                "Install: pip install karateclub  OR  pip install gensim"
            ) from exc

        docs: List[TaggedDocument] = []
        for i, g in enumerate(graphs):
            words: List[str] = []
            for node, data in g.nodes(data=True):
                words.append(f"n_{data.get('label', node)}")
            for u, v in g.edges():
                words.append(f"e_{g.nodes[u].get('label', u)}_{g.nodes[v].get('label', v)}")
            if not words:
                words = ["EMPTY"]
            docs.append(TaggedDocument(words=words, tags=[str(i)]))

        model = Doc2Vec(
            vector_size=self.dimensions,
            min_count=0,
            epochs=100,
            seed=self.seed,
            workers=self.workers,
            dm=1,
        )
        model.build_vocab(docs)
        model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)
        if self.backend_used is None:
            self.backend_used = "fallback_wl_doc2vec"
        return np.stack([model.dv[str(i)] for i in range(len(graphs))], axis=0).astype(
            np.float32
        )


def _kmeans_fit_predict(features: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """KMeans with n_init='auto' when supported, else n_init=10."""
    try:
        return KMeans(
            n_clusters=n_clusters, random_state=seed, n_init="auto"
        ).fit_predict(features)
    except TypeError:
        return KMeans(
            n_clusters=n_clusters, random_state=seed, n_init=10
        ).fit_predict(features)


def _fallback_flatten_cluster(
    binary_maps: List[np.ndarray],
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    flat = np.stack([m.reshape(-1).astype(np.float32) for m in binary_maps], axis=0)
    n_components = min(32, flat.shape[0], flat.shape[1])
    if n_components >= 2:
        svd = TruncatedSVD(n_components=n_components, random_state=seed)
        features = svd.fit_transform(flat)
    else:
        features = flat
    return _kmeans_fit_predict(features, n_clusters, seed)


def _fallback_jaccard_cluster(
    binary_maps: List[np.ndarray],
    n_clusters: int,
) -> np.ndarray:
    n = len(binary_maps)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            a = binary_maps[i].astype(bool).ravel()
            b = binary_maps[j].astype(bool).ravel()
            inter = float(np.logical_and(a, b).sum())
            union = float(np.logical_or(a, b).sum())
            d = 1.0 - (inter / union if union > 0 else 1.0)
            dist[i, j] = dist[j, i] = d
    clustering = AgglomerativeClustering(
        n_clusters=min(n_clusters, n),
        metric="precomputed",
        linkage="average",
    )
    return clustering.fit_predict(dist)


def cluster_head_embeddings(
    embeddings: np.ndarray,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    n_heads = embeddings.shape[0]
    if n_heads == 0:
        return np.array([], dtype=int)
    k = min(n_clusters, n_heads)
    if k == 1:
        return np.zeros(n_heads, dtype=int)
    return _kmeans_fit_predict(embeddings, k, seed)


def fix_empty_clusters(
    labels: np.ndarray,
    num_heads: int,
    n_clusters: int,
) -> np.ndarray:
    """Ensure every cluster has at least one head; split by head id if needed."""
    labels = labels.copy()
    for c in range(n_clusters):
        if np.sum(labels == c) == 0:
            largest_c = int(np.bincount(labels, minlength=n_clusters).argmax())
            members = np.where(labels == largest_c)[0]
            if len(members) >= 2:
                move = int(members[-1])
                labels[move] = c
            else:
                for h in range(num_heads):
                    if h not in labels:
                        labels[h % n_clusters] = c
                        break
                else:
                    labels[c % num_heads] = c
    return labels


def build_layer_binary_maps(
    layer_attn: torch.Tensor,
    query_abs_positions: torch.Tensor,
    args: Any,
) -> List[np.ndarray]:
    """Binarize each head attention map for clustering (fixed binarize_top_p, etc.)."""
    num_heads = layer_attn.shape[0]
    binary_maps: List[np.ndarray] = []
    for h in range(num_heads):
        binary = binarize_attention_map(
            layer_attn[h],
            args.binarize_method,
            query_abs_positions,
            top_p=args.binarize_top_p,
            top_k=args.binarize_top_k,
            threshold=args.binarize_threshold,
        )
        binary_maps.append(binary.cpu().numpy())
    return binary_maps


def binary_maps_to_graphs(
    binary_maps: List[np.ndarray],
    query_abs_positions: torch.Tensor,
    args: Any,
) -> List["nx.Graph"]:
    graphs: List["nx.Graph"] = []
    sink_tokens = int(getattr(args, "sink_tokens", 4))
    for binary_np in binary_maps:
        graphs.append(
            binary_attention_to_graph(
                binary_np,
                query_abs_positions,
                args.graph_type,
                local_window=args.local_window,
                sink_tokens=sink_tokens,
            )
        )
    return graphs


def _log_layer_cluster_diagnostics(
    layer_idx: int,
    cluster_method: str,
    labels: np.ndarray,
    extra_info: Dict[str, Any],
) -> None:
    """Log per-layer clustering diagnostics (labels before/after fix_empty_clusters)."""
    labels_list = labels.tolist()
    logger.info("[Layer %d] cluster_method=%s", layer_idx, cluster_method)
    logger.info("[Layer %d] labels=%s", layer_idx, labels_list)
    if "labels_before_fix" in extra_info:
        logger.info(
            "[Layer %d] labels_before_fix=%s",
            layer_idx,
            extra_info["labels_before_fix"],
        )
    logger.info("[Layer %d] labels_after_fix=%s", layer_idx, labels_list)

    if cluster_method == "svd_kmeans":
        logger.info(
            "[Layer %d] svd_components_used=%s",
            layer_idx,
            extra_info.get("svd_components_used"),
        )
    elif cluster_method == "bmm":
        logger.info(
            "[Layer %d] bmm_log_likelihood=%.4f",
            layer_idx,
            float(extra_info.get("log_likelihood", float("nan"))),
        )
        logger.info(
            "[Layer %d] bmm_n_iter=%s",
            layer_idx,
            extra_info.get("n_iter"),
        )
        logger.info(
            "[Layer %d] bmm_pi=%s",
            layer_idx,
            extra_info.get("pi"),
        )


def cluster_layer_heads_svd_kmeans(
    binary_maps: List[np.ndarray],
    n_clusters: int,
    seed: int,
    *,
    svd_components: int = 8,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Flatten binarized patterns -> TruncatedSVD -> KMeans(k=n_clusters)."""
    num_heads = len(binary_maps)
    flat = np.stack(
        [m.reshape(-1).astype(np.float32) for m in binary_maps],
        axis=0,
    )

    n_samples, n_features = flat.shape
    max_components = min(n_samples - 1, n_features - 1) if n_samples > 1 and n_features > 1 else 0
    n_components = min(int(svd_components), max_components)

    if n_components >= 2:
        svd = TruncatedSVD(n_components=n_components, random_state=seed)
        features = svd.fit_transform(flat)
        explained_variance_ratio = svd.explained_variance_ratio_.tolist()
    else:
        features = flat
        explained_variance_ratio = []
        n_components = 0

    labels = cluster_head_embeddings(features, n_clusters, seed)
    labels_before_fix = labels.copy()
    labels = fix_empty_clusters(labels, num_heads, n_clusters)

    extra_info: Dict[str, Any] = {
        "method": "svd_kmeans",
        "flat_shape": list(flat.shape),
        "feature_shape": list(features.shape),
        "svd_components_used": int(n_components),
        "svd_explained_variance_ratio": explained_variance_ratio,
        "labels_before_fix": labels_before_fix.tolist(),
        "labels_after_fix": labels.tolist(),
    }
    return labels, features.astype(np.float32), extra_info


def cluster_layer_heads_bmm(
    binary_maps: List[np.ndarray],
    n_clusters: int,
    seed: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-4,
    n_init: int = 5,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Bernoulli Mixture Model (NumPy EM) on flattened binary attention patterns.

    mu is the Bernoulli probability template for each cluster.
    Shape: [n_clusters, flattened_binary_map_dim].
    """
    num_heads = len(binary_maps)
    if num_heads == 0:
        empty_extra = {
            "method": "bmm",
            "flat_shape": [0, 0],
            "responsibility_shape": [0, 0],
            "mu_shape": [0, 0],
            "pi": [],
            "log_likelihood": float("nan"),
            "n_iter": 0,
            "best_init_id": -1,
            "labels_before_fix": [],
            "labels_after_fix": [],
        }
        return np.array([], dtype=int), np.zeros((0, 1), dtype=np.float32), empty_extra

    x = np.stack([m.astype(np.float64).reshape(-1) for m in binary_maps], axis=0)
    n_samples, dim = x.shape
    k = int(n_clusters)
    if k > n_samples:
        raise ValueError(
            f"n_clusters={k} cannot exceed n_samples={n_samples} for BMM."
        )
    if k <= 1:
        labels = np.zeros(n_samples, dtype=int)
        resp = np.ones((n_samples, 1), dtype=np.float32)
        mu = np.clip(x.mean(axis=0, keepdims=True), eps, 1.0 - eps)
        extra_info = {
            "method": "bmm",
            "flat_shape": list(x.shape),
            "responsibility_shape": list(resp.shape),
            "mu_shape": list(mu.shape),
            "pi": [1.0],
            "log_likelihood": float("nan"),
            "n_iter": 0,
            "best_init_id": 0,
            "labels_before_fix": labels.tolist(),
            "labels_after_fix": labels.tolist(),
            "_bmm_mu_array": mu.astype(np.float32),
        }
        return labels, resp, extra_info

    rng_master = np.random.default_rng(seed)
    best_ll = -np.inf
    best_result: Optional[Dict[str, Any]] = None

    for init_id in range(max(1, int(n_init))):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
        pi = np.full(k, 1.0 / k, dtype=np.float64)
        empirical = np.clip(x.mean(axis=0), eps, 1.0 - eps)
        mu = np.zeros((k, dim), dtype=np.float64)
        for cid in range(k):
            noise = rng.normal(loc=0.0, scale=0.05, size=dim)
            mu[cid] = np.clip(empirical + noise, eps, 1.0 - eps)

        prev_ll = -np.inf
        resp = np.full((n_samples, k), 1.0 / k, dtype=np.float64)
        n_iter_done = 0

        for it in range(max_iter):
            log_prob = np.zeros((n_samples, k), dtype=np.float64)
            for cid in range(k):
                muk = np.clip(mu[cid], eps, 1.0 - eps)
                log_px = x @ np.log(muk) + (1.0 - x) @ np.log(1.0 - muk)
                log_prob[:, cid] = np.log(pi[cid] + eps) + log_px

            max_log = np.max(log_prob, axis=1, keepdims=True)
            log_norm = max_log + np.log(
                np.sum(np.exp(log_prob - max_log), axis=1, keepdims=True) + eps
            )
            resp = np.exp(log_prob - log_norm)
            ll = float(np.sum(log_norm))

            nk = resp.sum(axis=0) + eps
            pi = nk / n_samples
            mu = (resp.T @ x) / nk[:, None]
            mu = np.clip(mu, eps, 1.0 - eps)
            n_iter_done = it + 1

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        if ll > best_ll:
            best_ll = ll
            best_result = {
                "pi": pi.copy(),
                "mu": mu.copy(),
                "resp": resp.copy(),
                "n_iter": n_iter_done,
                "log_likelihood": ll,
                "init_id": init_id,
            }

    if best_result is None:
        raise RuntimeError("BMM failed to fit.")

    resp = best_result["resp"]
    labels = resp.argmax(axis=1).astype(int)
    labels_before_fix = labels.copy()
    labels = fix_empty_clusters(labels, n_samples, k)

    extra_info = {
        "method": "bmm",
        "flat_shape": list(x.shape),
        "responsibility_shape": list(resp.shape),
        "mu_shape": list(best_result["mu"].shape),
        "pi": best_result["pi"].astype(np.float32).tolist(),
        "log_likelihood": float(best_result["log_likelihood"]),
        "n_iter": int(best_result["n_iter"]),
        "best_init_id": int(best_result["init_id"]),
        "labels_before_fix": labels_before_fix.tolist(),
        "labels_after_fix": labels.tolist(),
        # mu: Bernoulli template per cluster; save as .npy outside JSON dumps.
        "_bmm_mu_array": best_result["mu"].astype(np.float32),
    }
    return labels, resp.astype(np.float32), extra_info


def cluster_layer_heads_by_method(
    layer_attn: torch.Tensor,
    query_abs_positions: torch.Tensor,
    args: Any,
    *,
    layer_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List["nx.Graph"], Dict[str, Any]]:
    """
    Cluster heads in one layer using args.cluster_method:
      graph2vec | svd_kmeans | bmm

    All methods share the same binarized attention maps from build_layer_binary_maps().
    Returns (labels, embeddings, graphs, graph_stats/extra_info).
    """
    method = str(getattr(args, "cluster_method", "graph2vec")).lower()
    binary_maps = build_layer_binary_maps(layer_attn, query_abs_positions, args)
    num_heads = len(binary_maps)
    n_clusters = min(int(args.num_head_clusters), num_heads)
    seed = int(getattr(args, "cluster_seed", 42))

    graphs: List["nx.Graph"] = []
    extra_info: Dict[str, Any] = {"method": method}

    if method == "svd_kmeans":
        labels, embeddings, extra_info = cluster_layer_heads_svd_kmeans(
            binary_maps,
            n_clusters,
            seed,
            svd_components=int(getattr(args, "svd_components", 8)),
        )
        graph_stats = {
            "cluster_method": method,
            "num_graphs": 0,
            "graph_embedding_backend": method,
            **extra_info,
        }
        if layer_idx is not None:
            _log_layer_cluster_diagnostics(layer_idx, method, labels, extra_info)
        return labels, embeddings, graphs, graph_stats

    if method == "bmm":
        labels, embeddings, extra_info = cluster_layer_heads_bmm(
            binary_maps,
            n_clusters,
            seed,
            max_iter=int(getattr(args, "bmm_max_iter", 100)),
            tol=float(getattr(args, "bmm_tol", 1e-4)),
            n_init=int(getattr(args, "bmm_n_init", 5)),
        )
        graph_stats = {
            "cluster_method": method,
            "num_graphs": 0,
            "graph_embedding_backend": method,
            **extra_info,
        }
        if layer_idx is not None:
            _log_layer_cluster_diagnostics(layer_idx, method, labels, extra_info)
        return labels, embeddings, graphs, graph_stats

    if method != "graph2vec":
        raise ValueError(f"Unknown cluster_method: {method}")

    graphs = binary_maps_to_graphs(binary_maps, query_abs_positions, args)
    embedder = Graph2VecHeadEmbedder(
        dimensions=args.graph2vec_dim,
        wl_iterations=args.graph2vec_wl_iterations,
        workers=args.graph2vec_workers,
        seed=seed,
    )

    backend_used = "unknown"
    try:
        embeddings = embedder.fit_transform(graphs)
        backend_used = embedder.backend_used or "unknown"
        labels = cluster_head_embeddings(embeddings, n_clusters, seed)
    except Exception as exc:
        logger.warning(
            "Graph2Vec clustering failed for layer (%s); flatten+SVD fallback",
            exc,
        )
        backend_used = "fallback_flatten_svd_kmeans"
        labels, embeddings, fallback_extra = cluster_layer_heads_svd_kmeans(
            binary_maps,
            n_clusters,
            seed,
            svd_components=int(getattr(args, "svd_components", 8)),
        )
        extra_info.update(fallback_extra)
        extra_info["graph2vec_fallback_reason"] = str(exc)

    if "labels_before_fix" not in extra_info:
        labels_before_fix = labels.copy()
        labels = fix_empty_clusters(labels, num_heads, n_clusters)
        extra_info.update(
            {
                "method": "graph2vec",
                "embedding_shape": list(embeddings.shape),
                "labels_before_fix": labels_before_fix.tolist(),
                "labels_after_fix": labels.tolist(),
            }
        )
    else:
        labels = np.array(extra_info["labels_after_fix"], dtype=int)

    graph_stats = compute_graph_stats(graphs, backend_used)
    graph_stats["cluster_method"] = method
    graph_stats.update(extra_info)
    logger.info(
        "Layer graph stats: graphs=%d avg_nodes=%.1f avg_edges=%.1f max_nodes=%d backend=%s",
        graph_stats["num_graphs"],
        graph_stats["avg_num_nodes"],
        graph_stats["avg_num_edges"],
        graph_stats["max_num_nodes"],
        graph_stats["graph_embedding_backend"],
    )
    if layer_idx is not None:
        _log_layer_cluster_diagnostics(layer_idx, method, labels, extra_info)
    return labels, embeddings, graphs, graph_stats


def cluster_layer_heads_graph2vec(
    layer_attn: torch.Tensor,
    query_abs_positions: torch.Tensor,
    args: Any,
) -> Tuple[np.ndarray, np.ndarray, List["nx.Graph"], Dict[str, Any]]:
    """Backward-compatible alias for Graph2Vec clustering."""
    saved = getattr(args, "cluster_method", "graph2vec")
    args.cluster_method = "graph2vec"
    try:
        return cluster_layer_heads_by_method(layer_attn, query_abs_positions, args)
    finally:
        args.cluster_method = saved


def fallback_layer_clustering_no_graph2vec(
    layer_attn: torch.Tensor,
    query_abs_positions: torch.Tensor,
    args: Any,
    similarity_mask_builder: Any,
) -> Tuple[np.ndarray, np.ndarray, List["nx.Graph"], Dict[str, Any]]:
    """
    Non-Graph2Vec fallback for debug_layers skip: deterministic head-id split + coverage reps.
    """
    num_heads = layer_attn.shape[0]
    n_clusters = args.num_head_clusters
    labels = np.array([h % n_clusters for h in range(num_heads)], dtype=int)
    labels = fix_empty_clusters(labels, num_heads, n_clusters)

    binary_maps = build_layer_binary_maps(layer_attn, query_abs_positions, args)
    graphs = binary_maps_to_graphs(binary_maps, query_abs_positions, args)

    graph_stats = compute_graph_stats(graphs, "skipped_no_graph2vec_debug_layers")
    embeddings = np.zeros((num_heads, args.graph2vec_dim), dtype=np.float32)
    return labels, embeddings, graphs, graph_stats


# ---------------------------------------------------------------------------
# Cluster-internal representative head selection
# ---------------------------------------------------------------------------


def select_representative_in_cluster(
    layer_attn: torch.Tensor,
    cluster_heads: List[int],
    similarity_mask_builder: Any,
    query_abs_positions: torch.Tensor,
) -> Tuple[int, Dict[str, Any]]:
    """Directional coverage similarity within a cluster."""
    if len(cluster_heads) == 1:
        h = cluster_heads[0]
        return h, {
            "representative_head": h,
            "representative_score": 1.0,
            "mean_coverage": 1.0,
            "min_coverage": 1.0,
            "std_coverage": 0.0,
            "cluster_size": 1,
            "cluster_heads": cluster_heads,
        }

    cluster_attn = layer_attn[cluster_heads]

    sim = compute_directional_coverage_similarity(
        cluster_attn,
        similarity_mask_builder,
        query_abs_positions=query_abs_positions,
    )
    local_rep, score_per_head = select_representative_head(sim)
    rep_head = cluster_heads[int(local_rep)]
    rep_score = float(score_per_head[local_rep].item())

    outgoing = sim[local_rep]
    return rep_head, {
        "representative_head": rep_head,
        "representative_score": rep_score,
        "mean_coverage": float(outgoing.mean().item()),
        "min_coverage": float(outgoing.min().item()),
        "std_coverage": float(outgoing.std(unbiased=False).item()),
        "cluster_size": len(cluster_heads),
        "cluster_heads": cluster_heads,
        "similarity_matrix": sim,
    }


def cluster_assignments_to_dict(
    labels: np.ndarray,
    n_clusters: int,
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {c: [] for c in range(n_clusters)}
    for h, lab in enumerate(labels.tolist()):
        out[int(lab)].append(int(h))
    return out


def run_layer_clustering_and_selection(
    layer_attn: torch.Tensor,
    layer_idx: int,
    query_abs_positions: torch.Tensor,
    args: Any,
    similarity_mask_builder: Any,
) -> Dict[str, Any]:
    cluster_method = str(getattr(args, "cluster_method", "graph2vec")).lower()
    debug_layers = getattr(args, "debug_layers", None)

    if (
        cluster_method == "graph2vec"
        and debug_layers is not None
        and layer_idx not in debug_layers
    ):
        logger.info(
            "Layer %02d: skipping Graph2Vec because it is not in debug_layers=%s; "
            "using head-id split fallback.",
            layer_idx,
            sorted(debug_layers),
        )
        labels, embeddings, graphs, graph_stats = fallback_layer_clustering_no_graph2vec(
            layer_attn, query_abs_positions, args, similarity_mask_builder
        )
    else:
        labels, embeddings, graphs, graph_stats = cluster_layer_heads_by_method(
            layer_attn, query_abs_positions, args, layer_idx=layer_idx
        )

    bmm_mu = graph_stats.pop("_bmm_mu_array", None)
    if bmm_mu is not None:
        graph_stats["mu_shape"] = list(bmm_mu.shape)
        mu_dir: Optional[Path] = None

        bmm_mu_dir = getattr(args, "_bmm_mu_dir", None)
        if bmm_mu_dir:
            mu_dir = Path(bmm_mu_dir)
        else:
            output_dir = getattr(args, "output_dir", None)
            if output_dir:
                mu_dir = Path(output_dir) / "bmm_mu"

        if mu_dir is not None:
            mu_dir.mkdir(parents=True, exist_ok=True)
            mu_path = mu_dir / f"layer_{layer_idx:02d}_bmm_mu.npy"
            np.save(mu_path, bmm_mu)
            graph_stats["bmm_mu_path"] = str(mu_path)
            graph_stats["bmm_mu_saved"] = True
        else:
            graph_stats["bmm_mu_saved"] = False

    clusters = cluster_assignments_to_dict(labels, args.num_head_clusters)
    cluster_reps: Dict[int, Dict[str, Any]] = {}
    similarity_matrices: Dict[int, torch.Tensor] = {}

    for cid, heads in clusters.items():
        rep, stats = select_representative_in_cluster(
            layer_attn, heads, similarity_mask_builder, query_abs_positions
        )
        if "similarity_matrix" in stats:
            similarity_matrices[cid] = stats.pop("similarity_matrix")
        cluster_reps[cid] = stats

    return {
        "layer_idx": layer_idx,
        "labels": labels,
        "embeddings": embeddings,
        "graphs": graphs,
        "graph_stats": graph_stats,
        "clusters": clusters,
        "cluster_representatives": cluster_reps,
        "similarity_matrices": similarity_matrices,
    }


# ---------------------------------------------------------------------------
# Global cluster alignment and aggregation
# ---------------------------------------------------------------------------


def _jaccard_sets(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def align_cluster_labels_to_reference(
    ref_clusters: Dict[int, List[int]],
    sample_clusters: Dict[int, List[int]],
    n_clusters: int,
    *,
    layer_idx: Optional[int] = None,
    sample_idx: Optional[int] = None,
) -> Dict[int, List[int]]:
    """
    Align sample cluster labels to reference via Jaccard overlap on head sets.
    For k=2, swap labels if that improves overlap.
    """
    ref_sets = {c: set(ref_clusters.get(c, [])) for c in range(n_clusters)}
    sample_sets = {c: set(sample_clusters.get(c, [])) for c in range(n_clusters)}

    if n_clusters == 2:
        c0, c1 = 0, 1
        direct = _jaccard_sets(ref_sets[c0], sample_sets[c0]) + _jaccard_sets(
            ref_sets[c1], sample_sets[c1]
        )
        swapped = _jaccard_sets(ref_sets[c0], sample_sets[c1]) + _jaccard_sets(
            ref_sets[c1], sample_sets[c0]
        )
        swapped_flag = swapped > direct
        if layer_idx is not None and sample_idx is not None:
            logger.info(
                "Cluster alignment layer=%d sample=%d direct=%.4f swapped=%.4f swapped=%s",
                layer_idx,
                sample_idx,
                direct,
                swapped,
                swapped_flag,
            )
        if swapped_flag:
            return {c0: list(sample_sets[c1]), c1: list(sample_sets[c0])}
        return {c: list(sample_sets[c]) for c in range(n_clusters)}

    # Greedy matching for k > 2
    used_sample: set = set()
    aligned: Dict[int, List[int]] = {}
    for rc in range(n_clusters):
        best_sc, best_sc_id = -1.0, None
        for sc in range(n_clusters):
            if sc in used_sample:
                continue
            score = _jaccard_sets(ref_sets[rc], sample_sets[sc])
            if score > best_sc:
                best_sc, best_sc_id = score, sc
        if best_sc_id is not None:
            used_sample.add(best_sc_id)
            aligned[rc] = list(sample_sets[best_sc_id])
            if layer_idx is not None and sample_idx is not None:
                logger.info(
                    "Cluster alignment layer=%d sample=%d ref_cluster=%d -> sample_cluster=%d overlap=%.4f",
                    layer_idx,
                    sample_idx,
                    rc,
                    best_sc_id,
                    best_sc,
                )
        else:
            aligned[rc] = []
    return aligned


def align_cluster_labels_with_info(
    ref_clusters: Dict[int, List[int]],
    sample_clusters: Dict[int, List[int]],
    n_clusters: int,
    *,
    layer_idx: int,
    sample_idx: int,
) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
    """Align labels and return alignment metadata for logging."""
    ref_sets = {c: set(ref_clusters.get(c, [])) for c in range(n_clusters)}
    sample_sets = {c: set(sample_clusters.get(c, [])) for c in range(n_clusters)}

    info: Dict[str, Any] = {
        "layer": layer_idx,
        "sample": sample_idx,
    }

    if n_clusters == 2:
        c0, c1 = 0, 1
        direct = _jaccard_sets(ref_sets[c0], sample_sets[c0]) + _jaccard_sets(
            ref_sets[c1], sample_sets[c1]
        )
        swapped_score = _jaccard_sets(ref_sets[c0], sample_sets[c1]) + _jaccard_sets(
            ref_sets[c1], sample_sets[c0]
        )
        swapped_flag = swapped_score > direct
        info.update(
            {
                "direct_overlap": float(direct),
                "swapped_overlap": float(swapped_score),
                "swapped": swapped_flag,
            }
        )
        if swapped_flag:
            aligned = {c0: list(sample_sets[c1]), c1: list(sample_sets[c0])}
        else:
            aligned = {c: list(sample_sets[c]) for c in range(n_clusters)}
        logger.info(
            "Cluster alignment layer=%d sample=%d direct=%.4f swapped=%.4f swapped=%s",
            layer_idx,
            sample_idx,
            direct,
            swapped_score,
            swapped_flag,
        )
        return aligned, info

    aligned = align_cluster_labels_to_reference(
        ref_clusters,
        sample_clusters,
        n_clusters,
        layer_idx=layer_idx,
        sample_idx=sample_idx,
    )
    info["method"] = "greedy_matching_k_gt_2"
    return aligned, info


def labels_from_clusters(clusters: Dict[int, List[int]], num_heads: int) -> np.ndarray:
    labels = np.zeros(num_heads, dtype=int)
    for cid, heads in clusters.items():
        for h in heads:
            labels[h] = int(cid)
    return labels


def aggregate_global_cluster_assignments(
    selection_results: List[Dict[str, Any]],
    num_layers: int,
    num_heads: int,
    n_clusters: int,
) -> Dict[str, Any]:
    """Majority vote per head after aligning samples to reference (sample 0)."""
    ref = selection_results[0]
    ref_layer_clusters = ref["layer_clusters"]

    head_votes: Dict[int, Dict[int, List[int]]] = {
        layer: {h: [0] * n_clusters for h in range(num_heads)} for layer in range(num_layers)
    }
    alignment_records: List[Dict[str, Any]] = []

    for sample_idx, result in enumerate(selection_results):
        for layer_idx in range(num_layers):
            if sample_idx == 0:
                aligned = result["layer_clusters"][layer_idx]
            else:
                aligned, align_info = align_cluster_labels_with_info(
                    ref_layer_clusters[layer_idx],
                    result["layer_clusters"][layer_idx],
                    n_clusters,
                    layer_idx=layer_idx,
                    sample_idx=sample_idx,
                )
                alignment_records.append(align_info)
            labels = labels_from_clusters(aligned, num_heads)
            for h in range(num_heads):
                head_votes[layer_idx][h][int(labels[h])] += 1

    global_head_cluster: Dict[int, Dict[int, int]] = {}
    global_clusters: Dict[int, Dict[int, List[int]]] = {}
    vote_detail: Dict[str, Any] = {}

    for layer_idx in range(num_layers):
        ref_labels = labels_from_clusters(
            ref_layer_clusters[layer_idx], num_heads
        )
        layer_head_cluster: Dict[int, int] = {}
        layer_clusters: Dict[int, List[int]] = {c: [] for c in range(n_clusters)}

        for h in range(num_heads):
            votes = head_votes[layer_idx][h]
            max_votes = max(votes)
            winners = [c for c, v in enumerate(votes) if v == max_votes]
            if len(winners) == 1:
                cid = winners[0]
            else:
                cid = int(ref_labels[h])
            layer_head_cluster[h] = cid
            layer_clusters[cid].append(h)

        before_fix = {c: len(layer_clusters[c]) for c in range(n_clusters)}
        layer_clusters = fix_empty_clusters_dict(
            layer_clusters, num_heads, n_clusters, layer_idx=layer_idx
        )
        after_fix = {c: len(layer_clusters[c]) for c in range(n_clusters)}
        if before_fix != after_fix:
            logger.warning(
                "Layer %d: fixed empty global clusters %s -> %s",
                layer_idx,
                before_fix,
                after_fix,
            )
        layer_head_cluster = {
            h: cid
            for cid, heads in layer_clusters.items()
            for h in heads
        }
        global_head_cluster[layer_idx] = layer_head_cluster
        global_clusters[layer_idx] = layer_clusters
        vote_detail[str(layer_idx)] = {
            str(h): head_votes[layer_idx][h] for h in range(num_heads)
        }

    return {
        "layer_to_head_cluster": global_head_cluster,
        "layer_clusters": global_clusters,
        "num_selection_samples": len(selection_results),
        "per_head_vote_detail": vote_detail,
        "cluster_alignment_records": alignment_records,
    }


def fix_empty_clusters_dict(
    clusters: Dict[int, List[int]],
    num_heads: int,
    n_clusters: int,
    *,
    layer_idx: Optional[int] = None,
) -> Dict[int, List[int]]:
    out = {c: list(clusters.get(c, [])) for c in range(n_clusters)}
    for c in range(n_clusters):
        if not out[c]:
            largest_c = max(range(n_clusters), key=lambda x: len(out[x]))
            if out[largest_c]:
                moved = out[largest_c].pop()
                out[c].append(moved)
                if layer_idx is not None:
                    logger.warning(
                        "Layer %d cluster %d empty; moved head %d from cluster %d",
                        layer_idx,
                        c,
                        moved,
                        largest_c,
                    )
            else:
                for h in range(num_heads):
                    if all(h not in out[c2] for c2 in range(n_clusters)):
                        out[c].append(h)
                        break
    return out


def _select_best_representative(
    candidates: Dict[int, Tuple[int, float]],
    final_heads: set,
) -> int:
    """
    Select representative head: vote_count first, score_sum tie-break, then smaller head id.

    candidates: head_id -> (vote_count, score_sum)
    """
    if not candidates:
        if final_heads:
            return min(final_heads)
        return 0

    def sort_key(item: Tuple[int, Tuple[int, float]]) -> Tuple[int, float, int]:
        head, (votes, score_sum) = item
        return (votes, score_sum, -head)

    best_head = max(candidates.items(), key=sort_key)[0]
    if best_head in final_heads or not final_heads:
        return int(best_head)

    in_cluster = {h: candidates[h] for h in final_heads if h in candidates}
    if in_cluster:
        return int(max(in_cluster.items(), key=sort_key)[0])
    return min(final_heads)


def aggregate_global_cluster_representatives(
    selection_results: List[Dict[str, Any]],
    global_assignments: Dict[str, Any],
    num_layers: int,
    n_clusters: int,
) -> Dict[str, Any]:
    """Vote on representative heads per global cluster; vote count first, score sum tie-break."""
    ref = selection_results[0]
    ref_layer_clusters = ref["layer_clusters"]
    global_clusters = global_assignments["layer_clusters"]

    rep_vote_count: Dict[int, Dict[int, Dict[int, int]]] = {
        layer: {c: {} for c in range(n_clusters)} for layer in range(num_layers)
    }
    rep_score_sum: Dict[int, Dict[int, Dict[int, float]]] = {
        layer: {c: {} for c in range(n_clusters)} for layer in range(num_layers)
    }

    for sample_idx, result in enumerate(selection_results):
        for layer_idx in range(num_layers):
            if sample_idx == 0:
                aligned_clusters = result["layer_clusters"][layer_idx]
                aligned_reps = result["layer_cluster_reps"][layer_idx]
            else:
                aligned_clusters = align_cluster_labels_to_reference(
                    ref_layer_clusters[layer_idx],
                    result["layer_clusters"][layer_idx],
                    n_clusters,
                    layer_idx=layer_idx,
                    sample_idx=sample_idx,
                )
                raw_reps = result["layer_cluster_reps"][layer_idx]
                aligned_reps = {}
                sample_clusters = result["layer_clusters"][layer_idx]
                label_map: Dict[int, int] = {}
                for rc in range(n_clusters):
                    ref_set = set(ref_layer_clusters[layer_idx].get(rc, []))
                    best_sc, best_sc_id = -1.0, rc
                    for sc in range(n_clusters):
                        score = _jaccard_sets(ref_set, set(sample_clusters.get(sc, [])))
                        if score > best_sc:
                            best_sc, best_sc_id = score, sc
                    label_map[best_sc_id] = rc
                for sc, stats in raw_reps.items():
                    aligned_reps[label_map.get(int(sc), int(sc))] = stats

            for cid in range(n_clusters):
                stats = aligned_reps.get(cid, {})
                rep = stats.get("representative_head")
                score = float(stats.get("representative_score", 0.0))
                if rep is not None:
                    rep = int(rep)
                    rep_vote_count[layer_idx][cid][rep] = (
                        rep_vote_count[layer_idx][cid].get(rep, 0) + 1
                    )
                    rep_score_sum[layer_idx][cid][rep] = (
                        rep_score_sum[layer_idx][cid].get(rep, 0.0) + score
                    )

    global_reps: Dict[int, Dict[int, int]] = {}
    detail: Dict[str, Any] = {}

    for layer_idx in range(num_layers):
        global_reps[layer_idx] = {}
        detail[str(layer_idx)] = {}
        for cid in range(n_clusters):
            final_heads = set(global_clusters[layer_idx].get(cid, []))
            if not final_heads:
                all_layer_heads = sorted(
                    h
                    for c2 in range(n_clusters)
                    for h in global_clusters[layer_idx].get(c2, [])
                )
                if not all_layer_heads:
                    all_layer_heads = list(range(max(rep_vote_count[layer_idx][0].keys(), default=0) + 1))
                mid = len(all_layer_heads) // 2
                if cid == 0:
                    final_heads = set(all_layer_heads[:mid] or all_layer_heads[:1])
                else:
                    final_heads = set(all_layer_heads[mid:] or all_layer_heads[-1:])
                logger.warning(
                    "Layer %d cluster %d empty; fallback split heads=%s",
                    layer_idx,
                    cid,
                    sorted(final_heads),
                )

            candidates: Dict[int, Tuple[int, float]] = {}
            all_heads_in_layer = set()
            for c2 in range(n_clusters):
                all_heads_in_layer.update(global_clusters[layer_idx].get(c2, []))
            for rep, votes in rep_vote_count[layer_idx][cid].items():
                candidates[rep] = (votes, rep_score_sum[layer_idx][cid].get(rep, 0.0))

            if not candidates:
                global_reps[layer_idx][cid] = min(final_heads)
            else:
                best_head = _select_best_representative(candidates, final_heads)
                global_reps[layer_idx][cid] = int(best_head)

            detail[str(layer_idx)][str(cid)] = {
                "representative_head": global_reps[layer_idx][cid],
                "candidate_vote_counts": {
                    str(k): v for k, v in rep_vote_count[layer_idx][cid].items()
                },
                "candidate_score_sums": {
                    str(k): v for k, v in rep_score_sum[layer_idx][cid].items()
                },
                "cluster_heads": sorted(final_heads),
            }

    return {
        "layer_cluster_representatives": global_reps,
        "per_cluster_vote_detail": detail,
        "num_selection_samples": len(selection_results),
    }


# ---------------------------------------------------------------------------
# Cluster mask building
# ---------------------------------------------------------------------------


def build_layer_cluster_masks(
    attentions: torch.Tensor,
    global_cluster_reps: Dict[int, Dict[int, int]],
    mask_builder: Any,
    query_abs_positions: Optional[torch.Tensor] = None,
) -> Dict[int, Dict[int, torch.Tensor]]:
    """
    Build per-layer per-cluster masks from representative head attentions.

    Returns:
        layer_to_cluster_masks[layer][cluster] -> bool mask [last_q, seq_len]
    """
    num_layers = attentions.shape[0]
    out: Dict[int, Dict[int, torch.Tensor]] = {}
    for layer_idx in range(num_layers):
        out[layer_idx] = {}
        for cid, rep_head in global_cluster_reps[layer_idx].items():
            rep_attn = attentions[layer_idx, rep_head].to(torch.float32)
            out[layer_idx][cid] = mask_builder.build(
                rep_attn, query_abs_positions=query_abs_positions
            )
    return out


def build_layer_to_head_cluster_from_global(
    global_assignments: Dict[str, Any],
) -> Dict[int, Dict[int, int]]:
    return global_assignments["layer_to_head_cluster"]


# ---------------------------------------------------------------------------
# Cluster-aware sparse attention controller
# ---------------------------------------------------------------------------


@dataclass
class ClusterSharedMaskAttentionController:
    layer_to_cluster_masks: Dict[int, Dict[int, torch.Tensor]]
    layer_to_head_cluster: Dict[int, Dict[int, int]]
    apply_prefill: bool
    apply_decode: bool
    last_q: int
    analysis_seq_len: int

    def should_apply(self, stage: str, layer_idx: int) -> bool:
        if layer_idx not in self.layer_to_cluster_masks:
            return False
        if stage == "prefill":
            return self.apply_prefill
        if stage == "decode":
            return self.apply_decode
        return False

    def _base_mask_for_head(self, layer_idx: int, head_idx: int) -> Optional[torch.Tensor]:
        head_map = self.layer_to_head_cluster.get(layer_idx, {})
        cluster_masks = self.layer_to_cluster_masks.get(layer_idx, {})
        cid = head_map.get(head_idx)
        if cid is None:
            return None
        return cluster_masks.get(cid)

    def get_runtime_head_masks(
        self,
        layer_idx: int,
        ctx: AttentionForwardContext,
        *,
        actual_q_len: Optional[int] = None,
        actual_kv_len: Optional[int] = None,
        num_heads: int = 1,
    ) -> Optional[torch.Tensor]:
        """Build per-head bool masks [num_heads, q_len, kv_len]."""
        if not self.should_apply(ctx.stage, layer_idx):
            return None

        q_len = actual_q_len if actual_q_len is not None else ctx.q_len
        kv_len = actual_kv_len if actual_kv_len is not None else ctx.kv_len

        any_mask = next(iter(next(iter(self.layer_to_cluster_masks.values())).values()))
        device = any_mask.device
        head_masks = torch.ones((num_heads, q_len, kv_len), dtype=torch.bool, device=device)

        for head_idx in range(num_heads):
            base_mask = self._base_mask_for_head(layer_idx, head_idx)
            if base_mask is None:
                continue

            runtime = torch.ones((q_len, kv_len), dtype=torch.bool, device=device)

            if ctx.stage == "prefill":
                for local_q in range(q_len):
                    abs_q = ctx.query_abs_start + local_q
                    if abs_q < self.analysis_seq_len - self.last_q:
                        continue
                    rel = abs_q - (self.analysis_seq_len - self.last_q)
                    if 0 <= rel < base_mask.shape[0]:
                        row = base_mask[rel]
                        copy_len = min(row.shape[0], kv_len)
                        runtime[local_q, :copy_len] = row[:copy_len]

            elif ctx.stage == "decode":
                last_row = base_mask[-1]
                hist_len = min(last_row.shape[0], kv_len)
                runtime[0, :hist_len] = last_row[:hist_len]
                if kv_len > hist_len:
                    runtime[0, hist_len:kv_len] = True

            for local_q in range(q_len):
                abs_q = ctx.query_abs_start + local_q
                if abs_q + 1 < kv_len:
                    runtime[local_q, abs_q + 1 :] = False

            head_masks[head_idx] = runtime

        return head_masks


_CLUSTER_PATCH_STATE: Dict[str, Any] = {
    "patched": False,
    "original_fn": None,
    "controller": None,
    "context": None,
    "repeat_kv": None,
}


def apply_per_head_sparse_mask_to_attention_scores(
    attn_scores: torch.Tensor,
    head_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Apply per-head sparse masks before softmax.

    attn_scores: [batch, num_heads, q_len, kv_len]
    head_masks: [num_heads, q_len, kv_len] bool
    """
    if head_masks is None:
        return attn_scores

    b, nh, q_len, kv_len = attn_scores.shape
    mask = head_masks[:nh, :q_len, :kv_len]
    if mask.shape[-1] < kv_len:
        pad = torch.ones(
            (mask.shape[0], mask.shape[1], kv_len - mask.shape[-1]),
            dtype=torch.bool,
            device=mask.device,
        )
        mask = torch.cat([mask, pad], dim=-1)
    mask = mask.unsqueeze(0).expand(b, nh, q_len, kv_len)
    finfo = torch.finfo(attn_scores.dtype)
    return attn_scores.masked_fill(~mask, finfo.min)


def _patched_cluster_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    repeat_kv = _CLUSTER_PATCH_STATE["repeat_kv"]
    controller: Optional[ClusterSharedMaskAttentionController] = _CLUSTER_PATCH_STATE[
        "controller"
    ]
    ctx: Optional[AttentionForwardContext] = _CLUSTER_PATCH_STATE["context"]

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    layer_idx = getattr(module, "layer_idx", None)
    if (
        controller is not None
        and ctx is not None
        and layer_idx is not None
        and controller.should_apply(ctx.stage, layer_idx)
    ):
        head_masks = controller.get_runtime_head_masks(
            layer_idx,
            ctx,
            actual_q_len=attn_weights.shape[-2],
            actual_kv_len=attn_weights.shape[-1],
            num_heads=attn_weights.shape[1],
        )
        if head_masks is not None:
            attn_weights = apply_per_head_sparse_mask_to_attention_scores(
                attn_weights, head_masks.to(attn_weights.device)
            )

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def patch_model_attention_cluster(
    model,
    controller: ClusterSharedMaskAttentionController,
) -> None:
    if _CLUSTER_PATCH_STATE["patched"]:
        _CLUSTER_PATCH_STATE["controller"] = controller
        return

    from transformers.models.qwen2.modeling_qwen2 import (
        eager_attention_forward,
        repeat_kv,
    )
    import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2

    _CLUSTER_PATCH_STATE["original_fn"] = eager_attention_forward
    _CLUSTER_PATCH_STATE["repeat_kv"] = repeat_kv
    _CLUSTER_PATCH_STATE["controller"] = controller
    modeling_qwen2.eager_attention_forward = _patched_cluster_eager_attention_forward
    _CLUSTER_PATCH_STATE["patched"] = True
    logger.info(
        "Patched Qwen2 attention for cluster masks (prefill=%s, decode=%s)",
        controller.apply_prefill,
        controller.apply_decode,
    )


def unpatch_model_attention_cluster(model) -> None:
    if not _CLUSTER_PATCH_STATE["patched"]:
        return
    import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2

    if _CLUSTER_PATCH_STATE["original_fn"] is not None:
        modeling_qwen2.eager_attention_forward = _CLUSTER_PATCH_STATE["original_fn"]
    _CLUSTER_PATCH_STATE["patched"] = False
    _CLUSTER_PATCH_STATE["original_fn"] = None
    _CLUSTER_PATCH_STATE["controller"] = None
    _CLUSTER_PATCH_STATE["context"] = None


def set_cluster_attention_forward_context(
    ctx: Optional[AttentionForwardContext],
) -> None:
    _CLUSTER_PATCH_STATE["context"] = ctx


# ---------------------------------------------------------------------------
# Cluster layer stats
# ---------------------------------------------------------------------------


def compute_cluster_layer_stats(
    attentions: torch.Tensor,
    layer_to_cluster_masks: Dict[int, Dict[int, torch.Tensor]],
    layer_to_head_cluster: Dict[int, Dict[int, int]],
    cluster_reps: Dict[int, Dict[int, int]],
    cluster_rep_stats: Optional[Dict[int, Dict[int, Dict[str, Any]]]] = None,
    query_abs_positions: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    attentions = normalize_attention(attentions)
    num_layers, num_heads, last_q, seq_len = attentions.shape
    device = attentions.device

    if query_abs_positions is None:
        query_abs_positions = query_abs_positions_from_last_q(seq_len, last_q).to(device)

    valid = build_causal_valid_mask(last_q, seq_len, query_abs_positions, device=device)
    layer_stats: Dict[str, Any] = {}

    for layer_idx in range(num_layers):
        clusters_info: Dict[str, Any] = {}
        head_cluster = layer_to_head_cluster.get(layer_idx, {})

        cluster_to_heads: Dict[int, List[int]] = {}
        for h, cid in head_cluster.items():
            cluster_to_heads.setdefault(cid, []).append(h)

        for cid, heads in sorted(cluster_to_heads.items()):
            mask = layer_to_cluster_masks[layer_idx][cid].to(device)
            kept = int((mask & valid).sum().item())
            total = int(valid.sum().item())
            keep_ratio = kept / total if total > 0 else 0.0

            coverages: List[float] = []
            for h in heads:
                attn_h = attentions[layer_idx, h]
                num = float((attn_h * mask * valid).sum().item())
                den = float((attn_h * valid).sum().item())
                coverages.append(num / den if den > 0 else 0.0)

            cov_t = torch.tensor(coverages, dtype=torch.float32)
            rep = cluster_reps[layer_idx][cid]
            entry: Dict[str, Any] = {
                "heads": sorted(heads),
                "representative_head": rep,
                "mean_coverage": float(cov_t.mean().item()) if coverages else 0.0,
                "min_coverage": float(cov_t.min().item()) if coverages else 0.0,
                "std_coverage": float(cov_t.std(unbiased=False).item()) if coverages else 0.0,
                "keep_ratio": keep_ratio,
                "sparsity": 1.0 - keep_ratio,
            }
            if cluster_rep_stats and layer_idx in cluster_rep_stats:
                rs = cluster_rep_stats[layer_idx].get(cid, {})
                entry["representative_score"] = rs.get("representative_score", 0.0)
            clusters_info[str(cid)] = entry

        layer_stats[str(layer_idx)] = {"clusters": clusters_info}

    return layer_stats


def _smoke_test_cluster_methods() -> None:
    """Minimal sanity check for graph2vec / svd_kmeans / bmm clustering paths."""
    from types import SimpleNamespace

    last_q, seq_len, num_heads = 32, 512, 16
    dummy = torch.rand(num_heads, last_q, seq_len)
    dummy = dummy / dummy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    query_abs_positions = torch.arange(seq_len - last_q, seq_len)

    base_args = dict(
        binarize_method="top_p",
        binarize_top_p=0.95,
        binarize_top_k=128,
        binarize_threshold=0.0,
        graph_type="bipartite",
        local_window=256,
        sink_tokens=4,
        num_head_clusters=2,
        cluster_seed=42,
        graph2vec_dim=32,
        graph2vec_wl_iterations=2,
        graph2vec_workers=1,
        svd_components=8,
        bmm_max_iter=50,
        bmm_tol=1e-4,
        bmm_n_init=2,
    )

    for method in ("graph2vec", "svd_kmeans", "bmm"):
        args = SimpleNamespace(**base_args, cluster_method=method)
        labels, embeddings, graphs, stats = cluster_layer_heads_by_method(
            dummy,
            query_abs_positions,
            args,
            layer_idx=0,
        )
        binary_maps = build_layer_binary_maps(dummy, query_abs_positions, args)
        assert labels.shape == (num_heads,), f"{method}: labels shape {labels.shape}"
        assert len(binary_maps) == num_heads, f"{method}: binary_maps len"
        assert embeddings.shape[0] == num_heads, f"{method}: embeddings rows"
        assert "labels_before_fix" in stats, f"{method}: missing labels_before_fix"
        assert "labels_after_fix" in stats, f"{method}: missing labels_after_fix"
        if method == "graph2vec":
            assert embeddings.shape[1] == args.graph2vec_dim
            assert len(graphs) == num_heads
        elif method == "svd_kmeans":
            assert embeddings.shape[1] == stats.get("svd_components_used", embeddings.shape[1])
        elif method == "bmm":
            assert embeddings.shape[1] == args.num_head_clusters
        print(
            f"[smoke] {method}: labels={labels.tolist()} "
            f"embeddings={embeddings.shape} backend={stats.get('graph_embedding_backend', method)}"
        )


if __name__ == "__main__":
    _smoke_test_cluster_methods()
