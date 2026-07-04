"""Lightweight Graph2Vec: directed WL subtree features + Doc2Vec (Narayanan et al., 2017)."""

from __future__ import annotations

from typing import List, Sequence

import networkx as nx
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans


def binary_matrix_to_digraph(binary: np.ndarray) -> nx.DiGraph:
    """Build directed head graph: edge i->j if binary[i,j]==1."""
    n = binary.shape[0]
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and binary[i, j]:
                g.add_edge(i, j, weight=1)
    return g


def head_ego_graph(g: nx.DiGraph, head: int, *, radius: int = 2) -> nx.DiGraph:
    """
    Ego network for one head on a directed graph.

    Node set is chosen by undirected hop distance (in+out neighborhood),
    but induced subgraph preserves original directed edges only.
    """
    if head not in g:
        return nx.DiGraph()

    if radius <= 0:
        sub = nx.DiGraph()
        sub.add_node(head)
        return sub

    undirected = g.to_undirected()
    nodes = nx.single_source_shortest_path_length(
        undirected, head, cutoff=radius
    ).keys()

    return g.subgraph(nodes).copy()


def _initial_labels(g: nx.DiGraph) -> dict[int, str]:
    return {
        node: f"in{g.in_degree(node)}_out{g.out_degree(node)}"
        for node in g.nodes()
    }


def _wl_step(g: nx.DiGraph, labels: dict[int, str]) -> dict[int, str]:
    new_labels: dict[int, str] = {}

    for node in g.nodes():
        in_labels = sorted(labels[n] for n in g.predecessors(node))
        out_labels = sorted(labels[n] for n in g.successors(node))

        in_part = "_".join(in_labels) if in_labels else "none"
        out_part = "_".join(out_labels) if out_labels else "none"

        new_labels[node] = f"{labels[node]}|IN:{in_part}|OUT:{out_part}"

    return new_labels


def wl_subtree_words(
    g: nx.DiGraph,
    *,
    wl_iterations: int = 2,
    rooted_at: int | None = None,
) -> List[str]:
    """
    Extract anonymous directed WL subtree tokens for Graph2Vec.

    Does not encode raw head / node ids in tokens.
    """
    if g.number_of_nodes() == 0:
        return ["EMPTY"]

    if rooted_at is not None:
        nodes = [rooted_at] if rooted_at in g else []
    else:
        nodes = list(g.nodes())

    if not nodes:
        return ["EMPTY"]

    words: List[str] = []
    labels = _initial_labels(g)

    for node in nodes:
        words.append(f"t0_{labels[node]}")

    for t in range(wl_iterations):
        labels = _wl_step(g, labels)
        for node in nodes:
            words.append(f"t{t + 1}_{labels[node]}")

    return words


def graph_to_tagged_document(
    g: nx.DiGraph,
    tag: str,
    *,
    wl_iterations: int = 2,
    rooted_at: int | None = None,
) -> TaggedDocument:
    words = wl_subtree_words(g, wl_iterations=wl_iterations, rooted_at=rooted_at)
    return TaggedDocument(words=words, tags=[tag])


def embed_graphs_graph2vec(
    graphs: Sequence[nx.DiGraph],
    *,
    root_nodes: Sequence[int] | None = None,
    vector_size: int = 32,
    wl_iterations: int = 2,
    epochs: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """
    Graph2Vec embeddings for a list of graphs (one graph per head ego network).

    Returns (n_graphs, vector_size) array aligned with input order.
    """
    if not graphs:
        return np.zeros((0, vector_size), dtype=np.float32)

    if root_nodes is None:
        root_nodes = [None] * len(graphs)

    docs = [
        graph_to_tagged_document(
            g,
            tag=str(i),
            wl_iterations=wl_iterations,
            rooted_at=root,
        )
        for i, (g, root) in enumerate(zip(graphs, root_nodes))
    ]

    model = Doc2Vec(
        vector_size=vector_size,
        min_count=0,
        epochs=epochs,
        seed=seed,
        workers=1,
        dm=1,  # PV-DM
    )
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)

    embeddings = np.stack(
        [model.dv[str(i)] for i in range(len(graphs))],
        axis=0,
    ).astype(np.float32)
    return embeddings


def cluster_head_embeddings(
    embeddings: np.ndarray,
    *,
    k: int = 2,
    random_state: int | None = None,
) -> np.ndarray:
    if embeddings.shape[0] == 0:
        return np.array([], dtype=int)
    if embeddings.shape[0] < k:
        return np.arange(1, embeddings.shape[0] + 1, dtype=int)
    labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(
        embeddings
    )
    return labels + 1


def graph2vec_head_clustering(
    binary: np.ndarray,
    *,
    k: int = 2,
    ego_radius: int = 2,
    vector_size: int = 32,
    wl_iterations: int = 2,
    epochs: int = 200,
    random_state: int = 0,
) -> dict:
    """
    Graph2Vec pipeline for one layer's binarized head similarity matrix.

    Each head -> directed ego graph -> Graph2Vec embedding -> KMeans.
    """
    binary = np.asarray(binary).copy()
    if binary.ndim != 2 or binary.shape[0] != binary.shape[1]:
        raise ValueError(f"binary must be a square matrix, got shape={binary.shape}")

    np.fill_diagonal(binary, 0)
    n = binary.shape[0]

    base = binary_matrix_to_digraph(binary)
    ego_graphs = [
        head_ego_graph(base, head, radius=ego_radius)
        for head in range(n)
    ]
    embeddings = embed_graphs_graph2vec(
        ego_graphs,
        root_nodes=list(range(n)),
        vector_size=vector_size,
        wl_iterations=wl_iterations,
        epochs=epochs,
        seed=random_state,
    )
    graph2vec_labels = cluster_head_embeddings(
        embeddings, k=k, random_state=random_state
    )

    row_features = binary.astype(float)
    binary_row_labels = cluster_head_embeddings(
        row_features, k=k, random_state=random_state
    )

    row_col_features = np.concatenate([binary, binary.T], axis=1).astype(float)
    binary_row_col_labels = cluster_head_embeddings(
        row_col_features, k=k, random_state=random_state
    )

    return {
        "graph2vec_labels": graph2vec_labels.tolist(),
        "binary_row_labels": binary_row_labels.tolist(),
        "binary_row_col_labels": binary_row_col_labels.tolist(),
        "graph2vec_embeddings": embeddings,
        "Graph2Vec": {
            "labels": graph2vec_labels.tolist(),
            "embedding_shape": list(embeddings.shape),
        },
        "BinaryRow-KMeans": {
            "labels": binary_row_labels.tolist(),
            "embedding_shape": [n, n],
        },
        "BinaryRowCol-KMeans": {
            "labels": binary_row_col_labels.tolist(),
            "embedding_shape": [n, 2 * n],
        },
        "meta": {
            "ego_radius": ego_radius,
            "vector_size": vector_size,
            "wl_iterations": wl_iterations,
            "epochs": epochs,
        },
    }
