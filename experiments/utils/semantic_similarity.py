"""
Semantic Similarity Utilities

Provides embedding-based semantic similarity computation for self-consistency
clustering in ECR baseline experiments.

Uses sentence-transformers for efficient text embeddings.
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# Lazy loading for sentence-transformers (heavy import)
_embedder = None


def _get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load the sentence transformer model."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for semantic similarity. "
                "Install with: pip install sentence-transformers"
            )
    return _embedder


def compute_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Compute sentence embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        model_name: Name of the sentence-transformer model to use

    Returns:
        numpy array of shape (n_texts, embedding_dim)
    """
    if not texts:
        return np.array([])

    embedder = _get_embedder(model_name)
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    return embeddings


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings: numpy array of shape (n_texts, embedding_dim)

    Returns:
        Similarity matrix of shape (n_texts, n_texts) with values in [-1, 1]
    """
    if embeddings.size == 0:
        return np.array([[]])

    return cosine_similarity(embeddings)


def cluster_by_similarity(
    texts: List[str],
    threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2"
) -> List[List[int]]:
    """
    Cluster texts by semantic similarity using agglomerative clustering.

    Args:
        texts: List of text strings to cluster
        threshold: Similarity threshold for clustering (0-1, higher = stricter)
        model_name: Name of the sentence-transformer model

    Returns:
        List of clusters, where each cluster is a list of indices into the
        original texts list. Clusters are sorted by size (largest first).
    """
    if len(texts) == 0:
        return []

    if len(texts) == 1:
        return [[0]]

    # Compute embeddings and similarity
    embeddings = compute_embeddings(texts, model_name)
    similarity_matrix = compute_similarity_matrix(embeddings)

    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix

    # Ensure distance matrix is valid (no negative values due to numerical issues)
    distance_matrix = np.clip(distance_matrix, 0, 2)

    # Agglomerative clustering with distance threshold
    # distance_threshold = 1 - similarity_threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        metric='precomputed',
        linkage='average'
    )

    labels = clustering.fit_predict(distance_matrix)

    # Group indices by cluster label
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    # Sort clusters by size (largest first)
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    return sorted_clusters


def find_cluster_centroid(
    cluster_indices: List[int],
    embeddings: np.ndarray
) -> int:
    """
    Find the centroid (most representative element) of a cluster.

    The centroid is the element with highest average similarity to all other
    elements in the cluster.

    Args:
        cluster_indices: List of indices belonging to the cluster
        embeddings: Full embedding matrix for all texts

    Returns:
        Index of the centroid element (from original indices)
    """
    if len(cluster_indices) == 0:
        raise ValueError("Cannot find centroid of empty cluster")

    if len(cluster_indices) == 1:
        return cluster_indices[0]

    # Get embeddings for cluster members
    cluster_embeddings = embeddings[cluster_indices]

    # Compute pairwise similarity within cluster
    similarity_matrix = cosine_similarity(cluster_embeddings)

    # Find element with highest average similarity to others
    avg_similarities = similarity_matrix.mean(axis=1)
    centroid_local_idx = np.argmax(avg_similarities)

    return cluster_indices[centroid_local_idx]


def select_by_semantic_majority(
    candidates: List[str],
    similarity_threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2"
) -> Tuple[str, dict]:
    """
    Select the most representative candidate using semantic majority voting.

    This implements the core of self-consistency sampling:
    1. Cluster candidates by semantic similarity
    2. Find the largest cluster (majority)
    3. Return the centroid of the majority cluster

    Args:
        candidates: List of candidate response strings
        similarity_threshold: Threshold for considering responses similar (0-1)
        model_name: Sentence transformer model for embeddings

    Returns:
        Tuple of (selected_candidate, debug_info) where debug_info contains:
        - num_clusters: Number of distinct clusters found
        - majority_cluster_size: Size of the largest cluster
        - cluster_distribution: List of cluster sizes
        - selected_idx: Index of selected candidate
    """
    if not candidates:
        raise ValueError("Cannot select from empty candidate list")

    if len(candidates) == 1:
        return candidates[0], {
            'num_clusters': 1,
            'majority_cluster_size': 1,
            'cluster_distribution': [1],
            'selected_idx': 0
        }

    # Compute embeddings once
    embeddings = compute_embeddings(candidates, model_name)

    # Cluster candidates
    clusters = cluster_by_similarity(candidates, similarity_threshold, model_name)

    # Get majority cluster (first one, since sorted by size)
    majority_cluster = clusters[0]

    # Find centroid of majority cluster
    selected_idx = find_cluster_centroid(majority_cluster, embeddings)

    debug_info = {
        'num_clusters': len(clusters),
        'majority_cluster_size': len(majority_cluster),
        'cluster_distribution': [len(c) for c in clusters],
        'selected_idx': selected_idx
    }

    return candidates[selected_idx], debug_info
