"""Sinkhorn helpers for popularity-balanced RQ-KMeans."""

from __future__ import annotations

import numpy as np


def sinkhorn(
    cost: np.ndarray,
    row_marginal: np.ndarray,
    col_marginal: np.ndarray,
    epsilon: float,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Entropy-regularized optimal transport plan Q [N, K].

    Row sums match row_marginal, column sums match col_marginal.
    """
    cost = np.asarray(cost, dtype=np.float64)
    row = np.asarray(row_marginal, dtype=np.float64)
    col = np.asarray(col_marginal, dtype=np.float64)
    row = row / row.sum()
    col = col / col.sum()

    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    kernel = np.exp(-cost / epsilon)
    kernel = np.maximum(kernel, 1e-300)

    u = np.ones(cost.shape[0], dtype=np.float64)
    v = np.ones(cost.shape[1], dtype=np.float64)
    for _ in range(max_iter):
        u_prev = u
        u = row / (kernel @ v)
        v = col / (kernel.T @ u)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    return u[:, None] * kernel * v[None, :]


def natural_cluster_mass(
    codes: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Natural cluster mass: item-count share of each cluster after hard assignment."""
    counts = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.full(n_clusters, 1.0 / n_clusters, dtype=np.float64)
    return counts / total


def column_marginal_from_natural(
    natural_mass: np.ndarray,
    balance_lambda: float,
    n_clusters: int,
) -> np.ndarray:
    """b = (1-lambda)*p_nat + lambda*(1/K)."""
    if not 0.0 <= balance_lambda <= 1.0:
        raise ValueError(f"balance_lambda must be in [0, 1], got {balance_lambda}")
    uniform = np.full(n_clusters, 1.0 / n_clusters, dtype=np.float64)
    return (1.0 - balance_lambda) * natural_mass + balance_lambda * uniform


def cluster_popularity_mass(
    codes: np.ndarray,
    item_weights: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Total popularity mass assigned to each cluster."""
    mass = np.zeros(n_clusters, dtype=np.float64)
    np.add.at(mass, codes, item_weights)
    return mass
