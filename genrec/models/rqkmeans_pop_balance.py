
from __future__ import annotations

from collections import Counter
from typing import List, Literal

import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus

from genrec.models.rqkmeans import RqKmeans, RqKmeansOutput
from genrec.modules.sinkhorn import (
    cluster_popularity_mass,
    column_marginal_from_natural,
    natural_cluster_mass,
    sinkhorn,
)


BalanceScope = Literal["first", "second", "third", "all"]


class RqKmeansPopBalance(RqKmeans):

    def __init__(
        self,
        n_layers: int = 3,
        codebook_size: int = 256,
        random_state: int = 42,
        max_iter: int = 100,
        batch_size: int = 4096,
        backend: str = "auto",
        n_init: int = 10,
        dead_reinit: bool = True,
        balance_scope: BalanceScope = "first",
        balance_lambda: float = 1.0,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_max_iter: int = 100,
    ):
        super().__init__(
            n_layers=n_layers,
            codebook_size=codebook_size,
            random_state=random_state,
            max_iter=max_iter,
            batch_size=batch_size,
            backend=backend,
            n_init=n_init,
            dead_reinit=dead_reinit,
        )
        self.balance_scope = balance_scope
        self.balance_lambda = balance_lambda
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.item_weights: np.ndarray | None = None
        # Set by trainer: "popularity" (log-pop), "raw" (n_i), or "count" (uniform).
        self.weight_mode: str = "popularity"
        # Set by trainer: "ot" (embedding cost) or "random" (balanced-but-random).
        self.assignment_mode: str = "ot"

    def _use_balance(self, layer: int) -> bool:
        if self.item_weights is None:
            return False
        if self.balance_scope == "all":
            return True
        if self.balance_scope == "first":
            return layer == 0
        if self.balance_scope == "second":
            return layer == 1
        if self.balance_scope == "third":
            return layer == 2
        return False

    @staticmethod
    def _squared_distances(residual: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        x_sq = (residual ** 2).sum(axis=1, keepdims=True)
        c_sq = (centroids ** 2).sum(axis=1, keepdims=True).T
        return x_sq + c_sq - 2 * residual @ centroids.T

    def _reinit_dead_clusters(
        self,
        residual: np.ndarray,
        centroids: np.ndarray,
        codes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.dead_reinit:
            return centroids, codes

        used = set(codes.tolist())
        dead = [c for c in range(self.codebook_size) if c not in used]
        if not dead:
            return centroids, codes

        err = ((residual - centroids[codes]) ** 2).sum(axis=1)
        far_idx = np.argsort(-err)
        n_items = len(residual)
        for i, cluster_id in enumerate(dead):
            centroids[cluster_id] = residual[far_idx[i % n_items]]
        dist = self._squared_distances(residual, centroids)
        codes = dist.argmin(axis=1)
        return centroids, codes

    def _balance_mode_tag(self) -> str:
        if self.assignment_mode == "random":
            return "rand-balance"
        if self.weight_mode == "count":
            return "count-balance"
        if self.weight_mode == "raw":
            return "raw-balance"
        return "pop-balance"

    def _assignment_cost(
        self,
        residual: np.ndarray,
        centroids: np.ndarray,
        layer: int,
        step: int,
    ) -> np.ndarray:
        """Cost matrix for Sinkhorn: embedding distance, or random (semantic control)."""
        if self.assignment_mode == "random":
            rng = np.random.RandomState(self.random_state + 10007 * (layer + 1) + step)
            return rng.random((len(residual), self.codebook_size)).astype(np.float64)
        return self._squared_distances(residual, centroids)

    def _fit_balanced_layer(
        self,
        layer: int,
        residual: np.ndarray,
        item_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_items = len(residual)
        centroids, _ = kmeans_plusplus(
            residual,
            n_clusters=self.codebook_size,
            random_state=self.random_state + layer,
        )

        dist = self._assignment_cost(residual, centroids, layer, step=0)
        init_codes = dist.argmin(axis=1)
        p_nat = natural_cluster_mass(init_codes, self.codebook_size)
        col_marginal = column_marginal_from_natural(
            p_nat, self.balance_lambda, self.codebook_size
        )

        mode_tag = self._balance_mode_tag()
        print(
            f"  Layer {layer} [{mode_tag}]: lambda={self.balance_lambda}, "
            f"epsilon={self.sinkhorn_epsilon}, col_uniform={self.balance_lambda:.2f}, "
            f"assignment={self.assignment_mode}, p_nat=cluster_item_count "
            f"(count_max={p_nat.max():.4f})"
        )

        for step in range(self.max_iter):
            prev_centroids = centroids.copy()
            dist = self._assignment_cost(residual, centroids, layer, step=step)
            transport = sinkhorn(
                dist,
                row_marginal=item_weights,
                col_marginal=col_marginal,
                epsilon=self.sinkhorn_epsilon,
                max_iter=self.sinkhorn_max_iter,
            )
            denom = transport.sum(axis=0, keepdims=True).T
            denom = np.maximum(denom, 1e-12)
            centroids = (transport.T @ residual) / denom

            shift = np.linalg.norm(centroids - prev_centroids)
            if step > 0 and shift < 1e-4:
                break

        codes = transport.argmax(axis=1)
        # Distance-based dead reinit would reintroduce semantics for rand-balance.
        if self.assignment_mode != "random":
            centroids, codes = self._reinit_dead_clusters(residual, centroids, codes)
        return centroids, codes

    def _fit_standard_layer(self, layer: int, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        backend = self.backend
        n_items = len(residual)
        if backend == "auto":
            backend = "full" if n_items * self.codebook_size < 5e7 else "minibatch"

        if backend == "full":
            kmeans = KMeans(
                n_clusters=self.codebook_size,
                init="k-means++",
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=0,
            )
        else:
            from sklearn.cluster import MiniBatchKMeans

            kmeans = MiniBatchKMeans(
                n_clusters=self.codebook_size,
                random_state=self.random_state,
                max_iter=self.max_iter,
                batch_size=min(self.batch_size, n_items),
                n_init=self.n_init,
                verbose=0,
            )
        kmeans.fit(residual)
        centroids = kmeans.cluster_centers_.copy()
        codes = kmeans.predict(residual)
        centroids, codes = self._reinit_dead_clusters(residual, centroids, codes)
        return centroids, codes

    def fit(
        self,
        embeddings: np.ndarray,
        item_weights: np.ndarray | None = None,
    ) -> "RqKmeansPopBalance":
        """
        Fit popularity-balanced RQ-KMeans.

        Args:
            embeddings: [N, D]
            item_weights: normalized popularity mass [N]; required when balancing.
        """
        if item_weights is not None:
            item_weights = np.asarray(item_weights, dtype=np.float64)
            if len(item_weights) != len(embeddings):
                raise ValueError(
                    "item_weights length must match embeddings: "
                    f"{len(item_weights)} vs {len(embeddings)}"
                )
            item_weights = item_weights / item_weights.sum()
        self.item_weights = item_weights

        self.centroids = []
        residual = embeddings.copy()
        n_items = len(embeddings)

        for layer in range(self.n_layers):
            balanced = self._use_balance(layer)
            mode = self._balance_mode_tag() if balanced else self.backend
            print(
                f"Fitting layer {layer}/{self.n_layers} [{mode}], "
                f"N={n_items}, K={self.codebook_size}, "
                f"residual norm: {np.linalg.norm(residual, axis=1).mean():.4f}"
            )

            if balanced:
                centroids, codes = self._fit_balanced_layer(
                    layer, residual, item_weights
                )
            else:
                centroids, codes = self._fit_standard_layer(layer, residual)

            self.centroids.append(centroids)
            residual = residual - centroids[codes]

            cnt = Counter(codes.tolist())
            used_count = len(cnt)
            top_freq = max(cnt.values())
            print(
                f"  Layer {layer} usage: {used_count}/{self.codebook_size} "
                f"({100 * used_count / self.codebook_size:.1f}%), "
                f"max_freq={top_freq} ({100 * top_freq / n_items:.1f}% of items)"
            )
            if balanced and item_weights is not None:
                mass = cluster_popularity_mass(codes, item_weights, self.codebook_size)
                active = mass[mass > 0]
                cv = active.std() / active.mean() if active.mean() > 0 else 0.0
                if self.weight_mode == "count":
                    mass_label = "count-mass"
                elif self.weight_mode == "raw":
                    mass_label = "raw-mass"
                else:
                    mass_label = "pop-mass"
                print(
                    f"  Layer {layer} {mass_label}: min={active.min():.6f}, "
                    f"max={active.max():.6f}, cv={cv:.4f}"
                )

        print(f"Final residual norm: {np.linalg.norm(residual, axis=1).mean():.4f}")
        return self

    def save(self, path: str) -> None:
        import torch

        torch.save(
            {
                "model_type": "rqkmeans_pop_balance",
                "n_layers": self.n_layers,
                "codebook_size": self.codebook_size,
                "centroids": [c.copy() for c in self.centroids],
                "balance_scope": self.balance_scope,
                "balance_lambda": self.balance_lambda,
                "sinkhorn_epsilon": self.sinkhorn_epsilon,
                "sinkhorn_max_iter": self.sinkhorn_max_iter,
                "weight_mode": getattr(self, "weight_mode", "popularity"),
                "assignment_mode": getattr(self, "assignment_mode", "ot"),
                "p_nat_mode": "cluster_item_count",
            },
            path,
        )
        print(f"Saved Pop-Balance RQ-KMeans to {path}")

    @classmethod
    def load(cls, path: str) -> "RqKmeansPopBalance":
        import torch

        if not hasattr(np, "_core"):
            import sys as _sys

            _sys.modules.setdefault("numpy._core", np.core)
            _sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
            _sys.modules.setdefault("numpy._core.numeric", np.core.numeric)

        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("model_type") != "rqkmeans_pop_balance":
            raise ValueError(
                f"Checkpoint is not rqkmeans_pop_balance: {path}. "
                "Use RqKmeans.load() for baseline checkpoints."
            )
        obj = cls(
            n_layers=state["n_layers"],
            codebook_size=state["codebook_size"],
            balance_scope=state.get("balance_scope", "first"),
            balance_lambda=state.get("balance_lambda", 1.0),
            sinkhorn_epsilon=state.get("sinkhorn_epsilon", 0.1),
            sinkhorn_max_iter=state.get("sinkhorn_max_iter", 100),
        )
        obj.centroids = state["centroids"]
        obj.weight_mode = state.get("weight_mode", "popularity")
        obj.assignment_mode = state.get("assignment_mode", "ot")
        print(
            f"Loaded Pop-Balance RQ-KMeans from {path} "
            f"(layers={obj.n_layers}, codebook={obj.codebook_size}, "
            f"scope={obj.balance_scope}, weight_mode={obj.weight_mode}, "
            f"assignment={obj.assignment_mode}, dim={obj.centroids[0].shape[1]})"
        )
        return obj
