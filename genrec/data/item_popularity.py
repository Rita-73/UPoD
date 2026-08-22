"""Item popularity weights for Balance-ID (train-only log/raw mass)."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

ITEM_EMBEDDING_MODES = ("baseline",)


def embedding_mode_suffix(item_embedding_mode: str) -> str:
    """Parquet suffix for non-baseline embedding modes (UPoD uses baseline only)."""
    if item_embedding_mode == "baseline":
        return ""
    raise ValueError(
        f"Unknown item_embedding_mode: {item_embedding_mode!r}. "
        f"Expected one of {ITEM_EMBEDDING_MODES}."
    )


def format_item_metadata_text(info: dict) -> str:
    """Text string for sentence-t5-xl metadata encoding (baseline fields)."""
    return (
        f"'title':{info.get('title', '')}\n"
        f" 'price':{info.get('price', '')}\n"
        f" 'salesRank':{info.get('salesRank', '')}\n"
        f" 'brand':{info.get('brand', '')}\n"
        f" 'categories':{info.get('categories', '')}"
    )


def popularity_weights_from_counts(counts: np.ndarray) -> np.ndarray:
    """Normalize log1p interaction counts to a probability mass vector."""
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.log1p(np.maximum(counts, 0.0))
    total = weights.sum()
    if total <= 0:
        return np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    return weights / total


def raw_popularity_weights_from_counts(counts: np.ndarray) -> np.ndarray:
    """Normalize raw interaction counts (no log) to a probability mass vector."""
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.maximum(counts, 0.0)
    total = weights.sum()
    if total <= 0:
        return np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    return weights / total


def _weights_from_counts(counts: np.ndarray, mass_transform: str) -> np.ndarray:
    if mass_transform == "log":
        return popularity_weights_from_counts(counts)
    if mass_transform == "raw":
        return raw_popularity_weights_from_counts(counts)
    raise ValueError(
        f"mass_transform must be 'log' or 'raw', got {mass_transform!r}"
    )


def load_popularity_weights_for_item_dataset(
    item_dataset,
    root: str,
    split: str,
    holdout: int = 2,
    mass_transform: str = "log",
) -> np.ndarray:
    """Return train-only popularity weights aligned with item_dataset row order.

    Counts interactions from ``seq[:-holdout]`` for each user sequence (LOO train
    split, default holdout=2). Item IDs match ``ItemID`` in the item parquet
    (1-indexed, same as AmazonItemDataset).

    mass_transform:
      - "log": a_i ∝ log(1+n_i)  (default Balance ID)
      - "raw": a_i ∝ n_i         (raw-mass ablation)
    """
    import pandas as pd

    from genrec.data.amazon import (
        DATASET_CONFIGS,
        canonical_split,
        is_movielens_split,
        load_user2item,
        parse_gzip_json,
        raw_dir_for,
    )
    from genrec.data.fair_eval import build_item_popularity_from_train_sequences

    df = pd.read_parquet(item_dataset.parquet_path)
    if "ItemID" not in df.columns:
        raise ValueError(f"Missing ItemID column in {item_dataset.parquet_path}")

    if is_movielens_split(split):
        user2item = load_user2item(root, split)
        sequences: list[list[int]] = []
        for items in user2item["ItemID"]:
            seq = [int(i) for i in items if int(i) > 0]
            if len(seq) >= 5:
                sequences.append(seq)
        popularity = build_item_popularity_from_train_sequences(sequences, holdout=holdout)
        counts = np.array(
            [popularity.get(int(item_id), 0.0) for item_id in df["ItemID"].tolist()],
            dtype=np.float64,
        )
        return _weights_from_counts(counts, mass_transform)

    split = canonical_split(split)
    config = DATASET_CONFIGS[split]
    reviews_path = os.path.join(raw_dir_for(root, split), config["reviews"])

    user_sequences: dict[str, list[tuple]] = {}
    item_id_mapping: dict[str, int] = {}
    for review in parse_gzip_json(reviews_path):
        asin = review.get("asin")
        user_id = review.get("reviewerID")
        timestamp = review.get("unixReviewTime", 0)
        if asin and user_id:
            if asin not in item_id_mapping:
                item_id_mapping[asin] = len(item_id_mapping)
            item_id = item_id_mapping[asin]
            if user_id not in user_sequences:
                user_sequences[user_id] = []
            user_sequences[user_id].append((timestamp, item_id))

    sequences: list[list[int]] = []
    for seq in user_sequences.values():
        seq.sort(key=lambda x: x[0])
        items = [x[1] for x in seq]
        if len(items) >= 5:
            sequences.append(items)

    popularity_0idx = build_item_popularity_from_train_sequences(sequences, holdout=holdout)
    popularity = {int(item_id + 1): count for item_id, count in popularity_0idx.items()}
    counts = np.array(
        [popularity.get(int(item_id), 0.0) for item_id in df["ItemID"].tolist()],
        dtype=np.float64,
    )
    return _weights_from_counts(counts, mass_transform)
