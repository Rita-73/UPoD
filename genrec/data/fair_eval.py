
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from genrec.data.amazon import (
    DATASET_CONFIGS,
    canonical_split,
    ensure_amazon_raw_data,
    is_movielens_split,
    load_user2item,
    parse_gzip_json,
    raw_dir_for,
)
from genrec.modules.fair_metrics import load_item_popularity

logger = logging.getLogger(__name__)


def fair_eval_dir(root: str, split: str) -> str:
    return os.path.join(root, canonical_split(split), "processed", "fair_eval")


def mapping_cache_path(root: str, split: str) -> str:
    return os.path.join(root, canonical_split(split), "processed", "item_id_mapping.json")


def build_item_popularity_from_train_sequences(
    sequences: Sequence[Sequence[int]],
    holdout: int = 2,
) -> Dict[int, float]:
    counter: Counter[int] = Counter()
    for seq in sequences:
        if len(seq) <= holdout:
            continue
        for item_id in seq[:-holdout]:
            item_id = int(item_id)
            if item_id > 0:
                counter[item_id] += 1
    return {item_id: float(count) for item_id, count in counter.items()}


def build_item_popularity(
    root: str, split: str, asin_to_item_id: Dict[str, int]
) -> Dict[int, float]:
    config = DATASET_CONFIGS[canonical_split(split)]
    reviews_path = os.path.join(raw_dir_for(root, split), config["reviews"])
    counter: Counter[int] = Counter()
    for review in parse_gzip_json(reviews_path):
        asin = review.get("asin")
        if asin in asin_to_item_id:
            counter[asin_to_item_id[asin]] += 1
    return {item_id: float(count) for item_id, count in counter.items()}


def save_json(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def ensure_fair_eval_files(
    root: str,
    split: str,
    asin_to_item_id: Optional[Dict[str, int]] = None,
    train_sequences: Optional[Sequence[Sequence[int]]] = None,
) -> str:
    """Create fair-eval popularity JSON keyed by genrec item ids if missing."""
    out_dir = fair_eval_dir(root, split)
    pop_path = os.path.join(out_dir, "item_popularity.json")

    if os.path.exists(pop_path):
        return pop_path

    if is_movielens_split(split):
        if train_sequences is None:
            df = load_user2item(root, split)
            train_sequences = [list(items) for items in df["ItemID"]]
        popularity = build_item_popularity_from_train_sequences(train_sequences, holdout=2)
    else:
        if asin_to_item_id is None:
            raise ValueError(f"asin_to_item_id required for Amazon split {split}")
        ensure_amazon_raw_data(root, split, download_meta=False)
        popularity = build_item_popularity(root, split, asin_to_item_id)
    save_json(pop_path, {str(k): int(v) for k, v in popularity.items()})
    logger.info(
        "Built fair eval popularity for split=%s: %d entries",
        split,
        len(popularity),
    )
    return pop_path


def validate_external_fair_eval(
    external_popularity: Dict[int, float],
    genrec_popularity: Dict[int, float],
    min_match_ratio: float = 0.5,
) -> bool:
    """Return True if external popularity roughly matches genrec item ids."""
    if not external_popularity or not genrec_popularity:
        return False
    matched = 0
    compared = 0
    for item_id, count in genrec_popularity.items():
        if item_id in external_popularity:
            compared += 1
            if external_popularity[item_id] == count:
                matched += 1
    if compared == 0:
        return False
    return (matched / compared) >= min_match_ratio


def resolve_fair_eval_path(
    root: str,
    split: str,
    asin_to_item_id: Dict[str, int],
    item_popularity_path: Optional[str] = None,
) -> str:
    """Resolve fair-eval popularity JSON, preferring genrec-native files."""
    genrec_pop_path = ensure_fair_eval_files(root, split, asin_to_item_id)

    if item_popularity_path and os.path.exists(item_popularity_path):
        external_pop = load_item_popularity(item_popularity_path)
        genrec_pop = load_item_popularity(genrec_pop_path)
        if validate_external_fair_eval(external_pop, genrec_pop):
            return item_popularity_path
        logger.warning(
            "Ignoring external item_popularity_path=%s: item ids do not match "
            "genrec mapping (only ~%.1f%% exact matches). Using %s",
            item_popularity_path,
            100.0
            * sum(
                1
                for k, v in genrec_pop.items()
                if k in external_pop and external_pop[k] == v
            )
            / max(len(genrec_pop), 1),
            genrec_pop_path,
        )
    return genrec_pop_path


def load_fair_eval_metadata(
    root: str,
    split: str,
    asin_to_item_id: Optional[Dict[str, int]] = None,
    item_popularity_path: Optional[str] = None,
    train_sequences: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[Dict[int, float], str]:
    if is_movielens_split(split):
        pop_path = ensure_fair_eval_files(
            root,
            split,
            train_sequences=train_sequences,
        )
        if item_popularity_path and os.path.exists(item_popularity_path):
            pop_path = item_popularity_path
    else:
        if asin_to_item_id is None:
            raise ValueError(f"asin_to_item_id required for Amazon split {split}")
        pop_path = resolve_fair_eval_path(
            root,
            split,
            asin_to_item_id,
            item_popularity_path=item_popularity_path,
        )
    return load_item_popularity(pop_path), pop_path
