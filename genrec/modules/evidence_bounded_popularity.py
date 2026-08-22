
from __future__ import annotations

import math
from typing import Dict, Sequence

import torch

from genrec.data.fair_eval import build_item_popularity_from_train_sequences

__all__ = [
    "build_item_popularity_from_train_sequences",
    "normalize_item_popularity",
    "build_code_popularity_table_from_sem_ids",
]


def normalize_item_popularity(
    item_popularity: Dict[int, float],
    *,
    use_log: bool = True,
) -> Dict[int, float]:
    if not item_popularity:
        return {}
    if use_log:
        max_count = max(item_popularity.values())
        denom = math.log1p(max_count)
        if denom <= 0:
            return {k: 0.0 for k in item_popularity}
        return {
            k: math.log1p(v) / denom for k, v in item_popularity.items()
        }
    max_count = max(item_popularity.values())
    if max_count <= 0:
        return {k: 0.0 for k in item_popularity}
    return {k: v / max_count for k, v in item_popularity.items()}


def build_code_popularity_table_from_sem_ids(
    sem_ids_list: Sequence[Sequence[int]],
    item_popularity: Dict[int, float],
    num_levels: int,
    codebook_size: int,
) -> torch.Tensor:
    table_sum = torch.zeros(num_levels, codebook_size, dtype=torch.float32)
    table_cnt = torch.zeros(num_levels, codebook_size, dtype=torch.float32)

    for item_id, codes in enumerate(sem_ids_list):
        pop = item_popularity.get(item_id)
        if pop is None:
            continue
        for level in range(num_levels):
            if level >= len(codes):
                continue
            code = int(codes[level])
            if 0 <= code < codebook_size:
                table_sum[level, code] += pop
                table_cnt[level, code] += 1.0

    pop_table = table_sum / table_cnt.clamp(min=1.0)
    if item_popularity:
        global_mean = sum(item_popularity.values()) / len(item_popularity)
    else:
        global_mean = 0.0
    pop_table[table_cnt == 0] = global_mean
    return pop_table
