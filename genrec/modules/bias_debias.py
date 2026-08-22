from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

import torch


def build_level_log_hist_from_token_popularity(
    token_popularity_per_level: Sequence[Dict[int, float]],
    codebook_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build per-level log historical token distribution log P_hist(c)."""
    log_hist = []
    for pop_dict in token_popularity_per_level:
        counts = torch.zeros(codebook_size, dtype=torch.float32)
        for code, count in pop_dict.items():
            code = int(code)
            if 0 <= code < codebook_size:
                counts[code] = float(count)
        total = counts.sum()
        if total <= 0:
            probs = torch.full((codebook_size,), 1.0 / codebook_size)
        else:
            probs = counts / total
        log_hist.append(torch.log(probs + eps))
    return torch.stack(log_hist, dim=0)


def build_tiger_token_popularity_from_item_popularity(
    sem_ids_list: Sequence[Sequence[int]],
    item_popularity: Dict[int, float],
    num_levels: int,
    item_id_offset: int = 1,
) -> List[Dict[int, float]]:
    """Aggregate item popularity onto per-level semantic codes (0-indexed codes)."""
    counters = [Counter() for _ in range(num_levels)]
    for item_id, pop in item_popularity.items():
        idx = int(item_id) - item_id_offset
        if idx < 0 or idx >= len(sem_ids_list):
            continue
        codes = sem_ids_list[idx]
        for level in range(num_levels):
            if level < len(codes):
                counters[level][int(codes[level])] += float(pop)
    return [{int(k): float(v) for k, v in counter.items()} for counter in counters]
