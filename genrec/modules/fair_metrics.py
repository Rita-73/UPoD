"""UPoD eval metrics: overall Popularity Lift, popularity MGU vs history, and ARP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np


def bucket_by_percentile(
    popularity_scores: Dict[int, float], cutoffs: Sequence[float]
) -> Dict[int, int]:
    if not popularity_scores:
        return {}
    sorted_items = sorted(popularity_scores.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_items)
    buckets: Dict[int, int] = {}
    for rank, (key, _) in enumerate(sorted_items, start=1):
        percentile = rank / n
        bucket = 0
        while bucket < len(cutoffs) and percentile > cutoffs[bucket]:
            bucket += 1
        buckets[key] = bucket
    return buckets


def mean_item_popularity(
    items: Sequence[int], item_popularity: Dict[int, float]
) -> float:
    vals = [
        float(item_popularity.get(int(item_id), 0.0))
        for item_id in items
        if int(item_id) > 0
    ]
    return float(np.mean(vals)) if vals else 0.0


def _load_raw_item_int_map(path: Optional[str]) -> Dict[int, object]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {idx: v for idx, v in enumerate(raw)}
    return {int(k): v for k, v in raw.items()}


def load_item_popularity(path: Optional[str]) -> Dict[int, float]:
    item_map = _load_raw_item_int_map(path)
    if not item_map:
        return {}
    return {int(k): float(v) for k, v in item_map.items()}


class PopularityLiftTracker:
    """Overall Popularity Lift.

    PL = (GAP_q - GAP_p) / GAP_p
    GAP_p: mean item popularity in the user profile;
    GAP_q: mean item popularity in the top-k recommendations.
    """

    def __init__(
        self,
        topk_list: Sequence[int],
        item_popularity: Dict[int, float],
        enable: bool = True,
    ) -> None:
        self.enable = enable and bool(item_popularity)
        self.topk_list = list(topk_list)
        self.item_popularity = item_popularity
        self._overall_sum_gap_p: Dict[int, float] = {}
        self._overall_sum_gap_q: Dict[int, float] = {}
        self._overall_counts: Dict[int, int] = {}
        self.num_users = 0
        if self.enable:
            self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._overall_sum_gap_p = {k: 0.0 for k in self.topk_list}
        self._overall_sum_gap_q = {k: 0.0 for k in self.topk_list}
        self._overall_counts = {k: 0 for k in self.topk_list}
        self.num_users = 0

    def clone_for_eval(self) -> "PopularityLiftTracker":
        return PopularityLiftTracker(
            self.topk_list,
            self.item_popularity,
            self.enable,
        )

    def update(
        self,
        ranked_item_ids: Sequence[int],
        history_items: Sequence[int],
    ) -> None:
        if not self.enable:
            return

        self.num_users += 1
        gap_p = mean_item_popularity(history_items, self.item_popularity)
        for k in self.topk_list:
            topk_items = [int(x) for x in ranked_item_ids[:k] if int(x) > 0]
            gap_q = mean_item_popularity(topk_items, self.item_popularity)
            self._overall_sum_gap_p[k] += gap_p
            self._overall_sum_gap_q[k] += gap_q
            self._overall_counts[k] += 1

    def finalize(self, decimals: int = 4) -> Dict[str, float]:
        if not self.enable:
            return {}

        out: Dict[str, float] = {}
        for k in self.topk_list:
            overall_p = (
                self._overall_sum_gap_p[k] / self._overall_counts[k]
                if self._overall_counts[k] > 0
                else 0.0
            )
            overall_q = (
                self._overall_sum_gap_q[k] / self._overall_counts[k]
                if self._overall_counts[k] > 0
                else 0.0
            )
            overall_pl = (
                (overall_q - overall_p) / overall_p if overall_p > 0.0 else 0.0
            )
            out[f"pl_overall@{k}"] = float(np.round(overall_pl, decimals))
        return out


class HistPopularityMguAccumulator:
    """Dataset-level GP/GH counts for popularity MGU vs history."""

    def __init__(self, group_count: int) -> None:
        self.group_count = group_count
        self.pred_counts = np.zeros(group_count, dtype=np.float64)
        self.hist_counts = np.zeros(group_count, dtype=np.float64)

    def add_user(
        self,
        topk_items: Sequence[int],
        history_items: Sequence[int],
        item_to_group: Dict[int, int],
    ) -> None:
        for item_id in topk_items:
            group = item_to_group.get(int(item_id))
            if group is not None:
                self.pred_counts[group] += 1.0
        for item_id in history_items:
            if int(item_id) < 0:
                continue
            group = item_to_group.get(int(item_id))
            if group is not None:
                self.hist_counts[group] += 1.0

    def compute_mgu(self) -> float:
        if self.pred_counts.sum() <= 0 or self.hist_counts.sum() <= 0:
            return 0.0
        gp = self.pred_counts / self.pred_counts.sum()
        gh = self.hist_counts / self.hist_counts.sum()
        return float(np.abs(gp - gh).mean())


class FairMetricsTracker:
    """Accumulates ARP and popularity MGU vs history."""

    def __init__(
        self,
        topk_list: Sequence[int],
        item_popularity: Dict[int, float],
        popularity_group_cutoffs: Sequence[float],
        enable: bool = True,
    ) -> None:
        self.enable = enable
        self.topk_list = list(topk_list)
        self.item_popularity = item_popularity
        self.popularity_group_cutoffs = list(popularity_group_cutoffs)
        self.item_to_pop_bucket = bucket_by_percentile(
            item_popularity, popularity_group_cutoffs
        )
        self.num_pop_buckets = len(popularity_group_cutoffs) + 1
        self.pop_mgu_accumulators: Dict[int, HistPopularityMguAccumulator] = {}
        if self.item_to_pop_bucket:
            for k in self.topk_list:
                self.pop_mgu_accumulators[k] = HistPopularityMguAccumulator(
                    self.num_pop_buckets
                )
        self.result_dict = {f"arp@{k}": 0.0 for k in self.topk_list}
        self.num_users = 0

    def update(
        self,
        ranked_item_ids: Sequence[int],
        history_items: Sequence[int],
    ) -> None:
        if not self.enable:
            return

        self.num_users += 1
        history_list = [int(x) for x in history_items if int(x) > 0]
        for k in self.topk_list:
            topk_items = [int(x) for x in ranked_item_ids[:k]]
            if self.item_popularity:
                pop_vals = [
                    float(self.item_popularity.get(item_id, 0.0))
                    for item_id in topk_items
                ]
                self.result_dict[f"arp@{k}"] += (
                    float(np.mean(pop_vals)) if pop_vals else 0.0
                )
            else:
                self.result_dict[f"arp@{k}"] += 0.0
            if k in self.pop_mgu_accumulators:
                self.pop_mgu_accumulators[k].add_user(
                    topk_items,
                    history_list,
                    self.item_to_pop_bucket,
                )

    def finalize(self, num_users: Optional[int] = None, decimals: int = 4) -> Dict[str, float]:
        users = num_users if num_users is not None else self.num_users
        if not self.enable or users <= 0:
            return {}

        out: Dict[str, float] = {}
        for k in self.topk_list:
            out[f"arp@{k}"] = float(np.round(self.result_dict[f"arp@{k}"] / users, decimals))
            if k in self.pop_mgu_accumulators:
                out[f"pop_mgu_hist@{k}"] = float(
                    np.round(self.pop_mgu_accumulators[k].compute_mgu(), decimals)
                )
        return out
