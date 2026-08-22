from __future__ import annotations

import json
import logging
import math
from typing import Dict, Optional, Sequence

import torch

from genrec.data.fair_eval import build_item_popularity_from_train_sequences
from genrec.modules.bias_debias import (
    build_level_log_hist_from_token_popularity,
    build_tiger_token_popularity_from_item_popularity,
)

logger = logging.getLogger(__name__)


def _resolve_alpha_eff(
    alpha: float,
    level_weight: float,
    log_probs: torch.Tensor,
) -> torch.Tensor:
    """Effective debias strength [B, 1]."""
    return torch.full(
        (log_probs.size(0), 1),
        alpha * level_weight,
        dtype=log_probs.dtype,
        device=log_probs.device,
    )


def compute_entropy_gamma(
    log_probs: torch.Tensor,
    codebook_size: int,
    gamma_min: float,
    gamma_max: float,
) -> torch.Tensor:
    """Per-branch uncertainty factor gamma^(l)(u) in [gamma_min, gamma_max].

    H = -sum p log p on the RAW (pre-correction) log-softmax `log_probs`.
    High entropy -> larger gamma -> subtract more prior.
    """
    p = log_probs.exp()
    entropy = -(p * log_probs).sum(dim=-1)
    max_entropy = math.log(max(int(codebook_size), 2))
    h_norm = (entropy / max_entropy).clamp(0.0, 1.0)
    gamma = gamma_min + (gamma_max - gamma_min) * h_norm
    return gamma.unsqueeze(-1)


class EntropyGamma:
    """Per-(level, branch) uncertainty factor gamma^(l)(u)."""

    def __init__(
        self,
        *,
        enable: bool = False,
        gamma_min: float = 0.5,
        gamma_max: float = 1.5,
        levels: Optional[Sequence[int]] = None,
        share_level0: bool = False,
    ):
        self.enable = bool(enable)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.levels = {int(x) for x in levels} if levels is not None else None
        self.share_level0 = bool(share_level0)
        self._level0_gamma: Optional[torch.Tensor] = None
        self.stats: Dict[int, Dict[str, float]] = {}

    def active_at(self, level: int) -> bool:
        return self.enable and (self.levels is None or level in self.levels)

    def reset(self) -> None:
        self._level0_gamma = None

    @staticmethod
    def _expand_gamma(gamma: torch.Tensor, target_batch: int) -> torch.Tensor:
        n0 = int(gamma.size(0))
        if n0 == target_batch:
            return gamma
        if n0 == 0 or target_batch % n0 != 0:
            raise ValueError(
                f"Cannot expand entropy gamma from {n0} to {target_batch}"
            )
        return gamma.repeat_interleave(target_batch // n0, dim=0)

    def _record_stats(self, level: int, gamma: torch.Tensor) -> None:
        s = self.stats.setdefault(
            level, {"sum": 0.0, "count": 0.0, "min": math.inf, "max": -math.inf}
        )
        s["sum"] += float(gamma.sum())
        s["count"] += float(gamma.numel())
        s["min"] = min(s["min"], float(gamma.min()))
        s["max"] = max(s["max"], float(gamma.max()))

    @torch.no_grad()
    def apply(
        self,
        alpha_eff: torch.Tensor,
        log_probs: torch.Tensor,
        level: int,
        codebook_size: int,
    ) -> torch.Tensor:
        if not self.enable:
            return alpha_eff

        if self.share_level0 and level == 0:
            self._level0_gamma = compute_entropy_gamma(
                log_probs, codebook_size, self.gamma_min, self.gamma_max
            ).detach()

        if not self.active_at(level):
            return alpha_eff

        if self.share_level0:
            if self._level0_gamma is None:
                gamma = compute_entropy_gamma(
                    log_probs, codebook_size, self.gamma_min, self.gamma_max
                )
            else:
                gamma = self._expand_gamma(
                    self._level0_gamma.to(
                        device=log_probs.device, dtype=alpha_eff.dtype
                    ),
                    log_probs.size(0),
                )
        else:
            gamma = compute_entropy_gamma(
                log_probs, codebook_size, self.gamma_min, self.gamma_max
            )
        self._record_stats(level, gamma)
        return alpha_eff * gamma

    def summary(self) -> Dict[int, Dict[str, float]]:
        return {
            lvl: {
                "mean": v["sum"] / v["count"],
                "min": v["min"],
                "max": v["max"],
                "count": int(v["count"]),
            }
            for lvl, v in sorted(self.stats.items())
            if v["count"] > 0
        }


def resolve_level_weights(
    level_weights: Optional[Sequence[float]],
    num_levels: int,
) -> torch.Tensor:
    """Per-level debias strength multipliers w^(l) (default all-ones)."""
    if level_weights is None:
        return torch.ones(num_levels, dtype=torch.float32)
    weights = [float(x) for x in level_weights]
    if len(weights) != num_levels:
        raise ValueError(
            f"level_weights length {len(weights)} != num_levels {num_levels}"
        )
    return torch.tensor(weights, dtype=torch.float32)


def resolve_ugd_gamma_mode(*, entropy_gamma: bool) -> str:
    return "entropy" if entropy_gamma else "off"


class UgdDebiaser:
    """Analytic popularity subtraction: score = log p - alpha * w * gamma * log P_hist."""

    def __init__(
        self,
        log_p_hist: torch.Tensor,
        *,
        alpha: float = 1.0,
        num_levels: Optional[int] = None,
        codebook_size: Optional[int] = None,
        level_weights: Optional[Sequence[float]] = None,
        entropy_gamma: bool = False,
        gamma_mode: str = "off",
        gamma_min: float = 0.5,
        gamma_max: float = 1.5,
        gamma_levels: Optional[Sequence[int]] = None,
        entropy_gamma_share_level0: bool = False,
    ):
        self.alpha = float(alpha)
        self.register_log_p_hist(log_p_hist)
        self.num_levels = num_levels if num_levels is not None else log_p_hist.shape[0]
        self.codebook_size = (
            codebook_size if codebook_size is not None else log_p_hist.shape[1]
        )
        self.level_weights = resolve_level_weights(level_weights, self.num_levels)
        self.gamma_mode = gamma_mode
        self.entropy = EntropyGamma(
            enable=entropy_gamma and gamma_mode == "entropy",
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            levels=gamma_levels,
            share_level0=entropy_gamma_share_level0,
        )

    def register_log_p_hist(self, log_p_hist: torch.Tensor) -> None:
        if log_p_hist.ndim != 2:
            raise ValueError(
                f"log_p_hist must be [num_levels, codebook_size], "
                f"got {tuple(log_p_hist.shape)}"
            )
        self.log_p_hist = log_p_hist.detach().float()

    def to(self, device: torch.device) -> "UgdDebiaser":
        self.log_p_hist = self.log_p_hist.to(device)
        return self

    @torch.no_grad()
    def debias_log_probs(
        self,
        log_probs: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        log_hist = self.log_p_hist[level].to(log_probs.device).unsqueeze(0)
        log_hist = log_hist.expand_as(log_probs)
        delta = log_probs - log_hist
        alpha_eff = _resolve_alpha_eff(
            self.alpha,
            float(self.level_weights[level]),
            log_probs,
        )
        if self.gamma_mode == "entropy":
            alpha_eff = self.entropy.apply(
                alpha_eff, log_probs, level, self.codebook_size
            )
        return (1.0 - alpha_eff) * log_probs + alpha_eff * delta


def _resolve_item_popularity_for_p_hist(
    *,
    train_sequences: Optional[Sequence[Sequence[int]]] = None,
    item_popularity_path: Optional[str] = None,
    holdout: int = 2,
) -> Dict[int, float]:
    if train_sequences is not None:
        item_popularity = build_item_popularity_from_train_sequences(
            train_sequences, holdout=holdout
        )
        logger.info(
            "Built train-only item popularity from %d sequences (holdout=%d, %d items)",
            len(train_sequences),
            holdout,
            len(item_popularity),
        )
        return item_popularity
    if item_popularity_path is not None:
        with open(item_popularity_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        logger.warning(
            "Using item_popularity_path=%s (full-corpus counts). "
            "Prefer train_sequences for leakage-safe P_hist.",
            item_popularity_path,
        )
        return {int(k): float(v) for k, v in raw.items()}
    raise ValueError(
        "P_hist requires train_sequences (recommended) or item_popularity_path"
    )


def build_log_p_hist_for_tiger(
    *,
    sem_ids_list: Sequence[Sequence[int]],
    num_levels: int,
    codebook_size: int,
    train_sequences: Optional[Sequence[Sequence[int]]] = None,
    item_popularity_path: Optional[str] = None,
    holdout: int = 2,
) -> torch.Tensor:
    item_popularity = _resolve_item_popularity_for_p_hist(
        train_sequences=train_sequences,
        item_popularity_path=item_popularity_path,
        holdout=holdout,
    )
    token_pop = build_tiger_token_popularity_from_item_popularity(
        sem_ids_list,
        item_popularity,
        num_levels,
    )
    return build_level_log_hist_from_token_popularity(token_pop, codebook_size)


def setup_tiger_ugd_debiaser(
    model: torch.nn.Module,
    *,
    enable: bool = False,
    train_sequences: Optional[Sequence[Sequence[int]]] = None,
    item_popularity_path: Optional[str] = None,
    sem_ids_list: Optional[Sequence[Sequence[int]]] = None,
    num_levels: int = 4,
    codebook_size: int = 256,
    alpha: float = 1.0,
    level_weights: Optional[Sequence[float]] = None,
    entropy_gamma: bool = False,
    gamma_min: float = 0.5,
    gamma_max: float = 1.5,
    gamma_levels: Optional[Sequence[int]] = None,
    entropy_gamma_share_level0: bool = False,
    holdout: int = 2,
    device: Optional[torch.device] = None,
) -> Optional[UgdDebiaser]:
    """Analytic hist-pop subtract for TIGER: score = log p - α·w·γ·log P_hist."""
    if not enable:
        return None
    if sem_ids_list is None:
        raise ValueError("UGD debias requires sem_ids_list")
    if train_sequences is None and item_popularity_path is None:
        raise ValueError(
            "UGD debias requires train_sequences or item_popularity_path"
        )

    log_p_hist = build_log_p_hist_for_tiger(
        sem_ids_list=sem_ids_list,
        num_levels=num_levels,
        codebook_size=codebook_size,
        train_sequences=train_sequences,
        item_popularity_path=item_popularity_path,
        holdout=holdout,
    )
    gamma_mode = resolve_ugd_gamma_mode(entropy_gamma=entropy_gamma)
    debiaser = UgdDebiaser(
        log_p_hist,
        alpha=alpha,
        num_levels=num_levels,
        codebook_size=codebook_size,
        level_weights=level_weights,
        entropy_gamma=entropy_gamma,
        gamma_mode=gamma_mode,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        gamma_levels=gamma_levels,
        entropy_gamma_share_level0=entropy_gamma_share_level0,
    )
    if device is not None:
        debiaser.to(device)
    model.bias_debiaser = debiaser
    gamma_levels_log = (
        sorted(debiaser.entropy.levels)
        if gamma_mode == "entropy" and debiaser.entropy.levels
        else "all"
    )
    logger.info(
        "TIGER UGD debiaser enabled: alpha=%.3f level_weights=%s "
        "gamma_mode=%s gamma=[%.2f,%.2f] gamma_levels=%s "
        "entropy_share_level0=%s levels=%d "
        "(analytic log p - log P_hist)",
        alpha,
        debiaser.level_weights.tolist(),
        gamma_mode,
        gamma_min,
        gamma_max,
        gamma_levels_log,
        bool(entropy_gamma_share_level0),
        debiaser.num_levels,
    )
    return debiaser
