

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.modules.evidence_bounded_popularity import (
    build_code_popularity_table_from_sem_ids,
)


def compute_level0_next_token_entropy(
    step_logits: torch.Tensor,
    *,
    codebook_size: int,
    normalize: bool = True,
) -> torch.Tensor:
    """Per-sample entropy of the level-0 next-token distribution. Shape [B]."""
    log_probs = F.log_softmax(step_logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    if normalize:
        max_entropy = math.log(max(int(codebook_size), 2))
        entropy = entropy / max_entropy
    return entropy


@dataclass
class UalAssets:
    """Offline assets used by UalLoss."""

    code_bin_onehot: torch.Tensor  # [L, C, M]
    target_dist: torch.Tensor      # [L, M]
    lambda_per_level: List[float]  # [L]


def _item_codes_from_sem_ids(
    sem_ids_list: Sequence[Sequence[int]],
    num_levels: int,
) -> torch.Tensor:
    """Raw codes per item from sem_ids_list; shape [n_items, L]."""
    num_items = len(sem_ids_list)
    item_codes = torch.zeros(num_items, num_levels, dtype=torch.long)
    for item_id, codes in enumerate(sem_ids_list):
        for level in range(num_levels):
            if level < len(codes):
                item_codes[item_id, level] = int(codes[level])
    return item_codes


def compute_filter_rates(item_codes: torch.Tensor) -> torch.Tensor:

    item_codes = item_codes.long().cpu()
    num_levels = item_codes.shape[1]
    rates: List[float] = []
    for level in range(num_levels):
        prefix_map: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        for item_idx in range(item_codes.shape[0]):
            prefix = tuple(item_codes[item_idx, :level].tolist()) if level > 0 else ()
            prefix_map[prefix].append(item_idx)

        level_filters: List[float] = []
        for indices in prefix_map.values():
            group_size = len(indices)
            if group_size <= 1:
                continue
            child_counts: Dict[int, int] = defaultdict(int)
            for item_idx in indices:
                child_counts[item_codes[item_idx, level].item()] += 1
            sum_sq = sum(count ** 2 for count in child_counts.values())
            level_filters.append(1.0 - sum_sq / (group_size ** 2))
        rates.append(sum(level_filters) / len(level_filters) if level_filters else 0.0)
    return torch.tensor(rates, dtype=torch.float32)


def discriminative_lambda_from_filter_rates(
    filter_rates: Sequence[float],
    *,
    p: float = 1.0,
    q: float = 2.0,
    floor: float = 0.0,
    uniform_mix: float = 0.0,
) -> List[float]:
    discrim = torch.as_tensor(filter_rates, dtype=torch.float32)
    num_levels = discrim.numel()
    d_min, d_max = discrim.min(), discrim.max()
    if (d_max - d_min).abs() < 1e-12:
        d = torch.full_like(discrim, 0.5)
    else:
        d = (discrim - d_min) / (d_max - d_min)

    lambda_tilde = d.pow(p) * (1.0 - d).pow(q)
    if lambda_tilde.sum() < 1e-12:
        lam = torch.ones(num_levels, dtype=torch.float32)
    else:
        lam = num_levels * lambda_tilde / lambda_tilde.sum()

    uniform_mix = float(uniform_mix)
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError(f"uniform_mix must be in [0, 1], got {uniform_mix}")
    if uniform_mix > 0.0:
        uniform = torch.ones(num_levels, dtype=torch.float32)
        lam = (1.0 - uniform_mix) * lam + uniform_mix * uniform

    floor = float(floor)
    if floor < 0.0:
        raise ValueError(f"floor must be >= 0, got {floor}")
    if floor > 0.0:
        if floor * num_levels >= num_levels:
            raise ValueError(
                f"floor={floor} too large for {num_levels} levels "
                f"(need floor < 1)"
            )
        lam = floor + (num_levels - num_levels * floor) * (lam / lam.sum())

    return lam.tolist()


def resolve_ual_lambda(
    item_codes: Optional[torch.Tensor],
    *,
    num_levels: int,
    mode: str = "uniform",
    p: float = 1.0,
    q: float = 2.0,
    floor: float = 0.0,
    uniform_mix: float = 0.0,
    lambda_per_level: Optional[Sequence[float]] = None,
) -> List[float]:
    if mode == "manual":
        if lambda_per_level is None:
            raise ValueError("ual_lambda_mode='manual' requires lambda_per_level")
        if len(lambda_per_level) != num_levels:
            raise ValueError(
                f"lambda_per_level length {len(lambda_per_level)} != num_levels {num_levels}"
            )
        return list(lambda_per_level)

    if mode == "uniform":
        return uniform_lambda_levels(num_levels)

    if mode == "discriminative":
        if item_codes is None:
            raise ValueError("discriminative lambda mode requires item codes")
        return discriminative_lambda_from_filter_rates(
            compute_filter_rates(item_codes),
            p=p,
            q=q,
            floor=floor,
            uniform_mix=uniform_mix,
        )

    raise ValueError(
        f"Unknown ual_lambda_mode={mode!r}; "
        "expected 'uniform', 'discriminative', or 'manual'"
    )


def uniform_lambda_levels(num_levels: int) -> List[float]:
    return [1.0 / num_levels] * num_levels


def _build_target_dist(
    code_bin: torch.Tensor,
    item_codes: torch.Tensor,
    raw_counts: Dict[int, float],
    num_bins: int,
) -> torch.Tensor:
    """Target popularity distribution R^(l) over bins; shape [L, M]."""
    num_levels = code_bin.shape[0]

    weights = torch.zeros(item_codes.shape[0], dtype=torch.float32)
    for item_id, count in raw_counts.items():
        if 0 < item_id < weights.shape[0]:
            weights[item_id] = float(count)

    target = torch.zeros(num_levels, num_bins, dtype=torch.float32)
    for level in range(num_levels):
        bins_l = code_bin[level][item_codes[:, level]]
        target[level].index_add_(0, bins_l, weights)
    return target / target.sum(dim=1, keepdim=True).clamp(min=1e-12)


def _quantize_code_bins(
    pop_table: torch.Tensor, num_bins: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = pop_table.flatten()
    qs = torch.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = torch.quantile(flat, qs)
    bin_edges[0] = float("-inf")
    bin_edges[-1] = float("inf")

    interior = bin_edges[1:-1].contiguous()
    code_bin = torch.bucketize(pop_table, interior, right=False)
    code_bin = code_bin.clamp(0, num_bins - 1)
    code_bin_onehot = F.one_hot(code_bin, num_classes=num_bins).float()
    return code_bin, code_bin_onehot


def build_ual_assets_from_sem_ids(
    sem_ids_list: Sequence[Sequence[int]],
    item_popularity: Dict[int, float],
    raw_counts: Dict[int, float],
    *,
    num_levels: int,
    codebook_size: int,
    num_bins: int = 10,
    lambda_mode: str = "uniform",
    lambda_p: float = 1.0,
    lambda_q: float = 2.0,
    lambda_floor: float = 0.0,
    lambda_uniform_mix: float = 0.0,
    lambda_per_level: Optional[Sequence[float]] = None,
    target_mode: str = "uniform",
) -> UalAssets:
    """Build TIGER UAL tensors actually consumed by the loss."""
    pop_table = build_code_popularity_table_from_sem_ids(
        sem_ids_list, item_popularity, num_levels, codebook_size
    )
    code_bin, code_bin_onehot = _quantize_code_bins(pop_table, num_bins)

    need_item_codes = lambda_mode == "discriminative" or target_mode == "empirical"
    item_codes = (
        _item_codes_from_sem_ids(sem_ids_list, num_levels) if need_item_codes else None
    )
    if target_mode == "empirical":
        if item_codes is None:
            raise ValueError("empirical UAL target requires item codes")
        target = _build_target_dist(code_bin, item_codes, raw_counts, num_bins)
    elif target_mode == "uniform":
        target = torch.full(
            (num_levels, num_bins),
            1.0 / float(num_bins),
            dtype=torch.float32,
        )
    else:
        raise ValueError(
            f"Unknown ual_target_mode={target_mode!r}; "
            "expected 'empirical' or 'uniform'"
        )

    lambda_levels = resolve_ual_lambda(
        item_codes,
        num_levels=num_levels,
        mode=lambda_mode,
        p=lambda_p,
        q=lambda_q,
        floor=lambda_floor,
        uniform_mix=lambda_uniform_mix,
        lambda_per_level=lambda_per_level,
    )
    return UalAssets(
        code_bin_onehot=code_bin_onehot,
        target_dist=target,
        lambda_per_level=lambda_levels,
    )


class PopularityEvidenceHead(nn.Module):
    """Auxiliary next-item popularity head (genrec PE RNG parity).

    Hidden is detached by the caller. Kept even when aux_weight=0 so Linear
    init and the extra encoder dropout pass match genrec.
    """

    def __init__(
        self,
        n_embd: int,
        hidden: int = 128,
        *,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.ema_decay = ema_decay
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("error_ema", torch.ones(()))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)

    @staticmethod
    def mse_loss(rho_true: torch.Tensor, rho_hat: torch.Tensor) -> torch.Tensor:
        return ((rho_true - rho_hat) ** 2).mean()


class EntropyWeightNormalizer(nn.Module):

    def __init__(self, *, ema_decay: float = 0.99, num_levels: int = 1):
        super().__init__()
        self.ema_decay = ema_decay
        self.num_levels = int(num_levels)
        if self.num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        if self.num_levels == 1:
            self.register_buffer("entropy_ema", torch.ones(()))
        else:
            self.register_buffer("entropy_ema", torch.ones(self.num_levels))

    def weight(self, entropy: torch.Tensor, *, detach: bool = True) -> torch.Tensor:
        entropy_for_ema = entropy.detach()
        if detach:
            entropy = entropy_for_ema
        if self.training:
            with torch.no_grad():
                if entropy_for_ema.dim() == 1:
                    batch_mean = entropy_for_ema.mean().clamp(min=1e-8)
                else:
                    batch_mean = entropy_for_ema.mean(dim=0).clamp(min=1e-8)
                self.entropy_ema.mul_(self.ema_decay).add_(
                    batch_mean, alpha=1.0 - self.ema_decay
                )
        scale = self.entropy_ema.clamp(min=1e-8)
        return entropy / scale


def entropy_gate_sample_weights(
    step_logits: torch.Tensor,
    *,
    codebook_size: int,
    normalizer: EntropyWeightNormalizer,
    detach: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    entropy = compute_level0_next_token_entropy(
        step_logits,
        codebook_size=codebook_size,
        normalize=True,
    )
    gate_w = normalizer.weight(entropy, detach=detach)
    stats = {
        "entropy_mean": entropy.detach().mean(),
        "entropy_gate_mean": gate_w.detach().mean(),
        "entropy_gate_detach": torch.tensor(
            1.0 if detach else 0.0, device=gate_w.device
        ),
    }
    return gate_w, stats


def entropy_gate_per_level_sample_weights(
    hierarchy_logits: torch.Tensor,
    *,
    codebook_size: int,
    normalizer: EntropyWeightNormalizer,
    detach: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    log_probs = F.log_softmax(hierarchy_logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    max_entropy = math.log(max(int(codebook_size), 2))
    entropy = entropy / max_entropy
    gate_w = normalizer.weight(entropy, detach=detach)
    stats = {
        "entropy_mean": entropy.detach().mean(),
        "entropy_gate_mean": gate_w.detach().mean(),
        "entropy_gate_detach": torch.tensor(
            1.0 if detach else 0.0, device=gate_w.device
        ),
        "entropy_gate_per_level": torch.tensor(1.0, device=gate_w.device),
    }
    for level_idx in range(entropy.shape[-1]):
        stats[f"entropy_mean_l{level_idx}"] = entropy.detach()[:, level_idx].mean()
        stats[f"entropy_gate_mean_l{level_idx}"] = gate_w.detach()[:, level_idx].mean()
    return gate_w, stats


UAL_MATCH_MODES = (
    "onesided_cdf",       # default: ReLU(CDF_R - CDF_Q), cold→hot cumsum
    "pdf_l1",             # ||Q - R||_1
    "symmetric_cdf",      # |CDF_R - CDF_Q|
    "onesided_cdf_hot",   # reverse cumsum (hot→cold) + ReLU(CDF_Q - CDF_R)
)

UAL_REDUCE_MODES = (
    "aggregate",  # default: weighted-mean Q then one L_ual
    "per_user",   # per-user L_ual then weighted-mean of hinges
)


class UalLoss(nn.Module):

    def __init__(
        self,
        code_bin_onehot: torch.Tensor,   # [L, C, M]
        target_dist: torch.Tensor,       # [L, M]
        *,
        lambda_per_level: Optional[Sequence[float]] = None,
        ema_decay: float = 0.9,
        match_mode: str = "onesided_cdf",
        reduce_mode: str = "aggregate",
    ):
        super().__init__()
        num_levels, _, num_bins = code_bin_onehot.shape
        self.num_levels = num_levels
        self.num_bins = num_bins
        self.ema_decay = ema_decay
        if match_mode not in UAL_MATCH_MODES:
            raise ValueError(
                f"Unknown ual_match_mode={match_mode!r}; "
                f"expected one of {UAL_MATCH_MODES}"
            )
        if reduce_mode not in UAL_REDUCE_MODES:
            raise ValueError(
                f"Unknown ual_reduce_mode={reduce_mode!r}; "
                f"expected one of {UAL_REDUCE_MODES}"
            )
        self.match_mode = match_mode
        self.reduce_mode = reduce_mode

        if lambda_per_level is None:
            lambda_per_level = [1.0 / num_levels] * num_levels
        if len(lambda_per_level) != num_levels:
            raise ValueError(
                f"lambda_per_level length {len(lambda_per_level)} != "
                f"num_levels {num_levels}"
            )

        self.register_buffer("code_bin_onehot", code_bin_onehot.float())
        self.register_buffer("target_dist", target_dist.float())
        self.register_buffer(
            "lambda_per_level", torch.tensor(lambda_per_level, dtype=torch.float32)
        )
        self.register_buffer("q_ema", target_dist.float().clone())

    def _match_penalty(
        self,
        q_blend: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Per-level match penalty; ``q_blend``/``target`` shape [L, M] → [L]."""
        if self.match_mode == "onesided_cdf":
            cdf_q = torch.cumsum(q_blend, dim=-1)
            cdf_r = torch.cumsum(target, dim=-1)
            return F.relu(cdf_r - cdf_q).sum(dim=-1)
        if self.match_mode == "pdf_l1":
            return (q_blend - target).abs().sum(dim=-1)
        if self.match_mode == "symmetric_cdf":
            cdf_q = torch.cumsum(q_blend, dim=-1)
            cdf_r = torch.cumsum(target, dim=-1)
            return (cdf_r - cdf_q).abs().sum(dim=-1)
        if self.match_mode == "onesided_cdf_hot":
            # Accumulate from popular → cold; penalize Q hotter than R.
            q_rev = torch.flip(q_blend, dims=[-1])
            r_rev = torch.flip(target, dims=[-1])
            cdf_q = torch.cumsum(q_rev, dim=-1)
            cdf_r = torch.cumsum(r_rev, dim=-1)
            return F.relu(cdf_q - cdf_r).sum(dim=-1)
        raise ValueError(f"Unhandled match_mode={self.match_mode!r}")

    def _match_penalty_per_user(
        self,
        user_bin: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Per-user per-level match; ``user_bin`` [N, L, M], ``target`` [L, M] → [N, L]."""
        if self.match_mode == "onesided_cdf":
            cdf_q = torch.cumsum(user_bin, dim=-1)
            cdf_r = torch.cumsum(target, dim=-1).unsqueeze(0)
            return F.relu(cdf_r - cdf_q).sum(dim=-1)
        if self.match_mode == "pdf_l1":
            return (user_bin - target.unsqueeze(0)).abs().sum(dim=-1)
        if self.match_mode == "symmetric_cdf":
            cdf_q = torch.cumsum(user_bin, dim=-1)
            cdf_r = torch.cumsum(target, dim=-1).unsqueeze(0)
            return (cdf_r - cdf_q).abs().sum(dim=-1)
        if self.match_mode == "onesided_cdf_hot":
            q_rev = torch.flip(user_bin, dims=[-1])
            r_rev = torch.flip(target, dims=[-1]).unsqueeze(0)
            cdf_q = torch.cumsum(q_rev, dim=-1)
            cdf_r = torch.cumsum(r_rev, dim=-1)
            return F.relu(cdf_q - cdf_r).sum(dim=-1)
        raise ValueError(f"Unhandled match_mode={self.match_mode!r}")

    def forward(
        self,
        hierarchy_logits: torch.Tensor,         # [N, L, C]
        weights: Optional[torch.Tensor] = None,  # [N] or [N, L] (detached)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        code_bin_onehot = self.code_bin_onehot
        target = self.target_dist
        lambda_l = self.lambda_per_level

        probs = F.softmax(hierarchy_logits, dim=-1)
        user_bin = torch.einsum("nlc,lcm->nlm", probs, code_bin_onehot)

        if weights is None:
            weights = probs.new_ones(probs.shape[0])
        w = weights.detach().to(probs.dtype)
        if w.dim() == 1:
            # Shared sample weight across levels: [N]
            w_sum = w.sum().clamp(min=1e-8)
            w_nl = w.view(-1, 1)
            w_nlm = w.view(-1, 1, 1)
            q_batch = (w_nlm * user_bin).sum(dim=0) / w_sum
        elif w.dim() == 2:
            # Per-level sample weights: [N, L]
            if w.shape[0] != user_bin.shape[0] or w.shape[1] != user_bin.shape[1]:
                raise ValueError(
                    f"per-level weights shape {tuple(w.shape)} incompatible with "
                    f"user_bin {tuple(user_bin.shape)}"
                )
            w_sum = w.sum(dim=0).clamp(min=1e-8)  # [L]
            w_nl = w
            w_nlm = w.unsqueeze(-1)  # [N, L, 1]
            q_batch = (w_nlm * user_bin).sum(dim=0) / w_sum.view(-1, 1)
        else:
            raise ValueError(
                f"weights must be [N] or [N, L], got shape {tuple(w.shape)}"
            )

        # Batch-level Q for EMA / logging (always weighted aggregate).
        q_blend = self.ema_decay * self.q_ema + (1.0 - self.ema_decay) * q_batch

        if self.reduce_mode == "per_user":
            # Each user vs R, then weight the hinges (no pre-aggregation into Q).
            hinge_n = self._match_penalty_per_user(user_bin, target)  # [N, L]
            if w.dim() == 1:
                hinge = (w_nl * hinge_n).sum(dim=0) / w_sum
            else:
                hinge = (w_nl * hinge_n).sum(dim=0) / w_sum
        else:
            hinge = self._match_penalty(q_blend, target)

        total = (lambda_l * hinge).sum()

        with torch.no_grad():
            self.q_ema.copy_(q_blend.detach())
            half = self.num_bins // 2
            q_head = q_blend[:, half:].sum(dim=-1).mean()
            r_head = target[:, half:].sum(dim=-1).mean()
        stats = {
            "hinge_mean": hinge.mean(),
            "gate_mean": w.mean(),
            "q_head_mass": q_head,
            "r_head_mass": r_head,
            "reduce_mode_per_user": torch.tensor(
                1.0 if self.reduce_mode == "per_user" else 0.0,
                device=hinge.device,
            ),
            "per_level_weights": torch.tensor(
                1.0 if w.dim() == 2 else 0.0,
                device=hinge.device,
            ),
        }
        for level_idx in range(self.num_levels):
            stats[f"hinge_l{level_idx}"] = hinge[level_idx]
            stats[f"lambda_l{level_idx}"] = lambda_l[level_idx]
        return total, stats


