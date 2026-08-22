from __future__ import annotations

import logging
from typing import List, Sequence, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from genrec.data.amazon import AmazonSeqDataset

logger = logging.getLogger(__name__)


def deduplicate_semantic_ids(sem_ids_list: Sequence[Sequence[int]]) -> List[List[int]]:
    if not sem_ids_list:
        return []

    data = torch.tensor(sem_ids_list, dtype=torch.long)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D SID tensor, got shape {tuple(data.shape)}")

    unique_rows, inverse_indices, counts = torch.unique(
        data, dim=0, return_inverse=True, return_counts=True
    )
    del unique_rows

    output_indices = torch.zeros_like(inverse_indices)
    duplicate_indices = torch.where(counts > 1)[0]
    for i in range(len(duplicate_indices)):
        num_of_collisions = counts[duplicate_indices[i]]
        indices_to_change = torch.where(inverse_indices == duplicate_indices[i])[0]
        range_to_add = torch.arange(1, num_of_collisions + 1, dtype=torch.long)
        output_indices = output_indices.scatter(0, indices_to_change, range_to_add)

    result = torch.cat((data, output_indices.unsqueeze(1)), dim=1).long()
    return result.tolist()


def effective_sem_id_dim(base_layers: int, add_disambiguation: bool) -> int:
    """Return TIGER SID depth given quantization layers and dedup switch."""
    return base_layers + (1 if add_disambiguation else 0)


def compute_embeddings_per_hierarchy(sem_ids_list: Sequence[Sequence[int]]) -> int:
    """GRID-aligned width: max raw code + 1 across all hierarchies."""
    if not sem_ids_list:
        return 0
    return max(max(int(c) for c in codes) for codes in sem_ids_list) + 1


def resolve_embeddings_per_hierarchy(
    dataset: "AmazonSeqDataset",
    default_codebook_size: int = 256,
) -> int:
    """Resolve per-hierarchy embedding width (GRID: ``codebooks.max() + 1``)."""
    sem_ids_list = getattr(dataset, "sem_ids_list", None)
    if not sem_ids_list:
        return default_codebook_size

    computed = compute_embeddings_per_hierarchy(sem_ids_list)
    width = max(computed, default_codebook_size)
    if width > default_codebook_size:
        logger.info(
            "Dynamic embeddings_per_hierarchy=%d (codebook_size=%d, max raw code=%d)",
            width,
            default_codebook_size,
            width - 1,
        )
    return width


def compute_vocab_size(embeddings_per_hierarchy: int, sem_id_dim: int) -> int:
    """Offset-encoded vocab: one padding slot at id 0."""
    return embeddings_per_hierarchy * sem_id_dim + 1


def infer_sem_id_dim(dataset: "AmazonSeqDataset") -> int:
    """Infer SID depth from a loaded sequence dataset."""
    if getattr(dataset, "sem_ids_list", None):
        return len(dataset.sem_ids_list[0])
    add_disambiguation = getattr(dataset, "add_disambiguation", False)
    n_layers = getattr(dataset, "rqkmeans_n_layers", 3)
    return effective_sem_id_dim(n_layers, add_disambiguation)


def infer_vocab_size_from_tiger_state_dict(state_dict) -> int:
    """Read TIGER token embedding vocab size from a checkpoint state dict."""
    for key, weight in state_dict.items():
        if key.endswith("shared.weight"):
            return int(weight.shape[0])
    raise ValueError("Tiger checkpoint missing shared embedding weights")


def infer_sem_id_dim_from_tiger_state_dict(
    state_dict,
    codebook_size: int,
) -> int:
    """Infer ``sem_id_dim`` from TIGER embedding vocab (vocab = codebook * dim + 1)."""
    vocab_size = infer_vocab_size_from_tiger_state_dict(state_dict)
    if codebook_size <= 0 or (vocab_size - 1) % codebook_size != 0:
        raise ValueError(
            f"Cannot infer sem_id_dim from vocab_size={vocab_size} "
            f"and codebook_size={codebook_size}"
        )
    return (vocab_size - 1) // codebook_size


def _find_tiger_lm_head_weight(state_dict: dict) -> str | None:
    for key in ("model.lm_head.weight", "lm_head.weight"):
        if key in state_dict:
            return key
    for key in state_dict:
        if key.endswith("lm_head.weight"):
            return key
    return None


def prepare_tiger_state_dict_for_load(
    state_dict: dict,
    *,
    codebook_size: int,
    sem_id_dim: int | None = None,
) -> dict:
    state = dict(state_dict)
    state.pop("item_sid_embedding_table_encoder.weight", None)

    resolved_sem_id_dim = sem_id_dim
    if resolved_sem_id_dim is None:
        resolved_sem_id_dim = infer_sem_id_dim_from_tiger_state_dict(
            state, codebook_size
        )

    has_decoder_heads = any(k.startswith("decoder_heads.") for k in state)
    lm_key = _find_tiger_lm_head_weight(state)
    if has_decoder_heads and lm_key is not None:
        lm_weight = state[lm_key].clone()
        for level in range(resolved_sem_id_dim):
            head_key = f"decoder_heads.{level}.weight"
            if head_key in state:
                start = level * codebook_size + 1
                end = start + codebook_size
                lm_weight[start:end] = state[head_key]
        state[lm_key] = lm_weight

    for key in list(state.keys()):
        if (
            key.startswith("decoder_heads.")
            or key.startswith("pop_evidence_head.")
            or key.startswith("twmtl_loss.")
            or key.startswith("igd_loss.")
            or key.startswith("d2lr_loss.")
            or key.startswith("cft_loss.")
            or key == "log_vars"
            # TigerSprint: drop SPRINT power-iteration buffers (training only).
            or "sprint_emb" in key
            or key.endswith("_u")
            or key.endswith("_v")
        ):
            state.pop(key, None)

    return state


def resolve_sem_id_dim(
    dataset: "AmazonSeqDataset",
    sem_id_dim: int,
    *,
    strict: bool = False,
) -> int:
    """Resolve ``sem_id_dim`` from dataset state and optional config override.

    When ``add_disambiguation`` is enabled, TIGER uses
    ``rqkmeans_n_layers + 1`` tokens per item (e.g. 4 for 3-layer Balance-ID).
    """
    inferred = infer_sem_id_dim(dataset)
    if sem_id_dim == inferred:
        return sem_id_dim

    add_disambiguation = getattr(dataset, "add_disambiguation", False)
    n_layers = getattr(dataset, "rqkmeans_n_layers", 3)
    msg = (
        f"sem_id_dim={sem_id_dim} does not match dataset "
        f"(inferred={inferred}, rqkmeans_n_layers={n_layers}, "
        f"add_disambiguation={add_disambiguation})"
    )
    if strict:
        raise ValueError(msg)

    logger.warning("%s; using inferred value %d", msg, inferred)
    return inferred


def log_dedup_collision_stats(
    sem_ids_list: Sequence[Sequence[int]],
    *,
    base_dim: int = 3,
    codebook_size: int = 256,
) -> None:
    """Log collision statistics after deduplication suffix assignment."""
    if not sem_ids_list:
        return

    base_codes = [tuple(codes[:base_dim]) for codes in sem_ids_list]
    from collections import Counter

    counts = Counter(base_codes)
    collision_groups = sum(1 for c in counts.values() if c > 1)
    collision_items = sum(c for c in counts.values() if c > 1)
    max_collision = max(counts.values()) if counts else 0
    max_suffix = max(codes[-1] for codes in sem_ids_list)

    logger.info(
        "Semantic ID dedup: %d collision groups, %d items affected, "
        "max group size=%d, max dedup suffix=%d",
        collision_groups,
        collision_items,
        max_collision,
        max_suffix,
    )
    if max_suffix >= codebook_size:
        logger.info(
            "Max dedup suffix %d exceeds nominal codebook_size %d; "
            "TIGER will use embeddings_per_hierarchy=%d",
            max_suffix,
            codebook_size,
            max_suffix + 1,
        )
