"""Evaluate a trained TIGER checkpoint with UPoD metrics (Recall/NDCG/PL/MGU/ARP)."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import gin
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from genrec.data.amazon import is_movielens_split
from genrec.data.fair_eval import (
    load_fair_eval_metadata,
    mapping_cache_path,
)
from genrec.data.schemas import SeqData
from genrec.models.tiger import Tiger
from genrec.modules.ugd import setup_tiger_ugd_debiaser
from genrec.modules.fair_metrics import (
    FairMetricsTracker,
    PopularityLiftTracker,
)
from genrec.modules.semantic_id import (
    compute_vocab_size,
    infer_sem_id_dim_from_tiger_state_dict,
    prepare_tiger_state_dict_for_load,
    resolve_embeddings_per_hierarchy,
    resolve_sem_id_dim,
)
from genrec.modules.utils import get_run_split, parse_config, setup_logger
from genrec.trainers.tiger_trainer import (
    calculate_pos_index,
    ndcg_at_k,
    recall_at_k,
    t5_collate,
)
from genrec.trainers.trainer_utils import gpu_supports_bf16, setup_accelerator


def load_genrec_asin_to_item_id(root: str, split: str) -> Dict[str, int]:
    """Load 1-indexed ASIN->item_id mapping used by fair-eval JSON files."""
    path = mapping_cache_path(root, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Item-id mapping not found: {path}. "
            f"Process the split first (AmazonSeqDataset / fit_id)."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_sem_to_item_map(
    sem_ids_list: Sequence[Sequence[int]], sem_id_dim: int
) -> Dict[Tuple[int, ...], List[int]]:
    mapping: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for item_id, codes in enumerate(sem_ids_list):
        key = tuple(int(c) for c in codes[:sem_id_dim])
        mapping[key].append(int(item_id))
    return mapping


def decode_offset_tokens(
    offset_tokens: Sequence[int], codebook_size: int, sem_id_dim: int
) -> Tuple[int, ...]:
    codes = []
    for pos, tok in enumerate(offset_tokens):
        tok = int(tok)
        if tok <= 0:
            break
        cb_idx = pos % sem_id_dim
        codes.append(tok - 1 - cb_idx * codebook_size)
    return tuple(codes[:sem_id_dim])


def tiger_item_to_fair_item(item_id: int) -> int:
    """TIGER uses 0-indexed item ids; fair-eval JSON uses 1-indexed ids."""
    return int(item_id) + 1


def decode_ranked_tiger_items(
    preds_row: torch.Tensor,
    *,
    beam_size: int,
    codebook_size: int,
    sem_id_dim: int,
    sem_to_items: Dict[Tuple[int, ...], List[int]],
) -> List[int]:
    ranked: List[int] = []
    for beam_idx in range(beam_size):
        codes = decode_offset_tokens(
            preds_row[beam_idx].tolist(), codebook_size, sem_id_dim
        )
        candidates = sem_to_items.get(codes, [])
        ranked.append(int(candidates[0]) if candidates else -1)
    return ranked


class TigerEvalDataset(Dataset):
    """Wrap AmazonSeqDataset to expose raw item ids for PL/MGU/ARP."""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        seq: SeqData = self.base[idx]
        sample = self.base.samples[idx]
        return seq, self.base._truncate_history(sample["history"])


def t5_eval_collate(
    batch,
    codebook_size: int = 256,
    sem_id_dim: int = 3,
    max_history_tokens: int = 60,
):
    seq_batch = [item[0] for item in batch]
    history_items = [item[1] for item in batch]
    out = t5_collate(seq_batch, codebook_size, sem_id_dim, max_history_tokens)
    out["history_items"] = history_items
    return out


def evaluate(
    model,
    dataloader,
    accelerator,
    *,
    codebook_size: int,
    sem_id_dim: int,
    beam_size: int,
    decode_mode: str = "beam",
    eval_top_ks: Sequence[int] = (5, 10),
    fair_tracker: FairMetricsTracker | None = None,
    pl_tracker: PopularityLiftTracker | None = None,
    sem_to_items: Dict[Tuple[int, ...], List[int]] | None = None,
) -> Dict[str, float]:
    """Run beam-search eval; return Recall/NDCG/PL/MGU/ARP."""
    eval_model = accelerator.unwrap_model(model)
    eval_model.eval()
    device = accelerator.device
    eval_top_ks = list(eval_top_ks)

    all_recalls = {k: [] for k in eval_top_ks}
    all_ndcgs = {k: [] for k in eval_top_ks}
    local_fair_tracker = None
    if fair_tracker is not None and fair_tracker.enable:
        local_fair_tracker = FairMetricsTracker(
            topk_list=fair_tracker.topk_list,
            item_popularity=fair_tracker.item_popularity,
            popularity_group_cutoffs=fair_tracker.popularity_group_cutoffs,
            enable=True,
        )
    local_pl_tracker = pl_tracker.clone_for_eval() if pl_tracker is not None else None
    need_item_metrics = (
        sem_to_items is not None
        and (local_fair_tracker is not None or local_pl_tracker is not None)
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    eval_dtype = (
        torch.bfloat16
        if device.type == "cuda" and gpu_supports_bf16()
        else torch.float32
    )
    with torch.no_grad(), torch.amp.autocast(
        "cuda", dtype=eval_dtype, enabled=eval_dtype == torch.bfloat16
    ):
        for batch in tqdm(
            dataloader,
            desc="Evaluating",
            disable=not accelerator.is_main_process,
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu()
            history_items = batch["history_items"]
            batch_size = input_ids.size(0)

            preds = eval_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
                decode_mode=decode_mode,
            )
            preds = preds[:, 1:]
            preds = preds.reshape(batch_size, beam_size, -1).cpu()

            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            for k in eval_top_ks:
                all_recalls[k].append(recall_at_k(pos_index, k))
                all_ndcgs[k].append(ndcg_at_k(pos_index, k))

            if need_item_metrics:
                for i in range(batch_size):
                    history_tiger = [int(h) for h in history_items[i]]
                    ranked_tiger = decode_ranked_tiger_items(
                        preds[i],
                        beam_size=beam_size,
                        codebook_size=codebook_size,
                        sem_id_dim=sem_id_dim,
                        sem_to_items=sem_to_items,
                    )
                    history_fair = [
                        tiger_item_to_fair_item(h) for h in history_tiger
                    ]
                    ranked_fair = [
                        tiger_item_to_fair_item(item_id) if item_id >= 0 else 0
                        for item_id in ranked_tiger
                    ]
                    if local_fair_tracker is not None:
                        local_fair_tracker.update(
                            ranked_item_ids=ranked_fair,
                            history_items=history_fair,
                        )
                    if local_pl_tracker is not None:
                        local_pl_tracker.update(
                            ranked_item_ids=ranked_fair,
                            history_items=history_fair,
                        )

            del input_ids, attention_mask, preds, pos_index
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metrics: Dict[str, float] = {}
    for k in eval_top_ks:
        metrics[f"Recall@{k}"] = sum(all_recalls[k]) / max(len(all_recalls[k]), 1)
        metrics[f"NDCG@{k}"] = sum(all_ndcgs[k]) / max(len(all_ndcgs[k]), 1)

    if local_fair_tracker is not None:
        fair_metrics = local_fair_tracker.finalize(num_users=local_fair_tracker.num_users)
        for k in local_fair_tracker.topk_list:
            arp = fair_metrics.get(f"arp@{k}")
            mgu = fair_metrics.get(f"pop_mgu_hist@{k}")
            if isinstance(arp, (int, float)):
                metrics[f"ARP@{k}"] = float(arp)
            if isinstance(mgu, (int, float)):
                metrics[f"MGU@{k}"] = float(mgu)
    if local_pl_tracker is not None:
        for key, value in local_pl_tracker.finalize().items():
            if key.startswith("pl_overall@"):
                k = key.split("@", 1)[1]
                metrics[f"PL@{k}"] = float(value)

    model.train()
    return metrics


@gin.configurable
def eval_checkpoint(
    checkpoint_path=None,
    dataset_folder="dataset",
    split="beauty",
    save_dir_root="out/{split}/",
    pretrained_rqkmeans_path="./out/{split}/rqkmeans.pt",
    dataset=None,
    infer_batch_size=32,
    beam_size=30,
    decode_mode="beam",
    eval_top_ks=(5, 10),
    num_layers=4,
    num_decoder_layers=4,
    d_model=128,
    d_ff=1024,
    num_heads=6,
    d_kv=64,
    dropout_rate=0.1,
    feed_forward_proj="relu",
    codebook_size=256,
    sem_id_dim=3,
    max_seq_len=50,
    amp=False,
    mixed_precision_type="no",
    enable_fair_metrics=True,
    fair_eval_top_ks=(5, 10),
    item_popularity_path=None,
    popularity_group_cutoffs=(0.2, 0.4, 0.6, 0.8),
    token_num_levels=None,
    enable_pl_metrics=True,
    auto_infer_from_checkpoint=True,
    run_valid=True,
    run_test=True,
    output_json=None,
    enable_bias_debias=False,
    debias_backend="ugd",
    debias_alpha=1.0,
    debias_item_popularity_path=None,
    ugd_level_weights=None,
    ugd_entropy_gamma=False,
    ugd_gamma_min=0.5,
    ugd_gamma_max=1.5,
    ugd_gamma_levels=None,
    ugd_entropy_gamma_share_level0=False,
):
    """Load checkpoint and run valid/test evaluation."""
    if decode_mode != "beam":
        raise ValueError(f"decode_mode must be 'beam', got {decode_mode!r}")
    if enable_bias_debias and debias_backend != "ugd":
        raise ValueError(
            "UPoD eval only supports debias_backend='ugd', "
            f"got {debias_backend!r}"
        )

    split = get_run_split(default=split)
    save_dir_root = save_dir_root.replace("{split}", split)
    pretrained_rqkmeans_path = pretrained_rqkmeans_path.replace("{split}", split)

    if checkpoint_path is None:
        checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    if checkpoint_path is None:
        checkpoint_path = os.path.join(save_dir_root, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    inferred_sem_id_dim = None
    inferred_add_disambiguation = None
    if auto_infer_from_checkpoint and sem_id_dim != 1:
        inferred_sem_id_dim = infer_sem_id_dim_from_tiger_state_dict(
            state_dict, codebook_size
        )
        inferred_add_disambiguation = inferred_sem_id_dim > 3
        sem_id_dim = inferred_sem_id_dim
        if token_num_levels is None:
            token_num_levels = inferred_sem_id_dim

    logger = setup_logger(save_dir_root, name="UPoD_eval")
    accelerator = setup_accelerator(amp=amp, mixed_precision_type=mixed_precision_type)
    device = accelerator.device

    if auto_infer_from_checkpoint and inferred_sem_id_dim is not None:
        logger.info(
            "Inferred from checkpoint: sem_id_dim=%d add_disambiguation=%s "
            "vocab_size=%d",
            inferred_sem_id_dim,
            inferred_add_disambiguation,
            codebook_size * inferred_sem_id_dim + 1,
        )

    dataset_kwargs = dict(
        root=dataset_folder,
        split=split,
        max_seq_len=max_seq_len,
        pretrained_rqkmeans_path=pretrained_rqkmeans_path,
    )
    if inferred_add_disambiguation is not None:
        dataset_kwargs["add_disambiguation"] = inferred_add_disambiguation
    train_dataset = dataset(**dataset_kwargs, train_test_split="train")
    valid_dataset = (
        dataset(**dataset_kwargs, train_test_split="valid") if run_valid else None
    )
    test_dataset = (
        dataset(**dataset_kwargs, train_test_split="test") if run_test else None
    )

    id_type = getattr(train_dataset, "id_type", "rqkmeans_pop_balance")
    if id_type not in ("rqkmeans", "rqkmeans_pop_balance"):
        raise ValueError(
            f"Unknown id_type: {id_type!r}. "
            "Expected 'rqkmeans' or 'rqkmeans_pop_balance'."
        )
    if not os.path.exists(pretrained_rqkmeans_path):
        raise FileNotFoundError(
            "RQ-KMeans checkpoint not found: "
            f"{pretrained_rqkmeans_path}. "
            f"Fit first: SPLIT={split} bash scripts/fit_id.sh"
        )
    if not auto_infer_from_checkpoint:
        sem_id_dim = resolve_sem_id_dim(train_dataset, sem_id_dim)
    ds_codebook = getattr(train_dataset, "rqkmeans_codebook_size", codebook_size)
    codebook_size = max(int(codebook_size), int(ds_codebook))
    if getattr(train_dataset, "add_disambiguation", False):
        logger.info(
            "SID dedup enabled: sem_id_dim=%d (rqkmeans_n_layers=%d + dedup layer)",
            sem_id_dim,
            getattr(train_dataset, "rqkmeans_n_layers", 3),
        )

    embeddings_per_hierarchy = resolve_embeddings_per_hierarchy(
        train_dataset, codebook_size
    )
    logger.info(
        "SID mode=%s: sem_id_dim=%d, embeddings_per_hierarchy=%d, vocab_size=%d",
        id_type,
        sem_id_dim,
        embeddings_per_hierarchy,
        compute_vocab_size(embeddings_per_hierarchy, sem_id_dim),
    )

    sem_to_items = build_sem_to_item_map(train_dataset.sem_ids_list, sem_id_dim)
    collision_groups = sum(1 for items in sem_to_items.values() if len(items) > 1)
    if collision_groups:
        logger.warning(
            "Semantic-id collisions: %d groups may map multiple items to one code",
            collision_groups,
        )

    fair_tracker = None
    if enable_fair_metrics:
        asin_to_item_id = None
        if not is_movielens_split(split):
            asin_to_item_id = load_genrec_asin_to_item_id(dataset_folder, split)
        item_popularity, pop_path = load_fair_eval_metadata(
            root=dataset_folder,
            split=split,
            train_sequences=train_dataset.sequences,
            asin_to_item_id=asin_to_item_id,
            item_popularity_path=item_popularity_path,
        )
        fair_tracker = FairMetricsTracker(
            topk_list=fair_eval_top_ks,
            item_popularity=item_popularity,
            popularity_group_cutoffs=popularity_group_cutoffs,
            enable=True,
        )
        logger.info(
            "UPoD metrics enabled (Recall/NDCG/PL/MGU/ARP): top_ks=%s popularity=%s",
            fair_eval_top_ks,
            pop_path,
        )

    pl_tracker = None
    if enable_fair_metrics and enable_pl_metrics and fair_tracker is not None:
        pl_tracker = PopularityLiftTracker(
            topk_list=fair_eval_top_ks,
            item_popularity=fair_tracker.item_popularity,
            enable=True,
        )
        logger.info("Popularity Lift enabled: overall PL top_ks=%s", fair_eval_top_ks)

    max_history_tokens = max_seq_len * sem_id_dim
    collate_fn = lambda batch: t5_eval_collate(
        batch, embeddings_per_hierarchy, sem_id_dim, max_history_tokens
    )
    valid_loader = None
    if run_valid and valid_dataset is not None:
        valid_loader = DataLoader(
            TigerEvalDataset(valid_dataset),
            batch_size=infer_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    test_loader = None
    if run_test and test_dataset is not None:
        test_loader = DataLoader(
            TigerEvalDataset(test_dataset),
            batch_size=infer_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    vocab_size = compute_vocab_size(embeddings_per_hierarchy, sem_id_dim)
    model_config = {
        "num_layers": num_layers,
        "num_decoder_layers": num_decoder_layers,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_heads": num_heads,
        "d_kv": d_kv,
        "dropout_rate": dropout_rate,
        "vocab_size": vocab_size,
        "pad_token_id": 0,
        "eos_token_id": 0,
        "feed_forward_proj": feed_forward_proj,
        "sem_id_dim": sem_id_dim,
        "embeddings_per_hierarchy": embeddings_per_hierarchy,
        "codebook_size": codebook_size,
    }
    model = Tiger(model_config)
    state_dict = prepare_tiger_state_dict_for_load(
        state_dict,
        codebook_size=embeddings_per_hierarchy,
        sem_id_dim=sem_id_dim,
    )
    model.load_state_dict(state_dict, strict=False)

    item_pop_path = debias_item_popularity_path or item_popularity_path
    if enable_bias_debias and item_pop_path is None:
        default_pop = os.path.join(
            dataset_folder,
            "processed",
            split,
            "fair_eval",
            "item_popularity.json",
        )
        if os.path.exists(default_pop):
            item_pop_path = default_pop

    resolved_token_levels = token_num_levels
    if resolved_token_levels is None:
        resolved_token_levels = (
            len(train_dataset.sem_ids_list[0])
            if train_dataset.sem_ids_list
            else sem_id_dim
        )

    ugd_debiaser = setup_tiger_ugd_debiaser(
        model,
        enable=enable_bias_debias and debias_backend == "ugd",
        train_sequences=train_dataset.sequences,
        item_popularity_path=item_pop_path,
        sem_ids_list=train_dataset.sem_ids_list,
        num_levels=resolved_token_levels,
        codebook_size=embeddings_per_hierarchy,
        alpha=debias_alpha,
        level_weights=ugd_level_weights,
        entropy_gamma=ugd_entropy_gamma,
        gamma_min=ugd_gamma_min,
        gamma_max=ugd_gamma_max,
        gamma_levels=ugd_gamma_levels,
        entropy_gamma_share_level0=ugd_entropy_gamma_share_level0,
        device=device,
    )

    model = accelerator.prepare(model)

    logger.info(
        "Eval checkpoint=%s split=%s valid=%s test=%s decode=%s beam=%d "
        "infer_bs=%d device=%s",
        checkpoint_path,
        split,
        len(valid_dataset) if valid_dataset is not None else 0,
        len(test_dataset) if test_dataset is not None else 0,
        decode_mode,
        beam_size,
        infer_batch_size,
        device,
    )

    results = {
        "beam_size": beam_size,
        "checkpoint_path": checkpoint_path,
        "split": split,
    }

    eval_kwargs = dict(
        codebook_size=embeddings_per_hierarchy,
        sem_id_dim=sem_id_dim,
        beam_size=beam_size,
        decode_mode=decode_mode,
        eval_top_ks=eval_top_ks,
        fair_tracker=fair_tracker,
        sem_to_items=sem_to_items,
    )

    if run_valid and valid_loader is not None:
        valid_pl_tracker = (
            pl_tracker.clone_for_eval() if pl_tracker is not None else None
        )
        results["valid"] = evaluate(
            model,
            valid_loader,
            accelerator,
            pl_tracker=valid_pl_tracker,
            **eval_kwargs,
        )
        if accelerator.is_main_process:
            logger.info(
                "Valid: "
                + ", ".join(f"{k}={v:.4f}" for k, v in results["valid"].items())
            )

    if run_test and test_loader is not None:
        test_pl_tracker = (
            pl_tracker.clone_for_eval() if pl_tracker is not None else None
        )
        test_metrics = evaluate(
            model,
            test_loader,
            accelerator,
            pl_tracker=test_pl_tracker,
            **eval_kwargs,
        )
        results["test"] = test_metrics
        if accelerator.is_main_process:
            logger.info(
                "Test:  "
                + ", ".join(
                    f"{k}={v:.4f}"
                    for k, v in test_metrics.items()
                    if isinstance(v, (int, float))
                )
            )

    if ugd_debiaser is not None:
        if (
            getattr(ugd_debiaser, "entropy", None) is not None
            and ugd_debiaser.entropy.enable
        ):
            gamma_summary = ugd_debiaser.entropy.summary()
            logger.info(
                "Entropy gamma per-level summary (over decode steps): %s",
                gamma_summary,
            )

    if accelerator.is_main_process:
        if output_json is None:
            output_json = os.path.join(save_dir_root, f"eval_beam{beam_size}.json")
        payload = {
            "beam_size": results["beam_size"],
            "checkpoint_path": results["checkpoint_path"],
            "split": results["split"],
        }
        if "test" in results:
            payload["test"] = results["test"]
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        logger.info("Saved eval results to %s", output_json)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parse_config()
    # Running as a script registers @gin.configurable under __main__, while gin
    # bindings target genrec.trainers.tiger_eval.eval_checkpoint.
    import importlib

    importlib.import_module("genrec.trainers.tiger_eval").eval_checkpoint()
