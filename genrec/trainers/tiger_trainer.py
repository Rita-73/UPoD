import os
from typing import Dict, List

import gin
import torch
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from genrec.data.schemas import SeqData
from genrec.models.tiger import Tiger
from genrec.modules.evidence_bounded_popularity import (
    build_item_popularity_from_train_sequences,
    normalize_item_popularity,
)
from genrec.modules.ual import (
    EntropyWeightNormalizer,
    PopularityEvidenceHead,
    UalLoss,
    build_ual_assets_from_sem_ids,
    entropy_gate_per_level_sample_weights,
    entropy_gate_sample_weights,
)
from genrec.modules.semantic_id import (
    compute_vocab_size,
    resolve_embeddings_per_hierarchy,
    resolve_sem_id_dim,
)
from genrec.modules.utils import get_run_split, parse_config, setup_logger
from genrec.trainers.trainer_utils import (
    gpu_supports_bf16,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    setup_accelerator,
    setup_wandb,
)


def t5_collate(
    batch: List[SeqData],
    codebook_size: int = 256,
    sem_id_dim: int = 3,
    max_history_tokens: int = 60,
) -> Dict[str, torch.Tensor]:
    """Convert SeqData to flat T5 format with offset encoding.

    Offset encoding: token = code + codebook_idx * codebook_size + 1
    (0 is reserved for padding).
    """
    B = len(batch)

    input_ids = torch.zeros(B, max_history_tokens, dtype=torch.long)
    attention_mask = torch.zeros(B, max_history_tokens, dtype=torch.long)
    label_list = []

    for i, sample in enumerate(batch):
        raw_ids = sample.item_ids
        offset_ids = []
        for j, code in enumerate(raw_ids):
            cb_idx = j % sem_id_dim
            offset_ids.append(code + cb_idx * codebook_size + 1)

        L = len(offset_ids)
        if L > max_history_tokens:
            offset_ids = offset_ids[-max_history_tokens:]
            L = max_history_tokens

        start = max_history_tokens - L
        input_ids[i, start:] = torch.tensor(offset_ids, dtype=torch.long)
        attention_mask[i, start:] = 1

        offset_target = []
        for j, code in enumerate(sample.target_ids):
            cb_idx = j % sem_id_dim
            offset_target.append(code + cb_idx * codebook_size + 1)
        label_list.append(offset_target)

    target_items = torch.tensor([s.target_item for s in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor(label_list, dtype=torch.long),
        "target_items": target_items,
    }


def calculate_pos_index(preds, labels, maxk=20):
    labels_expanded = labels.unsqueeze(1).expand_as(preds)
    match = (preds == labels_expanded).all(dim=-1)
    first_match_mask = match.int().cumsum(dim=1) == 1
    return match & first_match_mask


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).float().mean().item()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1, device=pos_index.device).float()
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, device=pos_index.device))
    return dcg[:, :k].sum(dim=1).mean().item()


def per_sample_recall_ndcg(pos_index, k):
    """Per-sample recall hit and NDCG@k; each returns shape [B]."""
    recall = pos_index[:, :k].sum(dim=1).float()
    ranks = torch.arange(1, pos_index.shape[-1] + 1, device=pos_index.device).float()
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, device=pos_index.device))
    ndcg = dcg[:, :k].sum(dim=1)
    return recall, ndcg


def evaluate_tiger(
    model,
    loader,
    accelerator,
    *,
    beam_size: int,
    desc: str = "Eval",
):
    eval_model = accelerator.unwrap_model(model)
    eval_model.eval()
    device = accelerator.device
    top_ks = [5, 10]
    metrics = {f"Recall@{k}": 0.0 for k in top_ks}
    metrics.update({f"NDCG@{k}": 0.0 for k in top_ks})
    total = 0

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
            loader,
            desc=desc,
            disable=not accelerator.is_main_process,
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu()
            batch_size = input_ids.size(0)

            preds = eval_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
            )
            preds = preds[:, 1:]
            preds = preds.reshape(batch_size, beam_size, -1).cpu()

            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            for k in top_ks:
                recall, ndcg = per_sample_recall_ndcg(pos_index, k)
                metrics[f"Recall@{k}"] += recall.sum().item()
                metrics[f"NDCG@{k}"] += ndcg.sum().item()
            total += batch_size

            del input_ids, attention_mask, preds, pos_index
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def gather_scalar(value: float) -> float:
        tensor = torch.tensor([value], device=device, dtype=torch.float32)
        return accelerator.reduce(tensor, reduction="sum").item()

    total = int(gather_scalar(float(total)))
    result = {}
    for key, value in metrics.items():
        result[key] = gather_scalar(value) / total if total > 0 else 0.0
    return result


@gin.configurable
def train(
    epochs=200,
    batch_size=256,
    infer_batch_size=96,
    learning_rate=1e-4,
    weight_decay=0.0,
    dataset_folder="dataset",
    save_dir_root="out/{split}/",
    dataset=None,
    wandb_logging=True,
    wandb_project="UPoD_{split}",
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
    max_seq_len=20,
    lr_schedule="none",
    num_warmup_steps=0,
    beam_size=30,
    early_stop_patience=10,
    do_test_eval=True,
    eval_test_every_epoch=2,
    eval_test_on_improvement=True,
    eval_test_at_end=True,
    eval_valid_every_n_epochs=1,
    seed=2048,
    enable_ual=True,
    ual_beta=0.1,
    ual_beta_warmup_epochs=50,
    ual_num_bins=10,
    ual_ema_decay=0.4,
    ual_holdout=2,
    ual_use_log_norm=True,
    ual_lambda_mode="uniform",
    ual_lambda_p=1.0,
    ual_lambda_q=2.0,
    ual_lambda_floor=0.0,
    ual_lambda_uniform_mix=0.0,
    ual_lambda_per_level=None,
    ual_match_mode="onesided_cdf_hot",
    ual_target_mode="uniform",
    ual_reduce_mode="aggregate",
    enable_ual_entropy_gate=True,
    ual_entropy_gate_detach=True,
    ual_entropy_gate_ema_decay=0.99,
    ual_entropy_gate_per_level=False,
    ual_aux_weight=0.0,
    ual_aux_hidden=128,
    ual_gate_ema_decay=0.99,
    resume_from_checkpoint=None,
):
    split = get_run_split()
    save_dir_root = save_dir_root.replace("{split}", split)
    wandb_project = wandb_project.replace("{split}", split)

    set_seed(int(seed))

    accelerator = setup_accelerator(amp=False)
    device = accelerator.device
    logger = setup_logger(save_dir_root, name="UPoD")
    if accelerator.is_main_process:
        logger.info("Using train seed=%d", int(seed))

    if enable_ual_entropy_gate and not enable_ual:
        raise ValueError("enable_ual_entropy_gate requires enable_ual=True")

    train_dataset = dataset(
        root=dataset_folder,
        train_test_split="train",
        max_seq_len=max_seq_len,
    )
    valid_dataset = dataset(
        root=dataset_folder,
        train_test_split="valid",
        max_seq_len=max_seq_len,
    )
    test_dataset = dataset(
        root=dataset_folder,
        train_test_split="test",
        max_seq_len=max_seq_len,
    )

    id_type = getattr(train_dataset, "id_type", "rqkmeans_pop_balance")
    if id_type not in ("rqkmeans", "rqkmeans_pop_balance"):
        raise ValueError(
            f"Unknown id_type: {id_type!r}. "
            "Expected 'rqkmeans' or 'rqkmeans_pop_balance'."
        )
    sem_id_dim = resolve_sem_id_dim(train_dataset, sem_id_dim)
    codebook_size = getattr(train_dataset, "rqkmeans_codebook_size", codebook_size)
    if getattr(train_dataset, "add_disambiguation", False):
        logger.info(
            "SID dedup enabled: sem_id_dim=%d (rqkmeans_n_layers=%d + dedup layer)",
            sem_id_dim,
            getattr(train_dataset, "rqkmeans_n_layers", 3),
        )

    embeddings_per_hierarchy = resolve_embeddings_per_hierarchy(
        train_dataset, codebook_size
    )
    vocab_size = compute_vocab_size(embeddings_per_hierarchy, sem_id_dim)
    logger.info(
        "SID mode=%s: sem_id_dim=%d, embeddings_per_hierarchy=%d, vocab_size=%d",
        id_type,
        sem_id_dim,
        embeddings_per_hierarchy,
        vocab_size,
    )

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

    if wandb_logging and accelerator.is_main_process:
        setup_wandb(
            project=wandb_project,
            run_name=f"UPoD_{split}_{os.path.basename(save_dir_root.rstrip(os.sep)) or split}",
            config={
                **model_config,
                "lr": learning_rate,
                "batch_size": batch_size,
                "max_seq_len": max_seq_len,
                "beam_size": beam_size,
            },
        )

    max_history_tokens = max_seq_len * sem_id_dim
    collate_fn = lambda x: t5_collate(
        x, embeddings_per_hierarchy, sem_id_dim, max_history_tokens
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=infer_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=infer_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = Tiger(model_config)
    total_params, emb_params = model.num_parameters
    if accelerator.is_main_process:
        logger.info(f"Device: {device}, Params: {total_params:,} (emb: {emb_params:,})")

    ual_module = None
    ual_entropy_gate_normalizer = None
    item_pop_vec = None
    if enable_ual:
        raw_counts = build_item_popularity_from_train_sequences(
            train_dataset.sequences, holdout=ual_holdout
        )
        item_pop = normalize_item_popularity(raw_counts, use_log=ual_use_log_norm)
        ual_assets = build_ual_assets_from_sem_ids(
            train_dataset.sem_ids_list,
            item_pop,
            raw_counts,
            num_levels=sem_id_dim,
            codebook_size=embeddings_per_hierarchy,
            num_bins=ual_num_bins,
            lambda_mode=ual_lambda_mode,
            lambda_p=ual_lambda_p,
            lambda_q=ual_lambda_q,
            lambda_floor=ual_lambda_floor,
            lambda_uniform_mix=ual_lambda_uniform_mix,
            lambda_per_level=ual_lambda_per_level,
            target_mode=ual_target_mode,
        )
        ual_module = UalLoss(
            ual_assets.code_bin_onehot,
            ual_assets.target_dist,
            lambda_per_level=ual_assets.lambda_per_level,
            ema_decay=ual_ema_decay,
            match_mode=ual_match_mode,
            reduce_mode=ual_reduce_mode,
        ).to(device)
        if accelerator.is_main_process:
            logger.info(
                "UAL enabled: beta=%.4f (warmup %d epochs), "
                "num_bins=%d, ema_decay=%.3f, lambda_mode=%s, "
                "match_mode=%s, target_mode=%s, reduce_mode=%s, "
                "entropy_gate=%s, entropy_gate_detach=%s, seed=%d",
                ual_beta,
                ual_beta_warmup_epochs,
                ual_num_bins,
                ual_ema_decay,
                ual_lambda_mode,
                ual_match_mode,
                ual_target_mode,
                ual_reduce_mode,
                enable_ual_entropy_gate,
                ual_entropy_gate_detach,
                int(seed),
            )
        model.pop_evidence_head = PopularityEvidenceHead(
            d_model,
            hidden=ual_aux_hidden,
            ema_decay=ual_gate_ema_decay,
        )
        item_pop_vec = torch.zeros(train_dataset.num_items, dtype=torch.float32)
        for iid, pop in item_pop.items():
            if 0 < iid < item_pop_vec.shape[0]:
                item_pop_vec[iid] = pop
        item_pop_vec = item_pop_vec.to(device)

    if enable_ual_entropy_gate:
        gate_levels = int(sem_id_dim) if ual_entropy_gate_per_level else 1
        ual_entropy_gate_normalizer = EntropyWeightNormalizer(
            ema_decay=ual_entropy_gate_ema_decay,
            num_levels=gate_levels,
        ).to(device)
        if accelerator.is_main_process:
            logger.info(
                "UAL entropy gate enabled: detach=%s, ema_decay=%.3f, "
                "per_level=%s (g = [sg] H(p) / EMA(H))",
                ual_entropy_gate_detach,
                ual_entropy_gate_ema_decay,
                bool(ual_entropy_gate_per_level),
            )

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader, valid_loader, test_loader = accelerator.prepare(
        train_loader, valid_loader, test_loader
    )
    model, optimizer = accelerator.prepare(model, optimizer)

    if accelerator.is_main_process:
        logger.info(
            f"Train: {len(train_loader)} batches, Valid: {len(valid_loader)}, "
            f"Test: {len(test_loader)}"
        )
        if accelerator.num_processes > 1:
            logger.info(
                "Valid/Test eval sharded across %d processes",
                accelerator.num_processes,
            )

    scheduler = None
    if lr_schedule == "cosine" and epochs > 0:
        total_steps = len(train_loader) * epochs
        if num_warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.01, total_iters=num_warmup_steps
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=total_steps - num_warmup_steps
            )
            scheduler = SequentialLR(
                optimizer,
                [warmup_scheduler, cosine_scheduler],
                milestones=[num_warmup_steps],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        logger.info(
            f"Using cosine schedule, warmup={num_warmup_steps}, total_steps={total_steps}"
        )
    else:
        logger.info("Using plain Adam, no scheduler")

    best_valid_recall10 = 0.0
    best_epoch = -1
    early_stop_counter = 0
    start_epoch = 0
    last_epoch = -1

    if resume_from_checkpoint:
        resume_path = resume_from_checkpoint
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(save_dir_root, resume_path)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(
                f"resume_from_checkpoint not found: {resume_path}"
            )
        ckpt_data = load_checkpoint(
            resume_path,
            accelerator.unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger if accelerator.is_main_process else None,
        )
        start_epoch = int(ckpt_data.get("epoch", -1)) + 1
        best_valid_recall10 = float(ckpt_data.get("best_valid_recall10", 0.0))
        best_epoch = int(ckpt_data.get("best_epoch", -1))
        early_stop_counter = int(ckpt_data.get("early_stop_counter", 0))
        if accelerator.is_main_process:
            logger.info(
                "Resuming at epoch %d (best Valid R@10=%.4f @ epoch %d, "
                "early_stop=%d/%d)",
                start_epoch,
                best_valid_recall10,
                best_epoch,
                early_stop_counter,
                early_stop_patience,
            )

    if start_epoch >= epochs:
        if accelerator.is_main_process:
            logger.info(
                "Checkpoint already reached target epochs (%d >= %d); skipping train",
                start_epoch,
                epochs,
            )

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_ual_loss = 0.0
        beta_ual = 0.0
        if enable_ual and epoch >= ual_beta_warmup_epochs:
            beta_ual = ual_beta

        ual_stats = {}
        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_main_process,
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            ce_loss, logits, pop_rho_hat = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = ce_loss
            ual_loss = ce_loss.new_zeros(())
            aux_loss = ce_loss.new_zeros(())

            if enable_ual and ual_module is not None:
                hierarchy_logits = Tiger.hierarchy_logits_from_full(
                    logits, sem_id_dim, embeddings_per_hierarchy
                )
                sample_w = None
                if enable_ual_entropy_gate and ual_entropy_gate_normalizer is not None:
                    if ual_entropy_gate_per_level:
                        sample_w, entropy_gate_stats = (
                            entropy_gate_per_level_sample_weights(
                                hierarchy_logits,
                                codebook_size=embeddings_per_hierarchy,
                                normalizer=ual_entropy_gate_normalizer,
                                detach=ual_entropy_gate_detach,
                            )
                        )
                    else:
                        sample_w, entropy_gate_stats = entropy_gate_sample_weights(
                            hierarchy_logits[:, 0, :],
                            codebook_size=embeddings_per_hierarchy,
                            normalizer=ual_entropy_gate_normalizer,
                            detach=ual_entropy_gate_detach,
                        )
                ual_logits = (
                    hierarchy_logits if beta_ual > 0 else hierarchy_logits.detach()
                )
                ual_loss, ual_stats = ual_module(ual_logits, sample_w)
                if enable_ual_entropy_gate and ual_entropy_gate_normalizer is not None:
                    ual_stats = dict(ual_stats)
                    ual_stats.update(
                        {f"entropy_gate_{k}": v for k, v in entropy_gate_stats.items()}
                    )
                if pop_rho_hat is not None and item_pop_vec is not None:
                    head = accelerator.unwrap_model(model).pop_evidence_head
                    target_items = batch["target_items"].to(device)
                    rho_true = item_pop_vec[target_items.clamp(min=0)]
                    aux_loss = head.mse_loss(rho_true, pop_rho_hat)
                loss = ce_loss + ual_aux_weight * aux_loss
                if beta_ual > 0:
                    loss = loss + beta_ual * ual_loss
                else:
                    ual_loss = ual_loss.detach()

            accelerator.backward(loss)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            if enable_ual:
                total_ual_loss += ual_loss.item()

        avg_loss = total_loss / len(train_loader)
        log_dict = {
            "epoch": epoch,
            "train/loss": avg_loss,
            "train/loss_ce": total_ce_loss / len(train_loader),
        }
        if enable_ual:
            log_dict["train/ual_beta"] = beta_ual
            log_dict["train/loss_ual"] = total_ual_loss / len(train_loader)
            if ual_stats:
                for key, value in ual_stats.items():
                    log_dict[f"train/ual_{key}"] = value.item()

        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} - loss: {avg_loss:.4f}")

        if (epoch + 1) % eval_valid_every_n_epochs == 0 or epoch == 0:
            accelerator.wait_for_everyone()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            model.eval()
            valid_metrics = evaluate_tiger(
                model,
                valid_loader,
                accelerator,
                beam_size=beam_size,
                desc=f"Valid (Epoch {epoch})",
            )
            accelerator.wait_for_everyone()

            test_flag = torch.zeros(1, device=device)
            if accelerator.is_main_process:
                for k, v in valid_metrics.items():
                    log_dict[f"eval/valid_{k}"] = v
                logger.info(f"Epoch {epoch} - Valid: {valid_metrics}")

                cur_recall10 = valid_metrics["Recall@10"]
                if cur_recall10 > best_valid_recall10:
                    best_valid_recall10 = cur_recall10
                    best_epoch = epoch
                    early_stop_counter = 0
                    os.makedirs(save_dir_root, exist_ok=True)
                    torch.save(
                        accelerator.unwrap_model(model).state_dict(),
                        os.path.join(save_dir_root, "best_model.pt"),
                    )
                    logger.info(
                        f"New best Valid R@10={cur_recall10:.4f} at epoch {epoch}"
                    )
                    if do_test_eval and eval_test_on_improvement:
                        test_flag[0] = 1
                else:
                    early_stop_counter += 1
                    logger.info(
                        f"No improvement. Counter: {early_stop_counter}/{early_stop_patience}"
                    )
                    if (
                        do_test_eval
                        and eval_test_every_epoch > 0
                        and (epoch + 1) % eval_test_every_epoch == 0
                    ):
                        test_flag[0] = 1

            if accelerator.num_processes > 1:
                test_flag = accelerator.reduce(test_flag, reduction="max")
            if test_flag.item() == 1:
                test_metrics = evaluate_tiger(
                    model,
                    test_loader,
                    accelerator,
                    beam_size=beam_size,
                    desc=f"Test (Epoch {epoch})",
                )
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    logger.info(f"Epoch {epoch} - Test: {test_metrics}")
                    for k, v in test_metrics.items():
                        log_dict[f"eval/test_{k}"] = v

            model.train()

        if wandb_logging and accelerator.is_main_process:
            wandb.log(log_dict)

        stop_flag = torch.tensor([0], device=device)
        if accelerator.is_main_process and early_stop_counter >= early_stop_patience:
            stop_flag[0] = 1
        if accelerator.num_processes > 1:
            stop_flag = accelerator.reduce(stop_flag, reduction="max")
        if stop_flag.item() == 1:
            if accelerator.is_main_process:
                logger.info(
                    f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}"
                )
            last_epoch = epoch
            break

        if accelerator.is_main_process:
            last_ckpt_path = os.path.join(save_dir_root, "last_checkpoint.pt")
            save_checkpoint(
                last_ckpt_path,
                accelerator.unwrap_model(model),
                optimizer,
                scheduler=scheduler,
                epoch=epoch,
                extra={
                    "best_valid_recall10": best_valid_recall10,
                    "best_epoch": best_epoch,
                    "early_stop_counter": early_stop_counter,
                },
                logger=logger,
            )

        last_epoch = epoch
        accelerator.wait_for_everyone()

    run_final_test = torch.zeros(1, device=device)
    if (
        accelerator.is_main_process
        and do_test_eval
        and eval_test_at_end
        and best_epoch >= 0
    ):
        run_final_test[0] = 1
    if accelerator.num_processes > 1:
        run_final_test = accelerator.reduce(run_final_test, reduction="max")
    if run_final_test.item() == 1:
        best_path = os.path.join(save_dir_root, "best_model.pt")
        if os.path.exists(best_path):
            accelerator.unwrap_model(model).load_state_dict(
                torch.load(best_path, map_location=device, weights_only=True)
            )
            if accelerator.is_main_process:
                logger.info(f"Loaded best model from {best_path} for final test eval")
        accelerator.wait_for_everyone()
        model.eval()
        test_metrics = evaluate_tiger(
            model,
            test_loader,
            accelerator,
            beam_size=beam_size,
            desc="Test (final)",
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info(f"Final Test: {test_metrics}")
            if wandb_logging:
                wandb.log({f"eval/final_test_{k}": v for k, v in test_metrics.items()})

    if accelerator.is_main_process:
        logger.info(
            f"Training done. Best Valid R@10={best_valid_recall10:.4f} at epoch {best_epoch}"
        )

    if wandb_logging and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    import genrec.trainers.tiger_trainer as _mod

    _mod.train()
