
import os
import gin
import numpy as np

from genrec.data.amazon import AmazonItemDataset
from genrec.data.item_popularity import (
    load_popularity_weights_for_item_dataset,
)
from genrec.models.rqkmeans_pop_balance import RqKmeansPopBalance
from genrec.modules.utils import parse_config, get_run_split
from genrec.trainers.trainer_utils import set_seed


@gin.configurable
def fit_rqkmeans_pop_balance(
    dataset=AmazonItemDataset,
    dataset_folder: str = "dataset",
    encoder_model_name: str = "./models_hub/sentence-t5-xl",
    item_embedding_mode: str = "baseline",
    n_layers: int = 3,
    codebook_size: int = 256,
    save_dir_root: str = "out/{split}",
    random_state: int = 42,
    max_iter: int = 100,
    batch_size: int = 4096,
    balance_scope: str = "first",
    balance_lambda: float = 1.0,
    sinkhorn_epsilon: float = 0.1,
    sinkhorn_max_iter: int = 100,
    popularity_holdout: int = 2,
    weight_mode: str = "popularity",
    assignment_mode: str = "ot",
    seed: int = 42,
):
    set_seed(seed)
    split = get_run_split()
    save_dir_root = save_dir_root.replace("{split}", split)

    if balance_scope not in {"first", "second", "third", "all"}:
        raise ValueError(
            f"balance_scope must be 'first', 'second', 'third', or 'all', "
            f"got {balance_scope!r}"
        )
    if weight_mode not in {"popularity", "count", "raw"}:
        raise ValueError(
            f"weight_mode must be 'popularity', 'count', or 'raw', "
            f"got {weight_mode!r}"
        )
    if assignment_mode not in {"ot", "random"}:
        raise ValueError(
            f"assignment_mode must be 'ot' or 'random', got {assignment_mode!r}"
        )

    print("=== Pop-Balance RQ-KMeans Fitting ===")
    print(f"  split: {split}")
    print(f"  encoder: {encoder_model_name}")
    print(f"  item_embedding_mode: {item_embedding_mode}")
    print(f"  balance_scope: {balance_scope}")
    print(f"  balance_lambda: {balance_lambda}")
    print(f"  weight_mode: {weight_mode}")
    print(f"  assignment_mode: {assignment_mode}")
    print(f"  popularity_holdout: {popularity_holdout} (train-only counts)")
    print(f"  sinkhorn_epsilon: {sinkhorn_epsilon}")
    print(f"  codebook_size: {codebook_size}, n_layers: {n_layers}")
    print(f"  save_dir: {save_dir_root}")

    print("\n--- Loading embeddings ---")
    ds_kwargs = dict(
        root=dataset_folder,
        split=split,
        encoder_model_name=encoder_model_name,
        item_embedding_mode=item_embedding_mode,
    )
    item_dataset = dataset(**ds_kwargs)
    embeddings = item_dataset.embeddings.astype(np.float32)
    if weight_mode == "count":
        # Uniform item mass: balance cluster *counts*, not log-popularity.
        n_items = len(embeddings)
        item_weights = np.full(n_items, 1.0 / n_items, dtype=np.float64)
        weight_label = "Count (uniform) weights"
    else:
        mass_transform = "raw" if weight_mode == "raw" else "log"
        item_weights = load_popularity_weights_for_item_dataset(
            item_dataset,
            root=dataset_folder,
            split=split,
            holdout=popularity_holdout,
            mass_transform=mass_transform,
        )
        weight_label = (
            "Raw-mass weights" if weight_mode == "raw" else "Popularity (log) weights"
        )
    print(f"Loaded {len(embeddings)} embeddings, dim={embeddings.shape[1]}")
    print(
        "%s: min=%.6f max=%.6f mean=%.6f"
        % (weight_label, item_weights.min(), item_weights.max(), item_weights.mean())
    )

    print(
        f"\n--- Fitting Balance RQ-KMeans "
        f"(weight_mode={weight_mode}, assignment={assignment_mode}) ---"
    )
    rqkmeans = RqKmeansPopBalance(
        n_layers=n_layers,
        codebook_size=codebook_size,
        random_state=random_state,
        max_iter=max_iter,
        batch_size=batch_size,
        balance_scope=balance_scope,
        balance_lambda=balance_lambda,
        sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_max_iter=sinkhorn_max_iter,
    )
    rqkmeans.weight_mode = weight_mode
    rqkmeans.assignment_mode = assignment_mode
    rqkmeans.fit(embeddings, item_weights=item_weights)

    print("\n--- Collision Statistics ---")
    stats = rqkmeans.collision_rate(embeddings)
    print(f"  Total items: {stats['total']}")
    print(f"  Unique SIDs: {stats['unique']}")
    print(f"  Collision rate: {stats['collision_rate']:.4%}")
    print(f"  Max collision group: {stats['max_collision']}")

    os.makedirs(save_dir_root, exist_ok=True)
    save_path = os.path.join(save_dir_root, "rqkmeans.pt")
    rqkmeans.save(save_path)
    print(f"\n=== Done. Saved to {save_path} ===")


if __name__ == "__main__":
    parse_config()
    # gin files also `import` this module, so the bound configurable is the
    # package-qualified copy — not `__main__.fit_rqkmeans_pop_balance`.
    import genrec.trainers.rqkmeans_pop_balance_trainer as _mod

    _mod.fit_rqkmeans_pop_balance()
