import gzip
import json
import logging
import os
import shutil
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import gin

from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Dict, List, Optional

from genrec.data.item_popularity import (
    ITEM_EMBEDDING_MODES,
    embedding_mode_suffix,
    format_item_metadata_text,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = "dataset"
DEFAULT_ENCODER_MODEL_NAME = os.environ.get(
    "GENREC_ENCODER_MODEL_NAME",
    "sentence-transformers/sentence-t5-xl",
)

AMAZON_REVIEW_BASE_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
MOVIELENS_SPLIT = "movielens1m"
AMAZON_SPLITS = frozenset({"beauty", "sports", "toys"})
SPLIT_ALIASES = {
    "movielens-1m": MOVIELENS_SPLIT,
    "ml1m": MOVIELENS_SPLIT,
}
MOVIELENS_DOWNLOAD = {
    "urls": [
        "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "https://files.grouplens.org/privateset/MovieLens1M.zip",
    ],
    "archive": "ml-1m.zip",
    "files": {
        "ratings.dat": "ratings.dat",
        "movies.dat": "movies.dat",
    },
}

DATASET_CONFIGS = {
    "beauty": {
        "reviews": "reviews_Beauty_5.json.gz",
        "meta": "meta_Beauty.json.gz",
    },
    "sports": {
        "reviews": "reviews_Sports_and_Outdoors_5.json.gz",
        "meta": "meta_Sports_and_Outdoors.json.gz",
    },
    "toys": {
        "reviews": "reviews_Toys_and_Games_5.json.gz",
        "meta": "meta_Toys_and_Games.json.gz",
    },
}


def canonical_split(split: str) -> str:
    key = split.lower()
    return SPLIT_ALIASES.get(key, key)


def is_amazon_split(split: str) -> bool:
    return canonical_split(split) in AMAZON_SPLITS


def is_movielens_split(split: str) -> bool:
    return canonical_split(split) == MOVIELENS_SPLIT


def list_dataset_splits() -> List[str]:
    return sorted(AMAZON_SPLITS | {MOVIELENS_SPLIT})


def raw_dir_for(root: str, split: str) -> str:
    return os.path.join(root, canonical_split(split), "raw")


def processed_dir_for(root: str, split: str) -> str:
    return os.path.join(root, canonical_split(split), "processed")


def user2item_path(root: str, split: str) -> str:
    return os.path.join(processed_dir_for(root, split), "user2item.pkl")


def item2title_path(root: str, split: str) -> str:
    return os.path.join(processed_dir_for(root, split), "item2title.pkl")


def download_file(url: str, dest_path: str) -> None:
    """Download file with progress bar."""
    if os.path.exists(dest_path):
        logger.info(f"File already exists: {dest_path}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info(f"Downloading {url} -> {dest_path}")

    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

    logger.info(f"Downloaded: {dest_path}")


def ensure_amazon_raw_data(root: str, split: str, download_meta: bool = False) -> None:
    """Download Amazon 2014 raw files for a split if they are missing."""
    split = canonical_split(split)
    if not is_amazon_split(split):
        raise ValueError(f"Unknown Amazon split: {split}. Available: {sorted(AMAZON_SPLITS)}")

    config = DATASET_CONFIGS[split]
    raw_dir = raw_dir_for(root, split)
    os.makedirs(raw_dir, exist_ok=True)

    reviews_path = os.path.join(raw_dir, config["reviews"])
    if not os.path.exists(reviews_path):
        reviews_url = f"{AMAZON_REVIEW_BASE_URL}/{config['reviews']}"
        download_file(reviews_url, reviews_path)

    if download_meta:
        meta_path = os.path.join(raw_dir, config["meta"])
        if not os.path.exists(meta_path):
            meta_url = f"{AMAZON_REVIEW_BASE_URL}/{config['meta']}"
            download_file(meta_url, meta_path)


def parse_gzip_json(path: str):
    """Parse gzipped JSON file line by line."""
    with gzip.open(path, 'rt', encoding='utf-8') as g:
        for line in g:
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError:
                # Handle malformed lines
                try:
                    yield eval(line.strip())
                except:
                    continue


def _extract_zip_member(zip_path: str, member: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    if os.path.exists(dest_path):
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        member_path = member
        if member_path not in zf.namelist():
            candidates = [
                name
                for name in zf.namelist()
                if not name.endswith("/") and os.path.basename(name) == os.path.basename(member)
            ]
            if not candidates:
                raise KeyError(f"No zip member named {member!r} in {zip_path}")
            candidates.sort(key=len)
            member_path = candidates[0]
        with zf.open(member_path) as src, open(dest_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def ensure_movielens_raw_data(root: str, split: str = MOVIELENS_SPLIT) -> None:
    """Download MovieLens-1M ratings/movies if missing."""
    split = canonical_split(split)
    raw_dir = raw_dir_for(root, split)
    os.makedirs(raw_dir, exist_ok=True)
    needed = [os.path.join(raw_dir, dest) for dest in MOVIELENS_DOWNLOAD["files"].values()]
    if all(os.path.exists(p) for p in needed):
        return

    archive_path = os.path.join(raw_dir, MOVIELENS_DOWNLOAD["archive"])
    if not os.path.exists(archive_path):
        last_err: Optional[Exception] = None
        for url in MOVIELENS_DOWNLOAD["urls"]:
            try:
                download_file(url, archive_path)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning("MovieLens download failed from %s: %s", url, exc)
        if last_err is not None:
            raise RuntimeError("All MovieLens download URLs failed") from last_err

    for member, dest_name in MOVIELENS_DOWNLOAD["files"].items():
        _extract_zip_member(archive_path, member, os.path.join(raw_dir, dest_name))


def process_movielens(root: str, split: str = MOVIELENS_SPLIT, k_core: int = 5) -> str:
    """5-core MovieLens-1M processing; writes user2item/item2title under processed/."""
    split = canonical_split(split)
    raw_dir = raw_dir_for(root, split)
    processed_dir = processed_dir_for(root, split)
    os.makedirs(processed_dir, exist_ok=True)

    interactions = pd.read_csv(
        os.path.join(raw_dir, "ratings.dat"),
        sep="::",
        header=None,
        names=["UserID", "ItemID", "Rating", "Timestamp"],
        dtype={"UserID": int, "ItemID": int, "Rating": int, "Timestamp": int},
        engine="python",
        encoding="utf-8",
        encoding_errors="replace",
    ).dropna()[["UserID", "ItemID", "Timestamp"]]
    item2title = pd.read_csv(
        os.path.join(raw_dir, "movies.dat"),
        sep="::",
        header=None,
        names=["ItemID", "Title", "Genres"],
        dtype={"ItemID": int, "Title": str, "Genres": str},
        engine="python",
        encoding="utf-8",
        encoding_errors="replace",
    ).dropna()[["ItemID", "Title"]]
    item2title = item2title[item2title["Title"].notna() & (item2title["Title"] != "nan")]
    item2title["Title"] = item2title["Title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)

    valid_items = item2title[
        item2title["Title"].notna() & item2title["Title"].astype(str).str.strip().astype(bool)
    ]["ItemID"]
    valid_items = np.intersect1d(interactions["ItemID"].unique(), valid_items.unique())
    interactions = interactions[interactions["ItemID"].isin(valid_items)].reset_index(drop=True)
    item2title = item2title[item2title["ItemID"].isin(valid_items)].reset_index(drop=True)
    interactions = interactions.drop_duplicates(
        subset=["UserID", "ItemID", "Timestamp"], keep="first"
    ).reset_index(drop=True)
    item2title = item2title.drop_duplicates(subset=["ItemID"], keep="first").reset_index(drop=True)

    while True:
        user_cnt = interactions["UserID"].value_counts()
        item_cnt = interactions["ItemID"].value_counts()
        deleted_users = user_cnt[user_cnt < k_core].index
        deleted_items = item_cnt[item_cnt < k_core].index
        if deleted_users.empty and deleted_items.empty:
            break
        interactions = interactions[
            ~interactions["UserID"].isin(deleted_users)
            & ~interactions["ItemID"].isin(deleted_items)
        ].reset_index(drop=True)
        item2title = item2title[item2title["ItemID"].isin(interactions["ItemID"])].reset_index(
            drop=True
        )

    user2item = (
        interactions.sort_values(by=["UserID", "Timestamp"])
        .groupby("UserID")
        .agg({"ItemID": list, "Timestamp": list})
        .reset_index()
    )
    user_ids = user2item["UserID"].unique()
    item_ids = np.unique(np.concatenate(user2item["ItemID"].values))
    user_map = {user: idx for idx, user in enumerate(user_ids, start=0)}
    item_map = {item: idx for idx, item in enumerate(item_ids, start=1)}
    user2item = user2item.copy()
    user2item["UserID"] = user2item["UserID"].map(user_map)
    user2item["ItemID"] = user2item["ItemID"].apply(lambda xs: [item_map[i] for i in xs])
    user2item = user2item.sort_values(by=["UserID"]).reset_index(drop=True)
    item2title = item2title.copy()
    item2title["ItemID"] = item2title["ItemID"].map(item_map)
    item2title = item2title.sort_values(by=["ItemID"]).reset_index(drop=True)

    user2item.to_pickle(os.path.join(processed_dir, "user2item.pkl"))
    item2title.to_pickle(os.path.join(processed_dir, "item2title.pkl"))
    user_size = int(user2item["UserID"].nunique())
    interaction_lengths = user2item["ItemID"].apply(len)
    item_size = int(user2item["ItemID"].explode().nunique())
    n_inter = int(interaction_lengths.sum())
    stats = {
        "user_size": user_size,
        "item_size": item_size,
        "interactions": n_inter,
        "density": round(float(n_inter / (user_size * item_size)) if user_size and item_size else 0.0, 6),
    }
    with open(os.path.join(processed_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    logger.info("Saved MovieLens processed data to %s (%s)", processed_dir, stats)
    return processed_dir


def ensure_movielens_processed(root: str, split: str = MOVIELENS_SPLIT) -> str:
    split = canonical_split(split)
    path = user2item_path(root, split)
    if os.path.exists(path):
        return processed_dir_for(root, split)
    ensure_movielens_raw_data(root, split)
    return process_movielens(root, split)


def load_user2item(root: str, split: str) -> pd.DataFrame:
    ensure_movielens_processed(root, split)
    return pd.read_pickle(user2item_path(root, split))


def load_item2title(root: str, split: str) -> Optional[pd.DataFrame]:
    path = item2title_path(root, split)
    if not os.path.exists(path):
        ensure_movielens_processed(root, split)
    if not os.path.exists(path):
        return None
    return pd.read_pickle(path)


@gin.configurable
class AmazonItemDataset(Dataset):

    def __init__(
        self,
        root: str = DEFAULT_DATASET_ROOT,
        split: str = "beauty",
        encoder_model_name: str = DEFAULT_ENCODER_MODEL_NAME,
        force_regenerate: bool = False,
        item_embedding_mode: str = "baseline",
    ) -> None:
        self.root = root
        self.split = canonical_split(split)
        self.encoder_model_name = encoder_model_name
        if item_embedding_mode not in ITEM_EMBEDDING_MODES:
            raise ValueError(
                f"Unknown item_embedding_mode: {item_embedding_mode!r}. "
                f"Expected one of {ITEM_EMBEDDING_MODES}."
            )
        self.item_embedding_mode = item_embedding_mode

        if not is_amazon_split(self.split) and not is_movielens_split(self.split):
            raise ValueError(
                f"Unknown split: {split}. Available: {list_dataset_splits()}"
            )

        self.raw_dir = raw_dir_for(root, self.split)
        self.processed_dir = processed_dir_for(root, self.split)
        encoder_short_name = os.path.basename(encoder_model_name.rstrip("/"))
        mode_suffix = (
            ""
            if is_movielens_split(self.split)
            else embedding_mode_suffix(item_embedding_mode)
        )
        self.parquet_path = os.path.join(
            self.processed_dir,
            f"item_emb_{encoder_short_name}{mode_suffix}.parquet",
        )

        # Download if needed
        self._download_if_needed()

        # Load or generate embeddings
        if os.path.exists(self.parquet_path) and not force_regenerate:
            self._load_embeddings()
        else:
            self._generate_embeddings()

    def _download_if_needed(self) -> None:
        if is_movielens_split(self.split):
            ensure_movielens_processed(self.root, self.split)
        else:
            ensure_amazon_raw_data(self.root, self.split, download_meta=True)

    def _load_embeddings(self) -> None:
        df = pd.read_parquet(self.parquet_path)
        self.embeddings = np.stack(df['embedding'].values, axis=0)
        self.dim = self.embeddings.shape[-1]
        logger.info(f"Loaded {len(self.embeddings)} embeddings from {self.parquet_path}")

    def _generate_embeddings(self) -> None:
        os.makedirs(self.processed_dir, exist_ok=True)
        if is_movielens_split(self.split):
            self._generate_movielens_embeddings()
            return
        config = DATASET_CONFIGS[self.split]

        reviews_path = os.path.join(self.raw_dir, config["reviews"])
        meta_path = os.path.join(self.raw_dir, config["meta"])

        # Build item mapping from reviews (preserves order)
        logger.info("Building item mapping from reviews...")
        item_id_mapping: Dict[str, int] = {}
        for review in tqdm(parse_gzip_json(reviews_path), desc="Processing reviews"):
            asin = review.get('asin')
            if asin and asin not in item_id_mapping:
                item_id_mapping[asin] = len(item_id_mapping) + 1

        logger.info(f"Found {len(item_id_mapping)} unique items")

        # Load metadata
        logger.info("Loading metadata...")
        item_info: Dict[int, dict] = {}
        for meta in tqdm(parse_gzip_json(meta_path), desc="Processing metadata"):
            asin = meta.get('asin')
            if asin in item_id_mapping:
                item_id = item_id_mapping[asin]
                item_info[item_id] = {
                    'title': meta.get('title'),
                    'price': meta.get('price'),
                    'salesRank': meta.get('salesRank'),
                    'brand': meta.get('brand'),
                    'categories': meta.get('categories'),
                }

        logger.info(f"Loaded metadata for {len(item_info)} items")

        # Generate embeddings (batch encode for efficiency with large models)
        logger.info(
            "Generating embeddings with %s (item_embedding_mode=%s)...",
            self.encoder_model_name,
            self.item_embedding_mode,
        )
        model = SentenceTransformer(self.encoder_model_name)

        sorted_ids = sorted(item_info.keys())
        texts = [format_item_metadata_text(item_info[item_id]) for item_id in sorted_ids]
        logger.info(f"Batch encoding {len(texts)} items...")
        all_embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

        item_embeddings: List[dict] = []
        for i, item_id in enumerate(sorted_ids):
            item_embeddings.append({
                'ItemID': item_id,
                'embedding': all_embeddings[i].tolist()
            })

        # Save to parquet
        df = pd.DataFrame(item_embeddings)
        df.to_parquet(self.parquet_path, index=False)
        logger.info(f"Saved embeddings to {self.parquet_path}")

        # Store embeddings
        self.embeddings = np.stack(df['embedding'].values, axis=0)
        self.dim = self.embeddings.shape[-1]

    def _generate_movielens_embeddings(self) -> None:
        item2title = load_item2title(self.root, self.split)
        if item2title is None:
            raise FileNotFoundError(f"No item2title.pkl for split {self.split}")

        item2title = item2title.sort_values("ItemID").reset_index(drop=True)
        titles = item2title["Title"].astype(str).tolist()
        item_ids = item2title["ItemID"].astype(int).tolist()

        logger.info("Encoding %d MovieLens titles with %s", len(titles), self.encoder_model_name)
        model = SentenceTransformer(self.encoder_model_name)
        all_embeddings = model.encode(titles, batch_size=64, show_progress_bar=True)

        rows = [{"ItemID": iid, "embedding": emb.tolist()} for iid, emb in zip(item_ids, all_embeddings)]
        df = pd.DataFrame(rows)
        df.to_parquet(self.parquet_path, index=False)
        self.embeddings = np.stack(df["embedding"].values, axis=0).astype(np.float32)
        self.dim = self.embeddings.shape[-1]
        logger.info("Saved embeddings to %s", self.parquet_path)

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> List[float]:
        return self.embeddings[idx].tolist()


@gin.configurable
class AmazonSeqDataset(Dataset):
    """
    Amazon Sequence Dataset for TIGER training.

    Loads user interaction sequences and represents items with RQ-KMeans
    semantic IDs (Balance-ID or plain RQ-KMeans).

    Training uses sliding window strategy (same as reference):
    - For each user sequence, generate N-1 training samples
    - Each item (except first) serves as a target once

    Valid/Test uses leave-one-out:
    - Valid: history = seq[:-2], target = seq[-2]
    - Test: history = seq[:-1], target = seq[-1]
    """

    def __init__(
        self,
        root: str = DEFAULT_DATASET_ROOT,
        split: str = "beauty",
        train_test_split: str = "train",
        max_seq_len: int = 20,
        add_disambiguation: bool = True,
        id_type: str = "rqkmeans_pop_balance",
        item_dataset=AmazonItemDataset,
        random_seed: int = 42,
        encoder_model_name: str = DEFAULT_ENCODER_MODEL_NAME,
        pretrained_rqkmeans_path: str = "./out/{split}/rqkmeans.pt",
        rqkmeans_codebook_size: int = 256,
        rqkmeans_n_layers: int = 3,
        sem_ids_override_path: Optional[str] = None,
        item_embedding_mode: str = "baseline",
    ) -> None:
        self.root = root
        self.split = canonical_split(split)
        self.train_test_split = train_test_split
        self._max_seq_len = max_seq_len
        self.add_disambiguation = add_disambiguation
        self.rqkmeans_n_layers = rqkmeans_n_layers
        self.rqkmeans_codebook_size = rqkmeans_codebook_size
        self.sem_ids_override_path = sem_ids_override_path
        self.id_type = id_type
        self.random_seed = random_seed
        self.num_items = 0
        self.item_embedding_mode = item_embedding_mode

        self._load_sequences()

        if id_type not in ("rqkmeans", "rqkmeans_pop_balance"):
            raise ValueError(
                f"Unknown id_type: {id_type!r}. "
                "Expected 'rqkmeans' or 'rqkmeans_pop_balance'."
            )
        self._init_rqkmeans_semantic_ids(
            item_dataset=item_dataset,
            pretrained_rqkmeans_path=pretrained_rqkmeans_path,
            encoder_model_name=encoder_model_name,
            rqkmeans_codebook_size=rqkmeans_codebook_size,
            rqkmeans_n_layers=rqkmeans_n_layers,
            use_pop_balance=(id_type == "rqkmeans_pop_balance"),
        )

        self._generate_samples()

    def _init_rqkmeans_semantic_ids(
        self,
        item_dataset,
        pretrained_rqkmeans_path: str,
        encoder_model_name: str,
        rqkmeans_codebook_size: int,
        rqkmeans_n_layers: int,
        use_pop_balance: bool = False,
    ) -> None:
        """Assign SIDs via pretrained RQ-KMeans; optional pop-balance + dedup suffix."""
        if use_pop_balance:
            from genrec.models.rqkmeans_pop_balance import RqKmeansPopBalance as RqKmeansLoader
        else:
            from genrec.models.rqkmeans import RqKmeans as RqKmeansLoader

        pretrained_rqkmeans_path = pretrained_rqkmeans_path.format(split=self.split)

        item_ds = item_dataset(
            root=self.root,
            split=self.split,
            encoder_model_name=encoder_model_name,
            item_embedding_mode=self.item_embedding_mode,
        )
        embeddings = item_ds.embeddings.astype(np.float32)

        rqkmeans = RqKmeansLoader.load(pretrained_rqkmeans_path)
        if rqkmeans.n_layers != rqkmeans_n_layers:
            logger.warning(
                "RQ-KMeans checkpoint n_layers=%d != config rqkmeans_n_layers=%d",
                rqkmeans.n_layers,
                rqkmeans_n_layers,
            )
        if rqkmeans.codebook_size != rqkmeans_codebook_size:
            logger.warning(
                "RQ-KMeans checkpoint codebook_size=%d != config rqkmeans_codebook_size=%d",
                rqkmeans.codebook_size,
                rqkmeans_codebook_size,
            )

        self.rqkmeans_n_layers = rqkmeans.n_layers
        self.rqkmeans_codebook_size = rqkmeans.codebook_size

        override_path = self.sem_ids_override_path
        if override_path:
            override_path = override_path.format(split=self.split)
            if not os.path.exists(override_path):
                raise FileNotFoundError(
                    f"sem_ids_override_path not found: {override_path}"
                )
            loaded = np.load(override_path)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                if "sem_ids" in loaded:
                    sem_ids = np.asarray(loaded["sem_ids"])
                else:
                    key = loaded.files[0]
                    sem_ids = np.asarray(loaded[key])
            else:
                sem_ids = np.asarray(loaded)
            if sem_ids.ndim != 2:
                raise ValueError(
                    f"sem_ids_override must be 2D [N, L], got shape {sem_ids.shape}"
                )
            if sem_ids.shape[0] != len(embeddings):
                raise ValueError(
                    f"sem_ids_override N={sem_ids.shape[0]} != "
                    f"embeddings N={len(embeddings)} ({override_path})"
                )
            self.sem_ids_list = sem_ids.astype(np.int64).tolist()
            logger.info(
                "Loaded SID override from %s shape=%s",
                override_path,
                tuple(sem_ids.shape),
            )
        else:
            output = rqkmeans.assign(embeddings)
            self.sem_ids_list = output.sem_ids.tolist()

        if self.add_disambiguation:
            from genrec.modules.semantic_id import (
                deduplicate_semantic_ids,
                log_dedup_collision_stats,
            )

            self.sem_ids_list = deduplicate_semantic_ids(self.sem_ids_list)
            log_dedup_collision_stats(
                self.sem_ids_list,
                base_dim=rqkmeans.n_layers,
                codebook_size=rqkmeans.codebook_size,
            )

        self.num_items = len(self.sem_ids_list)

        if override_path:
            keys = ["-".join(str(c) for c in row) for row in self.sem_ids_list]
            unique = len(set(keys))
            from collections import Counter

            max_collision = max(Counter(keys).values()) if keys else 0
            stats = {
                "collision_rate": 1.0 - unique / max(len(keys), 1),
                "unique": unique,
                "max_collision": max_collision,
            }
        else:
            stats = rqkmeans.collision_rate(embeddings)
        if override_path:
            id_label = "Override SIDs (+ RQ checkpoint meta)"
        elif use_pop_balance:
            id_label = "Pop-Balance RQ-KMeans"
        else:
            id_label = "RQ-KMeans"
        logger.info(
            "%s SIDs: %d items, layers=%d, codebook=%d, "
            "collision_rate=%.4f (unique=%d, max_group=%d)",
            id_label,
            self.num_items,
            self.rqkmeans_n_layers,
            self.rqkmeans_codebook_size,
            stats["collision_rate"],
            stats["unique"],
            stats["max_collision"],
        )

    def _load_sequences(self) -> None:
        """Load user interaction sequences (Amazon reviews or MovieLens user2item)."""
        if is_movielens_split(self.split):
            df = load_user2item(self.root, self.split)
            # Convert 1-indexed ItemID to 0-indexed for sem_ids_list indexing.
            self.sequences = []
            self.user_ids = []
            for user_id, items in zip(df["UserID"], df["ItemID"]):
                if len(items) >= 5:
                    self.sequences.append([i - 1 for i in items])
                    self.user_ids.append(int(user_id))
            logger.info("Loaded %d MovieLens sequences", len(self.sequences))
            return

        config = DATASET_CONFIGS[self.split]
        reviews_path = os.path.join(raw_dir_for(self.root, self.split), config["reviews"])

        user_sequences: Dict[str, List[tuple]] = {}
        item_id_mapping: Dict[str, int] = {}

        logger.info("Loading user sequences...")
        for review in tqdm(parse_gzip_json(reviews_path), desc="Processing reviews"):
            asin = review.get('asin')
            user_id = review.get('reviewerID')
            timestamp = review.get('unixReviewTime', 0)

            if asin and user_id:
                if asin not in item_id_mapping:
                    item_id_mapping[asin] = len(item_id_mapping)

                item_id = item_id_mapping[asin]
                if user_id not in user_sequences:
                    user_sequences[user_id] = []
                user_sequences[user_id].append((timestamp, item_id))

        self.sequences = []
        self.user_ids = []
        for uid, seq in user_sequences.items():
            seq.sort(key=lambda x: x[0])
            items = [x[1] for x in seq]
            if len(items) >= 5:
                self.sequences.append(items)
                self.user_ids.append(uid)

        logger.info(f"Loaded {len(self.sequences)} user sequences")

    def _generate_samples(self) -> None:
        """Generate training/evaluation samples.

        Training: Sliding window strategy (same as reference)
        - For sequence [i1, i2, i3, i4, i5], generate:
          - history=[i1], target=i2
          - history=[i1,i2], target=i3
          - history=[i1,i2,i3], target=i4
          - history=[i1,i2,i3,i4], target=i5
        - Exclude last 2 items for leave-one-out eval

        Valid/Test: Single sample per user
        - Valid: history=seq[:-2], target=seq[-2]
        - Test: history=seq[:-1], target=seq[-1]
        """
        self.samples = []

        if self.train_test_split == "train":
            # Sliding window for training
            for user_idx, full_seq in enumerate(tqdm(self.sequences, desc="Generating train samples")):
                user_id = hash(self.user_ids[user_idx]) % 10000
                # Exclude last 2 items (for valid and test)
                seq = full_seq[:-2]
                # Generate sliding window samples
                for i in range(1, len(seq)):
                    history_items = seq[:i]
                    target_item = seq[i]
                    self.samples.append({
                        'user_id': user_id,
                        'history': history_items,
                        'target': target_item,
                    })
        elif self.train_test_split == "valid":
            # Valid: exclude last 1 item (test target)
            for user_idx, full_seq in enumerate(self.sequences):
                user_id = hash(self.user_ids[user_idx]) % 10000
                seq = full_seq[:-1]
                self.samples.append({
                    'user_id': user_id,
                    'history': seq[:-1],  # All except target
                    'target': seq[-1],    # Second to last of full_seq
                })
        else:  # test
            # Test: use full sequence
            for user_idx, full_seq in enumerate(self.sequences):
                user_id = hash(self.user_ids[user_idx]) % 10000
                self.samples.append({
                    'user_id': user_id,
                    'history': full_seq[:-1],  # All except last
                    'target': full_seq[-1],    # Last item
                })

        logger.info(f"Generated {len(self.samples)} samples for {self.train_test_split}")

    def _truncate_history(self, history: List[int]) -> List[int]:
        """Truncate history to max_seq_len (take last N items)."""
        if len(history) > self._max_seq_len:
            return history[-self._max_seq_len:]
        return history

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from genrec.data.schemas import SeqData

        sample = self.samples[idx]
        user_id = sample['user_id']
        history_items = self._truncate_history(sample['history'])
        target_item = sample['target']

        # Convert to semantic IDs
        item_sem_ids = []
        for item_id in history_items:
            if item_id < len(self.sem_ids_list):
                item_sem_ids.extend(self.sem_ids_list[item_id])

        if target_item < len(self.sem_ids_list):
            target_sem_ids = self.sem_ids_list[target_item]
        else:
            target_sem_ids = [0, 0, 0]

        return SeqData(
            user_id=user_id,
            item_ids=item_sem_ids,
            target_ids=target_sem_ids,
            target_item=target_item,
            history_items=list(history_items),
        )


if __name__ == "__main__":
    dataset = AmazonItemDataset(
        root=DEFAULT_DATASET_ROOT,
        split="beauty",
        encoder_model_name=DEFAULT_ENCODER_MODEL_NAME,
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Embedding dim: {dataset.dim}")
    print(f"First embedding shape: {len(dataset[0])}")
