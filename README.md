# UPoD

Implementation of **Uncertainty-Aware Popularity Debiasing for Generative Recommendation**.

UPoD mitigate popularity bias in SID-based generative recommenders with three components:

1. **Popularity-balanced encoding** (Balance-ID)
2. **Uncertainty-aware GR training** (UAL)
3. **Uncertainty-gated decoding** (UGD)


---

## Requirements

- Python 3.10
- PyTorch 2.6.0
- Transformers 4.57.3

```bash
cd UPoD
pip install -r requirements.txt
```

---

## Repository Structure

```
UPoD/
├── config/                 # Per-dataset gin configs
│   ├── tiger.gin           # Shared config
│   ├── beauty.gin
│   ├── sports.gin
│   ├── toys.gin
│   └── movielens1m.gin
├── genrec/                 # Models, losses, trainers
│   ├── models/             # TIGER + Balance-ID RQ-KMeans
│   ├── modules/            # UAL, UGD, Balanced-ID
│   ├── data/               # Amazon / MovieLens loaders
│   └── trainers/           # fit ID, train, eval entry points
└── dataset/                # Raw + processed data
```

---

## Dataset

| Split | `--split` | Config | Domain |
|---|---|---|---|
| Amazon Beauty | `beauty` | `config/beauty.gin` | Beauty products |
| Amazon Sports | `sports` | `config/sports.gin` | Sports & outdoors |
| Amazon Toys | `toys` | `config/toys.gin` | Toys & games |
| MovieLens-1M | `movielens1m` | `config/movielens1m.gin` | Movies |

---

## Quick start (Beauty)


### 1. Popularity-balanced encoding


```bash
python -u genrec/trainers/rqkmeans_pop_balance_trainer.py config/beauty.gin --split beauty
```

### 2. Uncertainty-aware GR training


```bash
python -u genrec/trainers/tiger_trainer.py config/beauty.gin --split beauty
```

Multi-GPU:

```bash
accelerate launch --multi_gpu --num_processes 4 \
  genrec/trainers/tiger_trainer.py config/beauty.gin --split beauty
```

### 3. Uncertainty-gated decoding


```bash
python -u genrec/trainers/tiger_eval.py config/beauty.gin --split beauty
```

Replace `beauty` / `config/beauty.gin` with `sports`, `toys`, or `movielens1m` for the other datasets.

---

Our implementation code is based on [GenRec](https://github.com/phonism/genrec).
