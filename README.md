# UPoD
This is the implementation of the paper: Uncertainty-Aware Popularity Debiasing for Generative Recommendation

### 1. Popularity-balanced encoding

python -u genrec/trainers/rqkmeans_pop_balance_trainer.py config/beauty.gin --split beauty

### 2. Uncertainty-aware GR training

python -u genrec/trainers/tiger_trainer.py config/beauty.gin --split beauty

### 3. Uncertainty-gated decoding

python -u genrec/trainers/tiger_eval.py config/beauty.gin --split beauty \
  --gin "genrec.trainers.tiger_eval.eval_checkpoint.checkpoint_path='out/beauty/best_model.pt'"

Our implementation code is based on [GenRec](https://github.com/phonism/genrec).
