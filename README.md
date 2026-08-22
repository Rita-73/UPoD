# UPoD

### 1. Fit Balance-ID

python -u genrec/trainers/rqkmeans_pop_balance_trainer.py config/beauty.gin --split beauty

### 2. Train UPoD

python -u genrec/trainers/tiger_trainer.py config/beauty.gin --split beauty \
  --gin "genrec.trainers.tiger_trainer.train.save_dir_root='out/beauty/'"

### 3. UGD eval

python -u genrec/trainers/tiger_eval.py config/beauty.gin --split beauty \
  --gin "genrec.trainers.tiger_eval.eval_checkpoint.checkpoint_path='out/beauty/best_model.pt'" \
  --gin "genrec.trainers.tiger_eval.eval_checkpoint.output_json='out/beauty/eval_beam30.json'"

Our implementation code is based on [GenRec](https://github.com/phonism/genrec).
