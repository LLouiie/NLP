# NLP Project 

## What this repository contains
- `BestModel/reconstruct_and_roberta_baseline.py`: main script for data reconstruction, training, and prediction.
- `BestModel/best_model/`: saved RoBERTa checkpoint used for inference.
- `dev.txt`, `test.txt`: prediction outputs for the dev and test sets.

## Runtime environment
- Python: 3.10+ (conda environment name: `nlp`)
- Key packages: torch, transformers, simpletransformers, pandas, scikit-learn

## How to run (simple steps)

Generate dev/test predictions using the saved best model:

```bash
python BestModel/reconstruct_and_roberta_baseline.py \
  --data-dir /rds/general/user/jy625/home/NLP_MODEL \
  --predict-only \
  --best-model-dir BestModel/best_model \
  --dev-out dev.txt \
  --test-out test.txt
```

## Notes on outputs
- `dev.txt` uses the dev IDs from `dev_semeval_parids-labels.csv`.
- `test.txt` uses the inputs from `task4_test.tsv`.
