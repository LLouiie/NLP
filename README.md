# NLP Project 

## Whats' in BestModel/
`BestModel/` contains the training/inference script used for prediction.

**Script**
- `BestModel/reconstruct_and_roberta_baseline.py`  
  Reconstructs the task1 train/dev splits from `dontpatronizeme_pcl.tsv` using `train_semeval_parids-labels.csv` and `dev_semeval_parids-labels.csv`.  
  Supports training (SimpleTransformers) and predict-only inference (Transformers).

**Saved model checkpoint**
- The RoBERTa checkpoint used for inference is passed via `--best-model-dir`.

## How to run 
 Generate dev/test predictions:

```bash
python BestModel/reconstruct_and_roberta_baseline.py \
  --data-dir /rds/general/user/jy625/home/NLP_MODEL \
  --predict-only \
  --best-model-dir /path/to/best_model \
  --dev-out Predicted_output/dev.txt \
  --test-out Predicted_output/test.txt
```

## Predicted_output (dev.txt and test.txt)
- `Predicted_output/dev.txt`: model predictions for dev IDs (one label per line).
- `Predicted_output/test.txt`: model predictions for task4 test (one label per line).

## Other files
- `dontpatronizeme_pcl.tsv`: original training data for task1.
- `dont_patronizeme.py`: helper loader that converts original labels into binary PCL labels.
- `train_semeval_parids-labels.csv`: task1 train split IDs.
- `dev_semeval_parids-labels.csv`: task1 dev split IDs.
- `task4_test.tsv`: official test input file for prediction.


## Runtime environment
- Python: 3.10+ (conda env name: `nlp`)
- Key packages: torch, transformers, simpletransformers, pandas, scikit-learn
