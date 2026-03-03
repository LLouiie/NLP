# NLP Project 

## Whats' in BestModel/
`BestModel/` contains the training/inference script and the saved checkpoint used for prediction.

**Script**
- `BestModel/reconstruct_and_roberta_baseline.py`  
  Reconstructs the task1 train/dev splits from `dontpatronizeme_pcl.tsv` using `train_semeval_parids-labels.csv` and `dev_semeval_parids-labels.csv`.  
  Supports training (SimpleTransformers) and predict-only inference (Transformers).

**Saved model checkpoint**
- `BestModel/best_model/` includes the RoBERTa weights and tokenizer artifacts:
  - `model.safetensors`: model weights
  - `config.json`: model architecture/config
  - `tokenizer.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`, `tokenizer_config.json`: tokenizer files
  - `training_args.bin`, `scheduler.pt`: training state (kept for reference)
  - `model_args.json`: training/inference arguments snapshot
  - `eval_results.txt`: evaluation results saved during training

## Predicted_output (dev.txt and test.txt)
- `Predicted_output/dev.txt`: model predictions for dev IDs (one label per line).
- `Predicted_output/test.txt`: model predictions for task4 test (one label per line).

## External files
- `optimizer.pt` is stored externally due to GitHub size limits:
  https://drive.google.com/drive/folders/1SzOSBEF-hqNkxnE_gijww5lTcoCIdcNf?usp=sharing


## How to run 
 Generate dev/test predictions:

```bash
python BestModel/reconstruct_and_roberta_baseline.py \
  --predict-only \
  --best-model-dir /path/to/best_model \
  --dev-out Predicted_output/dev.txt \
  --test-out Predicted_output/test.txt
```



## Other files
- `dontpatronizeme_pcl.tsv`: original training data for task1.
- `dont_patronizeme.py`: helper loader that converts original labels into binary PCL labels.
- `train_semeval_parids-labels.csv`: task1 train split IDs.
- `dev_semeval_parids-labels.csv`: task1 dev split IDs.
- `task4_test.tsv`: official test input file for prediction.


## Runtime environment
- Python: 3.10+ 
- Key packages: torch, transformers, simpletransformers, pandas, scikit-learn
