from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib import request

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore


def _maybe_print_nvidia_smi() -> None:
    if not shutil.which("nvidia-smi"):
        logging.info("nvidia-smi not found")
        return
    subprocess.run(["nvidia-smi", "-L"], check=False)


def _download_if_missing(module_url: str, module_path: Path) -> None:
    if module_path.exists():
        return
    module_path.parent.mkdir(parents=True, exist_ok=True)
    with request.urlopen(module_url) as f, module_path.open("wb") as outf:
        outf.write(f.read())


def _labels_to_file(preds: list[list[int]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as outf:
        for row in preds:
            outf.write(",".join(str(k) for k in row) + "\n")


def _binary_labels_to_file(preds: list[int], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as outf:
        for label in preds:
            outf.write(f"{label}\n")


def _load_dpm(data_dir: Path, module_url: str, download_module: bool) -> Any:
    module_path = data_dir / "dont_patronize_me.py"
    if download_module:
        _download_if_missing(module_url, module_path)
    sys.path.insert(0, str(data_dir))
    from dont_patronize_me import DontPatronizeMe  # type: ignore

    dpm = DontPatronizeMe(str(data_dir), str(data_dir))
    dpm.load_task1()
    return dpm


def _rebuild_task1(dpm_train_df: pd.DataFrame, ids_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for parid in ids_df["par_id"].astype(str).tolist():
        match = dpm_train_df.loc[dpm_train_df.par_id == parid]
        if match.empty:
            raise ValueError(f"par_id not found in task1 data: {parid}")
        rows.append(
            {
                "par_id": parid,
                "community": match.keyword.values[0],
                "text": match.text.values[0],
                "label": int(match.label.values[0]),
            }
        )
    return pd.DataFrame(rows)


def _build_task4_test(test_path: Path) -> pd.DataFrame:
    cols = ["par_id", "art_id", "keyword", "country", "text"]
    test_df = pd.read_csv(test_path, sep="\t", header=None, names=cols)
    test_df["input_text"] = "[COMM] " + test_df["keyword"].astype(str) + " [TEXT] " + test_df["text"].astype(str)
    return test_df


def _zip_submission(task1_path: Path, zip_path: Path) -> None:
    import zipfile

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if task1_path.exists():
            zf.write(task1_path, arcname=task1_path.name)


def _set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--train-ids", type=Path, default=Path("train_semeval_parids-labels.csv"))
    parser.add_argument("--dev-ids", type=Path, default=Path("dev_semeval_parids-labels.csv"))
    parser.add_argument("--task4-test", type=Path, default=Path("task4_test.tsv"))
    parser.add_argument("--task1-out", type=Path, default=Path("task1.txt"))
    parser.add_argument("--dev-out", type=Path, default=Path("dev.txt"))
    parser.add_argument("--test-out", type=Path, default=Path("test.txt"))
    parser.add_argument("--zip-out", type=Path, default=Path("submission.zip"))
    parser.add_argument("--best-model-dir", type=Path, default=Path("BestModel/best_model"))
    parser.add_argument("--predict-threshold", type=float, default=0.5)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--download-dpm-module", action="store_true")
    parser.add_argument(
        "--dpm-module-url",
        type=str,
        default="https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/semeval-2022/dont_patronize_me.py",
    )
    parser.add_argument("--model-type", type=str, default="roberta")
    parser.add_argument("--model-name", type=str, default="roberta-large")
    parser.add_argument("--task1-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pos-oversample", type=float, default=2.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.model_name = args.model_name.strip()
    args.model_type = args.model_type.strip()
    if args.model_name in {"deberta-v3-large", "deberta-v3-base"}:
        args.model_name = f"microsoft/{args.model_name}"
    if args.model_type in {"deberta-v2", "deberta-v3"}:
        args.model_type = "deberta"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if pd is None:
        raise RuntimeError("pandas is required. Install it in your environment (e.g. `conda install pandas`).")

    _set_seeds(args.seed)

    data_dir = args.data_dir.resolve()
    train_ids_path = (data_dir / args.train_ids).resolve() if not args.train_ids.is_absolute() else args.train_ids
    dev_ids_path = (data_dir / args.dev_ids).resolve() if not args.dev_ids.is_absolute() else args.dev_ids
    test_path = (data_dir / args.task4_test).resolve() if not args.task4_test.is_absolute() else args.task4_test
    best_model_dir = (data_dir / args.best_model_dir).resolve() if not args.best_model_dir.is_absolute() else args.best_model_dir

    if not train_ids_path.exists():
        raise FileNotFoundError(str(train_ids_path))
    if not dev_ids_path.exists():
        raise FileNotFoundError(str(dev_ids_path))
    if not test_path.exists():
        raise FileNotFoundError(str(test_path))

    _maybe_print_nvidia_smi()

    dpm = _load_dpm(data_dir=data_dir, module_url=args.dpm_module_url, download_module=args.download_dpm_module)
    dpm_train_df: pd.DataFrame = dpm.train_task1_df

    trids = pd.read_csv(train_ids_path)
    devids = pd.read_csv(dev_ids_path)

    trdf1 = _rebuild_task1(dpm_train_df, trids)
    devdf1 = _rebuild_task1(dpm_train_df, devids)
    trdf1 = trdf1[trdf1["text"].astype(str).str.strip().str.len() > 0]
    devdf1 = devdf1[devdf1["text"].astype(str).str.strip().str.len() > 0]
    trdf1["input_text"] = "[COMM] " + trdf1["community"].astype(str) + " [TEXT] " + trdf1["text"]
    devdf1["input_text"] = "[COMM] " + devdf1["community"].astype(str) + " [TEXT] " + devdf1["text"]

    logging.info("task1 train=%s dev=%s", len(trdf1), len(devdf1))

    if args.dry_run:
        return 0

    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"Missing dependencies. Install torch. (root cause: {e!r})") from e

    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    use_cuda = torch.cuda.is_available()
    logging.info("torch cuda available=%s", use_cuda)

    if args.predict_only:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as e:
            raise RuntimeError(f"Missing dependencies. Install transformers. (root cause: {e!r})") from e
        if not best_model_dir.exists():
            raise FileNotFoundError(str(best_model_dir))
        test_df = _build_task4_test(test_path)
        device = torch.device("cuda" if use_cuda else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(str(best_model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(best_model_dir))
        model.to(device)
        model.eval()

        def predict(texts: list[str]) -> list[float]:
            probs: list[float] = []
            for start in range(0, len(texts), args.batch_size):
                batch = texts[start : start + args.batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=args.max_seq_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    logits = model(**enc).logits
                    batch_probs = torch.softmax(logits, dim=-1)[:, 1].tolist()
                probs.extend(batch_probs)
            return probs

        dev_probs = predict(devdf1["input_text"].tolist())
        test_probs = predict(test_df["input_text"].tolist())
        dev_preds = [1 if p >= args.predict_threshold else 0 for p in dev_probs]
        test_preds = [1 if p >= args.predict_threshold else 0 for p in test_probs]
        _binary_labels_to_file(dev_preds, data_dir / args.dev_out)
        _binary_labels_to_file(test_preds, data_dir / args.test_out)
        logging.info("wrote %s lines=%s", (data_dir / args.dev_out).as_posix(), len(dev_preds))
        logging.info("wrote %s lines=%s", (data_dir / args.test_out).as_posix(), len(test_preds))
        return 0

    try:
        from simpletransformers.classification import ClassificationArgs, ClassificationModel  # type: ignore
        from sklearn.metrics import f1_score
    except Exception as e:
        msg = "Missing dependencies. Install simpletransformers and a CUDA-enabled torch if you want GPU."
        if "transformers.convert_graph_to_onnx" in str(e):
            msg = (
                msg
                + " Also ensure transformers<5 (simpletransformers currently expects transformers.convert_graph_to_onnx)."
            )
        raise RuntimeError(msg + f" (root cause: {e!r})") from e

    training_set1 = trdf1
    if args.pos_oversample > 1:
        pos_df = trdf1[trdf1.label == 1]
        extra = pos_df.sample(
            n=int(len(pos_df) * (args.pos_oversample - 1)),
            replace=True,
            random_state=args.seed,
        )
        training_set1 = pd.concat([training_set1, extra], ignore_index=True)
    training_set1 = training_set1.sample(frac=1.0, random_state=args.seed)

    def pcl_f1(labels: list[int], preds: list[int]) -> float:
        return float(f1_score(labels, preds, average="binary", pos_label=1))


    task1_model_args = ClassificationArgs(
        use_early_stopping=True,
        early_stopping_patience=3,
        early_stopping_delta=0.0,
        early_stopping_consider_epochs=True,
        num_train_epochs=args.task1_epochs,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        manual_seed=args.seed,
        evaluate_during_training=True,
        save_best_model=True,
        save_eval_checkpoints=False,
        save_model_every_epoch=False,
        early_stopping_metric="pcl_f1",
        early_stopping_metric_minimize=False,
        overwrite_output_dir=True,
        output_dir=str(data_dir / "outputs"),
        best_model_dir=str(data_dir / "outputs" / "best_model"),
        silent=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=use_cuda,
    )
    task1_model = ClassificationModel(
        args.model_type,
        args.model_name,
        args=task1_model_args,
        num_labels=2,
        use_cuda=use_cuda,
    )
    task1_model.train_model(training_set1[["input_text", "label"]], eval_df=devdf1[["input_text", "label"]], pcl_f1=pcl_f1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
