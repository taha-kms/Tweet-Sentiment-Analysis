#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a trained model and store results under the model's folder.

What it does
------------
- Loads a local HF checkpoint (best or last) or a hub model.
- Computes metrics on val/test: accuracy, F1 macro/micro, per-class P/R/F1, confusion matrix.
- Optional plots: reliability diagram + ROC-AUC (OvR).
- Saves misclassified examples (with logits/probs) + top-100 hardest cases.
- Stores ALL artifacts under: models/<MODEL_RUN>/eval/<TIMESTAMP[_tag]>/
- Maintains a global summary CSV at: <models_root>/overall_evaluations.csv
- Supports --last to auto-pick the most recent model run in <models_root>.

Expected layout from training
-----------------------------
models/
  <run_name>/                      # e.g. bert-base-uncased_20250820-1538
    best/                          # preferred checkpoint (weights + tokenizer)
    last/                          # last-epoch checkpoint
    tokenizer/
    params.json                    # (config + meta; used for flattening parameters)
    config.yaml
    train_log.csv
    ...
    eval/
      <YYYYMMDD-HHMM[_tag]>/
        metrics_test.json
        metrics_val.json
        metrics_overview.csv
        misclassified_val.csv
        misclassified_test.csv
        hard_cases.csv
        confusion_matrix.png
        *_reliability.png
        *_roc_auc.png

Usage examples
--------------
# Evaluate a specific run's BEST checkpoint
python src/evaluate.py --checkpoint models/bert-base-uncased_20250820-1538/best --include_plots

# Evaluate a parent run folder (auto-uses best/ if present, else last/)
python src/evaluate.py --checkpoint models/bert-base-uncased_20250820-1538 --include_plots

# Evaluate the most recent model run (auto-detected from models/), use BEST
python src/evaluate.py --last --include_plots

# Add an optional tag for this eval (folders under eval/)
python src/evaluate.py --last --run_tag "testA" --include_plots
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# -------------------------------
# Helpers: config, dirs, naming
# -------------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M")

def find_models_root_from_cfg(cfg_path: str) -> Path:
    cfg = load_yaml(cfg_path)
    mr = cfg.get("paths", {}).get("models_dir", "models")
    return Path(mr)

def is_model_run_dir(p: Path) -> bool:
    # A "run dir" should contain at least best/ or last/ or params.json
    if not p.is_dir():
        return False
    has_ckpt = (p / "best").is_dir() or (p / "last").is_dir()
    has_params = (p / "params.json").exists()
    return has_ckpt or has_params

def parse_timestamp_from_run_name(run_name: str) -> str:
    # Expects names like: "<modelname>_YYYYMMDD-HHMM"
    # We'll parse the last "_YYYYMMDD-HHMM" part if present.
    parts = run_name.split("_")
    if parts:
        ts = parts[-1]
        return ts
    return ""

def pick_latest_model_run(models_root: Path) -> Path:
    """Pick most recent run by parsing the trailing timestamp in folder names."""
    candidates = [p for p in models_root.iterdir() if is_model_run_dir(p)]
    if not candidates:
        raise FileNotFoundError(f"No model runs found under: {models_root}")
    def sort_key(p: Path):
        ts = parse_timestamp_from_run_name(p.name)
        return ts  # lexicographic OK because format is YYYYMMDD-HHMM
    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]

def resolve_checkpoint_dir(checkpoint_arg: str) -> Path:
    """
    If given a 'best' or 'last' folder, use it directly.
    If given a run folder, prefer 'best/' then fallback to 'last/'.
    If given a hub name, we just return the string-like Path (handled later).
    """
    p = Path(checkpoint_arg)
    if p.exists():
        if p.is_dir() and p.name in {"best", "last"}:
            return p
        if p.is_dir():
            if (p / "best").is_dir():
                return p / "best"
            if (p / "last").is_dir():
                return p / "last"
            # As a fallback, use the run folder itself (HF will load tokenizer/model if present)
            return p
    # not a local path; likely a hub id (handled by HF)
    return p


# -------------------------------
# Dataset & batching
# -------------------------------

class SimpleTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int,
                 text_col: str = "text", label_col: str = "label"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = text_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row[self.text_col]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.label_col in row and pd.notna(row[self.label_col]):
            item["labels"] = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        item["_idx"] = torch.tensor(idx, dtype=torch.long)
        return item

def make_loader(df, tokenizer, max_length, batch_size, num_workers=4):
    return DataLoader(
        SimpleTextDataset(df, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# -------------------------------
# Metrics / plots
# -------------------------------

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def cross_entropy_rowwise(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    log_probs = x - np.log(np.exp(x).sum(axis=1, keepdims=True) + 1e-12)
    return -log_probs[np.arange(len(labels)), labels]

def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cls_report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "per_class": {
            str(lbl): {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, lbl in enumerate(labels)
        },
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
    }

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: Path, normalize: bool = False):
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            plt.text(j, i, text, ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()

def plot_reliability(probs: np.ndarray, y_true: np.ndarray, out_path: Path, n_bins: int = 15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    accs, confs, cnts = [], [], []
    for b in range(n_bins):
        m = bin_ids == b
        if m.sum() == 0:
            accs.append(np.nan)
            confs.append((bins[b] + bins[b + 1]) / 2.0)
            cnts.append(0)
        else:
            accs.append(float(correct[m].mean()))
            confs.append(float(conf[m].mean()))
            cnts.append(int(m.sum()))
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1])
    plt.scatter(confs, accs)
    for x, y, c in zip(confs, accs, cnts):
        if not np.isnan(y):
            plt.text(x, y, str(c), fontsize=8, ha="center", va="bottom")
    plt.title("Reliability Diagram")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1); plt.xlim(0, 1); plt.grid(True, alpha=0.3)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()

def plot_ovr_roc(probs: np.ndarray, y_true: np.ndarray, labels: List[int], out_path: Path):
    y_true_bin = np.zeros((len(y_true), len(labels)), dtype=int)
    for i, lbl in enumerate(labels):
        y_true_bin[:, i] = (y_true == lbl).astype(int)
    aucs = []
    for i, _ in enumerate(labels):
        if y_true_bin[:, i].sum() in (0, len(y_true_bin)):
            aucs.append(np.nan)
            continue
        try:
            aucs.append(roc_auc_score(y_true_bin[:, i], probs[:, i]))
        except Exception:
            aucs.append(np.nan)
    plt.figure(figsize=(6, 4))
    plt.bar([str(l) for l in labels], [0 if np.isnan(a) else a for a in aucs])
    plt.ylim(0, 1)
    plt.title("ROC-AUC (OvR) per class")
    plt.xlabel("Class"); plt.ylabel("AUC")
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


# -------------------------------
# Eval core
# -------------------------------

@torch.no_grad()
def eval_step(
    model,
    df: pd.DataFrame,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    num_workers: int,
) -> Dict[str, np.ndarray]:
    model.eval()
    loader = make_loader(df, tokenizer, max_length, batch_size, num_workers)
    logits_list, labels_list, idx_list = [], [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        idxs = batch["_idx"].numpy()
        idx_list.append(idxs)
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ["labels", "_idx"]}
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()
        logits_list.append(logits)
        if "labels" in batch:
            labels_list.append(batch["labels"].detach().cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)
    indices = np.concatenate(idx_list, axis=0)
    labels = np.concatenate(labels_list, axis=0) if labels_list else None
    probs = softmax_np(logits)
    preds = probs.argmax(axis=1)
    return {"logits": logits, "probs": probs, "preds": preds, "labels": labels, "indices": indices}

def evaluate_split(
    name: str,
    df: pd.DataFrame,
    model,
    tokenizer,
    device,
    max_length: int,
    batch_size: int,
    num_workers: int,
    label_list: List[int],
    label_names: Dict[int, str],
    out_dir: Path,
    include_plots: bool,
) -> Tuple[Dict, pd.DataFrame]:
    res = eval_step(model, df, tokenizer, device, max_length, batch_size, num_workers)
    logits, probs, preds, labels = res["logits"], res["probs"], res["preds"], res["labels"]
    metrics = compute_basic_metrics(labels, preds, label_list)

    # Confusion matrix (for test split per DoD)
    if name == "test":
        cm_path = out_dir / "confusion_matrix.png"
        plot_confusion_matrix(np.array(metrics["confusion_matrix"]), [label_names[i] for i in label_list], cm_path)

    # Optional plots
    if include_plots:
        plot_reliability(probs, labels, out_dir / f"{name}_reliability.png")
        plot_ovr_roc(probs, labels, label_list, out_dir / f"{name}_roc_auc.png")

    # Misclassified examples (+ logits/probs)
    ce = cross_entropy_rowwise(logits, labels)
    mis_mask = preds != labels
    mis_df = df.iloc[res["indices"][mis_mask]].copy()
    mis_df["true_label"] = labels[mis_mask]
    mis_df["pred_label"] = preds[mis_mask]
    mis_df["loss"] = ce[mis_mask]
    mis_df["logits"] = [log.tolist() for log in logits[mis_mask]]
    mis_df["probs"] = [p.tolist() for p in probs[mis_mask]]
    mis_path = out_dir / f"misclassified_{name}.csv"
    mis_df.to_csv(mis_path, index=False)

    # Per-source metrics (if available)
    per_source = {}
    if "source_ds" in df.columns:
        for s, g in df.groupby("source_ds"):
            rr = eval_step(model, g, tokenizer, device, max_length, batch_size, num_workers)
            m = compute_basic_metrics(rr["labels"], rr["preds"], label_list)
            per_source[str(s)] = m

    return (
        {
            "split": name,
            "overall": metrics,
            "per_source": per_source,
            "misclassified_csv": str(mis_path.resolve()),
            "sizes": {"total": int(len(df))},
        },
        mis_df,
    )


# -------------------------------
# Global summary writer
# -------------------------------

def flatten(d: dict, prefix="") -> dict:
    """Flatten nested dict into dot.notation columns."""
    out = {}
    for k, v in d.items():
        kk = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten(v, kk))
        else:
            out[kk] = v
    return out

def append_overall_csv(models_root: Path, row: dict):
    idx_path = models_root / "overall_evaluations.csv"
    write_header = not idx_path.exists()
    df = pd.DataFrame([row])
    if write_header:
        df.to_csv(idx_path, index=False)
    else:
        df.to_csv(idx_path, mode="a", header=False, index=False)


# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", help="Project config (used for paths & labels).")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model run folder or to its 'best'/'last' subdir, or a HF hub id.")
    parser.add_argument("--last", action="store_true",
                        help="Evaluate the most recent model run found under models_root (ignores --checkpoint).")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val_path", default="data/processed/val.parquet")
    parser.add_argument("--test_path", default="data/processed/test.parquet")
    parser.add_argument("--include_plots", action="store_true")
    parser.add_argument("--run_tag", default=None,
                        help="Optional short tag for this eval. Folder is eval/<YYYYMMDD-HHMM[_tag]>/.")
    args = parser.parse_args()

    # Load config & infer models_root
    cfg = load_yaml(args.config)
    models_root = Path(cfg.get("paths", {}).get("models_dir", "models"))

    # Figure out which model to evaluate
    if args.last:
        run_dir = pick_latest_model_run(models_root)
        ckpt_dir = resolve_checkpoint_dir(str(run_dir))  # will prefer 'best' inside it
        model_run_dir = run_dir
    else:
        if not args.checkpoint:
            raise ValueError("Provide --checkpoint or use --last to auto-pick the newest model run.")
        ckpt_dir = resolve_checkpoint_dir(args.checkpoint)
        # Determine the parent run dir for eval placement
        p = Path(args.checkpoint)
        if p.exists():
            model_run_dir = p if is_model_run_dir(p) else p.parent
        else:
            # hub id case: use models_root/<sanitized_name> as a logical run dir
            model_run_dir = models_root / p.name.replace("/", "-")
            ensure_dir(model_run_dir)
    print(f"[checkpoint] {ckpt_dir}")

    # Create eval/<timestamp[_tag]> under the model run dir
    eval_root = Path(model_run_dir) / "eval"
    ensure_dir(eval_root)
    stamp = now_stamp()
    eval_dir = eval_root / (f"{stamp}_{args.run_tag}" if args.run_tag else stamp)
    ensure_dir(eval_dir)
    print(f"[eval_dir]   {eval_dir}")

    # Labels & names
    num_labels = int(cfg.get("model", {}).get("num_labels", 3))
    label_canonical = cfg.get("data", {}).get("label_canonical", {"negative": 0, "neutral": 1, "positive": 2})
    inv_map = {v: k for k, v in label_canonical.items()}
    label_list = list(range(num_labels))
    label_names = {i: inv_map.get(i, str(i)) for i in label_list}

    # Load model & tokenizer
    model_src = str(ckpt_dir)
    if Path(ckpt_dir).exists():
        tokenizer = AutoTokenizer.from_pretrained(model_src, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_src, num_labels=num_labels)
    else:
        # hub id
        tokenizer = AutoTokenizer.from_pretrained(model_src, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_src, num_labels=num_labels)

    device = torch.device(args.device)
    model.to(device)

    # Load splits
    val_df = pd.read_parquet(args.val_path)
    test_df = pd.read_parquet(args.test_path)

    # Max length from config (for tokenization during eval)
    max_len = int(cfg.get("model", {}).get("max_length", 64))

    # Evaluate splits
    val_metrics, val_mis = evaluate_split(
        "val", val_df, model, tokenizer, device, max_len,
        args.batch_size, args.num_workers, label_list, label_names, eval_dir, args.include_plots
    )
    test_metrics, test_mis = evaluate_split(
        "test", test_df, model, tokenizer, device, max_len,
        args.batch_size, args.num_workers, label_list, label_names, eval_dir, args.include_plots
    )

    # Save JSON metrics
    (eval_dir / "metrics_val.json").write_text(json.dumps(val_metrics, indent=2))
    (eval_dir / "metrics_test.json").write_text(json.dumps(test_metrics, indent=2))

    # Save hardest 100 across both
    both = pd.concat([val_mis.assign(split="val"), test_mis.assign(split="test")], ignore_index=True)
    hardest = both.sort_values("loss", ascending=False).head(100)
    hardest.to_csv(eval_dir / "hard_cases.csv", index=False)

    # Overview CSV
    pd.DataFrame([
        {"split": "val",
         "accuracy": val_metrics["overall"]["accuracy"],
         "f1_macro": val_metrics["overall"]["f1_macro"],
         "f1_micro": val_metrics["overall"]["f1_micro"]},
        {"split": "test",
         "accuracy": test_metrics["overall"]["accuracy"],
         "f1_macro": test_metrics["overall"]["f1_macro"],
         "f1_micro": test_metrics["overall"]["f1_micro"]},
    ]).to_csv(eval_dir / "metrics_overview.csv", index=False)

    print(f"[OK] wrote -> {eval_dir/'metrics_test.json'}")
    print(f"[OK] wrote -> {eval_dir/'confusion_matrix.png'} (test)")
    print(f"[OK] wrote -> {eval_dir/'hard_cases.csv'}")

    # ---------------------------------------------------------
    # Append/update global overall CSV at models_root
    # Collect parameters from params.json if present in run dir
    # ---------------------------------------------------------
    params_path = Path(model_run_dir) / "params.json"
    params_flat = {}
    if params_path.exists():
        try:
            params = json.loads(params_path.read_text())
            # flatten full params (config + meta)
            if isinstance(params, dict):
                params_flat = {}
                for key in ("config", "meta"):
                    if key in params and isinstance(params[key], dict):
                        params_flat.update(flatten(params[key], prefix=key))
        except Exception:
            pass

    # Build one summary row
    model_run_name = Path(model_run_dir).name
    summary_row = {
        "model_run": model_run_name,
        "eval_dir": str(eval_dir.resolve()),
        "val_accuracy": val_metrics["overall"]["accuracy"],
        "val_f1_macro": val_metrics["overall"]["f1_macro"],
        "val_f1_micro": val_metrics["overall"]["f1_micro"],
        "test_accuracy": test_metrics["overall"]["accuracy"],
        "test_f1_macro": test_metrics["overall"]["f1_macro"],
        "test_f1_micro": test_metrics["overall"]["f1_micro"],
    }
    # Merge flattened parameters (each param becomes its own column)
    summary_row.update(params_flat)

    append_overall_csv(models_root, summary_row)
    print(f"[OK] appended -> {models_root/'overall_evaluations.csv'}")


if __name__ == "__main__":
    main()
