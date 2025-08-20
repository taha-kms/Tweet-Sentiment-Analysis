#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a text classification model on val/test splits.

Features
- Metrics: accuracy, F1 (macro/micro), per-class P/R/F1, confusion matrix.
- Per-source metrics (if `source_ds` column exists).
- Misclassified rows with logits/probs saved to CSV.
- Optional reliability diagram & ROC-AUC (OvR) plots.
- Artifacts written to a unique timestamped run directory:
  runs/<YYYYMMDD-HHMMSS_[optional-name-or-auto]>/
    - params.json (config + env + args)
    - config.yaml (exact copy used)
    - metrics_val.json, metrics_test.json, metrics_overview.csv
    - misclassified_val.csv, misclassified_test.csv
    - hard_cases.csv (top-100 by cross-entropy)
    - confusion_matrix.png (test)
    - *_reliability.png / *_roc_auc.png (if --include_plots)
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
# I/O & run management
# -------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def timestamped_run_name(cfg: dict, base_name: str | None, args: argparse.Namespace) -> str:
    """Always prefix with timestamp; if base_name provided, append it."""
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if base_name:
        return f"{ts}_{base_name}"
    m = cfg.get("model", {})
    t = cfg.get("train", {})
    data_seed = cfg.get("data", {}).get("split", {}).get("seed", "?")
    auto = f"model={m.get('name','?')}_lr={t.get('lr','?')}_bs={t.get('batch_size','?')}_len={m.get('max_length','?')}_seed={data_seed}"
    return f"{ts}_{auto}"


def create_run_dir(out_root: str, run_name: str) -> str:
    """Create run directory; if it somehow exists, add _vN (extremely unlikely with timestamps)."""
    base = Path(out_root) / run_name
    out = base
    v = 1
    while out.exists():
        v += 1
        out = Path(f"{base}_v{v}")
    out.mkdir(parents=True, exist_ok=False)
    return str(out)


def dump_params(out_dir: str, cfg: dict, args: argparse.Namespace, notes: str):
    meta = {
        "timestamp": dt.datetime.now().isoformat(),
        "notes": notes,
        "args": vars(args),
        "config_hash": hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "sklearn": __import__("sklearn").__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({"config": cfg, "meta": meta}, f, indent=2)


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
        item["_idx"] = torch.tensor(idx, dtype=torch.long)  # track original row index
        return item


def make_loader(df, tokenizer, max_length, batch_size, num_workers=4):
    ds = SimpleTextDataset(df, tokenizer, max_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# -------------------------------
# Core evaluation helpers
# -------------------------------

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def cross_entropy_rowwise(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cross-entropy per example from raw logits and integer labels."""
    x = logits - logits.max(axis=1, keepdims=True)
    log_probs = x - np.log(np.exp(x).sum(axis=1, keepdims=True) + 1e-12)
    return -log_probs[np.arange(len(labels)), labels]


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

    return {
        "logits": logits,
        "probs": probs,
        "preds": preds,
        "labels": labels,
        "indices": indices,
    }


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


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str, normalize: bool = False):
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


def plot_reliability(probs: np.ndarray, y_true: np.ndarray, out_path: str, n_bins: int = 15):
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
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


def plot_ovr_roc(probs: np.ndarray, y_true: np.ndarray, labels: List[int], out_path: str):
    y_true_bin = np.zeros((len(y_true), len(labels)), dtype=int)
    for i, lbl in enumerate(labels):
        y_true_bin[:, i] = (y_true == lbl).astype(int)

    aucs = []
    for i, lbl in enumerate(labels):
        if y_true_bin[:, i].sum() == 0 or y_true_bin[:, i].sum() == len(y_true_bin):
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
    plt.xlabel("Class")
    plt.ylabel("AUC")
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


# -------------------------------
# Split-level evaluation
# -------------------------------

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
    out_dir: str,
    include_plots: bool,
) -> Tuple[Dict, pd.DataFrame]:
    res = eval_step(model, df, tokenizer, device, max_length, batch_size, num_workers)
    logits, probs, preds, labels = res["logits"], res["probs"], res["preds"], res["labels"]

    metrics = compute_basic_metrics(labels, preds, label_list)

    # Save confusion matrix for test split as per DoD
    if name == "test":
        cm_path = os.path.join(out_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]),
            [label_names[i] for i in label_list],
            cm_path,
            normalize=False,
        )

    # Optional plots
    if include_plots:
        plot_reliability(probs, labels, os.path.join(out_dir, f"{name}_reliability.png"))
        plot_ovr_roc(probs, labels, label_list, os.path.join(out_dir, f"{name}_roc_auc.png"))

    # Misclassified with logits/probs
    ce = cross_entropy_rowwise(logits, labels)
    mis_mask = preds != labels
    mis_df = df.iloc[res["indices"][mis_mask]].copy()
    mis_df["true_label"] = labels[mis_mask]
    mis_df["pred_label"] = preds[mis_mask]
    mis_df["loss"] = ce[mis_mask]
    mis_df["logits"] = [log.tolist() for log in logits[mis_mask]]
    mis_df["probs"] = [p.tolist() for p in probs[mis_mask]]
    mis_csv = os.path.join(out_dir, f"misclassified_{name}.csv")
    mis_df.to_csv(mis_csv, index=False)

    # Per-source metrics
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
            "misclassified_csv": os.path.abspath(mis_csv),
            "sizes": {"total": int(len(df))},
        },
        mis_df,
    )


# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="models/best")  # HF model dir or hub id
    parser.add_argument("--val_path", default="data/processed/val.parquet")
    parser.add_argument("--test_path", default="data/processed/test.parquet")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include_plots", action="store_true")
    # run management
    parser.add_argument("--out_root", default="runs")
    parser.add_argument("--run_name", default=None, help="Optional human-readable suffix for the run folder")
    parser.add_argument("--notes", default="", help="Free-form notes to store in params.json")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Build unique run dir
    run_name = timestamped_run_name(cfg, args.run_name, args)
    out_dir = create_run_dir(args.out_root, run_name)
    ensure_dir(out_dir)

    # Save provenance
    dump_params(out_dir, cfg, args, args.notes)
    # store an exact copy of the config used
    try:
        shutil.copy2(args.config, os.path.join(out_dir, "config.yaml"))
    except Exception:
        pass

    # Labels & names
    num_labels = int(cfg.get("model", {}).get("num_labels", 3))
    label_canonical = cfg.get("data", {}).get(
        "label_canonical", {"negative": 0, "neutral": 1, "positive": 2}
    )
    inv_map = {v: k for k, v in label_canonical.items()}
    label_list = list(range(num_labels))
    label_names = {i: inv_map.get(i, str(i)) for i in label_list}

    # Load model & tokenizer
    model_name = cfg.get("model", {}).get("name", args.checkpoint)
    max_len = int(cfg.get("model", {}).get("max_length", 64))
    tok_src = args.checkpoint if Path(args.checkpoint).exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        tok_src, num_labels=num_labels
    )
    device = torch.device(args.device)
    model.to(device)

    # Load splits
    val_df = pd.read_parquet(args.val_path)
    test_df = pd.read_parquet(args.test_path)

    print(f"[run] {run_name}")
    print(f"[dir] {out_dir}")
    print(f"[val] {len(val_df):,} rows  | [test] {len(test_df):,} rows")

    # Evaluate
    val_metrics, val_mis = evaluate_split(
        "val",
        val_df,
        model,
        tokenizer,
        device,
        max_len,
        args.batch_size,
        args.num_workers,
        label_list,
        label_names,
        out_dir,
        args.include_plots,
    )
    test_metrics, test_mis = evaluate_split(
        "test",
        test_df,
        model,
        tokenizer,
        device,
        max_len,
        args.batch_size,
        args.num_workers,
        label_list,
        label_names,
        out_dir,
        args.include_plots,
    )

    # Save required artifacts
    with open(os.path.join(out_dir, "metrics_val.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(out_dir, "metrics_test.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    both = pd.concat(
        [val_mis.assign(split="val"), test_mis.assign(split="test")],
        axis=0,
        ignore_index=True,
    )
    hardest = both.sort_values("loss", ascending=False).head(100)
    hardest.to_csv(os.path.join(out_dir, "hard_cases.csv"), index=False)

    # Compact overview CSV
    rows = []
    for split, metr in [("val", val_metrics), ("test", test_metrics)]:
        ov = metr["overall"]
        rows.append(
            {
                "split": split,
                "accuracy": ov["accuracy"],
                "f1_macro": ov["f1_macro"],
                "f1_micro": ov["f1_micro"],
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "metrics_overview.csv"), index=False)

    print(f"[OK] wrote -> {os.path.join(out_dir, 'metrics_test.json')}")
    print(f"[OK] wrote -> {os.path.join(out_dir, 'confusion_matrix.png')} (test)")
    print(f"[OK] wrote -> {os.path.join(out_dir, 'hard_cases.csv')}")
    print(f"[OK] misclassified -> {val_metrics['misclassified_csv']}, {test_metrics['misclassified_csv']}")


if __name__ == "__main__":
    main()
