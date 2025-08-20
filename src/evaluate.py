# src/evaluate.py
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


# -------------------------------
# IO / Config
# -------------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


# -------------------------------
# Dataset
# -------------------------------
class SimpleTextDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        text_col: str = "text",
        label_col: str = "label",
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = text_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.label_col in row and pd.notna(row[self.label_col]):
            item["labels"] = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        item["_idx"] = torch.tensor(idx, dtype=torch.long)  # for back-reference
        return item


def make_loader(df, tokenizer, max_length, batch_size, num_workers=2, shuffle=False):
    ds = SimpleTextDataset(df, tokenizer, max_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# -------------------------------
# Evaluation core
# -------------------------------
@torch.no_grad()
def run_inference(
    model,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels, all_indices = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k not in ["_idx"]}
        idxs = batch.pop("labels", None)
        # We kept true labels; but we also want to track dataset row indices
        # -> bring them separately:
        # (labels are already popped; we still need the true labels)
        true_labels = idxs
        # recover indices from batch (added as _idx in dataset)
        # we need to fetch from the original un-moved batch
        # Workaround: DataLoader collates "_idx"; pull it from loader again:
        # Better: re-add "_idx" to the batch dict before moving to device
        # So instead, we pass indices via a different handle:
        # Just re-run tokenizer? No. Simpler: loader dataset stores order; we rely on enumeration
        # To be robust, we’ll attach indices to attention_mask’s shape—too hacky.
        # Clean way: put _idx back:
        # (We still have access in outer scope? No.)
        # Quick fix: change dataset/getitem to put _idx and keep it out of 'batch' moved to device.
        raise NotImplementedError(
            "Internal error: expected _idx to be available. Please use eval_step below."
        )


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
    ds = SimpleTextDataset(df, tokenizer, max_length)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    logits_list, labels_list, idx_list = [], [], []
    for batch in loader:
        idxs = batch["_idx"].numpy()
        idx_list.append(idxs)

        # move inputs that belong on device
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


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_rowwise(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # logits -> probabilities via softmax for numerical stability
    log_probs = logits - logits.max(axis=1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=1, keepdims=True) + 1e-12)
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
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "per_class": {
            str(lbl): {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(support[i])}
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
    # confidence = max class prob
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
            accs.append(correct[m].mean())
            confs.append(conf[m].mean())
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
    # One-vs-rest ROC-AUC (macro)
    y_true_bin = np.zeros((len(y_true), len(labels)), dtype=int)
    for i, lbl in enumerate(labels):
        y_true_bin[:, i] = (y_true == lbl).astype(int)
    # AUC per class (if both positives and negatives exist)
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
# End-to-end evaluate
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
) -> Dict:
    res = eval_step(model, df, tokenizer, device, max_length, batch_size, num_workers)
    logits, probs, preds, labels = res["logits"], res["probs"], res["preds"], res["labels"]

    metrics = compute_basic_metrics(labels, preds, label_list)
    # save confusion matrix for test split (per DoD)
    if name == "test":
        cm_path = os.path.join(out_dir, "confusion_matrix.png")
        plot_confusion_matrix(np.array(metrics["confusion_matrix"]), [label_names[i] for i in label_list], cm_path, normalize=False)

    # optional plots
    if include_plots:
        rel_path = os.path.join(out_dir, f"{name}_reliability.png")
        plot_reliability(probs, labels, rel_path)
        roc_path = os.path.join(out_dir, f"{name}_roc_auc.png")
        plot_ovr_roc(probs, labels, label_list, roc_path)

    # misclassified with logits
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

    # per-source metrics & cross-domain
    per_source = {}
    if "source_ds" in df.columns:
        for s, g in df.groupby("source_ds"):
            rr = eval_step(model, g, tokenizer, device, max_length, batch_size, num_workers)
            m = compute_basic_metrics(rr["labels"], rr["preds"], label_list)
            per_source[str(s)] = m

    return {
        "split": name,
        "overall": metrics,
        "per_source": per_source,
        "misclassified_csv": os.path.abspath(mis_csv),
        "sizes": {"total": int(len(df))},
    }, mis_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="models/best")  # HF model dir or hub id
    parser.add_argument("--val_path", default="data/processed/val.parquet")
    parser.add_argument("--test_path", default="data/processed/test.parquet")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="runs")
    parser.add_argument("--include_plots", action="store_true")  # reliability & ROC
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dir(args.out_dir)

    # labels & names
    num_labels = int(cfg.get("model", {}).get("num_labels", 3))
    label_canonical = cfg.get("data", {}).get("label_canonical", {"negative": 0, "neutral": 1, "positive": 2})
    # reverse map for nicer axis labels if available
    inv_map = {v: k for k, v in label_canonical.items()}
    label_list = list(range(num_labels))
    label_names = {i: inv_map.get(i, str(i)) for i in label_list}

    # load model & tokenizer
    model_name = cfg.get("model", {}).get("name", args.checkpoint)
    max_len = int(cfg.get("model", {}).get("max_length", 64))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint if Path(args.checkpoint).exists() else model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint if Path(args.checkpoint).exists() else model_name,
        num_labels=num_labels,
    )
    device = torch.device(args.device)
    model.to(device)

    # load data
    val_df = pd.read_parquet(args.val_path)
    test_df = pd.read_parquet(args.test_path)

    # run evaluations
    results = {}

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
        args.out_dir,
        args.include_plots,
    )
    results["val"] = val_metrics

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
        args.out_dir,
        args.include_plots,
    )
    results["test"] = test_metrics

    # Save required artifacts
    # 1) JSON
    metrics_test_json = os.path.join(args.out_dir, "metrics_test.json")
    with open(metrics_test_json, "w") as f:
        json.dump(results["test"], f, indent=2)
    # also drop val metrics alongside
    with open(os.path.join(args.out_dir, "metrics_val.json"), "w") as f:
        json.dump(results["val"], f, indent=2)

    # 2) Hardest 100 misclassified (by CE) across val+test
    both = pd.concat([val_mis.assign(split="val"), test_mis.assign(split="test")], axis=0, ignore_index=True)
    hardest = both.sort_values("loss", ascending=False).head(100)
    hard_csv = os.path.join(args.out_dir, "hard_cases.csv")
    hardest.to_csv(hard_csv, index=False)

    # 3) Also export a compact CSV of overall metrics for quick scanning
    rows = []
    for split in ["val", "test"]:
        ov = results[split]["overall"]
        rows.append(
            {
                "split": split,
                "accuracy": ov["accuracy"],
                "f1_macro": ov["f1_macro"],
                "f1_micro": ov["f1_micro"],
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "metrics_overview.csv"), index=False)

    print(f"[OK] Wrote: {metrics_test_json}")
    print(f"[OK] Wrote: {os.path.join(args.out_dir, 'confusion_matrix.png')} (test)")
    print(f"[OK] Wrote: {hard_csv}")
    print(f"[OK] Misclassified CSVs: {results['val']['misclassified_csv']}, {results['test']['misclassified_csv']}")


if __name__ == "__main__":
    main()
