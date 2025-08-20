#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal-change training script.

- Reads settings from configs/config.yaml
- Trains one or more HF models
- Saves each run under: {paths.models_dir}/{model-name}_{YYYYMMDD-HHMM}/
  with subfolders: best/, last/, tokenizer/ and logs/metadata.

Assumes:
  data/processed/train.parquet
  data/processed/val.parquet
(or override via CLI).
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

# ------------------------
# Config / utils
# ------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M")

def make_run_dir(models_root: str, model_name: str) -> str:
    # "bert-base-uncased_20250820-1538"
    safe = model_name.replace("/", "-")
    base = f"{safe}_{now_stamp()}"
    d = Path(models_root) / base
    i, out = 1, d
    while out.exists():  # extremely unlikely, but safe
        i += 1
        out = Path(f"{str(d)}_v{i}")
    out.mkdir(parents=True, exist_ok=False)
    # subdirs
    (out / "best").mkdir(exist_ok=True)
    (out / "last").mkdir(exist_ok=True)
    (out / "tokenizer").mkdir(exist_ok=True)
    return str(out)

def dump_params(out_dir: str, cfg: dict, args: argparse.Namespace, model_name: str):
    meta = {
        "timestamp": dt.datetime.now().isoformat(),
        "model_name": model_name,
        "args": vars(args),
        "config_hash": hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({"config": cfg, "meta": meta}, f, indent=2)

# ------------------------
# Dataset
# ------------------------

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
        item["labels"] = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return item

def make_loader(df, tokenizer, max_length, batch_size, shuffle, num_workers=4):
    ds = SimpleTextDataset(df, tokenizer, max_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

# ------------------------
# Metrics
# ------------------------

@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        out = model(**inputs)
        logits = out.logits.detach().cpu().numpy()
        y_hat = logits.argmax(axis=1)
        y = inputs["labels"].detach().cpu().numpy()
        preds.append(y_hat)
        labels.append(y)
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }

# ------------------------
# Training loop
# ------------------------

def train_one(
    cfg: dict,
    model_name: str,
    models_root: str,
    train_path: str,
    val_path: str,
    device_str: str,
):
    # --- config bits
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    data_cfg  = cfg.get("data", {})

    text_col  = data_cfg.get("text_col", "text")
    label_col = data_cfg.get("label_col", "label")

    num_labels = int(model_cfg.get("num_labels", 3))
    max_len    = int(model_cfg.get("max_length", 64))
    freeze_enc = bool(model_cfg.get("freeze_encoder", False))

    batch_size = int(train_cfg.get("batch_size", 64))
    eval_bs    = int(train_cfg.get("eval_batch_size", max(64, batch_size)))
    epochs     = int(train_cfg.get("epochs", 5))
    lr         = float(train_cfg.get("lr", 2e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    grad_acc_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
    fp16       = bool(train_cfg.get("fp16", False))
    patience   = int(train_cfg.get("early_stopping_patience", 3))
    metric_best = str(train_cfg.get("metric_for_best_model", "f1_macro"))
    use_class_w = bool(train_cfg.get("use_class_weights", False))
    num_workers = int(train_cfg.get("num_workers", 4))

    # seed
    seed = int(cfg.get("data", {}).get("split", {}).get("seed", 42))
    set_seed(seed)

    # --- output dir
    # uses paths.models_dir from config for parent
    models_root = models_root or cfg.get("paths", {}).get("models_dir", "models")
    out_dir = make_run_dir(models_root, model_name)
    print(f"[run] {Path(out_dir).name}")
    print(f"[dir] {out_dir}")

    # provenance
    dump_params(out_dir, cfg, argparse.Namespace(), model_name)
    # keep a copy of the config
    try:
        # Config path might be different; best-effort copy if exists
        default_cfg_path = Path("configs/config.yaml")
        if default_cfg_path.exists():
            shutil.copy2(str(default_cfg_path), os.path.join(out_dir, "config.yaml"))
    except Exception:
        pass

    # --- data
    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if freeze_enc:
        for p in model.base_model.parameters():
            p.requires_grad = False
    device = torch.device(device_str)
    model.to(device)

    train_loader = make_loader(train_df, tok, max_len, batch_size, True,  num_workers)
    val_loader   = make_loader(val_df,   tok, max_len, eval_bs,   False, num_workers)

    # class weights
    class_weights = None
    if use_class_w:
        counts = train_df[label_col].value_counts().sort_index()
        inv = 1.0 / (counts + 1e-6)
        class_weights = (inv / inv.sum() * len(counts)).to_numpy()
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # opt / sched
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = math.ceil(len(train_loader) / max(1, grad_acc_steps)) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=fp16 and device.type == "cuda")

    # train
    best_score = -1e9
    best_epoch = -1
    patience_left = patience
    log_rows: List[Dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"[{model_name}] epoch {epoch}/{epochs}", leave=False)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=fp16 and device.type == "cuda"):
                out = model(**inputs)
                loss = out.loss
                if class_weights is not None:
                    ce = F.cross_entropy(out.logits, inputs["labels"], weight=class_weights)
                    loss = ce
                loss = loss / max(1, grad_acc_steps)

            scaler.scale(loss).backward()
            if step % grad_acc_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            bs = inputs["labels"].size(0)
            running_loss += loss.item() * bs
            n_seen += bs
            pbar.set_postfix(loss=f"{running_loss / max(1, n_seen):.4f}")

        # validate
        val_metrics = evaluate(model, val_loader, device)
        current = float(val_metrics.get(metric_best, -1e9))

        # log row
        row = {"epoch": epoch, "train_loss": running_loss / max(1, n_seen)}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(os.path.join(out_dir, "train_log.csv"), index=False)

        # save last
        model.save_pretrained(os.path.join(out_dir, "last"))
        tok.save_pretrained(os.path.join(out_dir, "last"))

        # track best
        improved = current > best_score
        if improved:
            best_score, best_epoch, patience_left = current, epoch, patience
            model.save_pretrained(os.path.join(out_dir, "best"))
            tok.save_pretrained(os.path.join(out_dir, "best"))
            with open(os.path.join(out_dir, "val_metrics_best.json"), "w") as f:
                json.dump({"epoch": epoch, "metrics": val_metrics}, f, indent=2)
        else:
            patience_left -= 1

        print(f"[{model_name}] epoch {epoch}: val {metric_best}={current:.4f} "
              f"{'**BEST**' if improved else f'(best={best_score:.4f} @ {best_epoch})'}")

        if patience_left <= 0:
            print(f"[{model_name}] early stopping at epoch {epoch}. best @ {best_epoch}.")
            break

    print(f"[{model_name}] done. best {metric_best}={best_score:.4f} @ epoch {best_epoch}")
    return out_dir, best_score, best_epoch

# ------------------------
# Main
# ------------------------

def parse_models_arg(models: str | None, cfg_default: str) -> List[str]:
    if not models:
        return [cfg_default]
    return [m.strip() for m in models.split(",") if m.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--models", default=None,
                        help="Comma-separated HF model names. If omitted, uses model.name from config.")
    parser.add_argument("--models_root", default=None,
                        help="Parent folder for trained models. Defaults to paths.models_dir in config.")
    parser.add_argument("--train_path", default="data/processed/train.parquet")
    parser.add_argument("--val_path", default="data/processed/val.parquet")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    default_model = cfg.get("model", {}).get("name", "bert-base-uncased")
    model_list = parse_models_arg(args.models, default_model)

    models_root = args.models_root or cfg.get("paths", {}).get("models_dir", "models")

    results = []
    for m in model_list:
        out_dir, best_score, best_epoch = train_one(
            cfg=cfg,
            model_name=m,
            models_root=models_root,
            train_path=args.train_path,
            val_path=args.val_path,
            device_str=args.device,
        )
        results.append({"model": m, "dir": out_dir, "best_metric": best_score, "best_epoch": best_epoch})

    # brief summary
    df = pd.DataFrame(results).sort_values("best_metric", ascending=False)
    print("\n=== Summary ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
