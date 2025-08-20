from __future__ import annotations
import os
import json
from xml.parsers.expat import model
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset as HFDataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

# ---------------------------
# Config helpers
# ---------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def cfg_get(dct, path, default=None):
    cur = dct
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def maybe_get_encoder_module(model):
    """
    Try to find the encoder/backbone module across common architectures.
    Returns the module or None if not found.
    """
    for attr in ("base_model", "bert", "roberta", "distilbert", "xlm_roberta", "deberta", "deberta_v2"):
        if hasattr(model, attr):
            return getattr(model, attr)
    return None

def freeze_encoder(model):
    enc = maybe_get_encoder_module(model)
    if enc is None:
        return
    for p in enc.parameters():
        p.requires_grad = False
 


# ---------------------------
# Tokenizer + collator
# ---------------------------
def get_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None and hasattr(tok, "eos_token"):
        tok.pad_token = tok.eos_token
    return tok

def build_collator(tokenizer) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")


# ---------------------------
# HF Dataset builder (from existing parquet)
# ---------------------------
def hf_from_parquet(df: pd.DataFrame, text_col: str, label_col: str) -> HFDataset:
    use_df = df[[text_col, label_col]].rename(columns={label_col: "labels"})
    return HFDataset.from_pandas(use_df, preserve_index=False)

def tokenize_ds(ds: HFDataset, tokenizer, max_length: int) -> HFDataset:
    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )
    ds = ds.map(_tok, batched=True, remove_columns=["text"])
    ds = ds.with_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


# ---------------------------
# Class weights (address imbalance)
# ---------------------------
def compute_class_weights(train_df: pd.DataFrame, label_col: str) -> torch.Tensor:
    counts = train_df[label_col].value_counts().sort_index().to_numpy()
    weights = counts.sum() / (len(counts) * counts)  # inverse frequency, normalized
    return torch.tensor(weights, dtype=torch.float)


# ---------------------------
# Metrics (accuracy, macro-F1, per-class F1)
# ---------------------------
def build_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro")
        f1_per = f1_score(labels, preds, average=None, labels=[0, 1, 2])
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_neg": f1_per[0],
            "f1_neu": f1_per[1],
            "f1_pos": f1_per[2],
        }
    return compute_metrics


# ---------------------------
# Weighted Trainer (uses class-weighted CE)
# ---------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------
# Main
# ---------------------------
def main():


    # ---- Device check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
    else:
        print("No GPU detected, training will run on CPU")


    # ---- Load config
    cfg = load_config("configs/config.yaml")
    set_seed(int(cfg_get(cfg, "data.split.seed", 42)))

    proc_dir   = cfg_get(cfg, "paths.proc_dir", "data/processed")
    models_dir = cfg_get(cfg, "paths.models_dir", "models")
    runs_dir   = cfg_get(cfg, "paths.runs_dir", "runs")
    text_col   = cfg_get(cfg, "data.text_col", "text")      
    label_col  = cfg_get(cfg, "data.label_col", "label")        
       

    model_name = cfg_get(cfg, "model.name", "bert-base-uncased")   
    max_len    = int(cfg_get(cfg, "model.max_length", 64))          
    num_labels   = int(cfg_get(cfg, "data.num_labels", 3))  # :contentReference[oaicite:8]{index=8}

    
    epochs     = int(cfg_get(cfg, "train.epochs", 5))
    lr         = float(cfg_get(cfg, "train.lr", 2e-5))
    weight_decay = float(cfg_get(cfg, "train.weight_decay", 0.01))
    warmup_ratio = float(cfg_get(cfg, "train.warmup_ratio", 0.1))
    train_bs     = int(cfg_get(cfg, "train.batch_size", 64))
    eval_bs      = int(cfg_get(cfg, "train.eval_batch_size", 128))
    fp16         = bool(cfg_get(cfg, "train.fp16", True))
    early_pat    = int(cfg_get(cfg, "train.early_stopping_patience", 3))
    metric_best  = cfg_get(cfg, "train.metric_for_best_model", "f1_macro")  # :contentReference[oaicite:7]{index=7}
    

    # ---- IO
    ensure_dir("runs")
    ensure_dir("models")
    best_dir = os.path.join("models", "best")

    # ---- Load parquet produced by your dataset pipeline
    train_df = pd.read_parquet(os.path.join(proc_dir, "train.parquet"))
    val_df   = pd.read_parquet(os.path.join(proc_dir, "val.parquet"))

    # ---- Build tokenizer & collator
    tokenizer = get_tokenizer(model_name)
    collator  = build_collator(tokenizer)

    # ---- Build HF datasets
    train_hf = hf_from_parquet(train_df, text_col, label_col)
    val_hf   = hf_from_parquet(val_df,   text_col, label_col)

    train_hf = tokenize_ds(train_hf, tokenizer, max_len)
    val_hf   = tokenize_ds(val_hf,   tokenizer, max_len)

    # ---- Class weights to address imbalance
    class_weights = compute_class_weights(train_df, label_col)

    # ---- Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    # ---- Training args
    args = TrainingArguments(
        output_dir="models/checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model=metric_best,
        greater_is_better=True,
        save_total_limit=2,
        fp16=fp16,
        report_to=["tensorboard"],
        logging_dir="runs",
        dataloader_num_workers=2,
        remove_unused_columns=False,  # we already control columns
    )

    # ---- Trainer (+ early stopping + weighted loss)
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=build_compute_metrics(),
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_pat)],
    )

    # ---- Train
    trainer.train()

    # ---- Save best to models/best/
    ensure_dir(best_dir)
    trainer.save_model(best_dir)                 # weights + config
    if trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(best_dir)

    print(f"\nBest model saved to: {best_dir}")
    print("TensorBoard logs in: runs/  (launch with: tensorboard --logdir runs/)")


if __name__ == "__main__":
    main()
